# MIT License
#
# Copyright (c) 2021 Marek Jacob
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import xarray
import datetime
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import matplotlib as mpl
import matplotlib.ticker
import os
import abc
import cartopy.crs as ccrs
import geopy.distance

# Import some support utilities to help with the plots
from AbsoluteGrid import get_HAMP_label, AbsoluteGrid, center_to_edge


"""Investigate HAMP - GPM collocations

This script provides methods for a point-to-point comparison of HAMP
and GPM measurements and products.

This script is just a starting point. It provides functions to handle
the GPM data a bit easier.
The GPM data comes in a lot of different processing levels:
https://gpm.nasa.gov/data/directory
The files are in hdf5 format and use the Group feature a lot. However,
no verbose dimensions are used. To handle those inconvenience features,
helper functions like rename_phony_dims() and several open_*() functions
were written. Those function try to provide a useful xarray interface to
the GPM observations.


The GPM Core satellite:
https://pmm.nasa.gov/gpm/flight-project/core-observatory

"""


def rename_phony_dims(ds):
    """rename phony dims by the given DimensionNames

    Scans all variables in ds for the DimensionNames attribute and
    use this to decode "phony_dim_<num>" dimensions.

    Parameters
    ----------
    ds : {xarray.Dataset, xarray.DataArray}
        input dataset

    Returns
    -------
    {xarray.Dataset, xarray.DataArray}
        renamed dataset
    """

    ds = ds.copy()
    for key, var in ds.items():
        names = var.DimensionNames.split(",")
        ds[key] = var.rename(
            {phony_dim: name for phony_dim, name in zip(var.dims, names) if phony_dim.startswith("phony_dim")}
        )
        if hasattr(ds[key], "CodeMissingValue"):
            cmv = ds[key].CodeMissingValue
            int_cmv = int(cmv.split(".")[0])
            if ds[key].min() == int_cmv:
                print(f"Note: Variable `{key}' uses {int_cmv:d} instead if {cmv} as missing value.")
                ds[key] = ds[key].where(ds[key] != int_cmv)

    return ds


@np.vectorize
def time_converter_ymdhms(year, month, dayOfMonth, hour, minute, second, milliSecond):
    """Convert a time that is given by its individual components to np.datetime64"""
    # conversion through datetime is just for convenience and to be sure to get dates right.
    return np.datetime64(datetime.datetime(year, month, dayOfMonth, hour, minute, second, milliSecond * 1000))


def open_gmi(filename):
    """xarray.open_dataset wrapper for gmi hdf5"""
    ds_S1 = xarray.open_dataset(filename, group="S1")
    ds_S2 = xarray.open_dataset(filename, group="S2")

    ds_S1 = ds_S1.rename(phony_dim_18="S1_time", phony_dim_19="S1_across", phony_dim_21="S1_frequency")
    ds_S2 = ds_S2.rename(phony_dim_40="S2_time", phony_dim_41="S2_across", phony_dim_43="S2_frequency")

    ds_S1_ScanTime = xarray.open_dataset(filename, group="S1/ScanTime", decode_times=False, mask_and_scale=False)
    ds_S2_ScanTime = xarray.open_dataset(filename, group="S2/ScanTime", decode_times=False, mask_and_scale=False)

    assert np.all(ds_S1_ScanTime.MilliSecond.values == ds_S2_ScanTime.MilliSecond.values)
    assert np.all(ds_S1_ScanTime.DayOfMonth.values == ds_S2_ScanTime.DayOfMonth.values)
    assert np.all(ds_S1_ScanTime.Year.values == ds_S2_ScanTime.Year.values)

    datetime64 = time_converter_ymdhms(
        ds_S1_ScanTime.Year.values,
        ds_S1_ScanTime.Month.values,
        ds_S1_ScanTime.DayOfMonth.values,
        ds_S1_ScanTime.Hour.values,
        ds_S1_ScanTime.Minute.values,
        ds_S1_ScanTime.Second.values,
        ds_S1_ScanTime.MilliSecond.values,
    )

    ds = xarray.Dataset()
    for key in ds_S1.keys():
        ds["S1_" + key] = ds_S1[key]
        ds["S2_" + key] = ds_S2[key]

    ds["S1_time"] = datetime64
    ds["S2_time"] = datetime64

    ds["S1_frequency"] = [
        "10.65 V",
        "10.65 H",
        "18.70 V",
        "18.70 H",
        "23.80 V",
        "36.50 V",
        "36.50 H",
        "89.00 V",
        "89.00 H",
    ]  # GHz
    ds["S2_frequency"] = ["166.00 V", "166.00 H", "183.31+-3.0 V", "183.31+-7.0 V"]  # GHz

    ds = transfer_Wband_conical_to_nadir(ds)
    return ds


def transfer_Wband_conical_to_nadir(ds):
    """Convert the GMI measurements at 89 GHz to pseudo nadir measurements at 90 GHz

    The GMI scans at an earth-incidence-angle of 52.8 degree, while HAMP measures about
    straight nadir.
    This results in different length of the beam path through the atmosphere.
    However, this effect can be compensated for by using an empirical function. The
    coefficients of such functions can be found by comparing forward simulations
    of the different setup. Such function can than also compensate the little frequency
    difference between the HAMP W-band channel at 90 GHz and the GMI channels at 89 GHz
    with horizontal and vertical polarizations.
    One can use both or just one of the 89 GHz channels to convert it to a nadir-90-GHZ
    signal. Of course such conversion has some error, but it is never the less better to
    compare the converted GMI measurements with HAMP than the raw measurements.

    Here I suppose two conversion functions. One uses H and V, the other only one channel.
    There seems to be a definition difference between PAMTRA and GMI what is H and V.
    Thus we will apply a "H" correction on "V" measurements in the second correction.
    The converted GMI observations are stored in the "89.00 HV" and "89.00 HH" entries.
    """

    def fit_HV_pol(x, a, b, c, d, e):
        return a + b * x[0] + c * x[0] ** 2 + d * x[1] + e * x[1] ** 2

    def fit_mono_pol(x, a, b, c):
        return a + b * x + c * x ** 2

    param90 = [0.20539126, 3.80749043, -0.00869966, -2.39947559, 0.00718512]
    tb = xarray.zeros_like(ds.S1_Tb.sel(S1_frequency=["89.00 H"]))
    tb.values = fit_HV_pol(
        np.concatenate(
            [
                [ds.S1_Tb.sel(S1_frequency="89.00 V")],  # polarization is defined differently than in PAMTRA
                [ds.S1_Tb.sel(S1_frequency="89.00 H")],
            ]
        ),
        *param90,
    )[:, :, np.newaxis]
    tb = tb.assign_coords(S1_frequency=["89.00 HV"])
    tb_concat = xarray.concat((ds["S1_Tb"], tb), "S1_frequency")
    ds = ds.drop("S1_Tb").drop("S1_frequency")  # drop variables in order to extend them
    ds["S1_Tb"] = tb_concat

    param90_V = [5.40000658e02, -3.20591358e00, 8.01888816e-03]
    tb = xarray.zeros_like(ds.S1_Tb.sel(S1_frequency=["89.00 H"]))
    tb.values = fit_mono_pol(ds.S1_Tb.sel(S1_frequency=["89.00 H"]), *param90_V)
    tb = tb.assign_coords(
        S1_frequency=["89.00 HH"]
    )  # polarization is defined differently than in PAMTRA. Therefore we use H channel here with V function
    tb_concat = xarray.concat((ds["S1_Tb"], tb), "S1_frequency")
    ds = ds.drop("S1_Tb").drop("S1_frequency")  # drop variables in order to extend them
    ds["S1_Tb"] = tb_concat

    return ds


def open_gprof(filename):
    """xarray.open_dataset wrapper for gmi gprof L2A hdf5

    GPROF is a precipitation product."""
    ds_S1 = xarray.open_dataset(filename, group="S1")

    ds_S1 = rename_phony_dims(ds_S1)
    ds_S1 = ds_S1.rename(
        nscan="S1_time",
        npixel="S1_across",
    )

    ds_S1_ScanTime = xarray.open_dataset(filename, group="S1/ScanTime", decode_times=False, mask_and_scale=False)

    datetime64 = time_converter_ymdhms(
        ds_S1_ScanTime.Year.values,
        ds_S1_ScanTime.Month.values,
        ds_S1_ScanTime.DayOfMonth.values,
        ds_S1_ScanTime.Hour.values,
        ds_S1_ScanTime.Minute.values,
        ds_S1_ScanTime.Second.values,
        ds_S1_ScanTime.MilliSecond.values,
    )

    ds = xarray.Dataset()
    for key in ds_S1.keys():
        ds["S1_" + key] = ds_S1[key]

    ds["S1_time"] = datetime64

    return ds


def open_dpr_ka(filename):
    """xarray.open_dataset wrapper for dpr Ka-band precipitation radar KaPR hdf5

    The GPM radar data is stored in two different groups.
    These groups represent two different scan strategies.
    Originally these were the "main"-scan (MS) and the "high sensitivity" scan
    (HS) which was interlarded between the MS beams. However on  May 21, 2018,
    JAXA and NASA changed the scanning pattern of the KaPR such that the HS bins
    are used to sample the outer swath of the KuPR:
    See KaPR_scan_pattern.pdf or https://www.eorc.jaxa.jp/en/news/2020/nw200604.html
    However, this means, we have to handle a MS and a HS dataset with its
    individual set of coordinates and stuff.
    """

    def open_scan(S):
        """opener function to open HS or MS datasets"""
        ds = xarray.open_dataset(filename, group=S)

        ds = rename_phony_dims(ds)

        ds_ScanTime = xarray.open_dataset(filename, group=S + "/ScanTime", decode_times=False, mask_and_scale=False)

        datetime64 = time_converter_ymdhms(
            ds_ScanTime.Year.values,
            ds_ScanTime.Month.values,
            ds_ScanTime.DayOfMonth.values,
            ds_ScanTime.Hour.values,
            ds_ScanTime.Minute.values,
            ds_ScanTime.Second.values,
            ds_ScanTime.MilliSecond.values,
        )

        ds_SLV = xarray.open_dataset(filename, group=S + "/SLV")
        ds_SLV = rename_phony_dims(ds_SLV)

        ds_PRE = xarray.open_dataset(filename, group=S + "/PRE")
        ds_PRE = rename_phony_dims(ds_PRE)

        ds = ds.merge(ds_SLV)
        ds = ds.merge(ds_PRE)

        ds = ds.rename(nscan="time")
        ds["time"] = datetime64

        return ds

    ds_HS = open_scan("HS").rename(time="HS_time", nrayHS="HS_across")
    ds_MS = open_scan("MS").rename(time="MS_time", nrayMS="MS_across")

    assert list(ds_HS.keys()) == list(ds_MS.keys())
    ds = xarray.Dataset()
    for key in ds_HS.keys():
        ds["HS_" + key] = ds_HS[key]

        ds["MS_" + key] = ds_MS[key]

    ds = ds.rename(nbin="MS_nbin", nbinHS="HS_nbin")

    return ds


def open_imerg(filename):
    """Open and serve the IMERG product hdf5 files with xarray"""
    ds = xarray.open_dataset(
        filename,
        group="Grid",
        decode_cf=False,
        # use_cftime=False
    )
    assert ds.time.attrs["calendar"] == "julian"
    del ds.time.attrs[
        "calendar"
    ]  # remove this information as it confuses xarray resulting in an object(cftime.DatetimeJulian) time vector instead of np.datetime64
    ds = xarray.decode_cf(ds)

    assert ds.dims["time"] == 1
    ds = ds.isel(time=0)

    Longitude, Latitude = np.meshgrid(ds.lon, ds.lat)
    ds["Latitude"] = ("lat", "lon"), Latitude
    ds["Latitude"] = ds["Latitude"].transpose(*ds.precipitationCal.dims)
    ds["Longitude"] = ("lat", "lon"), Longitude
    ds["Longitude"] = ds["Longitude"].transpose(*ds.precipitationCal.dims)
    ds["HQobservationTime"] = ds.time + ds["HQobservationTime"]
    ds = ds.rename(**{key: "S1_" + key for key in ds.variables.keys()})
    return ds


def frequency_pairs(hamp, gmi_ds):
    """List of best matching frequencies of HAMP and GMI

    not used at the moment"""
    n = {"method": "nearest"}
    freq_pairs = [  # list of tuples
        (hamp.tb.sel(frequency=23.84, **n), gmi_ds.S1_Tb.sel(S1_frequency="23.80 V")),
        (hamp.tb.sel(frequency=90, **n), gmi_ds.S1_Tb.sel(S1_frequency="89.00 V")),
        (hamp.tb.sel(frequency=90, **n), gmi_ds.S1_Tb.sel(S1_frequency="89.00 H")),
        (hamp.tb.sel(frequency=31.4, **n), gmi_ds.S1_Tb.sel(S1_frequency="36.50 V")),
        (hamp.tb.sel(frequency=31.4, **n), gmi_ds.S1_Tb.sel(S1_frequency="36.50 H")),
        (hamp.tb.sel(frequency=186.81, **n), gmi_ds.S2_Tb.sel(S2_frequency="183.31+-3.0 V")),
        (hamp.tb.sel(frequency=190.81, **n), gmi_ds.S2_Tb.sel(S2_frequency="183.31+-7.0 V")),
    ]
    return freq_pairs


@np.vectorize
def distance(lat1, lon1, lat2, lon2):
    """return great circle distance between two lists of points"""
    gc = geopy.distance.great_circle((lat1, lon1), (lat2, lon2))
    return gc.kilometers


class GPM_kdtree(object):
    """Collocations between HAMP and GMP

    This class helps finding spatial collocations of the HAMP and GPM data sets.
    The key of this class is to use the scipy.spatial.cKDTree as fast search
    algorithm. The need for this class comes from the fact, that many GPM data
    sets like the GMI or KaPR use different scan patterns with different
    spatial grids. To handle these, we build one kdtree per scan pattern. The
    scan pattern is identified here by the variable "swath".

    """

    def __init__(self, gpm_ds):
        self.gpm_ds = gpm_ds

        self.kdtree = {}

        for var_name in gpm_ds.variables:
            if not var_name.endswith("_Longitude"):
                continue
            swath = var_name[: -len("_Longitude")]
            self.kdtree[swath] = scipy.spatial.cKDTree(
                np.asarray(
                    [
                        gpm_ds[swath + "_Latitude"].values.flatten(),
                        gpm_ds[swath + "_Longitude"].values.flatten(),
                    ]
                ).T,
                leafsize=10,
            )

    def query(self, hamp_ds, time_window=np.timedelta64(20, "m"), max_distance=10, swath=None):
        """get gpm_ds along hamp coordinates

        Parameters
        ----------
        hamp_ds : {xarray.Dataset}
            unified HAMP dataset
        max_distance : numeric
            Maximum distance in km.
        swath : 'S1', 'S2', 'HS', 'MS' or None
            either a specific swath pattern (S1, S2, HS, MS) or both (None)

        Returns
        -------
        A 2-tuple of HAMP and GPM xarray Datasets that share the same spatial
        domain. GPM data is selected spatially along the HALO track.
        """
        if swath is None:
            swath_set = self.kdtree.keys()
        else:
            swath_set = (swath,)

        gmi_time_mean = self.gpm_ds[swath_set[0] + "_time"].mean().values
        hamp_ds = hamp_ds.sel(time=slice(gmi_time_mean - time_window, gmi_time_mean + time_window))

        gpm_ds = self.gpm_ds

        for swath in swath_set:
            print("swath", swath)
            latlon_distance, indices1d = self.kdtree[swath].query(
                np.asarray([hamp_ds.lat.values, hamp_ds.lon.values]).T
            )

            gpm_ds = gpm_ds.stack(halo_time=self.gpm_ds[swath + "_Latitude"].dims)
            gpm_ds = gpm_ds.isel(halo_time=indices1d)

            d_km = distance(
                gpm_ds[swath + "_Latitude"].values,
                gpm_ds[swath + "_Longitude"].values,
                hamp_ds.lat.values,
                hamp_ds.lon.values,
            )
            hamp_ds = hamp_ds.isel(time=d_km <= max_distance)
            gpm_ds = gpm_ds.isel(halo_time=d_km <= max_distance)

            gpm_ds = gpm_ds.rename(halo_time=swath + "_halo_time")

        return hamp_ds, gpm_ds


def gpm_filter_area(
    gpm_ds,
    center_longitude=-57.717,
    center_latitude=13.3,
    margin=2,
):
    """Filter GMI Dataset

    return subset which fully covers the given circle area (and more)
    No sweeps are cropped
    """

    for var_name in gpm_ds:
        if not var_name.endswith("_Longitude"):
            continue
        swath = var_name[: -len("_Longitude")]

        longitude = gpm_ds[swath + "_Longitude"]
        latitude = gpm_ds[swath + "_Latitude"]

        mask = (
            (center_longitude - margin < longitude)
            & (longitude < center_longitude + margin)
            & (center_latitude - margin < latitude)
            & (latitude < center_latitude + margin)
        )

        across_mask = mask.any(swath + "_time")
        time_mask = mask.any(swath + "_across")
        gpm_ds = gpm_ds.isel(
            **{
                swath + "_across": across_mask,
                swath + "_time": time_mask,
            }
        )

    return gpm_ds


# Next we define some handy plotting containers using some prepared stuff from
# mj_utils.
# These following figures are composed of different subplots (axes).
# The advantage of classes is here, that they inherit methods from their parents
# such that we can reuse plotting codes for certain subplot components, but change
# others.
class GPM_HAMP_grid_plot_base(AbsoluteGrid, abc.ABC):
    """abc.ABC: this is a absctract class. Derive a new class from it before using it."""

    vertical_space_inch = 0.2  # vertical space in inch
    horizonzal_space_inch = 0.8  # horizontal space in inch
    left_inch = 0.6  # left margin in inch
    bottom_inch = 0.5  # bottom margin in inch
    right_inch = 0.8  # right margin in inch
    top_inch = 0.25  # top margin in inch
    width_inch = 6.0
    height_inch = 7.0

    def get_nearest_hamp_frequency(self, hamp_frequency, tolerance=0.1):
        return float(self.hamp_ds.frequency.sel(frequency=hamp_frequency, method="nearest", tolerance=tolerance))


class GMI_HAMP_grid(GPM_HAMP_grid_plot_base):
    """Plot GMI and HAMP brightness temperature"""

    def __init__(self, gmi_ds, hamp_ds, gmi_frequency, hamp_frequency, time_window=np.timedelta64(20, "m")):
        self.gmi_ds = gmi_ds
        self.hamp_ds = hamp_ds
        self.gmi_frequency = gmi_frequency

        hamp_frequency = self.get_nearest_hamp_frequency(hamp_frequency)
        self.hamp_frequency = hamp_frequency

        kdtree = GPM_kdtree(gmi_ds)
        swath = self.get_gmi_swath(gmi_frequency)

        self.co_hamp, self.co_gmi = kdtree.query(
            hamp_ds,
            swath=swath,
            time_window=time_window,
        )

        # is there actually data to plot?
        if self.co_hamp.dims["time"] == 0 or np.all(np.isnan(self.co_hamp.tb.sel(frequency=hamp_frequency))):
            return

        # Prepare the matplotlib.gridspec.GridSpec:
        height_ratios = [2, 1.4, 0.6]
        width_ratios = [1]
        ax_pos_grid = self.grid_from_inch(height_ratios, width_ratios)

        # Plot stuff into the figure
        self.plot_map(ax_pos_grid[0, 0], gmi_frequency, hamp_frequency)
        ax = self.plot_tb(ax_pos_grid[1, 0], gmi_frequency, hamp_frequency)
        self.hide_xticklabels(ax)
        ax = self.plot_distance(ax_pos_grid[2, 0], gmi_frequency, sharex=ax)
        self.set_major_formatter_HMS(ax)

    def plot_tb(self, ax_pos, gmi_frequency, hamp_frequency, sharex=None):
        """Plot GMI and HAMP brightness temperature as collocated time series"""
        ax = self.fig.add_subplot(ax_pos, sharex=sharex)

        co_hamp = self.co_hamp
        co_gmi = self.co_gmi

        swath = self.get_gmi_swath(gmi_frequency)

        ax.plot(
            co_hamp.time.values,
            co_hamp.tb.sel(frequency=hamp_frequency),
            label="HAMP %s" % get_HAMP_label(hamp_frequency),
        )
        # Try a 30s average for HAMP to mimic the coarser spatial resolution of GMI
        ax.plot(
            co_hamp.time.values,
            co_hamp.tb.sel(frequency=hamp_frequency).rolling(time=30, center=True).mean(),
            label="HAMP 30 s avg %s" % get_HAMP_label(hamp_frequency),
            linewidth=0.5,
        )

        ax.plot(
            co_hamp.time.values,
            co_gmi[swath + "_Tb"].sel(**{swath + "_frequency": gmi_frequency}).values,
            label="GMI %s" % gmi_frequency,
        )

        ax.set_ylabel("TB (K)")
        ax.legend()
        ax.grid()
        return ax

    def plot_distance(self, ax_pos, gmi_frequency, sharex=None):
        """Plot the panel with spatial and temporal distance of HAMP and GPM"""
        ax = self.fig.add_subplot(ax_pos, sharex=sharex)
        co_hamp = self.co_hamp
        co_gmi = self.co_gmi

        swath = self.get_gmi_swath(gmi_frequency)

        one_minute = np.timedelta64(1, "m")
        time_distance = (co_hamp.time.values - co_gmi[swath + "_time"].values) / one_minute
        ax.plot(co_hamp.time.values, time_distance, label="HAMP - GMI", color="tab:blue")
        ax.set_ylabel("Time offset (min)", color="tab:blue")

        ax.legend()
        ax2 = ax.twinx()

        d = distance(
            co_gmi[swath + "_Latitude"].values,
            co_gmi[swath + "_Longitude"].values,
            co_hamp.lat.values,
            co_hamp.lon.values,
        )
        ax2.plot(co_hamp.time.values, d, color="tab:orange")
        ax2.set_ylabel("Distance (km)", color="tab:orange")
        ax2.spines["right"].set_visible(True)
        self.show_xticklabels(ax2)

        ax.grid()
        return ax

    def plot_map(self, ax_pos, gmi_frequency, hamp_frequency):
        """Plot the map with the satellite swath and HALO track.

        The HALO track is plotted with a colorful scatter plot that can encode
        the value of a HAMP brightness temperature."""
        ax = self.fig.add_subplot(ax_pos, projection=ccrs.PlateCarree())

        swath = self.get_gmi_swath(gmi_frequency)

        longitude = self.gmi_ds[swath + "_Longitude"]
        latitude = self.gmi_ds[swath + "_Latitude"]

        # Cut out Barbados from GMI data to remove very bright land measurements
        barbados_longitude = -59.556
        barbados_latitude = 13.1906
        barbados_margin = 0.3
        barbados_mask = (
            (barbados_longitude - barbados_margin < longitude)
            & (longitude < barbados_longitude + barbados_margin)
            & (barbados_latitude - barbados_margin < latitude)
            & (latitude < barbados_latitude + barbados_margin)
        )
        gl = ax.gridlines(
            draw_labels=True,
        )
        gl.right_labels = False
        gl.top_labels = False

        gmi_tb = self.gmi_ds[swath + "_Tb"].where(~barbados_mask).sel(**{swath + "_frequency": gmi_frequency})
        hamp_tb = self.co_hamp.tb.sel(frequency=hamp_frequency)

        # When we plot the GMI BT converted to nadir, we can use the same colorbar
        # and range for HAMP and GMI
        same_colorbar = "HV" in gmi_frequency or "HH" in gmi_frequency

        vmin = vmax = None
        if same_colorbar:
            vmin = min(gmi_tb.min(), hamp_tb.min())
            vmax = max(gmi_tb.max(), hamp_tb.max())

        ax.coastlines("50m")
        PC = ax.pcolormesh(
            center_to_edge(longitude),
            center_to_edge(latitude),
            gmi_tb.values,
            cmap="plasma" if same_colorbar else "viridis",
            # vmin=gpm_EC_exact.tb1.sel(frequency1=GMI_frequency1).min(),
            # vmax=gpm_EC_exact.tb1.sel(frequency1=GMI_frequency1).max(),
        )
        GMI_frequency1_str = str(gmi_frequency)
        cbar = self.fig.colorbar(PC)
        cbar.set_label("GMI BT (K) at %s" % GMI_frequency1_str)

        # Plot HAMP BT in consecutive groups in order to have highest values
        # plotted last, i.e. on top.
        if not same_colorbar:
            vmin = hamp_tb.min()
            vmax = hamp_tb.max()
        ax.plot(self.hamp_ds.lon, self.hamp_ds.lat, linewidth=0.7, color="#cccccc", zorder=1)
        for index, ds in self.co_hamp.sel(frequency=hamp_frequency).groupby_bins("tb", bins=5):
            SC = ax.scatter(ds.lon, ds.lat, c=ds.tb, vmin=vmin, vmax=vmax, cmap="plasma", s=5, zorder=2)

        if same_colorbar:
            cbar.set_label(
                "GMI BT (K) at %s\n" % GMI_frequency1_str
                + "HAMP BT (K) at %s"
                % get_HAMP_label(self.co_hamp.frequency.sel(frequency=hamp_frequency))
            )
        else:
            cbar_sc = self.fig.colorbar(SC)
            cbar_sc.set_label(
                "HAMP BT (K) at %s"
                % get_HAMP_label(self.co_hamp.frequency.sel(frequency=hamp_frequency))
            )

        # Zoom the map to HALO's flight track
        lon_max = self.co_hamp.lon.max()
        lon_min = self.co_hamp.lon.min()
        lat_max = self.co_hamp.lat.max()
        lat_min = self.co_hamp.lat.min()

        center_longitude = (lon_max + lon_min) / 2
        center_latitude = (lat_max + lat_min) / 2

        margin = max(lon_max - lon_min, lat_max - lat_min) * 0.6
        ax.set_xlim(center_longitude - margin, center_longitude + margin)
        ax.set_ylim(center_latitude - margin, center_latitude + margin)
        return ax

    def get_gmi_swath(self, gmi_frequency):
        """Find out in which swath the given gmi_frequency was measured"""
        if gmi_frequency in self.gmi_ds.S1_frequency:
            return "S1"
        elif gmi_frequency in self.gmi_ds.S2_frequency:
            return "S2"
        else:
            raise KeyError("Unknown GMI frequency %r" % gmi_frequency)

    def savefig(self, filename, *args, **kwargs):
        filename += "_%s_%.2f" % (self.gmi_frequency, self.hamp_frequency)
        if not hasattr(self, "fig"):
            print("Nothing to plot for %s" % filename)
            return
        super().savefig(filename, *args, **kwargs)


class product_HAMP_grid(GPM_HAMP_grid_plot_base, abc.ABC):
    """Prepare a class to plot GPM products and HAMP

    abc.ABC: this is a absctract class. Derive a new class from it before using it."""

    product_name = "surfacePrecipitation"
    lon_name = "S1_Longitude"
    lat_lane = "S1_Latitude"
    time_name = "S1_time"

    def plot_tb(self, ax_pos, hamp_frequency, sharex=None):
        """Plot HAMP brightness temperature and the GPM product in one time
        series"""
        ax = self.fig.add_subplot(ax_pos, sharex=sharex)

        co_hamp = self.co_hamp
        co_product = self.co_product

        ax.plot(
            co_hamp.time.values,
            co_hamp.tb.sel(frequency=hamp_frequency),
            label="HAMP %s" % get_HAMP_label(hamp_frequency),
            alpha=0.5,
        )

        ax.set_ylabel("TB (K)")
        ax.legend(loc="upper left")

        ax2 = ax.twinx()
        ax2.set_yscale("log")
        ax2.plot(
            co_hamp.time.values,
            co_product[self.product_name].values,
            label="Satellite",
            color="tab:orange",
        )

        if "radar_rainrate" in co_hamp:
            ax2.plot(
                co_hamp.time.values,
                co_hamp.radar_rainrate.values,
                label="HAMP Z-R",
                color="tab:red",
            )

        ax2.spines["right"].set_visible(True)

        units = co_product[self.product_name].units
        ax2.set_ylabel(f"{self.product_name} ({units})", color="tab:orange")
        ax2.legend(loc="upper right")

        return ax

    def plot_distance(self, ax_pos, sharex=None):
        """Plot the panel with spatial and temporal distance of HAMP and GPM.

        Difference to GMI_HAMP_grid.plot_distance: here we use self.co_product
        instead of self.co_gmi and don't need a gmi_frequency."""
        ax = self.fig.add_subplot(ax_pos, sharex=sharex)
        co_hamp = self.co_hamp
        co_product = self.co_product

        one_minute = np.timedelta64(1, "m")
        time_distance = (co_hamp.time.values - co_product[self.time_name].values) / one_minute
        ax.plot(co_hamp.time.values, time_distance, label="HAMP - GMI", color="tab:blue")
        ax.set_ylabel("Time offset (min)", color="tab:blue")

        ax.legend()
        ax2 = ax.twinx()

        d = distance(
            co_product[self.lat_lane].values,
            co_product[self.lon_name].values,
            co_hamp.lat.values,
            co_hamp.lon.values,
        )
        ax2.plot(co_hamp.time.values, d, color="tab:orange")
        ax2.set_ylabel("Distance (km)", color="tab:orange")
        ax2.spines["right"].set_visible(True)
        self.show_xticklabels(ax2)

        ax.grid()
        return ax

    def plot_map(self, ax_pos, hamp_frequency):
        """Plot the map with the satellite swath and HALO track.

        The HALO track is plotted with a colorful scatter plot that can encode
        the value of a HAMP brightness temperature.

        Difference to GMI_HAMP_grid.plot_distance: here we use self.co_product
        instead of self.co_gmi and don't need a gmi_frequency."""
        ax = self.fig.add_subplot(ax_pos, projection=ccrs.PlateCarree())

        longitude = self.product_ds[self.lon_name]
        latitude = self.product_ds[self.lat_lane]

        barbados_longitude = -59.556
        barbados_latitude = 13.1906
        barbados_margin = 0.3
        barbados_mask = (
            (barbados_longitude - barbados_margin < longitude)
            & (longitude < barbados_longitude + barbados_margin)
            & (barbados_latitude - barbados_margin < latitude)
            & (latitude < barbados_latitude + barbados_margin)
        )
        gl = ax.gridlines(
            draw_labels=True,
        )
        gl.right_labels = False
        gl.top_labels = False

        ax.coastlines("50m")
        PC = ax.pcolormesh(
            center_to_edge(longitude),
            center_to_edge(latitude),
            self.product_ds[self.product_name].where(~barbados_mask).values,
        )
        cbar = self.fig.colorbar(PC)
        cbar.set_label(f"GMI {self.product_name}")

        # Plot HAMP BT in consecutive groups in order to have highest values
        # plotted last, i.e. on top.
        vmin = self.co_hamp.tb.sel(frequency=hamp_frequency).min()
        vmax = self.co_hamp.tb.sel(frequency=hamp_frequency).max()
        ax.plot(self.hamp_ds.lon, self.hamp_ds.lat, linewidth=0.7, color="#cccccc", zorder=1)
        for index, ds in self.co_hamp.sel(frequency=hamp_frequency).groupby_bins("tb", bins=5):
            SC = ax.scatter(ds.lon, ds.lat, c=ds.tb, vmin=vmin, vmax=vmax, cmap="plasma", s=5, zorder=2)
        cbar_sc = self.fig.colorbar(SC)
        cbar_sc.set_label(
            "HAMP BT (K) at %s" % get_HAMP_label(self.co_hamp.frequency.sel(frequency=hamp_frequency))
        )

        lon_max = self.co_hamp.lon.max()
        lon_min = self.co_hamp.lon.min()
        lat_max = self.co_hamp.lat.max()
        lat_min = self.co_hamp.lat.min()

        center_longitude = (lon_max + lon_min) / 2
        center_latitude = (lat_max + lat_min) / 2

        margin = max(lon_max - lon_min, lat_max - lat_min) * 0.6
        ax.set_xlim(center_longitude - margin, center_longitude + margin)
        ax.set_ylim(center_latitude - margin, center_latitude + margin)
        return ax

    def savefig(self, filename, *args, **kwargs):
        filename += "_%.2f" % (self.hamp_frequency)
        if not hasattr(self, "fig"):
            print("Nothing to plot for %s" % filename)
            return
        super().savefig(filename, *args, **kwargs)


class GPROF_HAMP_grid(product_HAMP_grid):
    """Plot HAMP and GPROF product"""

    product_name = "S1_surfacePrecipitation"

    def __init__(self, product_ds, hamp_ds, hamp_frequency, time_window=np.timedelta64(20, "m")):
        self.product_ds = product_ds
        self.hamp_ds = hamp_ds

        hamp_frequency = self.get_nearest_hamp_frequency(hamp_frequency)
        self.hamp_frequency = hamp_frequency

        kdtree = GPM_kdtree(product_ds)

        self.co_hamp, self.co_product = kdtree.query(
            hamp_ds,
            swath="S1",
            time_window=time_window,
        )

        if self.co_hamp.dims["time"] == 0 or np.all(np.isnan(self.co_hamp.tb.sel(frequency=hamp_frequency))):
            return

        height_ratios = [2, 1.4, 0.6]
        width_ratios = [1]
        ax_pos_grid = self.grid_from_inch(height_ratios, width_ratios)

        self.plot_map(ax_pos_grid[0, 0], hamp_frequency)
        ax = self.plot_tb(ax_pos_grid[1, 0], hamp_frequency)
        self.hide_xticklabels(ax)
        ax = self.plot_distance(ax_pos_grid[2, 0], sharex=ax)
        self.set_major_formatter_HMS(ax)


class IMERG_HAMP_grid(product_HAMP_grid):
    """Plot HAMP and IMERG product"""

    time_name = "S1_HQobservationTime"

    def __init__(
        self,
        product_ds,
        hamp_ds,
        hamp_frequency,
        product="surfacePrecipitation",
        time_window=np.timedelta64(20, "m"),
    ):
        self.product_name = product

        self.product_ds = product_ds
        self.hamp_ds = hamp_ds

        hamp_frequency = self.get_nearest_hamp_frequency(hamp_frequency)
        self.hamp_frequency = hamp_frequency

        kdtree = GPM_kdtree(product_ds)

        self.co_hamp, self.co_product = kdtree.query(
            hamp_ds,
            swath="S1",
            time_window=time_window,
        )

        if self.co_hamp.dims["time"] == 0 or np.all(np.isnan(self.co_hamp.tb.sel(frequency=hamp_frequency))):
            return

        height_ratios = [2, 1.4, 0.6]
        width_ratios = [1]
        ax_pos_grid = self.grid_from_inch(height_ratios, width_ratios)

        self.plot_map(ax_pos_grid[0, 0], hamp_frequency)
        ax = self.plot_tb(ax_pos_grid[1, 0], hamp_frequency)
        self.hide_xticklabels(ax)
        ax = self.plot_distance(ax_pos_grid[2, 0], sharex=ax)
        self.set_major_formatter_HMS(ax)

    def plot_map(self, ax_pos, hamp_frequency):
        ax = self.fig.add_subplot(ax_pos, projection=ccrs.PlateCarree())

        longitude = self.product_ds[self.lon_name]
        latitude = self.product_ds[self.lat_lane]

        barbados_longitude = -59.556
        barbados_latitude = 13.1906
        barbados_margin = 0.3
        barbados_mask = (
            (barbados_longitude - barbados_margin < longitude)
            & (longitude < barbados_longitude + barbados_margin)
            & (barbados_latitude - barbados_margin < latitude)
            & (latitude < barbados_latitude + barbados_margin)
        )
        gl = ax.gridlines(
            draw_labels=True,
        )
        gl.right_labels = False
        gl.top_labels = False

        ax.coastlines("50m")
        PC = ax.pcolormesh(
            center_to_edge(longitude),
            center_to_edge(latitude),
            self.product_ds[self.product_name].where(~barbados_mask).values,
            vmax=2,
            vmin=0,
        )
        cbar = self.fig.colorbar(PC)
        cbar.set_label(f"IMERG {self.product_name}")

        # Plot HAMP BT in consecutive groups in order to have highest values
        # plotted last, i.e. on top.
        vmin = self.co_hamp.tb.sel(frequency=hamp_frequency).min()
        vmax = self.co_hamp.tb.sel(frequency=hamp_frequency).max()
        ax.plot(self.hamp_ds.lon, self.hamp_ds.lat, linewidth=0.7, color="#cccccc", zorder=1)
        for index, ds in self.co_hamp.sel(frequency=hamp_frequency).groupby_bins("tb", bins=5):
            SC = ax.scatter(ds.lon, ds.lat, c=ds.tb, vmin=vmin, vmax=vmax, cmap="plasma", s=5, zorder=2)
        cbar_sc = self.fig.colorbar(SC)
        cbar_sc.set_label(
            "HAMP BT (K) at %s" % get_HAMP_label(self.co_hamp.frequency.sel(frequency=hamp_frequency))
        )

        lon_max = self.co_hamp.lon.max()
        lon_min = self.co_hamp.lon.min()
        lat_max = self.co_hamp.lat.max()
        lat_min = self.co_hamp.lat.min()

        center_longitude = (lon_max + lon_min) / 2
        center_latitude = (lat_max + lat_min) / 2

        margin = max(lon_max - lon_min, lat_max - lat_min) * 0.6
        ax.set_xlim(center_longitude - margin, center_longitude + margin)
        ax.set_ylim(center_latitude - margin, center_latitude + margin)


class DPR_Ka_HAMP_grid(AbsoluteGrid):
    """Plot HAMP and KaPR data

    This plotting class tries to plot a "time"-height slice of the KaPR.
    However, figuring out the height coordinate of KaPR data is kind of complicated
    and _not_ implemented here (as there was no radar signal above the noise
    signal in the cases I looked at.)
    """

    vertical_space_inch = 0.2  # vertical space in inch
    horizonzal_space_inch = 0.8  # horizontal space in inch
    left_inch = 0.6  # left margin in inch
    bottom_inch = 0.5  # bottom margin in inch
    right_inch = 0.8  # right margin in inch
    top_inch = 0.25  # top margin in inch
    width_inch = 6.0
    height_inch = 9.0

    def __init__(self, dpr_ds, hamp_ds, DPR_swath, time_window=np.timedelta64(20, "m")):
        self.dpr_ds = dpr_ds
        self.hamp_ds = hamp_ds
        self.DPR_swath = DPR_swath

        kdtree = GPM_kdtree(dpr_ds)

        self.co_hamp, self.co_dpr = kdtree.query(
            hamp_ds,
            swath=DPR_swath,
            time_window=time_window,
        )

        if self.co_hamp.dims["time"] == 0 or np.all(np.isnan(self.co_hamp.dBZ.max("height"))):
            return

        height_ratios = [2, 1.4, 1.4, 0.6]
        width_ratios = [1]
        ax_pos_grid = self.grid_from_inch(height_ratios, width_ratios)

        self.plot_map(ax_pos_grid[0, 0], DPR_swath)
        ax = self.plot_radar_hamp(ax_pos_grid[1, 0])
        ax = self.plot_radar_dpr(ax_pos_grid[2, 0], DPR_swath)
        self.hide_xticklabels(ax)
        ax = self.plot_distance(ax_pos_grid[-1, 0], DPR_swath, sharex=ax)
        self.set_major_formatter_HMS(ax)

    def plot_radar_hamp(self, ax_pos, sharex=None):
        """
        Plot the HAMP radar curtain
        """
        ax = self.fig.add_subplot(ax_pos, sharex=sharex)

        dBZ = np.ma.masked_array(self.co_hamp.dBZ.transpose("height", "time"))
        # maybe one has to adjust this for updates of the unified datasets.
        # dBZ[np.isnan(self.co_hamp.dBZ)] = -np.inf
        # dBZ.mask = self.co_hamp.radar_not_available
        x = center_to_edge(self.co_hamp.time.values)
        y = center_to_edge(self.co_hamp.height.values)

        pcolormesh = ax.pcolormesh(x, y, dBZ, cmap="plasma", vmin=-35, vmax=20, rasterized=True, zorder=1)
        ax.set_ylim(0, 12e3)

        return ax

    def plot_radar_dpr(self, ax_pos, swath, sharex=None):
        """
        Plot the KaPR radar curtain.
        """
        ax = self.fig.add_subplot(ax_pos, sharex=sharex)

        # There are different variables with different processing levels for
        # reflectivity
        # Z = self.co_dpr[swath+'_zFactorCorrected']
        Z = self.co_dpr[swath + "_zFactorMeasured"]
        Z = Z.where(Z > -1000)  # There are some strange error values with unrealistic values
        assert Z.units == "dBZ"
        dBZ = np.ma.masked_array(Z.transpose(swath + "_nbin", swath + "_halo_time"))

        # This calculation of altitude is completely bull sh**! To do it right,
        # it is much more complicated and involves several DPR variables. Read
        # the note from PSS Helpdesk:
        pps_helpdesk = """
PRE/binClutterFreeBottom stores the one-based bin number where the 2D array
SLV/zFactorCorrectedNearSurface is extracted from the 3D array SLV/zFactorCorrected.
PRE/binRealSurface stores the one-based bin number for the altitude where the 2A-DPR algorithm
estimates the earth's physical surface exists. The 2A-DPR algorithm extrapolates from the near-
surface observed echo (in SLV/zFactorCorrectedNearSurface) to calculate an estimated echo at the
earth's physical surface, which is stored in SLV/zFactorCorrectedESurface.
The GPM Ku convention is that bin number 1 is at the top of the altitude grid which is 21.875 km
altitude above the earth ellipsoid. GPM Ku Bin 176 is at the earth ellipsoid. The ellipsoid is a
mathematical abstraction of the earth's surface
(see: http://http://en.wikipedia.org/wiki/Earth_ellipsoid ) that does not exactly match up with the
hard, physical surface of the earth (land or water body). This is why PRE/binRealSurface isn't
always equal to 176. Over moutains, for example, PRE/binRealSurface is much less than 176 since the
mountains stick up above the earth's ellipsoid.
The bin spacing for GPM DPR Ka is 0.25 km (1/8th km), and for GPM DPR Ku is 0.125 km (1/16th km).
The Ku 176 bin arrays are at 125m spacing and the 88 bin arrays (HS Ka) swath are at 250m.
176*125m = 22,000m total.
That flag value means No Observation at that specific height/ifov.
The -28888 is no rain (there was an observation)
The -29999 is no observation

Please refer to the GPM documentation file  PPS/Global Precipitation Measurement File Specification
for GPM Products for details of GPM product specification which is available at our website:
http://pps.gsfc.nasa.gov/
under the Documentation tab PPS/GPM Documentation
https://pps.gsfc.nasa.gov/thorrelease.html
-Also look in the filespec for 2AKa, Dimension definition for nbin or nbinHS?
ftp://gpmweb2.pps.eosdis.nasa.gov/pub/GPMfilespec/filespec.GPM.pdf Starts on pg.1466
        """
        if swath == "HS":
            GPM_altitude = np.arange(self.co_dpr.dims[swath + "_nbin"])[::-1] * 250  # , 0, -1
        elif swath == "MS":
            GPM_altitude = np.arange(self.co_dpr.dims[swath + "_nbin"])[::-1] * 125  # , 0, -1

        x = center_to_edge(self.co_dpr[swath + "_time"].values)
        y = center_to_edge(GPM_altitude)

        pcolormesh = ax.pcolormesh(
            x,
            y,
            dBZ,
            cmap="viridis",
            # vmin=-35, vmax=20,
            rasterized=True,
            zorder=1,
        )
        self.fig.colorbar(pcolormesh)
        ax.set_ylim(0, 12e3)

        return ax

    def plot_distance(self, ax_pos, swath, sharex=None):
        ax = self.fig.add_subplot(ax_pos, sharex=sharex)
        co_hamp = self.co_hamp
        co_dpr = self.co_dpr

        one_minute = np.timedelta64(1, "m")
        time_distance = (co_hamp.time.values - co_dpr[swath + "_time"].values) / one_minute
        ax.plot(co_hamp.time.values, time_distance, label="HAMP - DPR", color="tab:blue")
        ax.set_ylabel("Time offset (min)", color="tab:blue")

        ax.legend()
        ax2 = ax.twinx()

        d = distance(
            co_dpr[swath + "_Latitude"].values,
            co_dpr[swath + "_Longitude"].values,
            co_hamp.lat.values,
            co_hamp.lon.values,
        )
        ax2.plot(co_hamp.time.values, d, color="tab:orange")
        ax2.set_ylabel("Distance (km)", color="tab:orange")
        ax2.spines["right"].set_visible(True)
        self.show_xticklabels(ax2)

        return ax

    def plot_map(self, ax_pos, swath):
        ax = self.fig.add_subplot(ax_pos, projection=ccrs.PlateCarree())

        longitude = self.dpr_ds[swath + "_Longitude"]
        latitude = self.dpr_ds[swath + "_Latitude"]

        barbados_longitude = -59.556
        barbados_latitude = 13.1906
        barbados_margin = 0.3
        barbados_mask = (
            (barbados_longitude - barbados_margin < longitude)
            & (longitude < barbados_longitude + barbados_margin)
            & (barbados_latitude - barbados_margin < latitude)
            & (latitude < barbados_latitude + barbados_margin)
        )
        gl = ax.gridlines(
            draw_labels=True,
        )
        gl.right_labels = False
        gl.top_labels = False

        product_name = swath + "_zFactorCorrectedNearSurface"
        ax.coastlines("50m")
        PC = ax.pcolormesh(
            center_to_edge(longitude),
            center_to_edge(latitude),
            self.dpr_ds[product_name].values,
            # self.dpr_ds[swath+'_zFactorCorrected'].where(~barbados_mask).max(swath+'_nbin').values,
        )
        cbar = self.fig.colorbar(PC)
        cbar.set_label(f"DRP Ka {product_name} ({self.dpr_ds[product_name].units})")

        # Plot HAMP dBZ in consecutive groups in order to have highest values
        # plotted last, i.e. on top.
        vmin = -35
        vmax = 20
        ax.plot(self.hamp_ds.lon, self.hamp_ds.lat, linewidth=0.7, color="#cccccc", zorder=1)
        for index, ds in self.co_hamp.max("height").groupby_bins("dBZ", bins=np.arange(-40, 41, 10)):
            SC = ax.scatter(ds.lon, ds.lat, c=ds.dBZ, vmin=vmin, vmax=vmax, cmap="plasma", s=5, zorder=2)
        cbar_sc = self.fig.colorbar(SC)
        cbar_sc.set_label("HAMP reflectivity (dBZ)")

        lon_max = self.co_hamp.lon.max()
        lon_min = self.co_hamp.lon.min()
        lat_max = self.co_hamp.lat.max()
        lat_min = self.co_hamp.lat.min()

        center_longitude = (lon_max + lon_min) / 2
        center_latitude = (lat_max + lat_min) / 2

        margin = max(lon_max - lon_min, lat_max - lat_min) * 0.6
        ax.set_xlim(center_longitude - margin, center_longitude + margin)
        ax.set_ylim(center_latitude - margin, center_latitude + margin)

    def savefig(self, filename, *args, **kwargs):
        filename += self.DPR_swath
        if not hasattr(self, "fig"):
            print("Nothing to plot for %s" % filename)
            return
        super().savefig(filename, *args, **kwargs)


def correct_radiometer(hamp):
    """Apply the brightness correction to HAMP data

    This is needed as long as it is not included in the unified data set.
    Like in v0.9 and before this is not the case and this correction here is required.

    Works as in-place on the `hamp' xarray datset"""
    bias_filename = (
        os.path.realpath(os.path.dirname(__file__))
        + "/../out/tb_bias3/EUREC4A_CSSC/clear_sky_sonde_comparison_ALL_J3v0.9.2_radar_daily.nc"
    )
    if not os.path.isfile(bias_filename):
        bias_filename = (
            os.path.realpath(os.path.dirname(__file__))
            + "/../../out/tb_bias3/EUREC4A_CSSC/clear_sky_sonde_comparison_ALL_J3v0.9.2_radar_daily.nc"
        )

    d = xarray.open_dataset(bias_filename)
    d = d.reindex(frequency=hamp.frequency, method="nearest", tolerance=0.1)

    date = hamp.time.dt.strftime("%Y%m%d").values[0]
    print("date:", date)
    if date in d.date.values:
        offset = d.offset.sel(date=date)
        offset = offset.drop("date")
        offset = offset.fillna(d.mean_bias)
        slope = d.slope.sel(date=date)
        slope = slope.drop("date")
        slope = slope.fillna(d.mean_bias)
    else:
        raise ValueError(f"date {date} not in bias_file")
    offset = offset.values  # xarray.Dataset to np.array
    slope = slope.values  # xarray.Dataset to np.array
    hamp["tb_raw"] = hamp["tb"].copy()  # in-place change. should keep attributes
    hamp["tb"].values = slope * hamp["tb"] + offset  # in-place change. should keep attributes
    assert hamp.tb.units == "K"


if __name__ == "__main__":

    dates = [
        "20200124",
        "20200211",
        "20200205",
        "20200119",
        "20200122",  # no HAMP radar
        "20200130",
        "20200131",
        "20200202",
        # "20200207",  # needs more time difference and GMI data is not downloaded
        "20200209",
        "20200213",
    ]

    time_window = {
        "20200211": 80,
        "20200131": 80,
        "20200205": 160,
    }

    ###
    # Now let's plot different products in different scenes!
    #
    # Some code settings like frequency pairs or certain plots are commented out,
    # but they should all work You can try to uncomment some of these blocks and
    # try the code out. Some products are not yet downloaded for all dates, so
    # certain combinations might fail. Feel free to download more data!
    # To get data, note the /data/obs/campaigns/eurec4a/sat/GPM/README.txt and
    # https://gpm.nasa.gov/data
    #

    for date in dates:
        hamp = xarray.open_dataset("/data/obs/campaigns/eurec4a/HALO/unified/radiometer_%s_v0.8.nc" % date)
        correct_radiometer(hamp)
        filenames = glob.glob("/data/obs/campaigns/eurec4a/sat/GPM/1B.GPM.GMI.TB2016.%s-S*-E*.*.V05A.HDF5" % date)
        assert len(filenames) == 1, "%s, 1 != len(%r)" % (date, filenames)
        gmi_ds = open_gmi(filenames[0])

        gmi_ds = gpm_filter_area(gmi_ds, center_longitude=-55, margin=5)

        for hamp_frequency, gmi_frequency in (
            # other frequency pairs work as well, but the 90 GHz might be the most interesting one for precipitation
            # (23.84, "23.80 V"),
            # (90, "89.00 V"),
            # (90, "89.00 H"),
            (90, "89.00 HH"),
            (90, "89.00 HV"),
            # (31.4, "36.50 V"),
            # (31.4, "36.50 H"),
            # (186.81, "183.31+-3.0 V"),
            # (190.81, "183.31+-7.0 V"),
        ):
            fig = GMI_HAMP_grid(
                gmi_ds,
                hamp,
                gmi_frequency,
                hamp_frequency,
                time_window=np.timedelta64(time_window.get(date, 20), "m"),
            )
            fig.savefig("./out/bt_collocation/%s" % date)

    for date in dates:
        if date in '20200122':
            # HAMP radar was not working on this day
            continue
        elif date in '20200209':
            # This day is not working. Probably no spatial overlap?
            continue

        # GPROF
        hamp = xarray.open_dataset("/data/obs/campaigns/eurec4a/HALO/unified/radiometer_%s_v0.8.nc" % date)
        correct_radiometer(hamp)
        hamp_radar = xarray.open_dataset("/data/obs/campaigns/eurec4a/HALO/unified/radar_%s_v0.6.nc" % date)
        # https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/met.1825 Z = 10.09 * R**0.59
        if "Z" in hamp_radar:
            hamp["radar_rainrate"] = (hamp_radar.Z.sel(height=300, method="nearest") / 10.09) ** (1 / 0.59)
            hamp["radar_rainrate"] = (10 ** (hamp_radar.dBZ.sel(height=300, method="nearest") / 10) / 10.09) ** (
                1 / 0.59
            )
        filenames = glob.glob(
            "/data/obs/campaigns/eurec4a/sat/GPM/2A-CLIM.GPM.GMI.GPROF2017v1.%s-S*-E*.*.V05C.HDF5" % date
        )
        assert len(filenames) == 1
        gprof_ds = open_gprof(filenames[0])

        gprof_ds = gpm_filter_area(gprof_ds, center_longitude=-55, margin=5)

        fig = GPROF_HAMP_grid(gprof_ds, hamp, 90, time_window=np.timedelta64(time_window.get(date, 20), "m"))
        fig.savefig("./out/gprof_collocation/%s" % date)

        # DPR
        filenames = glob.glob("/data/obs/campaigns/eurec4a/sat/GPM/2A.GPM.Ka.V8-*.%s-S*-E*.*.V06A.HDF5" % date)
        assert len(filenames) == 1
        dpr_ds = open_dpr_ka(filenames[0])

        dpr_ds = gpm_filter_area(dpr_ds, center_longitude=-55, margin=5)

        for DPR_swath in (
            "MS",
            "HS",
        ):
            fig = DPR_Ka_HAMP_grid(dpr_ds, hamp_radar, DPR_swath, time_window=np.timedelta64(time_window.get(date, 20), "m"))
            fig.savefig("./out/radar_collocation/radar_%s" % date)

    for date in ["20200211"]:
        # IMERG data was only downloaded for one day.
        hamp = xarray.open_dataset("/data/obs/campaigns/eurec4a/HALO/unified/radiometer_%s_v0.8.nc" % date)
        correct_radiometer(hamp)

        for product in ["S1_HQprecipitation", "S1_precipitationCal"]:
            imerg_ds = open_imerg(
                "/data/obs/campaigns/eurec4a/sat/IMERG/3B-HHR.MS.MRG.3IMERG.20200211-S193000-E195959.1170.V06B.HDF5"
            )
            fig = IMERG_HAMP_grid(
                imerg_ds,
                hamp,
                90,
                product=product,
                time_window=np.timedelta64(time_window.get(date, 120), "m"),
            )
            fig.savefig("./out/imerg_collocation/%s_%s_S193000" % (product, date), save_pdf=False)
            imerg_ds = open_imerg(
                "/data/obs/campaigns/eurec4a/sat/IMERG/3B-HHR.MS.MRG.3IMERG.20200211-S200000-E202959.1200.V06B.HDF5"
            )
            fig = IMERG_HAMP_grid(
                imerg_ds,
                hamp,
                90,
                product=product,
                time_window=np.timedelta64(time_window.get(date, 120), "m"),
            )
            fig.savefig("./out/imerg_collocation/%s_%s_S200000" % (product, date), save_pdf=False)
            imerg_ds = open_imerg(
                "/data/obs/campaigns/eurec4a/sat/IMERG/3B-HHR.MS.MRG.3IMERG.20200211-S203000-E205959.1230.V06B.HDF5"
            )
            fig = IMERG_HAMP_grid(
                imerg_ds,
                hamp,
                90,
                product=product,
                time_window=np.timedelta64(time_window.get(date, 120), "m"),
            )
            fig.savefig("./out/imerg_collocation/%s_%s_S203000" % (product, date), save_pdf=False)
