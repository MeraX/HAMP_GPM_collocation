#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc

import matplotlib.pyplot as plt
import matplotlib.gridspec
import matplotlib.dates
import numpy as np
import datetime

def get_HAMP_label(frequency, units=True, show_center=True):
    """Generate label for HAMP channel.

    Parameters
    ----------
    frequency : {float}
        Frequency in GHz
    units : {bool}, optional
        Append units, i.e., "GHz" (the default is True)
    show_center : {bool}, optional
        Show frequencies near and above 118.75 and 183.31 GHz as "center ± Offset". Frequencies
        from the center to center+25 are considered. (the default is True)

    Returns
    -------
    str
        Label
    """
    if 118.75 <= frequency < (118.75 + 25):
        label = "± %.1f" % (frequency - 118.75)
        if show_center:
            label = "118.75 " + label
    elif 183.31 <= frequency < (183.31 + 25):
        label = "± %.1f" % (frequency - 183.31)
        if show_center:
            label = "183.31 " + label
    else:
        label = "%.2f" % frequency
    if units:
        if "±" in label:
            label = "(%s) GHz" % label
        else:
            label = "%s GHz" % label

    return label


def center_to_edge(center):
    """convert a coordinate array that marks the centers of cells to an array that describes the
    edges 1D or 2D fload arrays and 1D datetime arrays are supported"""
    center = np.asarray(center)
    if len(center.shape) == 1:
        if isinstance(center[0], datetime.datetime):
            edge = np.empty(center.shape[0] + 1, dtype=object)
        else:
            try:
                center - np.datetime64("1")
            except TypeError:
                edge = np.empty(center.shape[0] + 1, dtype=float)
                edge.fill(np.nan)
            else:
                edge = np.empty(center.shape[0] + 1, dtype=center.dtype)
        # the strange notation in the following lines is used to make the calculations datetime compatible
        edge[1:-1] = center[1:] + (center[:-1] - center[1:]) / 2.0
        edge[0] = edge[1] + (edge[1] - edge[2])
        edge[-1] = edge[-2] + (edge[-2] - edge[-3])
    elif len(center.shape) == 2:
        edge = np.empty([center.shape[0] + 1, center.shape[1] + 1])
        edge.fill(np.nan)
        edge[1:-1, 1:-1] = (
            center[1:, 1:]
            + center[1:, :-1]
            + center[:-1, 1:]
            + center[:-1, :-1]
        ) / 4.0
        edge[0, :] = 2 * edge[1, :] - edge[2, :]
        edge[-1, :] = 2 * edge[-2, :] - edge[-3, :]
        edge[:, 0] = 2 * edge[:, 1] - edge[:, 2]
        edge[:, -1] = 2 * edge[:, -2] - edge[:, -3]
    else:
        raise ValueError("input data has wrong shape: %s", center.shape)
    return edge


class AbsoluteGrid(metaclass=abc.ABCMeta):
    """This Meta Class helps to create a multi-axis plots with GridSpec.

    The advantage of this class compared to bare matplotlib.gridspec.GridSpec is that it is based on
    absolute parameter for the margins between the sub-plots.

    The space between top_inch and bottom_inch will be distributed according to height_ratios while
    keeping a spacing of vertical_space_inch between the sub-plots.

    All *_inch attributes are in inches, which means that
        1 inch == dpi * px.

    The "Meta Class" simply means, that one has to derive an own class to use AbsoluteGrid.
    """

    vertical_space_inch = 0.8  # vertical space in inch
    horizonzal_space_inch = 0.8  # horizontal space in inch
    left_inch = 0.6  # left margin in inch
    bottom_inch = 0.5  # bottom margin in inch
    right_inch = 0.5  # right margin in inch
    top_inch = 0.25  # top margin in inch
    width_inch = 5.0
    height_inch = 7.0

    @abc.abstractmethod
    def __init__(self, height_ratios, width_ratios):
        """Initialization

        You can define your sub-plots here.

        Important steps are to define are outlined below

        Parameters
        ----------
        height_ratios : tuple of int
            Ratio of the axis heights.
        width_ratios :  tuple of int
            Ratio of the axis widths.
        """
        ax_pos_grid = self.grid_from_inch(height_ratios, width_ratios)

        ax = self.fig.add_subplot(ax_pos_grid[0, 0])

        ## then use this ax to have fun and make more axes.

    def grid_from_inch(self, height_ratios, width_ratios):
        """Call matplotlib.gridspec.GridSpec with absolute spacings

        Geometry magic happens here.
        Can also make self.fig, if not defined before.

        Parameters
        ----------
        height_ratios : tuple of int
            Ratio of the axis heights.
        width_ratios :  tuple of int
            Ratio of the axis widths.
        """
        width_inch = self.width_inch
        height_inch = self.height_inch
        nrows = len(height_ratios)
        ncols = len(width_ratios)
        vertical_space_inch = self.vertical_space_inch
        horizonzal_space_inch = self.horizonzal_space_inch

        # Calculate relative sizes.
        left = self.left_inch / width_inch
        bottom = self.bottom_inch / height_inch
        right = 1 - self.right_inch / width_inch
        top = 1 - self.top_inch / height_inch

        average_height = ((top - bottom) - (vertical_space_inch / height_inch) * (nrows - 1)) / nrows
        # hspace is the vertical space between subplots as a fraction of the average subplot height
        hspace = (vertical_space_inch / height_inch) / average_height

        average_width = ((right - left) - (horizonzal_space_inch / width_inch) * (ncols - 1)) / ncols
        # wspace is the horizontal space between subplots as a fraction of the average subplot width
        wspace = (horizonzal_space_inch / width_inch) / average_width

        if not hasattr(self, "fig"):
            self.fig = plt.figure(figsize=(self.width_inch, self.height_inch))

        ax_pos_grid = matplotlib.gridspec.GridSpec(
            nrows=nrows,
            ncols=ncols,
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            wspace=wspace,
            hspace=hspace,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
        )

        return ax_pos_grid

    def hide_xticklabels(self, ax):
        """Hide xticklabels of given axis ax.

        Parameters
        ----------
        ax : matplotlib.figure.Axes
        """
        try:
            xticklabels = ax.get_xticklabels()
        except ValueError:
            pass
        else:
            return plt.setp(xticklabels, visible=False)

    def show_xticklabels(self, ax):
        """Show xticklabels of given axis ax.

        Parameters
        ----------
        ax : matplotlib.figure.Axes
        """
        try:
            xticklabels = ax.get_xticklabels()
        except ValueError:
            pass
        else:
            return plt.setp(xticklabels, visible=True)

    def hide_yticklabels(self, ax):
        """Hide yticklabels of given axis ax.

        Parameters
        ----------
        ax : matplotlib.figure.Axes
        """
        try:
            yticklabels = ax.get_yticklabels()
        except ValueError:
            pass
        else:
            return plt.setp(yticklabels, visible=False)

    def show_yticklabels(self, ax):
        """Show yticklabels of given axis ax.

        Parameters
        ----------
        ax : matplotlib.figure.Axes
        """
        try:
            yticklabels = ax.get_yticklabels()
        except ValueError:
            pass
        else:
            return plt.setp(yticklabels, visible=True)

    def hide_spines(self, ax):
        """Hide all spines and x- and y-ticks of given axis ax.

        Parameters
        ----------
        ax : matplotlib.figure.Axes
        """
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_visible(False)
        ax.xaxis.set_ticks_position("none")
        ax.yaxis.set_ticks_position("none")

    def set_major_formatter_HMS(self, ax):
        """Set the formatter of ax to HH:MM:SS.

        Parameters
        ----------
        ax : matplotlib.figure.Axes
        """
        formatter = matplotlib.dates.DateFormatter("%H:%M:%S")
        ax.xaxis.set_major_formatter(formatter)

    def savefig(
        self,
        outpath="out/figure01",
        save_pdf=True,
        save_png=True,
        png_dpi=200,
        verbose=True,
    ):
        """Save and close Figure

        Parameters
        ----------
        outpath : {str}, optional
            Output directory and path. Should not contain extension. (the default is 'out/figure01')
        save_pdf : {bool}, optional
            Save as PDF. (the default is True)
        save_png : {bool}, optional
            Save as PNG. (the default is True)
        png_dpi : {int}, optional
            Resolution of save png. (the default is 100)
        verbose : {bool}, optional
            Tell where plots are saved (the default is True)
        """
        if save_png:
            self.fig.savefig(outpath + ".png", dpi=png_dpi)
            if verbose and not save_pdf:
                print("save png:", outpath + ".png")
        if save_pdf:
            self.fig.savefig(outpath + ".pdf", dpi=400)
            if verbose and not save_png:
                print("save pdf", outpath + ".pdf")
        if verbose and save_pdf and save_png:
            print("save png, pdf", outpath + ".{png,pdf}")
        plt.close(self.fig)
