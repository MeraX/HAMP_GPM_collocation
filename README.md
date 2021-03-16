[![DOI](https://zenodo.org/badge/348316876.svg)](https://zenodo.org/badge/latestdoi/348316876)

# Investigate HAMP - GPM collocations

This script published here provides methods for a point-to-point comparison
of HAMP and GPM measurements and products.
HAMP is the [*HALO Microwave Package*](https://doi.org/10.5194/amt-7-4539-2014) that was one of the instruments operated
on board of the German HALO research aircraft during the [EUREC4A](http://eurec4a.eu/) field
experiment.

The script can be used as a starting point for further studies of airborne
and spaceborne atmospheric microwave observations.

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

# Plot Example
![20200211_89 00 HH_90 00](https://user-images.githubusercontent.com/5948670/111300806-c3c6f100-8651-11eb-9a3f-b595b38c500a.png)
