import numpy  as np
import tables as tb
import pandas as pd

from typing import Tuple
from typing import Callable

from invisible_cities.reco.corrections     import read_maps

from invisible_cities.core.core_functions  import in_range


###################################
############# UTILS ###############
###################################
def create_xyz_function(H, bins):
    """Given a 3D array and a list of bins for
    each dim, it returns a x,y,z function"""

    xbins, ybins, zbins = bins
    if not H.shape == (len(xbins)-1, len(ybins)-1, len(zbins)-1):
        raise Exception("bins and array shapes not consistent")

    def function(x, y, z):
        if not x.shape==y.shape==z.shape:
            raise Exception("x, y and z must have same size")

        out = np.zeros(x.shape)
        #select values inside bin extremes
        selx = in_range(x, xbins[0], xbins[-1])
        sely = in_range(y, ybins[0], ybins[-1])
        selz = in_range(z, zbins[0], zbins[-1])
        sel = selx & sely & selz

        ix = np.digitize(x[sel], xbins)-1
        iy = np.digitize(y[sel], ybins)-1
        iz = np.digitize(z[sel], zbins)-1

        out[sel] = H[ix, iy, iz]
        return out
    return function

def create_xy_function(H, bins):
    """Given a 2D array and a list of bins for
    each dim, it returns a x,y,z function"""

    xbins, ybins = bins
    if not H.shape == (len(xbins)-1, len(ybins)-1):
        raise Exception("bins and array shapes not consistent")

    def function(x, y):
        if not x.shape==y.shape:
            raise Exception("x, y and z must have same size")

        out = np.zeros(x.shape)
        #select values inside bin extremes
        selx = in_range(x, xbins[0], xbins[-1])
        sely = in_range(y, ybins[0], ybins[-1])
        sel = selx & sely

        ix = np.digitize(x[sel], xbins)-1
        iy = np.digitize(y[sel], ybins)-1

        out[sel] = H[ix, iy]
        return out
    return function


def binedges_from_bincenters(bincenters):

    binedges = np.zeros(len(bincenters)+1)

    binedges[1:-1] = (bincenters[1:] + bincenters[:-1])/2.
    binedges[0]  = bincenters[0]
    binedges[-1] = bincenters[-1]

    return binedges

# def binedges_from_bincenters(bincenters):
#     ds = np.diff(bincenters)
#     if not np.allclose(ds, ds[0]):
#         raise Exception("Bin distances must be equal")
#
#     d = ds[0]
#     return np.arange(bincenters[0]-d/2., bincenters[-1]+d/2.+d, d)

##################################
############# PSF ################
##################################
def get_psf(filename):
    with tb.open_file(filename) as h5file:
        PSF = h5file.root.LightTable.table.read()
        info = dict(h5file.root.Config.table.read())

    PSF  = np.sort(PSF, order="index")
    bins = binedges_from_bincenters(PSF["index"])
    PSF = np.array(PSF.tolist())[:, 1:]

    def psf(d):
        out = np.zeros((*d.shape, PSF.shape[1]))
        sel = in_range(d, bins[0], bins[-1])
        idxs = np.digitize(d[sel], bins)-1
        out[sel] = PSF[idxs]
        return out            #(nsensors, nhits, npartitions)

    return psf, info


##################################
######### LIGTH TABLE ############
##################################
def get_ligthtables(filename: str,
                    signal  : str)->Callable:
    ##### Load LT ######
    with tb.open_file(filename) as h5file:
        LT = h5file.root.LightTable.table.read()
        info = dict(h5file.root.Config.table.read())

    sensor = str(info[b'sensor'], errors="ignore")
    sensors = [name for name in LT.dtype.names if sensor in name and "total" not in name]
    sensors.sort(key=lambda name: int(name.split("_")[-1]))

    if signal == "S1":
        #### XYZ binning #####
        x, y, z = LT["x"], LT["y"], LT["z"]

        xcenters, ycenters, zcenters = np.unique(x), np.unique(y), np.unique(z)
        xbins = binedges_from_bincenters(xcenters)
        ybins = binedges_from_bincenters(ycenters)
        zbins = binedges_from_bincenters(zcenters)
        bins  = [xbins, ybins, zbins]

        ###### CREATE XYZ FUNCTION FOR EACH SENSOR ######
        func_per_sensor = []
        for sensor in sensors:
            w = LT[sensor]
            H, _ = np.histogramdd((x, y, z), weights=w, bins=bins)
            fxyz = create_xyz_function(H, bins)
            func_per_sensor.append(fxyz)

        ###### CREATE XYZ CALLABLE FOR LIST OF XYZ FUNCTIONS #####
        def merge_list_of_functions(list_of_functions):
            def merged(x, y, z):
                return np.array([f(x, y, z) for f in list_of_functions]).T
            return merged
        return merge_list_of_functions(func_per_sensor)

    elif signal == "S2":
        #### XYZ binning #####
        x, y = LT["x"], LT["y"]

        xcenters, ycenters = np.unique(x), np.unique(y)
        xbins = binedges_from_bincenters(xcenters)
        ybins = binedges_from_bincenters(ycenters)
        bins  = [xbins, ybins]

        ###### CREATE XY FUNCTION FOR EACH SENSOR ######
        func_per_sensor = []
        # sensors = [f"PmtR11410_{i}" for i in range(0, 12)]
        for sensor in sensors:
            w = LT[sensor]

            H, _ = np.histogramdd((x, y), weights=w, bins=bins)
            fxy = create_xy_function(H, bins)

            func_per_sensor.append(fxy)

        ###### CREATE XY CALLABLE FOR LIST OF XY FUNCTIONS #####
        def merge_list_of_functions(list_of_functions):
            def merged(x, y):
                return np.array([f(x, y) for f in list_of_functions]).T
            return merged
        return merge_list_of_functions(func_per_sensor)
