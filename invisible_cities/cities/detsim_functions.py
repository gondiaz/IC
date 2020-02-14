import numpy  as np
import tables as tb
import pandas as pd

from typing    import   List, Tuple, Callable, Generator

from invisible_cities.io.mcinfo_io import read_mchits_df
from invisible_cities.reco.corrections_new import read_maps


#######################################
############### SOURCE ################
#######################################
def load_MC(files_in : List[str]) -> Generator:
    for filename in files_in:
        with tb.open_file(filename) as h5in:
            extents = pd.read_hdf(filename, 'MC/extents')
            event_ids  = extents.evt_number
            hits_df    = read_mchits_df(h5in, extents)
            for evt in event_ids:
                hits = hits_df.loc[evt, :, :]
                yield dict(event_number = evt,
                           x      = hits["x"]     .values,
                           y      = hits["y"]     .values,
                           z      = hits["z"]     .values,
                           energy = hits["energy"].values)


#######################################
######### ELECTRON SIMULATION #########
#######################################
def generate_electrons(energies    : np.array,
                       wi          : float,
                       fano_factor : float ) -> np.array:
    """ generate secondary electrons from energy deposits
    """
    nes  = np.array(energies/wi, dtype = int)
    pois = nes < 10
    nes[ pois] = np.random.poisson(nes[pois])
    nes[~pois] = np.round(np.random.normal(nes[~pois], np.sqrt(nes[~pois] * fano_factor)))
    return nes


def drift_electrons(zs             : np.array,
                    electrons      : np.array,
                    lifetime       : float,
                    drift_velocity : float) -> np.array:
    """ returns number of electrons due to lifetime loses from secondary electrons
    """
    ts  = zs / drift_velocity
    nes = electrons - np.random.poisson(electrons * (1. - np.exp(-ts/lifetime)))
    nes[nes < 0] = 0
    return nes


def diffuse_electrons(xs                     : np.array,
                      ys                     : np.array,
                      zs                     : np.array,
                      electrons              : np.array,
                      transverse_diffusion   : float,
                      longitudinal_diffusion : float)\
                      -> Tuple[np.array, np.array, np.array, np.array]:
    """
    starting from a voxelized electrons with positions xs, ys, zs, and number of electrons,
    apply diffusion and return voxelixed electrons with positions xs, ys, zs, an electrons
    the voxel_size arguement controls the size of the voxels for the diffused electrons
    """
    xs = np.repeat(xs, electrons.astype(int))
    ys = np.repeat(ys, electrons.astype(int))
    zs = np.repeat(zs, electrons.astype(int))

    sqrtz = zs ** 0.5
    dxs  = np.random.normal(xs, sqrtz * transverse_diffusion)
    dys  = np.random.normal(ys, sqrtz * transverse_diffusion)
    dzs  = np.random.normal(zs, sqrtz * longitudinal_diffusion)

    return (dxs, dys, dzs)


#######################################
########## PHOTON SIMULATION ##########
#######################################
def generate_s1_photons(energies : np.array,
                        ws       : float) -> np.array:
    """ generate s1 photons
    """
    return np.random.poisson(energies / ws)


def generate_s2_photons(x              : np.array,
                        el_gain        : float,
                        el_gain_sigma  : float) -> np.array:
    """ generate number of EL-photons produced by secondary electrons that reach
    the EL (after drift and diffusion)
    """
    n = len(x)
    nphs = np.random.normal(el_gain, el_gain_sigma, size = n)
    return nphs


def photons_at_sensors(xs         : np.array,
                       ys         : np.array,
                       zs         : np.array,
                       photons    : np.array,
                       x_sensors  : np.array,
                       y_sensors  : np.array,
                       z_sensors  : float   ,
                       psf : Callable) -> np.array:
    """Compute the photons that reach each sensor, based on
    the sensor psf"""

    dxs = xs[:, np.newaxis] - x_sensors
    dys = ys[:, np.newaxis] - y_sensors
    dzs = zs[:, np.newaxis] - z_sensors
    photons = photons[:, np.newaxis]

    phs = photons * psf(dxs, dys, dzs)
    phs = np.random.poisson(phs)
    return phs.T


##################################
############# PSF ################
##################################
def _psf(dx, dy, dz, factor = 1.):
    """ generic analytic PSF function
    """
    return factor * np.abs(dz) / (2 * np.pi) / (dx**2 + dy**2 + dz**2)**1.5


def create_xy_function(array, xaxis, yaxis):
    xmin, xmax, dx = xaxis
    ymin, ymax, dy = yaxis
    def function(x, y, z=0):
        x = np.clip(x, xmin, xmax-0.1*dx)
        y = np.clip(y, ymin, ymax-0.1*dx)
        ix = ((x-xmin)//dx).astype(int)
        iy = ((y-ymin)//dy).astype(int)
        return array[ix, iy]
    return function


def get_psf_from_krmap(filename, factor=1.):
    """ reads KrMap and generate a psf function with the E0-map
    """
    maps = read_maps(filename)
    xmin, xmax, ymin, ymax, nx, ny, _ = maps.mapinfo.values
    dx   = (xmax - xmin)/ float(nx)
    dy   = (ymax - ymin)/ float(ny)
    e0map  = factor * np.nan_to_num(np.array(maps.e0), 0.)

    return create_xy_function(e0map, (xmin, xmax, dx), (ymin, ymax, dy))


def get_sipm_psf_from_file(filename):
    with tb.open_file(filename) as h5file:
        psf = h5file.root.PSF.PSFs.read()

    #select psf z
    sel = (psf["z"] == np.min(psf["z"]))
    psf = psf[sel]
    xr, yr, factor = psf["xr"], psf["yr"], psf["factor"]

    #create binning
    xbins, ybins = np.unique(xr), np.unique(yr)
    dx = (xbins[-1] - xbins[0])/(len(xbins)-1)
    dy = (ybins[-1] - ybins[0])/(len(ybins)-1)
    xbins = np.append(xbins-dx/2., xbins[-1]+dx/2.)
    ybins = np.append(ybins-dy/2., ybins[-1]+dy/2.)

    #histogram
    psf, _ = np.histogramdd((xr, yr), weights=factor, bins=(xbins, ybins))
    xaxis = (np.min(xbins), np.max(xbins), dx)
    yaxis = (np.min(ybins), np.max(ybins), dy)

    return create_xy_function(psf, xaxis, yaxis)


##################################
######### WAVEFORMS ##############
##################################
def bincounter(xs, dx = 1., x0 = 0.):
    #ixs = ((xs + x0) // dx).astype(int)
    ixs = ((xs - x0) // dx).astype(int)
    return np.unique(ixs, return_counts=True)


def create_waveform(times   : np.array,
                    photons : np.array,
                    bins    : np.array,
                    wf_bin_time : float,
                    nsamples    : float) -> np.array:
    wf   = np.zeros(len(bins))

    if np.sum(photons)==0:
        return wf

    sel = photons>0
    t = np.repeat(times[sel], photons[sel])
    t = np.clip  (t, 0, bins[-nsamples])
    indexes, counts = bincounter(t, wf_bin_time)

    spread_photons = np.repeat(counts[:, np.newaxis]/nsamples, nsamples, axis=1)
    for index, counts in zip(indexes, spread_photons):
        wf[index:index+nsamples] = wf[index:index+nsamples] + counts

    return wf


def create_sensor_waveforms(times   : np.array,
                            photons : np.array,
                            wf_buffer_time : float,
                            wf_bin_time    : float,
                            nsamples : float,
                            poisson  : bool=False) -> np.array:
    bins = np.arange(0, wf_buffer_time, wf_bin_time)
    wfs = np.array([create_waveform(times, phs, bins, wf_bin_time, nsamples) for phs in photons])

    if poisson:
        wfs = np.random.poisson(wfs)

    return wfs


# def _wf(its, iphs, iwf, wf_bin_time, nsamples):
#     if (np.sum(iphs) <= 0): return iwf
#     isel       = iphs > 0
#     nts        = np.repeat(its[isel], iphs[isel])
#     sits, sphs = bincounter(nts, wf_bin_time)
#     sphsn      = np.random.poisson(sphs/nsamples, size = (nsamples, sphs.size))
#     for kk, kphs in enumerate(sphsn):
#         iwf[sits + kk] = iwf[sits + kk] + kphs
#     return iwf
#
#
# def sample_photons_and_fill_wfs(ts          : np.array,
#                                 phs         : np.array,
#                                 wfs         : np.array,
#                                 wf_bin_time : float,
#                                 nsamples    : int):
#     """ Create the wfs starting from the photons arrived at each sensor.
#     The control parameters are the wf_bin_time and the nsamples.
#     Returns: waveforms
#     """
#     out = np.array([_wf(ts, iphs, iwf, wf_bin_time, nsamples) for iphs, iwf in zip(phs, wfs)])
#     return out
