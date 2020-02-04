import numpy  as np
import tables as tb
import pandas as pd

from typing    import   List, Tuple, Callable, Generator

from invisible_cities.io.mcinfo_io import read_mchits_df


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
                yield dict(evt    = evt,
                           x      = hits["x"]     .values,
                           y      = hits["y"]     .values,
                           z      = hits["z"]     .values,
                           energy = hits["energy"].values)


#######################################
######### ELECTRON SIMULATION #########
#######################################

def bincounterdd(xxs, dxs = 1., x0s = 0., n = 1000):
    xs = np.array(xxs, dtype = float).T
    dxs = np.ones_like(xs) * dxs
    x0s = np.ones_like(xs) * x0s

    ixs = ((xs - x0s) // dxs).astype(int)
    ids, ccs =  np.unique(ixs, axis = 0, return_counts = True)
    return ids.T, ccs


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
                      longitudinal_diffusion : float,
                      voxel_sizes            : np.array)\
                      -> Tuple[np.array, np.array, np.array, np.array]:
    """
    starting from a voxelized electrons with positions xs, ys, zs, and number of electrons,
    apply diffusion and return voxelixed electrons with positions xs, ys, zs, an electrons
    the voxel_size arguement controls the size of the voxels for the diffused electrons
    """
    nes = electrons.astype(int)
    voxel_sizes = np.array(voxel_sizes)
    xs = np.repeat(xs, nes); ys = np.repeat(ys, nes); zs = np.repeat(zs, nes)

    sqrtz = zs ** 0.5
    vxs  = np.random.normal(xs, sqrtz * transverse_diffusion)
    vys  = np.random.normal(ys, sqrtz * transverse_diffusion)
    vzs  = np.random.normal(zs, sqrtz * longitudinal_diffusion)
    vnes = np.ones(vxs.size)
    vpos, vnes = bincounterdd((vxs, vys, vzs), voxel_sizes)
    vxs, vys, vzs = voxel_sizes[:, np.newaxis] * vpos
    return (vxs, vys, vzs, vnes)
