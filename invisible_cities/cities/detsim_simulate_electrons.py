import numpy as np

from typing import List
from typing import Tuple
from typing import Callable

#######################################
######### ELECTRON SIMULATION #########
#######################################
def generate_ionization_electrons(energies    : np.array,
                                  wi          : float,
                                  fano_factor : float ) -> np.array:
    """ generate ionization secondary electrons from energy deposits
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
                      -> Tuple[np.array, np.array, np.array]:
    """
    starting from hits with positions xs, ys, zs, and number of electrons,
    apply diffusion and return diffused positions xs, ys, zs for each electron
    """
    xs = np.repeat(xs, electrons.astype(int))
    ys = np.repeat(ys, electrons.astype(int))
    zs = np.repeat(zs, electrons.astype(int))

    sqrtz = zs ** 0.5
    dxs  = np.random.normal(xs, sqrtz * transverse_diffusion)
    dys  = np.random.normal(ys, sqrtz * transverse_diffusion)
    dzs  = np.random.normal(zs, sqrtz * longitudinal_diffusion)

    return (dxs, dys, dzs)
