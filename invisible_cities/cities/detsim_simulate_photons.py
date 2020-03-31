import numpy  as np

from typing import Callable


#######################################
########## PHOTON SIMULATION ##########
#######################################
def generate_s1_photons(energies : np.ndarray,
                        ws       : float) -> np.ndarray:
    """generate s1 photons"""
    return np.random.poisson(energies / ws)


def generate_s2_photons(n              : int,
                        el_gain        : float,
                        el_gain_sigma  : float) -> np.ndarray:
    """generate number of EL-photons produced by secondary electrons that reach
    the EL (after drift and diffusion)
    """
    nphs = np.random.normal(el_gain, el_gain_sigma, size = n)
    return nphs


def pes_at_sensors(xs         : np.ndarray,
                   ys         : np.ndarray,
                   photons    : np.ndarray,
                   zs         : np.ndarray = None,
                   LT         : Callable   = None,
                   psf        : Callable   = None,
                   x_sensors  : np.ndarray = None,
                   y_sensors  : np.ndarray = None,
                   z_sensors  : np.ndarray = None) -> np.ndarray:
    """compute the pes that reach each sensor, based on
    the sensor psf"""

    if psf:
        dxs = xs[:, np.newaxis] - x_sensors
        dys = ys[:, np.newaxis] - y_sensors
        
        pes = photons[:, np.newaxis] * psf(dxs, dys)
        pes = np.random.poisson(pes)
    elif LT:
        if np.any(zs):
            pes = photons[:, np.newaxis] * LT(xs, ys, zs)
        else:
            pes = photons[:, np.newaxis] * LT(xs, ys)
        pes = np.random.poisson(pes)
    return pes.T
