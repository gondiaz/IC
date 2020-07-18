import numpy  as np

from typing import Callable


#######################################
########## PHOTON SIMULATION ##########
#######################################
def generate_s1_photons(ws       : float,
                        energies : np.ndarray) -> np.ndarray:
    """generate s1 photons"""
    return np.random.poisson(energies / ws)


def generate_s2_photons(el_gain        : float,
                        el_gain_sigma  : float,
                        nes            : np.ndarray) -> np.ndarray:
    """generate number of EL-photons produced by secondary electrons that reach
    the EL (after drift and diffusion)
    """
    nphs = np.random.normal(el_gain*nes, np.sqrt(el_gain_sigma**2*nes))
    return nphs
