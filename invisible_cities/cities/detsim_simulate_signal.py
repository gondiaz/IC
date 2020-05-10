import numpy as np

from typing import Callable


##################################
############## PES ###############
##################################
def pes_at_pmts(LT      : Callable  ,
                photons : np.ndarray,
                xs      : np.ndarray,
                ys      : np.ndarray,
                zs      : np.ndarray = None):

    """Compute the pes generated in each PMT for photons generated at some point.
    xs, ys : positions of the generated photons.
    photons: number of photons generated at xs, ys.
    LT     : lightTable
    zs     : position in z of the generated photons. This is used just for S1 signal.
    """
    if np.any(zs): #S1
        pes = photons[:, np.newaxis] * LT(xs, ys, zs)
    else:          #S2
        pes = photons[:, np.newaxis] * LT(xs, ys)
    pes = np.random.poisson(pes)
    return pes.T


def pes_at_sipms(PSF     : Callable,
                 xsensors: np.ndarray,
                 ysensors: np.ndarray,
                 photons : np.ndarray,
                 xs      : np.ndarray,
                 ys      : np.ndarray):

    xdistance = xs[:, np.newaxis] - xsensors
    ydistance = ys[:, np.newaxis] - ysensors
    distances = (xdistance**2 + ydistance**2)**0.5

    psf = PSF(distances.T)
    nsensors, nhits, npartitions = psf.shape
    psf = np.reshape(psf, (nsensors, nhits*npartitions))

    pes = np.multiply(psf, np.repeat(photons, npartitions))
    pes = np.random.poisson(pes/npartitions)
    return pes


##################################
############## TIMES #############
##################################
def generate_S1_time(size=1):
    """This function generates an array of size with a time random variable
    distributed: 0.1*exp(t/4.5) + 0.9*exp(t/100). This is the S1 emission time."""
    r = []
    for i in range(size):
        t1 = np.random.exponential(4.5)
        t2 = np.random.exponential(100)
        r.append(np.random.choice([t1, t2], p=[0.1, 0.9]))
    return np.array(r)

def generate_S1_times_from_pes(S1pes_at_pmts):
    """Given the S1pes_at_pmts, this function returns the times at which the pes
    are be distributed according with its distribution (see generate_S1_time function).
    It returns a list whose elements are the times at which the photoelectrons in that PMT
    are generated.
    """
    S1pes_pmt = np.sum(S1pes_at_pmts, axis=1)
    S1times = [generate_S1_time(size=pes) for pes in S1pes_pmt]
    return S1times

# def pes_at_sensors(xs         : np.ndarray,
#                    ys         : np.ndarray,
#                    photons    : np.ndarray,
#                    zs         : np.ndarray = None,
#                    LT         : Callable   = None,
#                    psf        : Callable   = None,
#                    x_sensors  : np.ndarray = None,
#                    y_sensors  : np.ndarray = None,
#                    z_sensors  : np.ndarray = None) -> np.ndarray:
#     """compute the pes that reach each sensor, based on
#     the sensor psf"""
#
#     if psf:
#         dxs = xs[:, np.newaxis] - x_sensors
#         dys = ys[:, np.newaxis] - y_sensors
#
#         pes = photons[:, np.newaxis] * psf(dxs, dys)
#         pes = np.random.poisson(pes)
#     elif LT:
#         if np.any(zs):
#             pes = photons[:, np.newaxis] * LT(xs, ys, zs)
#         else:
#             pes = photons[:, np.newaxis] * LT(xs, ys)
#         pes = np.random.poisson(pes)
#     return pes.T
