import numpy as np
import pandas as pd

from typing import Tuple
from typing import Callable

from scipy.optimize import brentq

from .. core.core_functions import in_range
from .. core import system_of_units as units

from .. io.dst_io  import load_dst


def compute_S1_pes_at_pmts(xs      : np.ndarray,
                           ys      : np.ndarray,
                           zs      : np.ndarray,
                           photons : np.ndarray,
                           LT      : Callable  )->np.ndarray:
    """Compute the pes generated in each PMT from S1 photons
    Parameters:
        :LT: function
            The Light Table in functional form
        :photons: np.ndarray
            The photons generated at each hit
        :xs, ys, zs: np.ndarray
            hit position
    Returns:
        :pes: np.ndarray
            photoelectrons at each PMT produced by all hits.
    """
    pes = photons[:, np.newaxis] * LT(xs, ys, zs)
    pes = np.random.poisson(pes)
    return pes.T

tau1 = 4.5; c1 = 0.1
tau2 = 100; c2 = 0.9

N = 1/(c1*tau1 + c2*tau2)
A1 = tau1*c1*N
A2 = tau2*c2*N

def S1_minimize(x, P):
    return A1*np.exp(-x/tau1) + A2*np.exp(-x/tau2) - P

def generate_S1_time(size : int)->np.ndarray:
    """Generates random numbers following the hyper-exponential
    distribution: 0.9 exp(-x/4.5) + 0.1 exp(-x/100).
    Parameters:
        :size: int
            number of random numbers
    Returns:
        :sol: np.ndarray
            the random number values
    """
    sol = []
    for i in range(size):
        P = np.random.random()
        sol.append(brentq(S1_minimize, 0, 10000, args=P))
    return np.array(sol)


def generate_S1_times_from_pes(S1_pes_at_pmts : np.ndarray,
                               hit_times      : np.ndarray)->list:
    """Given the S1_pes_at_pmts, this function returns the times at which the pes
    are be distributed (see generate_S1_time function).
    It returns a list whose elements are the times at which the photoelectrons in that PMT
    are generated.
    Parameters:
        :S1_pes_at_pmts: np.ndarray
            the pes at each PMT generated by each hit
    Returns:
        :S1_times: list[np.ndarray,..]
            Each element are the S1 times for a PMT. If certain sensor
            do not see any pes, the array is empty.
    """
    S1_times = []
    for pes_at_pmt in S1_pes_at_pmts:
        repeated_times = np.repeat(hit_times, pes_at_pmt)
        times = generate_S1_time(np.sum(pes_at_pmt))
        S1_times.append(times + repeated_times)
    return S1_times


def create_S1_waveforms(S1_times      : list,
                        buffer_length : float,
                        bin_width     : float)->np.ndarray:
    """
    Create S1 waveform : S1_times are histogramed in a waveform
    of given buffer_length and bin_width
    Parameters
        :S1_times: list
            output of generate_S1_times_from_pes, list of size equal to the
            number of pmts. Each element is an array with the times of the S1
    Returns:
        :wfs: np.ndarray
            waveforms with buffer_length and bin_width
    """
    bins = np.arange(0, buffer_length + bin_width, bin_width)
    wfs  = np.stack([np.histogram(times, bins=bins)[0] for times in S1_times])
    return wfs


def create_lighttable_function(filename : str)->Callable:
    """ From a lighttable file, it returns a function of (x, y) for S2 signal
    or (x, y, z) for S1 signal type. Signal type is read from the table.
    Parameters:
        :filename: str
            name of the lighttable file
    Returns:
        :get_lt_values: Callable
            this is a function which access the desired value inside
            the lighttable. The lighttable values would be the nearest
            points to the input positions. If the input positions are
            outside the lighttable boundaries, zero is returned.
            Input values must be vectors of same lenght, I. The output
            shape will be (I, number_of_pmts).
    """

    lt     = load_dst(filename, "LT", "LightTable")
    Config = load_dst(filename, "LT", "Config")    .set_index("parameter")

    signaltype = Config.loc["signal_type"].value
    sensor     = Config.loc["sensor"]     .value

    lt = lt.drop(sensor + "_total", axis=1) # drop total column

    if signaltype == "S1":

        lt = lt.set_index(["x", "y", "z"])
        nsensors = lt.shape[-1]

        xcenters = np.sort(np.unique(lt.index.get_level_values('x')))
        ycenters = np.sort(np.unique(lt.index.get_level_values('y')))
        zcenters = np.sort(np.unique(lt.index.get_level_values('z')))

        xbins=binedges_from_bincenters(xcenters)
        ybins=binedges_from_bincenters(ycenters)
        zbins=binedges_from_bincenters(zcenters)

        def get_lt_values(xs, ys, zs):
            if not (xs.shape == ys.shape == zs.shape):
                raise Exception("input arrays must be of same shape")
            xindices = pd.cut(xs, xbins, labels=xcenters)
            yindices = pd.cut(ys, ybins, labels=ycenters)
            zindices = pd.cut(zs, zbins, labels=zcenters)
            indices = pd.Index(zip(xindices, yindices, zindices), name=("x", "y", "z"))

            mask = indices.isin(lt.index)
            values = np.zeros((len(xs), nsensors))
            values[mask] = lt.loc[indices[mask]]
            return values

    elif signaltype == "S2":

        lt = lt.set_index(["x", "y"])
        nsensors = lt.shape[-1]

        xcenters = np.sort(np.unique(lt.index.get_level_values('x')))
        ycenters = np.sort(np.unique(lt.index.get_level_values('y')))

        xbins=binedges_from_bincenters(xcenters)
        ybins=binedges_from_bincenters(ycenters)

        def get_lt_values(xs, ys):
            if not (xs.shape == ys.shape):
                raise Exception("input arrays must be of same shape")
            xindices = pd.cut(xs, xbins, labels=xcenters)
            yindices = pd.cut(ys, ybins, labels=ycenters)
            indices = pd.Index(zip(xindices, yindices), name=("x", "y"))

            mask = indices.isin(lt.index)
            values = np.zeros((len(xs), nsensors))
            values[mask] = lt.loc[indices[mask]]
            return values

    return get_lt_values


def binedges_from_bincenters(bincenters: np.ndarray)->np.ndarray:
    """
    computes bin-edges from bin-centers.
    The lowest bin edge is assigned to the lowest bin center.
    The highest bin edge is assigned to the highest bin center extended a 1%
    Parameters:
        :bincenters: np.ndarray
            bin centers
    Returns:
        :binedges: np.ndarray
            bin edges
    """
    if np.any(bincenters[:-1] >= bincenters[1:]):
        raise Exception("Unsorted or repeted bin centers")

    binedges = np.zeros(len(bincenters)+1)

    binedges[1:-1] = (bincenters[1:] + bincenters[:-1])/2.
    binedges[0]  = bincenters[0]
    binedges[-1] = bincenters[-1]*1.1

    return binedges
