import numpy as np
import scipy
from typing import Callable
import pandas as pd
from .. core.core_functions import in_range
from .. detsim.detsim_loop  import electron_loop
from .. detsim.detsim_loop  import create_waveform


def create_pmt_waveforms(signal_type   : str,
                         buffer_length : float,
                         bin_width     : float) -> Callable:
    """
    This function calls recursively to create_waveform. See create_waveform for
    an explanation of the arguments not explained below.

    Parameters
        :pes_at_sensors:
            an array with size (#sensors, len(times)). It is the same
            as pes argument in create_waveform but for each sensor in axis 0.
        :wf_buffer_time:
            a float with the waveform extent (in default IC units)
        :bin_width:
            a float with the time distance between bins in the waveform buffer.
    Returns:
        :create_sensor_waveforms_: function
    """
    bins = np.arange(0, buffer_length + bin_width, bin_width)

    if signal_type=="S1":
        # @profile
        def create_pmt_waveforms_(S1times : list):
            wfs = np.stack([np.histogram(times, bins=bins)[0] for times in S1times])
            return wfs

    elif signal_type=="S2":
        # @profile
        def create_pmt_waveforms_(times          : np.ndarray,
                                  pes_at_sensors : np.ndarray,
                                  nsamples       : int = 1):
            wfs = np.stack([create_waveform(times, pes, bins, nsamples) for pes in pes_at_sensors])
            wfs = np.random.poisson(wfs)
            return wfs
    else:
        ValueError("signal_type must be one of S1 or S1")

    return create_pmt_waveforms_


def create_sipm_waveforms(wf_buffer_length  : float,
                          wf_sipm_bin_width : float,
                          datasipm : pd.DataFrame,
                          PSF : pd.DataFrame,
                          EL_dz : float,
                          el_pitch : float,
                          drift_velocity_EL : float):
    ELtimes = np.arange(el_pitch/2., EL_dz, el_pitch)/drift_velocity_EL

    xsipms, ysipms = datasipm["X"].values, datasipm["Y"].values
    PSF_distances = PSF.index.values
    PSF_values = PSF.values

    # @profile
    def create_sipm_waveforms_(times,
                               photons,
                               dx,
                               dy):
        waveform_nbins = wf_buffer_length//wf_sipm_bin_width
        sipmwfs =  electron_loop(dx.astype(np.float64), dy.astype(np.float64), times.astype(np.float64), photons.astype(np.uint),
                                 xsipms.astype(np.float64), ysipms.astype(np.float64), PSF_values.astype(np.float64), PSF_distances.astype(np.float64),
                                 ELtimes.astype(np.float64), wf_sipm_bin_width, waveform_nbins)
        sipmwfs = np.random.poisson(sipmwfs)
        return sipmwfs

    return create_sipm_waveforms_
