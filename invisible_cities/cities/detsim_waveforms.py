import numpy as np

from invisible_cities.core.core_functions import in_range

##################################
######### WAVEFORMS ##############
##################################
def create_waveform(times    : np.ndarray,
                    bins     : np.ndarray,
                    pes      : np.ndarray = None,
                    nsamples : int = None) -> np.ndarray:
    """
    This function builds a waveform from a set of (buffer_time, pes) values.
    This set is of values come from the times and pes arguments.

    Input
    times: a vector with the buffer times in which a photoelectron is produced in the
    detector.
    pes (from photoelecton): a vector with the photoelectrons produced in the detector in
    each of the buffer times in times argument.
    bins: a vector with the output waveform bin times (for example [0, 25, 50, ...] if
    the detector has a sampling time of 25).
    nsamples: an integer that controlls the distribution of the photoelectrons in each of
    the waveform bins. The counts (N) in a given time bin (T) are distributed in the waveform
    such as the nsamples posterior to T would have N/nsamples counts (included T).
    nsamples must be >=1 an <len(bins).
    """
    ##### S1 type signal ######
    if not nsamples:
        wf, _ = np.histogram(times, bins=bins)
        return wf

    ##### S2 type signal ######
    if (nsamples<1) or (nsamples>len(bins)):
        raise ValueError("nsamples must lay betwen 1 and len(bins) (inclusive)")

    wf = np.zeros(len(bins)-1 + nsamples-1)
    if np.sum(pes)==0:
        return wf[:len(bins)-1]

    t = np.repeat(times, pes)
    sel = in_range(t, bins[0], bins[-1])

    indexes = np.digitize(t[sel], bins)-1
    indexes, counts = np.unique(indexes, return_counts=True)

    i_sample = np.arange(nsamples)
    for index, c in zip(indexes, counts):
        idxs = np.random.choice(i_sample, size=c)
        idx, sp = np.unique(idxs, return_counts=True)
        wf[index + idx] += sp
    return wf[:len(bins)-1]


def create_sensor_waveforms(wf_buffer_time : float,
                            bin_width      : float,
                            signal_type    : str,
                            times          : np.ndarray,
                            pes_at_sensors : np.ndarray = None,
                            nsamples       : int = None) -> np.ndarray:
    """
    This function calls recursively to create_waveform. See create_waveform for
    an explanation of the arguments not explained below.

    Input:
    pes_at_sensors: an array with size (#sensors, len(times)). It is the same
    as pes argument in create_waveform but for each sensor in axis 0.
    wf_buffer_time: a float with the waveform extent (in default IC units)
    bin_width: a float with the time distance between bins in the waveform buffer.
    poisson: a bool. This must be set to True to poisson distribute each bin count.
    """

    bins = np.arange(0, wf_buffer_time + bin_width, bin_width)
    if signal_type == "S1":
        wfs = np.stack([create_waveform(time, bins) for time in times])

    if signal_type == "S2":
        if not np.any(pes_at_sensors):
            raise ValueError("pes_at_sensors must be specified for S2 signal type")
        if not nsamples:
            raise ValueError("nsamples must be specified for S2 signal type")
        wfs = np.stack([create_waveform(times, bins, pes, nsamples) for pes in pes_at_sensors])

    return wfs
