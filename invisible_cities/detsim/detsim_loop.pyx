import numpy as np
cimport numpy as np
from libc.math cimport sqrt, round, ceil, floor
cimport cython
from ..core.core_functions import in_range


#@cython.boundscheck(False)
#@cython.wraparound(False)
def electron_loop(np.ndarray[double, ndim=1] dx,
                  np.ndarray[double, ndim=1] dy,
                  np.ndarray[double, ndim=1] times,
                  np.ndarray[unsigned long, ndim=1] photons,
                  np.ndarray[double, ndim=1] xsipms,
                  np.ndarray[double, ndim=1] ysipms,
                  np.ndarray[double, ndim=2] PSF,
                  np.ndarray[double, ndim=1] distance_centers,
                  np.ndarray[double, ndim=1] EL_times,
                  double sipm_time_bin,
                  int len_sipm_time_bins
):
    cdef int nsipms = xsipms.shape[0]
    cdef np.ndarray[double, ndim=2] sipmwfs = np.zeros([nsipms, len_sipm_time_bins], dtype=np.float64)
    cdef int numel = dx.shape[0]
    cdef int npartitions = PSF.shape[1]
    cdef double EL_bin = distance_centers[1]-distance_centers[0]
    cdef double max_dist = max(distance_centers)
    cdef int sipmwf_timeindx
    cdef double ts
    cdef int psf_bin
    cdef double signal
    for sipmindx in range(nsipms):
        for elindx in range(numel):
            dist = sqrt((dx[elindx]-xsipms[sipmindx])**2+(dy[elindx]-ysipms[sipmindx])**2)
            if dist>max_dist:
                continue
            for partindx in range(npartitions):
                ts = times[elindx] + EL_times[partindx]
                psf_bin = <int> floor(dist/EL_bin)
                signal = PSF[psf_bin, partindx]/npartitions*photons[elindx]
                sipmwf_timeindx = <int> floor(ts/sipm_time_bin)
                sipmwfs[sipmindx,sipmwf_timeindx] += signal
    return sipmwfs


############################
###### create_waveform #####
############################

# cdef np.ndarray[np.int_t, ndim=1] weighted_histogram(np.ndarray[double, ndim=1] values,
#                                                      np.ndarray[np.int_t, ndim=1] weights,
#                                                      np.ndarray[double, ndim=1] bins):
#     cdef np.ndarray[np.int_t, ndim=1] histogram = np.zeros(len(bins)-1, dtype=np.int)
#     cdef int nbins = len(bins)
#     cdef double binwidth = bins[1] - bins[0]
#     cdef int i, j
#
#     for i in range(len(values)):
#         j = <int> floor(values[i]/binwidth)
#         histogram[j] += weights[i]
#     return histogram

cdef np.ndarray[double, ndim=1] weighted_histogram(np.ndarray [double, ndim=1] values,
                                                    np.ndarray[double, ndim=1] weights,
                                                    np.ndarray[double, ndim=1] bins):
    cdef np.ndarray[double, ndim=1] histogram = np.zeros(len(bins)-1)
    cdef int nbins = len(bins)
    cdef double binwidth = bins[1] - bins[0]
    cdef int i, j

    for i in range(len(values)):
        j = <int> floor(values[i]/binwidth)
        histogram[j] += weights[i]
    return histogram


cdef np.ndarray[double, ndim=1] spread_histogram(np.ndarray[double, ndim=1] histogram, int nsamples):

    cdef int l = len(histogram)
    cdef np.ndarray[double, ndim=1] spreaded = np.zeros(l + nsamples-1)
    cdef int h, i
    cdef double v

    for h in range(l):
        if histogram[h]>0:
            v = histogram[h]/nsamples
            for i in range(nsamples):
                spreaded[h + i] += v
    return spreaded[:l]


def create_waveform(values, weights, bins, nsamples):

    if (nsamples<1) or (nsamples>len(bins)):
        raise ValueError("nsamples must lay betwen 1 and len(bins) (inclusive)")

    cdef np.ndarray sel = in_range(values, bins[0], bins[-1])

    cdef np.ndarray[double, ndim=1] wf = weighted_histogram(values[sel], weights[sel], bins)
    if nsamples>1:
        return spread_histogram(wf, nsamples)

    return wf
