import os
import numpy as np

from . simulate_S1 import generate_S1_time
from . simulate_S1 import generate_S1_times_from_pes
from . simulate_S1 import create_S1_waveforms
from . simulate_S1 import compute_S1_pes_at_pmts
from . simulate_S1 import create_lighttable_function
from . simulate_S1 import binedges_from_bincenters

from pytest import raises

from hypothesis import given
from hypothesis import settings
from hypothesis.strategies import integers


@settings(deadline=1000)
@given(integers(min_value=0, max_value=10000))
def test_generate_S1_time(size):

    s1times = generate_S1_time(size)

    # size
    assert len(s1times) == size
    # np.ndarray
    assert issubclass(type(s1times), (np.ndarray))


def test_generate_S1_times_from_pes():

    S1_pes_at_pmts = np.array([[1, 2, 0, 1, 2, 1, 0]])
    hit_times      = np.array([1, 0, 0, 1, 0, 1, 1])

    S1times = generate_S1_times_from_pes(S1_pes_at_pmts, hit_times)

    # shape is same as npmts
    assert len(S1times) == S1_pes_at_pmts.shape[0]
    # shape of each S1-times per pmt is same as total number of pes
    for S1t, pes_at_pmt in zip(S1times, S1_pes_at_pmts):
        assert len(S1t) == np.sum(pes_at_pmt)


def test_create_S1_waveforms():

    S1_times = [np.array([0, 1, 2, 3, 3, 10]), np.array([2, 2])]
    buffer_length = 5
    bin_width = 1

    wfs = create_S1_waveforms(S1_times, buffer_length, bin_width)

    assert wfs.shape == (2, 5)
    assert np.all(wfs[0] == np.array([1, 1, 1, 2, 0]))
    assert np.all(wfs[1] == np.array([0, 0, 2, 0, 0]))


def test_compute_S1_pes_at_pmts(ICDATADIR):
    s1_lighttable = os.path.join(ICDATADIR, "NEXT_NEW.energy.S1.PmtR11410.LightTable.h5")
    S1_LT = create_lighttable_function(s1_lighttable)

    xs = np.array([1, 2])
    ys = np.array([1, 2])
    zs = np.array([1, 2])
    photons = np.array([1000, 10000])
    S1_pes_at_pmts = compute_S1_pes_at_pmts(xs, ys, zs, photons, S1_LT)

    assert S1_pes_at_pmts.shape[-1] == len(xs)


def test_create_lighttable_function(ICDATADIR):
    s1_lighttable = os.path.join(ICDATADIR, "NEXT_NEW.energy.S1.PmtR11410.LightTable.h5")
    S1_LT = create_lighttable_function(s1_lighttable)

    # is function
    assert callable(S1_LT)

    # different size input
    with raises(Exception, match="input arrays must be of same shape"):
        x = np.array([1])
        y = np.array([1, 2])
        z = np.array([1])
        S1_LT(x, y, z)


def test_binedges_from_bincenters():

    bincenters = np.array([-1, 0, 2, 10])
    binedges = binedges_from_bincenters(bincenters)
    # particular case
    assert np.all(binedges == np.array([-1, -0.5, 1, 6, 10]))

    bincenters = np.array([-1, 0, 2, 10, 9])
    # exception
    with raises(Exception, match="Unsorted or repeted bin centers"):
        binedges_from_bincenters(bincenters)
