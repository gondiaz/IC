import os
import pytest

import numpy as np

from pytest import warns

from .. database.load_db import DataSiPM

from .. core             import system_of_units as units

from . lighttables     import LT
from . lighttables     import LT_SiPM
from . lighttables     import LT_PMT
from . ielectrons_loop import electron_loop

from hypothesis                import given
from hypothesis                import settings
from hypothesis.strategies     import floats
from hypothesis.strategies     import integers
from hypothesis.extra.numpy    import arrays


@given(xs=arrays(np.float, 10, elements = floats  (min_value = -500, max_value = 500)),
       ys=arrays(np.float, 10, elements = floats  (min_value = -500, max_value = 500)),
       ts=arrays(np.float, 10, elements = floats  (min_value =    0, max_value = 100)),
       ps=arrays(np.int32, 10, elements = integers(min_value =    10, max_value = 100*units.mus)))
def test_ielectron_loop_sipms_shape(get_dfs, xs, ys, ts, ps):
    datasipm = DataSiPM('new')
    fname, psf_df, psf_conf = get_dfs['psf']
    EL_drift_velocity = 2.5*units.mm/units.mus
    sensor_time_bin = 1*units.mus
    buffer_length = 200*units.mus
    LT = LT_SiPM(fname=fname, sipm_database=datasipm)
    n_sensors = len(LT.sensor_ids)
    waveform = electron_loop(xs, ys, ts, ps, LT, EL_drift_velocity, sensor_time_bin, buffer_length)
    assert isinstance(waveform, np.ndarray)
    assert waveform.shape ==  (n_sensors, buffer_length//sensor_time_bin)


@given(xs=arrays(np.float, 10, elements = floats  (min_value = -500, max_value = 500)),
       ys=arrays(np.float, 10, elements = floats  (min_value = -500, max_value = 500)),
       ts=arrays(np.float, 10, elements = floats  (min_value =    0, max_value = 100)),
       ps=arrays(np.int32, 10, elements = integers(min_value =    10, max_value = 100*units.mus)))
def test_ielectron_loop_pmts_shape(get_dfs, xs, ys, ts, ps):
    fname, lt_df, lt_conf = get_dfs['lt']
    EL_drift_velocity = 2.5*units.mm/units.mus
    sensor_time_bin = 100*units.ns
    buffer_length = 200*units.mus
    LT = LT_PMT(fname=fname)
    n_sensors = len(LT.sensor_ids)
    waveform = electron_loop(xs, ys, ts, ps, LT, EL_drift_velocity, sensor_time_bin, buffer_length)
    assert isinstance(waveform, np.ndarray)
    assert waveform.shape == (n_sensors, buffer_length//sensor_time_bin)
