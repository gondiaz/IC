import os
import pytest

import numpy as np

from pytest import warns

from .. database.load_db import DataSiPM
from .. io      .dst_io  import load_dst

from . lighttables import LT
from . lighttables import LT_SiPM
from . lighttables import LT_PMT

from hypothesis                import given
from hypothesis                import settings
from hypothesis.strategies     import floats
from hypothesis.strategies     import integers

@pytest.fixture(scope='module')
def get_dfs(ICDATADIR):
    psffname = os.path.join(ICDATADIR, 'NEXT_NEW.tracking.S2.SiPM.LightTable.h5')
    ltfname  = os.path.join(ICDATADIR, 'NEXT_NEW.energy.S2.PmtR11410.LightTable.h5')
    psf_df   = load_dst(psffname, 'PSF', 'LightTable').set_index('dist_xy')
    lt_df    = load_dst( ltfname,  'LT', 'LightTable').set_index(['x', 'y'])
    psf_conf = load_dst(psffname, 'PSF', 'Config'    ).set_index('parameter')
    lt_conf  = load_dst( ltfname,  'LT', 'Config'    ).set_index('parameter')
    return  {'psf':(psffname, psf_df, psf_conf),
             'lt' : (ltfname, lt_df, lt_conf)}


def test_LT_SiPM_optional_arguments(get_dfs):
    datasipm = DataSiPM('new')
    fname, psf_df, psf_conf = get_dfs['psf']
    LT = LT_SiPM(fname=fname, sipm_database=datasipm)
    #check the values are read from teh table
    assert LT.el_gap   == psf_conf.loc['EL_GAP'    ].astype(float).value
    assert LT.active_r == psf_conf.loc['ACTIVE_rad'].astype(float).value
    #check optional arguments are set with User Warning
    with warns(UserWarning):
        LT = LT_SiPM(fname=fname, sipm_database=datasipm, el_gap=2, active_r=150)
        assert LT.el_gap   == 2
        assert LT.active_r == 150

@given(xs=floats(min_value=-500, max_value=500),
       ys=floats(min_value=-500, max_value=500),
       sipm_indx=integers(min_value=0, max_value=1500))
def test_LT_SiPM_values(get_dfs, xs, ys, sipm_indx):
    datasipm = DataSiPM('new')
    fname, psf_df, psf_conf = get_dfs['psf']

    r_active = psf_conf.loc['ACTIVE_rad'].astype(float).value
    r = np.sqrt(xs**2 + ys**2)
    psfbins = psf_df.index.values
    lenz = psf_df.shape[1]
    psf_df = psf_df /lenz
    LT = LT_SiPM(fname=fname, sipm_database=datasipm)
    x_sipm, y_sipm = datasipm.iloc[sipm_indx][['X', 'Y']]
    dist = np.sqrt((xs-x_sipm)**2+(ys-y_sipm)**2)
    psf_bin = np.digitize(dist, psfbins)-1
    max_psf = psf_df.index.max()
    if (dist>=max_psf) or (r>=r_active):
        values = np.zeros(psf_df.shape[1])
    else:
        values = (psf_df.loc[psf_bin].values)

    ltvals = LT.get_values(xs, ys, sipm_indx)
    np.testing.assert_allclose(values, ltvals)


def find_nearest_(array, value):
    """Finds nearest element of an array"""
    idx = (np.abs(array - value)).argmin()
    return array[idx]


@given(xs=floats(min_value=-500, max_value=500),
       ys=floats(min_value=-500, max_value=500),
       pmt_indx=integers(min_value=0, max_value=11))
def test_LT_PMTs_values(get_dfs, xs, ys, pmt_indx):
    fname, lt_df, lt_conf = get_dfs['lt']
    r_active = lt_conf.loc['ACTIVE_rad'].astype(float).value
    r = np.sqrt(xs**2 + ys**2)

    LT = LT_PMT(fname=fname)

    xs_lt =  find_nearest_(np.sort(np.unique(lt_df.index.get_level_values('x'))), xs)
    ys_lt =  find_nearest_(np.sort(np.unique(lt_df.index.get_level_values('y'))), ys)
    if (r>=r_active):
        values = np.array([0]) #the values are one dimension only
    else:
        values = lt_df.loc[xs_lt, ys_lt].values[pmt_indx]
    ltvals = LT.get_values(xs, ys, pmt_indx)
    np.testing.assert_allclose(values, ltvals)


@given(xs=floats(min_value=-500, max_value=500),
       ys=floats(min_value=-500, max_value=500),
       pmt_indx=integers(min_value=0, max_value=11))
def test_LT_PMTs_values_extended(get_dfs, xs, ys, pmt_indx):
    fname, lt_df, lt_conf = get_dfs['lt']
    r_active = lt_conf.loc['ACTIVE_rad'].astype(float).value
    r_new = 2*r_active
    r = np.sqrt(xs**2 + ys**2)
    with warns(UserWarning):
        LT = LT_PMT(fname=fname,active_r=r_new)
    xs_lt =  find_nearest_(np.sort(np.unique(lt_df.index.get_level_values('x'))), xs)
    ys_lt =  find_nearest_(np.sort(np.unique(lt_df.index.get_level_values('y'))), ys)
    if (r>=r_new):
        values = np.array([0]) #the values are one dimension only
    else:
        values = lt_df.loc[xs_lt, ys_lt].values[pmt_indx]
    ltvals = LT.get_values(xs, ys, pmt_indx)
    np.testing.assert_allclose(values, ltvals)


def test_LT_SiPM_non_physical_sensor(get_dfs):
    datasipm = DataSiPM('new')
    fname, psf_df, psf_conf = get_dfs['psf']
    LT = LT_SiPM(fname=fname, sipm_database=datasipm)
    xs = 0
    ys = 0
    sipm_id = len(datasipm)
    values = LT.get_values(xs, ys, sipm_id)
    assert not values.any()

    fname, lt_df, lt_conf = get_dfs['lt']
    LT = LT_PMT(fname=fname)
    pmt_id = 12
    values = LT.get_values(xs, ys,  pmt_id)
    assert not values.any()

