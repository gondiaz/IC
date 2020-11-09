import numpy  as np
import pandas as pd
import warnings

cimport cython
cimport numpy as np


from ..core import system_of_units as units
from ..io.dst_io import               load_dst

cdef class LT:
    """Base abstract class to be inherited from for all LightTables classes.
    It needs get_values method implemented"""
    cdef double* get_values_(self, const double x, const double y, const int sensor_id):
        raise NotImplementedError

    @property
    def zbins(self):
        return np.asarray(self.zbins_)
    @property
    def sensor_ids(self):
        return np.asarray(self.sensor_ids_)

    def get_values(self, const double x, const double y, const int sns_id):
        """ This is only for using within python"""
        cdef double* pointer
        pointer = self.get_values_(x, y, sns_id)
        if pointer!=NULL:
            return np.asarray(<np.double_t[:self.zbins_.shape[0]]> pointer)
        else:
            return np.zeros(self.zbins_.shape[0])

    def __str__(self):
        return f'{self.__class__.__name__}'


    __repr__ =     __str__



def extract_info_lighttables_(fname, group_name, el_gap, active_r):
    lt_df      = load_dst(fname, group_name, "LightTable")
    config_df  = load_dst(fname, group_name, "Config").set_index('parameter')
    el_gap_f   = float(config_df.loc["EL_GAP"    ].value) * units.mm
    active_r_f = float(config_df.loc["ACTIVE_rad"].value) * units.mm
    if el_gap and (el_gap != el_gap_f):
        warnings.warn('el_gap parameter mismatch, setting to user defined one',
                      UserWarning)
    else:
        el_gap = el_gap_f

    if active_r and (active_r != active_r_f):
            warnings.warn('active_r parameter mismatch, setting to user defined one',
                          UserWarning)
    else:
        active_r = active_r_f

    return lt_df, config_df, el_gap, active_r

