import os

from pytest import mark, fixture

import numpy  as np
import tables as tb

from invisible_cities.cities.detsim import detsim
from invisible_cities.core.configure import configure



def test_detsim_empty_input_file(config_tmpdir, ICDATADIR):
    # 
    # Test that a empty MC file produces an empty rwf file

    PATH_IN  = os.path.join(ICDATADIR    , 'empty_MCfile.h5')
    PATH_OUT = os.path.join(config_tmpdir, 'empty_rwf.h5')

    conf = configure('dummy invisible_cities/config/detsim.conf'.split())
    conf.update(dict(files_in      = PATH_IN,
                     file_out      = PATH_OUT))

    cnt  = detsim(**conf)

    with tb.open_file(PATH_OUT) as h5out:

        pmtwfs  = h5out.get_node("/RD/pmtrwf")
        sipmwfs = h5out.get_node("/RD/sipmrwf")
        assert len( pmtwfs) == 0
        assert len(sipmwfs) == 0

    # nevt_in  = cnt.events_in
    # nevt_out = cnt.events_out
