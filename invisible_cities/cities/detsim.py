import numpy  as np
import tables as tb
import pandas as pd

from functools import partial

import invisible_cities.core.system_of_units_c as system_of_units
units = system_of_units.SystemOfUnits()
import invisible_cities.database.load_db       as db

from invisible_cities.cities.components import city

from invisible_cities.dataflow import dataflow as fl

from invisible_cities.cities import detsim_functions as fn




@city
def detsim(files_in, file_out, event_range, detector_db, run_number, krmap_filename, psfsipm_filename,
           ws, wi, fano_factor, conde_policarpo_factor, drift_velocity, lifetime, transverse_diffusion, longitudinal_diffusion, voxel_sizes):


    #### SIMULATE ELECTRONS ######
    generate_electrons = fl.map(partial(fn.generate_electrons, wi=wi, fano_factor=fano_factor),
                                args=("energy"), out=("secondary_electrons"))

    drift_electrons    = fl.map(partial(fn.drift_electrons, lifetime=lifetime, drift_velocity=drift_velocity),
                                args=("z", "secondary_electrons"), out=("secondary_electrons"))

    diffuse_electrons  = fl.map(partial(fn.diffuse_electrons, transverse_diffusion  =transverse_diffusion,
                                                              longitudinal_diffusion=longitudinal_diffusion,
                                                              voxel_sizes = voxel_sizes),
                                args = ( "x",  "y",  "z", "secondary_electrons"),
                                out  = ("dx", "dy", "dz", "diffused_electrons"))

    #simulate_electrons = fl.pipe()

    with tb.open_file(file_out, "w") as h5out:

        sinkpipe = fl.sink(lambda x: print(x), args=("evt"))

        return fl.push(source=fn.load_MC(files_in),
                       pipe  = fl.pipe(generate_electrons,
                                       drift_electrons   ,
                                       diffuse_electrons,
                                       fl.spy(print),
                                       sinkpipe),
                        result = ())
