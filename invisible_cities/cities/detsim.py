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
           ws, wi, fano_factor, drift_velocity, lifetime, transverse_diffusion, longitudinal_diffusion,
           el_gain, conde_policarpo_factor, EP_z, EL_dz,
           wf_buffer_time, wf_pmt_bin_time, wf_sipm_bin_time):

    ########################
    ######## Globals #######
    ########################
    datapmt  = db.DataPMT (detector_db, run_number)
    datasipm = db.DataSiPM(detector_db, run_number)

    el_gain_sigma = np.sqrt( el_gain * conde_policarpo_factor)

    S2pmt_psf  = partial(fn._psf, factor=1e5)
    S2sipm_psf = partial(fn._psf, factor=1e5)
    S1pmt_psf  = partial(fn._psf, factor=1e5)

    ###########################################################
    ################### SIMULATE ELECTRONS ####################
    ###########################################################
    generate_electrons = fl.map(partial(fn.generate_electrons, wi = wi, fano_factor = fano_factor),
                                args = ("energy"), out  = ("electrons"))

    drift_electrons    = fl.map(partial(fn.drift_electrons, lifetime = lifetime, drift_velocity = drift_velocity),
                                args = ("z", "electrons"), out  = ("electrons"))

    diffuse_electrons  = fl.map(partial(fn.diffuse_electrons, transverse_diffusion  =transverse_diffusion, longitudinal_diffusion=longitudinal_diffusion),
                                args = ( "x",  "y",  "z", "electrons"), out  = ("dx", "dy", "dz"))

    simulate_electrons = fl.pipe( generate_electrons, drift_electrons, diffuse_electrons )

    ###########################################################
    #################### SIMULATE PHOTONS #####################
    ###########################################################
    generate_S1_photons = fl.map(partial(fn.generate_s1_photons, ws = ws),
                                 args = ("energy"), out  = ("S1photons"))

    generate_S2_photons = fl.map(partial(fn.generate_s2_photons, el_gain = el_gain, el_gain_sigma = el_gain_sigma),
                                 args = ("dx"), out  = ("S2photons"))

    S1photons_at_pmts = fl.map(partial(fn.photons_at_sensors, x_sensors = datapmt["X"].values, y_sensors = datapmt["Y"].values, z_sensors = EP_z,
                                                              psf       = S1pmt_psf),
                               args = ("x", "y", "z", "S1photons"), out  = ("S1photons_pmt"))

    S2photons_at_pmts = fl.map(partial(fn.photons_at_sensors, x_sensors = datapmt["X"].values, y_sensors = datapmt["Y"].values, z_sensors = 0,
                                                              psf       = S2pmt_psf),
                               args = ("dx", "dy", "dz", "S2photons"), out  = ("S2photons_pmt"))

    S2photons_at_sipms= fl.map(partial(fn.photons_at_sensors, x_sensors = datasipm["X"].values, y_sensors = datasipm["Y"].values, z_sensors = 0,
                                                              psf       = S2sipm_psf),
                               args = ("dx", "dy", "dz", "S2photons"), out  = ("S2photons_sipm"))

    simulate_photons = fl.pipe(generate_S1_photons, S1photons_at_pmts, generate_S2_photons, S2photons_at_pmts, S2photons_at_sipms)

    ############################
    ###### BUFFER TIMES ########
    ############################
    S1_buffer_times = fl.map(lambda x   : wf_buffer_time/2. - np.min(x)/drift_velocity, args = ("dz")                  , out=("S1_buffer_times"))
    S2_buffer_times = fl.map(lambda x, y: x/drift_velocity + y                        , args = ("dz", "S1_buffer_time"), out=("S2_buffer_times"))


    with tb.open_file(file_out, "w") as h5out:

        ######################################
        ############# WRITE WFS ##############
        ######################################
        sinkpipe = fl.sink(lambda x: print(x), args=("evt"))

        return fl.push(source=fn.load_MC(files_in),
                       pipe  = fl.pipe(fl.filter(lambda x: x==0, args=("evt")),
                                       simulate_electrons,
                                       simulate_photons,
                                       S1_buffer_times,
                                       S2_buffer_times,
                                       fl.spy(print),
                                       sinkpipe),
                        result = ())
