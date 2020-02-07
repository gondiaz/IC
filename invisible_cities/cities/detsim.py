import numpy  as np
import tables as tb
import pandas as pd

from functools import partial

import invisible_cities.core.system_of_units_c as system_of_units
units = system_of_units.SystemOfUnits()
import invisible_cities.database.load_db       as db

from invisible_cities.cities.components import city

from invisible_cities.dataflow  import dataflow as fl

from invisible_cities.cities    import detsim_functions as fn

from invisible_cities.io.rwf_io           import rwf_writer
from invisible_cities.io.run_and_event_io import run_and_event_writer



@city
def detsim(files_in, file_out, event_range, detector_db, run_number, krmap_filename, psfsipm_filename,
           ws, wi, fano_factor, drift_velocity, lifetime, transverse_diffusion, longitudinal_diffusion,
           el_gain, conde_policarpo_factor, EP_z, EL_dz,
           wf_buffer_time, wf_pmt_bin_time, wf_sipm_bin_time, s1_dtime):

    ########################
    ######## Globals #######
    ########################
    datapmt  = db.DataPMT (detector_db, run_number)
    datasipm = db.DataSiPM(detector_db, run_number)
    npmts  = len(datapmt)
    nsipms = len(datasipm)

    S2pmt_psf  = partial(fn._psf, factor=1e5)
    S2sipm_psf = partial(fn._psf, factor=1e5)
    S1pmt_psf  = partial(fn._psf, factor=1e5)

    el_gain_sigma = np.sqrt(el_gain * conde_policarpo_factor)
    EL_dtime      =  EL_dz / drift_velocity

    wf_pmt_nbins     = int(wf_buffer_time // wf_pmt_bin_time)
    wf_sipm_nbins    = int(wf_buffer_time // wf_sipm_bin_time)

    s1_nsamples      = np.max((int(s1_dtime // wf_pmt_bin_time ), 1))
    s2_pmt_nsamples  = np.max((int(EL_dtime // wf_pmt_bin_time ), 1))
    s2_sipm_nsamples = np.max((int(EL_dtime // wf_sipm_bin_time), 1))

    ##########################################
    ############ SIMULATE ELECTRONS ##########
    ##########################################
    generate_electrons = fl.map(partial(fn.generate_electrons, wi = wi, fano_factor = fano_factor),
                                args = ("energy"), out  = ("electrons"))
    drift_electrons    = fl.map(partial(fn.drift_electrons, lifetime = lifetime, drift_velocity = drift_velocity),
                                args = ("z", "electrons"), out  = ("electrons"))
    diffuse_electrons  = fl.map(partial(fn.diffuse_electrons, transverse_diffusion = transverse_diffusion, longitudinal_diffusion = longitudinal_diffusion),
                                args = ( "x",  "y",  "z", "electrons"), out  = ("dx", "dy", "dz"))

    simulate_electrons = fl.pipe( generate_electrons, drift_electrons, diffuse_electrons )

    ############################################
    ############ SIMULATE PHOTONS ##############
    ############################################
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
    S1_buffer_times = fl.map(lambda dz, z: (wf_buffer_time/2. - np.min(dz)/drift_velocity)*np.ones_like(z), args = ("dz", "z"), out=("S1_buffer_times"))
    S2_buffer_times = fl.map(lambda x, y: x/drift_velocity + np.max(y), args = ("dz", "S1_buffer_times"), out=("S2_buffer_times"))

    ############################
    ######### FILL WFS #########
    ############################
    # create_empty_pmt_waveforms  = fl.map(lambda x: np.zeros(  (npmts, wf_pmt_nbins), dtype = int), args=("event_number"), out=("pmtwfs"))
    # create_empty_sipm_waveforms = fl.map(lambda x: np.zeros((nsipms, wf_sipm_nbins), dtype = int), args=("event_number"), out=("sipmwfs"))
    # fill_S1_pmts = fl.map(partial(fn.sample_photons_and_fill_wfs, wf_bin_time = wf_pmt_bin_time, nsamples = s1_nsamples),
    #                       args = ("S1_buffer_times", "S1photons_pmt", "pmtwfs"), out=("pmtwfs"))
    # fill_S2_pmts = fl.map(partial(fn.sample_photons_and_fill_wfs, wf_bin_time = wf_pmt_bin_time, nsamples = s2_pmt_nsamples),
    #                       args = ("S2_buffer_times", "S2photons_pmt", "pmtwfs"), out=("pmtwfs"))
    # fill_S2_sipms = fl.map(partial(fn.sample_photons_and_fill_wfs, wf_bin_time = wf_sipm_bin_time, nsamples = s2_sipm_nsamples),
    #                       args = ("S2_buffer_times", "S2photons_sipm", "sipmwfs"), out=("sipmwfs"))

    fill_S1_pmts = fl.map(partial(fn.create_sensor_waveforms, wf_buffer_time = wf_buffer_time, wf_bin_time = wf_pmt_bin_time, nsamples = s2_pmt_nsamples, poisson = True),
                          args = ("S1_buffer_times", "S1photons_pmt"), out=("S1pmtwfs"))
    fill_S2_pmts = fl.map(partial(fn.create_sensor_waveforms, wf_buffer_time = wf_buffer_time, wf_bin_time = wf_pmt_bin_time, nsamples = s2_pmt_nsamples, poisson = True),
                          args = ("S2_buffer_times", "S2photons_pmt"), out=("S2pmtwfs"))
    fill_S2_sipms = fl.map(partial(fn.create_sensor_waveforms, wf_buffer_time = wf_buffer_time, wf_bin_time = wf_sipm_bin_time, nsamples = s2_sipm_nsamples, poisson = True),
                          args = ("S2_buffer_times", "S2photons_sipm"), out=("sipmwfs"))

    add_pmt_wfs = fl.map(lambda x, y: x + y, args=("S1pmtwfs", "S2pmtwfs"), out=("pmtwfs"))

    convert_pmtwfs_to_adc  = fl.map(lambda x: x*datapmt ["adc_to_pes"].values[:, np.newaxis], args = ("pmtwfs") , out=("pmtwfs"))
    convert_sipmwfs_to_adc = fl.map(lambda x: x*datasipm["adc_to_pes"].values[:, np.newaxis], args = ("sipmwfs"), out=("sipmwfs"))

    with tb.open_file(file_out, "w") as h5out:

        ######################################
        ############# WRITE WFS ##############
        ######################################
        write_pmtwfs_  = rwf_writer(h5out, group_name = "RD", table_name = "pmtrwf" , n_sensors = npmts , waveform_length = wf_pmt_nbins)
        write_sipmwfs_ = rwf_writer(h5out, group_name = "RD", table_name = "sipmrwf", n_sensors = nsipms, waveform_length = wf_sipm_nbins)
        write_pmtwfs  = fl.sink(write_pmtwfs_ , args=("pmtwfs"))
        write_sipmwfs = fl.sink(write_sipmwfs_, args=("sipmwfs"))

        return fl.push(source=fn.load_MC(files_in),
                       pipe  = fl.pipe(fl.filter(lambda x: x==0, args=("event_number")),
                                       simulate_electrons,
                                       simulate_photons,
                                       S1_buffer_times,
                                       S2_buffer_times,
                                       #create_empty_pmt_waveforms,
                                       #create_empty_sipm_waveforms,
                                       fill_S1_pmts,
                                       fill_S2_pmts,
                                       fill_S2_sipms,
                                       add_pmt_wfs,
                                       convert_pmtwfs_to_adc,
                                       convert_sipmwfs_to_adc,
                                       fl.spy(print),
                                       fl.fork(write_pmtwfs,
                                               write_sipmwfs)),
                        result = ())
