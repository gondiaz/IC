import numpy  as np
import tables as tb
import pandas as pd

from functools import partial

import invisible_cities.core.system_of_units_c as system_of_units
units = system_of_units.SystemOfUnits()
import invisible_cities.database.load_db       as db

from invisible_cities.cities.components import city

from invisible_cities.dataflow  import dataflow as fl

from invisible_cities.io.rwf_io           import rwf_writer
from invisible_cities.io.run_and_event_io import run_and_event_writer

# DETSIM IMPORTS
from invisible_cities.cities.detsim_source             import load_MC
from invisible_cities.cities.detsim_simulate_electrons import generate_ionization_electrons
from invisible_cities.cities.detsim_simulate_electrons import drift_electrons
from invisible_cities.cities.detsim_simulate_electrons import diffuse_electrons

from invisible_cities.cities.detsim_simulate_photons   import generate_s1_photons
from invisible_cities.cities.detsim_simulate_photons   import generate_s2_photons
from invisible_cities.cities.detsim_simulate_photons   import pes_at_sensors

from invisible_cities.cities.detsim_waveforms          import create_sensor_waveforms

from invisible_cities.cities.detsim_get_psf            import get_sipm_psf_from_file
from invisible_cities.cities.detsim_get_psf            import get_ligthtables
from invisible_cities.cities.detsim_get_psf            import get_krmaps_as_ligthtables


def get_derived_parameters(detector_db, run_number,
                           s1_ligthtable, s2_ligthtable, psfsipm_filename,
                           wi, el_gain, conde_policarpo_factor, EL_dz,
                           drift_velocity_EL,
                           wf_buffer_time, wf_pmt_bin_time, wf_sipm_bin_time):
    ########################
    ######## Globals #######
    ########################
    datapmt  = db.DataPMT (detector_db, run_number)
    datasipm = db.DataSiPM(detector_db, run_number)

    # nphotons = (41.5 * units.keV / wi) * el_gain * npmts

    # S1pmt_psf  = partial(fn._psf, factor=1e5)
    # S2pmt_psf  = fn.get_psf_from_krmap     (krmap_filename, factor=1./nphotons)
    S1_LT = get_ligthtables(s1_ligthtable, "S1")
    S2_LT = get_ligthtables(s2_ligthtable, "S2")
    S2sipm_psf = get_sipm_psf_from_file(psfsipm_filename)

    el_gain_sigma = np.sqrt(el_gain * conde_policarpo_factor)
    EL_dtime      =  EL_dz / drift_velocity_EL

    s2_pmt_nsamples  = np.max((int(EL_dtime // wf_pmt_bin_time ), 1))
    s2_sipm_nsamples = np.max((int(EL_dtime // wf_sipm_bin_time), 1))

    return datapmt, datasipm,\
           S1_LT, S2_LT, S2sipm_psf,\
           el_gain_sigma,\
           s2_pmt_nsamples, s2_sipm_nsamples


@city
def detsim(files_in, file_out, event_range, detector_db, run_number, s1_ligthtable, s2_ligthtable, psfsipm_filename,
           ws, wi, fano_factor, drift_velocity, lifetime, transverse_diffusion, longitudinal_diffusion,
           el_gain, conde_policarpo_factor, EP_z, EL_dz, drift_velocity_EL,
           wf_buffer_time, wf_pmt_bin_time, wf_sipm_bin_time):
    ########################
    ######## Globals #######
    ########################
    datapmt, datasipm,\
    S1_LT, S2_LT, S2sipm_psf,\
    el_gain_sigma,\
    s2_pmt_nsamples, s2_sipm_nsamples = get_derived_parameters(detector_db, run_number,
                                                               s1_ligthtable, s2_ligthtable, psfsipm_filename,
                                                               wi, el_gain, conde_policarpo_factor, EL_dz, drift_velocity,
                                                               wf_buffer_time, wf_pmt_bin_time, wf_sipm_bin_time)

    ##########################################
    ############ SIMULATE ELECTRONS ##########
    ##########################################
    generate_electrons_ = partial(generate_ionization_electrons, wi = wi, fano_factor = fano_factor)
    generate_electrons_ = fl.map(generate_electrons_, args = ("energy"), out  = ("electrons"))

    drift_electrons_ = partial(drift_electrons, lifetime = lifetime, drift_velocity = drift_velocity)
    drift_electrons_ = fl.map(drift_electrons_, args = ("z", "electrons"), out  = ("electrons"))
    count_electrons  = fl.map(lambda x: np.sum(x), args=("electrons"), out=("nes"))

    diffuse_electrons_ = partial(diffuse_electrons, transverse_diffusion = transverse_diffusion, longitudinal_diffusion = longitudinal_diffusion)
    diffuse_electrons_ = fl.map(diffuse_electrons_, args = ("x",  "y",  "z", "electrons"), out  = ("dx", "dy", "dz"))

    simulate_electrons = fl.pipe(generate_electrons_, drift_electrons_, count_electrons, diffuse_electrons_)

    ############################################
    ############ SIMULATE PHOTONS ##############
    ############################################
    # S1#
    generate_S1_photons = partial(generate_s1_photons, ws = ws)
    generate_S1_photons = fl.map(generate_S1_photons, args = ("energy"), out  = ("S1photons"))

    S1pes_at_pmts = partial(pes_at_sensors, LT = S1_LT)
    S1pes_at_pmts = fl.map(S1pes_at_pmts, args = ("x", "y", "S1photons", "z"), out  = ("S1pes_pmt"))

    # S2 #
    generate_S2_photons = partial(generate_s2_photons, el_gain = el_gain, el_gain_sigma = el_gain_sigma)
    generate_S2_photons = fl.map(generate_S2_photons, args = ("nes"), out  = ("S2photons"))

    S2pes_at_pmts = partial(pes_at_sensors, LT = S2_LT)
    S2pes_at_pmts = fl.map(S2pes_at_pmts, args = ("dx", "dy", "S2photons"), out  = ("S2pes_pmt"))

    S2pes_at_sipms = partial(pes_at_sensors, x_sensors = datasipm["X"].values, y_sensors = datasipm["Y"].values, psf = S2sipm_psf)
    S2pes_at_sipms = fl.map(S2pes_at_sipms, args = ("dx", "dy", "S2photons"), out  = ("S2pes_sipm"))

    simulate_photons = fl.pipe(generate_S1_photons, S1pes_at_pmts, generate_S2_photons, S2pes_at_pmts, S2pes_at_sipms)

    ############################
    ###### BUFFER TIMES ########
    ############################
    def generate_S1_time(size=1):
        r = []
        for i in range(size):
            t1 = np.random.exponential(4.5)
            t2 = np.random.exponential(100)
            r.append(np.random.choice([t1, t2], p=[0.1, 0.9]))
        return np.array(r)

    def generate_S1_times_from_pes(S1pes_pmt):
        S1pes_pmt = np.sum(S1pes_pmt, axis=1)
        S1times = [generate_S1_time(size=pes) for pes in S1pes_pmt]
        return S1times

    S1times_        = fl.map(generate_S1_times_from_pes, args=("S1pes_pmt"), out=("S1times"))
    S1_buffer_times = fl.map(lambda S1times: [100 * units.mus + t for t in S1times], args=("S1times"), out=("S1_buffer_times"))
    S2_buffer_times = fl.map(lambda dz     :  100 * units.mus + dz/drift_velocity,   args=("dz"),      out=("S2_buffer_times"))

    # S1_buffer_times = fl.map(lambda dz, z: (wf_buffer_time/2. - np.min(dz)/drift_velocity)*np.ones_like(z), args = ("dz", "z"), out=("S1_buffer_times"))
    # S2_buffer_times = fl.map(lambda dz   :  wf_buffer_time/2. + (dz - np.min(dz))/drift_velocity          , args = ("dz")     , out=("S2_buffer_times"))

    ############################
    ######### FILL WFS #########
    ############################
    create_S1_waveform = create_sensor_waveforms("S1", wf_buffer_time, wf_pmt_bin_time)
    fill_S1_pmts = fl.map(create_S1_waveform, args = ("S1_buffer_times"), out=("S1pmtwfs"))

    create_S2pmt_waveforms = create_sensor_waveforms("S2", wf_buffer_time, wf_pmt_bin_time)
    fill_S2_pmts  = fl.map(partial(create_S2pmt_waveforms, s2_pmt_nsamples), args = ("S2_buffer_times", "S2pes_pmt"), out=("S2pmtwfs"))

    create_S2sipm_waveforms = create_sensor_waveforms("S2", wf_buffer_time, wf_sipm_bin_time)
    fill_S2_sipms = fl.map(partial(create_S2sipm_waveforms, s2_sipm_nsamples), args = ("S2_buffer_times", "S2pes_sipm"), out=("sipmwfs"))

    add_pmt_wfs = fl.map(lambda x, y: x + y, args=("S1pmtwfs", "S2pmtwfs"), out=("pmtwfs"))


    with tb.open_file(file_out, "w") as h5out:

        ######################################
        ############# WRITE WFS ##############
        ######################################
        write_pmtwfs  = rwf_writer(h5out, group_name = None, table_name = "pmtrd" , n_sensors = len(datapmt) , waveform_length = int(wf_buffer_time // wf_pmt_bin_time))
        write_sipmwfs = rwf_writer(h5out, group_name = None, table_name = "sipmrd", n_sensors = len(datasipm), waveform_length = int(wf_buffer_time // wf_sipm_bin_time))
        write_pmtwfs  = fl.sink(write_pmtwfs, args=("pmtwfs"))
        write_sipmwfs = fl.sink(write_sipmwfs, args=("sipmwfs"))

        write_run_event = partial(run_and_event_writer(h5out), run_number, timestamp=0)
        write_run_event = fl.sink(write_run_event, args=("event_number"))

        return fl.push(source=load_MC(files_in),
                       pipe  = fl.pipe(fl.slice(*event_range, close_all=True),
                                       simulate_electrons,
                                       simulate_photons,
                                       # fl.spy(lambda d: [print(k) for k in d]),
                                       S1times_,
                                       S1_buffer_times,
                                       S2_buffer_times,
                                       fill_S1_pmts,
                                       fill_S2_pmts,
                                       fill_S2_sipms,
                                       add_pmt_wfs,
                                       fl.fork(write_pmtwfs,
                                               write_sipmwfs,
                                               write_run_event)),
                        result = ())
