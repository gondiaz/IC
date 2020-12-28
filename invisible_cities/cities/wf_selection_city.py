"""
---------------------------------
        selectioncity
---------------------------------

This city selects the events and its data from
whichever IC data type.

"""
import os
import numpy  as np
import tables as tb

from functools import partial

from invisible_cities.reco import tbl_functions as tbl

from invisible_cities.dataflow import dataflow as fl

from invisible_cities.cities.components import city
from invisible_cities.cities.components import WfType
from invisible_cities.cities.components import get_pmt_wfs
from invisible_cities.cities.components import get_sipm_wfs
from invisible_cities.cities.components import get_event_info
from invisible_cities.cities.components import get_run_number

from invisible_cities.io.run_and_event_io import run_and_event_writer
from invisible_cities.io.rwf_io           import rwf_writer



def get_source(selection_filename, wf_type):
    selection_filename = os.path.expandvars(selection_filename)

    def source(files_in):
        for filename in files_in:
            selected_events = np.load(selection_filename)
            if len(selected_events) == 0 : return

            with tb.open_file(filename) as h5in:
                try:
                    event_info  = get_event_info  (h5in)
                    run_number  = get_run_number  (h5in)
                    pmt_wfs     = get_pmt_wfs     (h5in, wf_type)
                    sipm_wfs    = get_sipm_wfs    (h5in, wf_type)
                except tb.exceptions.NoSuchNodeError:
                    continue

                event_info_read = event_info.read()
                all_events_in_file = event_info_read["evt_number"]

                selidxs            = np.argwhere(np.isin(all_events_in_file, selected_events)).flatten()
                sel_events_in_file = all_events_in_file[selidxs]

                selected_events = selected_events[~np.isin(selected_events, all_events_in_file)]
                np.save(selection_filename, selected_events)

                for i, event in zip(selidxs, sel_events_in_file):
                    pmtwf  = pmt_wfs [i]
                    sipmwf = sipm_wfs[i]

                    d =  dict(#filename = filename,
                              pmtwf = pmtwf, sipmwf = sipmwf,
                              run_number   = run_number,
                              event_number = event,
                              timestamp    = event_info_read["timestamp"][i])

                    yield d

    return source



@city
def wf_selection_city(files_in, file_out, selection_filename,
                      compression, event_range):

    # compute waveform shapes from input file
    for file_in in files_in:
        with tb.open_file(file_in) as h5in:
            try:
                run_number  = get_run_number  (h5in)

                if   run_number >  0: wf_type = WfType.rwf
                elif run_number <= 0: wf_type = WfType.mcrd
                else                : raise ValueError("Invalid wf_type")

                pmt_wfs     = get_pmt_wfs     (h5in, wf_type)
                sipm_wfs    = get_sipm_wfs    (h5in, wf_type)

                npmt , pmt_wf_length  = pmt_wfs .shape[-2:]
                nsipm, sipm_wf_length = sipm_wfs.shape[-2:]
                break
            except tb.exceptions.NoSuchNodeError:
                continue

    source = get_source(selection_filename, wf_type)

    ###### define counters #####
    count  = fl.spy_count()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        #write event info
        write_event_info_ = run_and_event_writer(h5out)
        write_event_info  = fl.sink(write_event_info_, args=("run_number", "event_number", "timestamp"))

        #write waveforms
        if   run_number >  0:
            pmt_writer  = rwf_writer(h5out, group_name="RD", table_name= "pmtrwf", compression=compression, n_sensors= npmt, waveform_length=pmt_wf_length)
            sipm_writer = rwf_writer(h5out, group_name="RD", table_name="sipmrwf", compression=compression, n_sensors=nsipm, waveform_length=sipm_wf_length)

        elif run_number <= 0:
            pmt_writer  = rwf_writer(h5out, group_name=None, table_name= "pmtrd", compression=compression, n_sensors= npmt, waveform_length=pmt_wf_length)
            sipm_writer = rwf_writer(h5out, group_name=None, table_name="sipmrd", compression=compression, n_sensors=nsipm, waveform_length=sipm_wf_length)

        write_pmt_wfs  = fl.sink(pmt_writer , args=( "pmtwf"))
        write_sipm_wfs = fl.sink(sipm_writer, args=("sipmwf"))

        result = fl.push(source = source(files_in),
                         pipe   = fl.pipe(count.spy,
                                          fl.spy(lambda d: [print(d[nodename]) for nodename in d if isinstance(d[nodename], str)]),
                                          fl.fork(write_event_info,
                                                  write_pmt_wfs,
                                                  write_sipm_wfs)),
                         result = dict(n_total     = count.future))

        # add dummy trigger group
        if run_number > 0:
            nevents = result.n_total
            h5out.create_earray("/Trigger", "events" , obj = np.zeros((nevents, npmt)), createparents=True)
            h5out.create_table ("/Trigger", "trigger", obj = np.zeros(nevents, dtype=[("trigger_type", "float")]))
        return result
