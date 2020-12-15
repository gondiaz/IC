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

from invisible_cities.reco import tbl_functions        as tbl

from invisible_cities.cities.components import city

from invisible_cities.dataflow import dataflow as fl


def get_file_structure(filename):
    """
    From a given filename, it returns a dictionary whose
    elements are the atributes of each of the nodes of the .h5,
    being the nodes of type tb.Table or tb.EArray.
    """
    d = dict()
    with tb.open_file(filename) as h5file:
        for node in h5file.walk_nodes():
            name = node._v_pathname
            if "MC" in name:
                continue
            ####### Table #########
            if isinstance(node, tb.Table):
                d[node._v_pathname] = dict(nodetype = tb.Table               ,
                                           where = node._v_parent._v_pathname,
                                           name  = node.name                 ,
                                           description = node.description    ,
                                           title   = node.title              ,
                                           filters = node.filters)
            ####### EArray #######
            if isinstance(node, tb.EArray):
                shape = [*node.shape]
                shape[node.maindim] = 0
                d[node._v_pathname] = dict(nodetype = tb.EArray              ,
                                           where = node._v_parent._v_pathname,
                                           name  = node.name                 ,
                                           title = node.title                ,
                                           atom  = node.atom                 ,
                                           shape = shape)
    return d


def copy_file_structure(h5file, structure):
    for node in structure:
        if structure[node]["nodetype"] is tb.Table:
            h5file.create_table(structure[node]["where"]                   ,
                                structure[node]["name"]                    ,
                                description= structure[node]["description"],
                                title      = structure[node]["title"]      ,
                                filters    = structure[node]["filters"]    ,
                                createparents=True)

        if structure[node]["nodetype"] is tb.EArray:
            h5file.create_earray(structure[node]["where"]        ,
                                 structure[node]["name"]         ,
                                 atom  = structure[node]["atom"] ,
                                 shape = structure[node]["shape"],
                                 title = structure[node]["title"],
                                 createparents = True)


def get_general_source(selection):

    def general_source(files_in):
        selection_filename = os.path.expandvars(selection)
        selected_events = np.load(selection_filename)

        for file in files_in:
            d = dict()
            d[file] = file
            if len(selected_events)==0:
                return

            with tb.open_file(file, "r") as h5file:

                events = h5file.root.Run.events.read()["evt_number"]
                selidxs = np.argwhere(np.isin(events, selected_events)).flatten()
                events_in_file  = events[selidxs]
                selected_events = selected_events[~np.isin(selected_events, events)]
                np.save(selection_filename, selected_events)

                for i, event in zip(selidxs, events_in_file):
                    for node in h5file.walk_nodes():
                        name = node._v_pathname
                        if "MC" in name:
                            continue
                        if isinstance(node, tb.EArray):
                            d[name] = node[i][np.newaxis, :]
                        if isinstance(node, tb.Table):
                            table = node.read()
                            try:
                                sel = (table["evt_number"] == event)
                                d[name] = table[sel]
                            except (IndexError, ValueError):
                                d["g" + name] = node.read()
                    yield d

    return general_source


def general_writer(h5file, d):
    for nodename in d:
        # check if global node has been filled, if not, fill it
        if nodename[0] == "g":
            globalnode = h5file.get_node(nodename[1:])
            if globalnode.nrows == 0:
                globalnode.append(d[nodename])
                continue
            else:
                continue

        if isinstance(d[nodename], np.ndarray):
            # fill event data
            node = h5file.get_node(nodename)
            node.append(d[nodename])
    h5file.flush()


@city
def selectioncity(files_in, file_out, selection,
                  event_range, compression):

    general_source = get_general_source(selection)

    ###### get file structure and create empty file_out ####
    structure = get_file_structure(np.random.choice(files_in))

    ###### define counters #####
    count  = fl.spy_count()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5file:

        copy_file_structure(h5file, structure)

        writer = fl.sink(partial(general_writer, h5file ))

        result = fl.push(source = general_source(files_in),
                         pipe   = fl.pipe(count.spy,
                                          fl.spy(lambda d: [print(d[nodename]) for nodename in d if isinstance(d[nodename], str)]),
                                          fl.fork(writer)),
                         result = dict(n_total = count.future))
        return result
