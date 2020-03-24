import tables as tb
import pandas as pd

from typing import List
from typing import Generator

from invisible_cities.io.mcinfo_io import load_mchits_df

#######################################
############### SOURCE ################
#######################################
def load_MC(files_in : List[str]) -> Generator:
    for filename in files_in:
        hits_df = load_mchits_df(filename)

        for evt, hits in hits_df.groupby(level=0):
            yield dict(event_number = evt,
                       x            = hits.x     .values,
                       y            = hits.y     .values,
                       z            = hits.z     .values,
                       energy       = hits.energy.values,
                       time         = hits.time  .values)
