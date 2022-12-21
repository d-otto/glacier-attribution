# -*- coding: utf-8 -*-
"""
download_cmip6.py

Description.

Author: drotto
Created: 12/15/2022 @ 11:25 AM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyesgf.search import SearchConnection
from pathlib import Path
from esmvalcore.esgf import find_files, download
import logging

logging.basicConfig(level=logging.DEBUG)

#%%

ensembles = [f"r{d}i1p1" for d in range(1, 100)]
p1k_ensembles = [f"rii1p{d}" for d in [121, 122, 1221, 123, 124, 125, 126, 127, 128]]
ensembles.extend(p1k_ensembles)

files = []
fs = find_files(
    project='CMIP5',
    # output='output1',
    mip='Amon',
    short_name='tas',
    dataset='*',
    # dataset='MPI-ESM-P',
    # exp=['historical', 'historicalNat', 'past1000', 'past2k'],
    parent_experiment_id='past1000',
    # start_year=['185001, 185101'],
    # end_year=['200012']
    # ensemble=ensembles[0],
    # ensemble='r1i1p1',
)
files.extend(fs)
logging.info('Search complete.')

fs = find_files(
    project='CMIP5',
    mip='Amon',
    short_name='tas',
    dataset='*',
    exp=['past1000', 'historical', 'historicalNat', 'past2k'],
)
files.extend(fs)
logging.info('Search complete.')

dirname = Path(r'C:\sandbox\glacier-attribution\data\external\gcm\cmip5')
download(files, dest_folder=dirname)