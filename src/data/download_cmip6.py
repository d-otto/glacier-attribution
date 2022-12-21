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

logging.basicConfig(level=logging.INFO)

#%%

ensembles = [f"r{d}i1f1p1" for d in range(1, 200)]
# p1k_ensembles = [f"rii1f1p1" for d in [121, 122, 1221, 123, 124, 125, 126, 127, 128]]
# ensembles.extend(p1k_ensembles)

files = find_files(
    project='CMIP6',
    mip='Amon',
    short_name='tas',
    dataset='*',
    exp=['historical', 'hist-nat', 'past1000', 'past2k'],
    # ensemble=ensembles,
)  

dirname = Path(r'C:\sandbox\glacier-attribution\data\external\gcm\cmip6')
download(files, dest_folder=dirname)

#%%

# conn = SearchConnection('https://esgf-node.llnl.gov/esg-search')
# ctx = conn.new_context(project='CMIP6', experiment_id='past1000', variable='tas')
# print('Hits: {}, Realms: {}, Ensembles: {}'.format(
#     ctx.hit_count,
#     ctx.facet_counts['realm'],
#     ctx.facet_counts['ensemble']))
# print(ctx.get_facet_options())
# 
# #%%
# 
# conn = SearchConnection('https://esgf-data.dkrz.de/esg-search', distrib=True, timeout=240)
# ctx = conn.new_context(
#     project='CMIP6',
#     experiment='past1000',
#     facets='*',
# )
# ctx.hit_count