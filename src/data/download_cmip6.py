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

#%%
conn = SearchConnection('https://esgf-node.llnl.gov/esg-search')
ctx = conn.new_context(project='CMIP6', experiment='historical', variable='tas')
print('Hits: {}, Realms: {}, Ensembles: {}'.format(
    ctx.hit_count,
    ctx.facet_counts['realm'],
    ctx.facet_counts['ensemble']))
print(ctx.get_facet_options())

#%%

conn = SearchConnection('https://esgf-data.dkrz.de/esg-search', distrib=True, timeout=240)
ctx = conn.new_context(
    project='CMIP6',
    experiment='past1000',
    facets='*',
)
ctx.hit_count