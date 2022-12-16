# -*- coding: utf-8 -*-
"""
concat_cmip_points.py

Description.

Author: drotto
Created: 12/12/2022 @ 3:08 PM
Project: glacier-attribution
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocess as mp
from itertools import product

import src.data as data
from config import ROOT

#%%

use_mp = True
mip=6
#processes = mp.cpu_count() - 2
processes = 2

#%%

def mp_loop(rgiid):
    pattern = f'^{rgiid}'
    fps = [fp for fp in ps if re.match(pattern, fp.name)]
    identifiers = fps[0].name.split('_')[1:3]
    ds = xr.open_mfdataset(fps, coords='minimal', data_vars='minimal')
    p = Path(ROOT, f'features/gcm/cmip{mip}_{rgiid}_{"_".join(identifiers)}.nc')
    ds.to_netcdf(p)
    print(p.name)

# def mp_loop(rgiid):
#     pattern = f'^{rgiid}'
#     fps = [fp for fp in ps if re.match(pattern, fp.name)]
#     identifiers = fps[0].name.split('_')[1:3]
#     concat=[]
#     # for fp in fps:
#     #     concat.append(xr.open_dataset(fp))
#     #     print(fp)
#     #ds = xr.merge(concat)
#     # ds = xr.concat(concat, dim='rgiid', compat='no_conflicts', coords='minimal', data_vars='minimal')
#     #ds = xr.combine_nested(concat, concat_dim='rgiid', compat='no_conflicts', data_vars='minimal', coords='minimal')
#     print('Meeeeeeeeeeeeeeeerrrrrrrrrrrgggeeedd.')
#     p = Path(ROOT, f'features/gcm/cmip{mip}_{rgiid}_{"_".join(identifiers)}.nc')
#     ds.to_netcdf(p)
#     print(p.name)


#%%
p = Path(ROOT, f'data/interim/gcm/cmip{mip}')
ps = list(p.iterdir())
fnames = [fp.name for fp in ps]
rgiids = list(set([fname.split('_')[0] for fname in fnames]))

if __name__ == '__main__':
    if use_mp:
        with mp.Pool(processes=processes) as pool:
            pool.starmap(mp_loop, product(rgiids))
    else: 
        for rgiid in rgiids:
            mp_loop(rgiid)



#%% For debugging

# concat = []
# for fp in fps[0:10]:
#     ds = xr.open_dataset(fp)
#     concat.append(ds)
#     
# test = xr.combine_by_coords(concat)
# test = xr.concat(concat, dim='time', coords='all')