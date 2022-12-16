# -*- coding: utf-8 -*-
"""
data.py

Description.

Author: drotto
Created: 11/9/2022 @ 3:48 PM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import xarray as xr
import netCDF4
import h5netcdf
from pathlib import Path
from tqdm import tqdm
import multiprocess as mp
from itertools import product

from config import ROOT, cfg
from src.data import get_cmip6, get_rgi, get_cmip6_run_attrs, extract_cmip_points, read_cmip_model, cmip_fname_to_dict

# %%

rgiids = {'hintereisferner': 'RGI60-11.00897',
          'argentiere'     : 'RGI60-11.03638',
          'south cascade'  : 'RGI60-02.18778',
          'wolverine'      : 'RGI60-01.09162'}
rgi = get_rgi(list(rgiids.values()), from_sqllite=True)
rgi = rgi.set_index('RGIId')

# %%

use_mp = False
use_cache = True
mip = 6
experiments = ['past1000', 'hist-nat', 'past2k', 'historical']
# mip = 5
# experiments = ['past1000', 'historical', 'historicalNat']
variant = None
freq = 'jjas'
variable = 'tas'

def mp_loop(variable, fps, freq):
    fp = fps[0]
    segments = cmip_fname_to_dict(fp)
    ps = [Path(ROOT, f"data/interim/gcm/cmip{mip}/{rgiid}_{variable}_{freq}_{segments['model']}_{segments['experiment']}_{segments['variant']}.nc")
        for rgiid in rgiids.values()]
    exists = [p.exists() for p in ps]
    if any(exists) & use_cache:
        print(f'skippity bippity: {ps[0]}')
    else:
        try:
            ds = read_cmip_model(fps=fps, freq=freq, mip=mip)
            extract_cmip_points(ds, glaciers=rgi, variable=variable, freq=freq, mip=mip)
            try:
                print(f'Success! {fps[0]}')
            except:
                print(f'Success! {fps}')
        except:
            print(f'Failed in the set including {fps[0]}')
    # ds = read_cmip_model(fps=fps, freq=freq, mip=mip)
    # extract_cmip_points(ds, glaciers=rgi, variable=variable, freq=freq)
    return None

#%%
if __name__ == '__main__':
    
    fps = []
    dirname = Path(ROOT, f'data/external/gcm/cmip{mip}')
    for experiment in experiments:
        models = get_cmip6_run_attrs(dirname=dirname, experiment=experiment, by='model')
        for model in models:
            variants = get_cmip6_run_attrs(dirname=dirname, experiment=experiment, model=model, by='variant')
            for variant in variants:
                fp = get_cmip6_run_attrs(dirname=dirname, experiment=experiment, model=model, variant=variant)
                fps.append(fp)
        
    if use_mp:
        with mp.Pool(processes=mp.cpu_count() - 2) as pool:
            pool.starmap(mp_loop, product([variable], fps, [freq]))
    else:
        for tup in product([variable], fps, [freq]):
            mp_loop(*tup)
            