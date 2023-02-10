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
import logging
from pathlib import Path
from tqdm import tqdm
import multiprocess as mp
from itertools import product
import re

from config import ROOT, cfg
from src.data import get_cmip6, get_rgi, get_cmip6_run_attrs, extract_cmip_points, read_cmip_model, cmip_fname_to_dict
from src import data

logging.basicConfig(level=logging.INFO)

# %%

rgiids = list({k: v for k, v in cfg['glaciers'].items()}.keys())
rgi = get_rgi(rgiids, from_sqllite=True)
rgi = rgi.set_index('RGIId')


# %%

use_mp = False
use_cache = False

mip = 5
experiments = ['past1000', 'historicalNat', 'past2k', 'historical']

# mip = 6
# experiments = ['past1000', 'historical', 'hist-nat', 'past2k']
freq = 'jjas'
variable = ['tas']

def mp_loop(variable, fps, freq):
    # variable, fps, freq = tup
    if isinstance(fps, list):
        fp = fps[0]
    else:
        fp = fps
    segments = data.cmip_fname_to_dict(fp)
    ps = [Path(ROOT, f"data/interim/gcm/cmip{mip}/{rgiid}_{variable}_{freq}_{segments['model']}_{segments['experiment']}_{segments['variant']}.nc")
        for rgiid in rgi.index]
    exists = [p.exists() for p in ps]
    if all(exists) & use_cache:
        print(f'skippity bippity: {ps[0]}')
    else:
        try:
            with read_cmip_model(fps=fps, freq=freq, mip=mip) as ds:  # maybe help make sure dataset closes?
                # ds = read_cmip_model(fps=fps, freq=freq, mip=mip)
                extract_cmip_points(ds, glaciers=rgi, variable=variable, freq=freq, mip=mip)
                ds.close()
            logging.info(f'Success! {fp}')
        except:
            logging.warning(f'Failed in the set including {fp}')
    # ds = read_cmip_model(fps=fps, freq=freq, mip=mip)
    # extract_cmip_points(ds, glaciers=rgi, variable=variable, freq=freq)
    return None

#%%
if __name__ == '__main__':
    #dirname = Path(rf'H:\data\gcm\cmip{mip}')
    dirname = Path(rf'H:\data\gcm\cmip{mip}')
    #fps = data.get_cmip_paths(dirname, mip={mip}, experiment=experiments, variant=variant, grouped=True)
    runs = data.get_cmip_facets(dirname, mip=mip)
    runs = runs.loc[runs.realization <= 10, :]
    print(runs.loc[~runs.experiment.isin(experiments), :])  # there shouldn't be any of these
    runs = runs.loc[runs.experiment.isin(experiments), :]

    group_order = ["experiment", "model", "variant"]
    fps = runs.groupby(group_order)["local_path"].apply(list).tolist()
        
    if use_mp:
        with mp.Pool(processes=mp.cpu_count() - 6) as pool:
            pool.starmap(mp_loop, product(variable, fps, [freq]))
    else:
        for tup in product(variable, fps, [freq]):
            mp_loop(*tup)
            