# -*- coding: utf-8 -*-
"""
test.py

Description.

Author: drotto
Created: 9/28/2022 @ 8:53 AM
Project: oggm
"""

import os
import pickle
import gzip
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import netCDF4 as nc
import matplotlib as mpl
import matplotlib.pyplot as plt
import oggm
from oggm import cfg, utils
from oggm import workflow
from oggm import graphics
from oggm import tasks
from oggm.core.massbalance import MultipleFlowlineMassBalance, LinearMassBalance


mpl.use('qtagg')

# %%

cfg.initialize(logging_level='WARNING')
cfg.PATHS['working_dir'] = utils.mkdir('/home/drotto/oggm_out')
cfg.PARAMS['use_tstar_calibration'] = True
cfg.PARAMS['prcp_scaling_factor'] = 2.5
cfg.PARAMS['climate_qc_months'] = 3
cfg.PARAMS['use_winter_prcp_factor'] = False
cfg.PARAMS['min_mu_star'] = 50
cfg.PARAMS['use_multiprocessing'] = False
cfg.PARAMS['check_calib_params'] = False
cfg.PATHS['working_dir'] = utils.gettempdir(dirname='OGGM-ebands', reset=True)


# %%

rgi_ids = ['RGI60-01.09162']

# %%

# Where to fetch the pre-processed directories
# todo: just until 1.6 transition is officially documented then this should be changed 
prepro_path = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/elev_bands'
gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=1, prepro_border=80, prepro_base_url=prepro_path, reset=True, force=False)


# %%

workflow.execute_entity_task(tasks.process_dem, gdirs)
workflow.execute_entity_task(tasks.simple_glacier_masks, gdirs, write_hypsometry=True)
#workflow.execute_entity_task(tasks.glacier_masks, gdirs)
list_tasks = [
    
    tasks.elevation_band_flowline,
    tasks.fixed_dx_elevation_band_flowline,
    tasks.compute_downstream_line,
    tasks.compute_downstream_bedshape,

]
for task in list_tasks:
    # The order matters!
    workflow.execute_entity_task(task, gdirs)


#%% climate tasks

base_url = r'https://cluster.klima.uni-bremen.de/~oggm/ref_mb_params/oggm_v1.4/RGIV62/CRU/elev_bands/qc3/pcp2.5/'
workflow.download_ref_tstars(base_url=base_url)
list_tasks = [
    tasks.process_climate_data,
    tasks.local_t_star,
    tasks.mu_star_calibration
]
for task in list_tasks:
    workflow.execute_entity_task(task, gdirs)


#%% Inversion tasks

list_tasks = [
    tasks.prepare_for_inversion,
    tasks.mass_conservation_inversion,
    tasks.filter_inversion_output,
]
for task in list_tasks:
    workflow.execute_entity_task(task, gdirs)

    
#%% init model

workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

#%%

gdir = gdirs[0]

#%%

fp = gdir.get_filepath('model_flowlines')
flowlines = []
with gzip.open(fp, "rb") as openfile:
    while True:
        try:
            flowlines.append(pickle.load(openfile))
        except EOFError:
            break
            
#%%

fp = gdir.get_filepath('inversion_flowlines')
inv_flowlines = []
with gzip.open(fp, "rb") as openfile:
    while True:
        try:
            inv_flowlines.append(pickle.load(openfile))
        except EOFError:
            break

#%%
fp = gdir.get_filepath('inversion_output')
inv_output = []
with gzip.open(fp, "rb") as openfile:
    while True:
        try:
            inv_output.append(pickle.load(openfile))
        except EOFError:
            break

#%%

fp = gdir.get_filepath('downstream_line')
downstream_line = []
with gzip.open(fp, "rb") as openfile:
    while True:
        try:
            downstream_line.append(pickle.load(openfile))
        except EOFError:
            break


         
#%%

fp = gdir.get_filepath('centerlines')
centerlines = []
with gzip.open(fp, "rb") as openfile:
    while True:
        try:
            centerlines.append(nc.Dataset(openfile))
        except EOFError:
            break
            
            
            
            
#%%

model = oggm.FlowlineModel(flowlines=mbmod.fls, mb_model=MultipleFlowlineMassBalance, y0=0, glen_a=3, fs=5.7e-20)
model.run_until(5)

inversion_flowlines