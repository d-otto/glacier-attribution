# -*- coding: utf-8 -*-
"""
get_geom.py

Requires using WSL oggm_env interpreter!

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

#%%
def flatten(items, seqtypes=(list, tuple)):
    try:
        for i, x in enumerate(items):
            while isinstance(x, seqtypes):    
                items[i:i+1] = x
                x = items[i]
    except IndexError:
        pass
    return items

# %%

cfg.initialize(logging_level='DEBUG')
cfg.PATHS['working_dir'] = utils.mkdir('~/oggm_out')
cfg.PARAMS['use_tstar_calibration'] = True
cfg.PARAMS['prcp_scaling_factor'] = 2.5
cfg.PARAMS['climate_qc_months'] = 3
cfg.PARAMS['use_winter_prcp_factor'] = False
cfg.PARAMS['min_mu_star'] = 50
cfg.PARAMS['use_multiprocessing'] = False
cfg.PARAMS['check_calib_params'] = False
cfg.PARAMS['use_rgi_area'] = True
#cfg.PATHS['working_dir'] = utils.gettempdir(dirname='OGGM-ebands', reset=True)

# %%

rgi_ids = ['RGI60-11.00897',  # hintereisferner
           'RGI60-11.03638',  # argentiere
           'RGI60-02.18778',  # south cascade
           'RGI60-01.09162']  # wolverine

# %%

# Where to fetch the pre-processed directories
# todo: just until 1.6 transition is officially documented then this should be changed 
prepro_path = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/elev_bands'
base_url = 'https://cluster.klima.uni-bremen.de/data/gdirs/dems_v1/highres/'
# gdirs = workflow.init_glacier_directories(rgi_ids, reset=False,
#                                           prepro_base_url=prepro_path,
#                                           from_prepro_level=1,
#                                           prepro_rgi_version='62')
gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=2, prepro_border=240, prepro_base_url=prepro_path, reset=False, force=False)

# %%

cfg.PARAMS['elevation_band_flowline_binsize'] = 10  # meters

workflow.execute_entity_task(tasks.define_glacier_region, gdirs, source=['ALASKA', 'NASADEM'])#, 'ARCTICDEM', 'COPDEM30', 'TANDEM', 'ASTER'])
workflow.execute_entity_task(tasks.process_dem, gdirs)
workflow.execute_entity_task(tasks.simple_glacier_masks, gdirs, write_hypsometry=True)
workflow.execute_entity_task(tasks.elevation_band_flowline, gdirs)
workflow.execute_entity_task(tasks.fixed_dx_elevation_band_flowline, gdirs)
workflow.execute_entity_task(tasks.compute_downstream_line, gdirs)
workflow.execute_entity_task(tasks.compute_downstream_bedshape, gdirs)
#workflow.execute_entity_task(tasks.catchment_width_correction, gdirs)  # not sure if I should do this or not


# %% climate tasks

base_url = r'https://cluster.klima.uni-bremen.de/~oggm/ref_mb_params/oggm_v1.4/RGIV62/CRU/elev_bands/qc3/pcp2.5/'
workflow.download_ref_tstars(base_url=base_url)
list_tasks = [
    tasks.process_climate_data,
    tasks.local_t_star,
    tasks.mu_star_calibration
]
for task in list_tasks:
    workflow.execute_entity_task(task, gdirs)

# %% Inversion tasks

list_tasks = [
    tasks.prepare_for_inversion,
    tasks.mass_conservation_inversion,
    tasks.filter_inversion_output,
]
for task in list_tasks:
    workflow.execute_entity_task(task, gdirs)

# %% init model

workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

# %%


for i, rgiid in enumerate(rgi_ids):
    gdir = gdirs[i]
    gdir_objs = ['model_flowlines', 'downstream_line']
    objs = []
    for gdir_obj in gdir_objs:
        fp = gdir.get_filepath(gdir_obj)
        with gzip.open(fp, "rb") as openfile:
            while True:
                try:
                    obj = pickle.load(openfile)
                    objs.append(obj)
                except EOFError:
                    break
        # rewrite to other location
        fp = f'data/interim/oggm_flowlines/{rgiid}.{gdir_obj}.pickle'
        with gzip.open(fp, "wb") as f:
            # hopefully removes dependencie on oggm/salem
            objs = flatten(objs)
            #objs = [obj.__dict__ for obj in objs]
            pickle.dump(objs, f)

# %%
# 
# fp = gdir.get_filepath('inversion_flowlines')
# inv_flowlines = []
# with gzip.open(fp, "rb") as openfile:
#     while True:
#         try:
#             inv_flowlines.append(pickle.load(openfile))
#         except EOFError:
#             break
# 
# # %%
# 
# fp = gdir.get_filepath('downstream_line')
# downstream_line = []
# with gzip.open(fp, "rb") as openfile:
#     while True:
#         try:
#             downstream_line.append(pickle.load(openfile))
#         except EOFError:
#             break
# 
# # %%
# 
# fp = gdir.get_filepath('inversion_output')
# inv_output = []
# with gzip.open(fp, "rb") as openfile:
#     while True:
#         try:
#             inv_output.append(pickle.load(openfile))
#         except EOFError:
#             break


#%%

