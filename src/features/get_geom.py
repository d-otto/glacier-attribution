# -*- coding: utf-8 -*-
"""
get_geom.py

Requires using WSL oggm_env interpreter!

Author: drotto
Created: 9/28/2022 @ 8:53 AM
Project: oggm
"""
import shutil
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
from pathlib import Path
import oggm
from oggm import utils
from oggm import workflow
from oggm import graphics
from oggm import tasks
from oggm.core.massbalance import MultipleFlowlineMassBalance, LinearMassBalance
from pathlib import Path

from config import cfg, ROOT

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

#%%



# %%

#oggm.cfg.initialize(logging_level='DEBUG')
oggm.cfg.initialize(logging_level='DEBUG')
# oggm.cfg.PATHS['dl_cache_dir'] = Path(r'\\wsl.localhost\Ubuntu\home\drotto\OGGM\download_cache')
# oggm.cfg.PATHS['rgi_dir'] = Path(r'\\wsl.localhost\Ubuntu\home\drotto\OGGM\rgi')
# oggm.cfg.PATHS['test_dir'] = Path(r'\\wsl.localhost\Ubuntu\home\drotto\OGGM\tests')
# oggm.cfg.PATHS['tmp_dir'] = Path(r'\\wsl.localhost\Ubuntu\home\drotto\OGGM\tmp')
# oggm.cfg.PATHS['working_dir'] = Path(r'\\wsl.localhost\Ubuntu\home\drotto\oggm_out')
oggm.cfg.PATHS['dl_cache_dir'] = Path(r'~/OGGM/download_cache')
oggm.cfg.PATHS['rgi_dir'] = Path(r'~/OGGM/rgi')
oggm.cfg.PATHS['test_dir'] = Path(r'~/OGGM/tests')
oggm.cfg.PATHS['tmp_dir'] = Path(r'~/OGGM/tmp')
oggm.cfg.PATHS['working_dir'] = Path(r'~/oggm_out')
oggm.cfg.PARAMS['prcp_scaling_factor'] = 2.5
oggm.cfg.PARAMS['climate_qc_months'] = 3
oggm.cfg.PARAMS['use_winter_prcp_factor'] = False
oggm.cfg.PARAMS['min_mu_star'] = 50
oggm.cfg.PARAMS['max_mu_star'] = 1000
oggm.cfg.PARAMS['use_multiprocessing'] = False
oggm.cfg.PARAMS['check_calib_params'] = False
oggm.cfg.PARAMS['use_rgi_area'] = True
oggm.cfg.PARAMS['use_tstar_calibration'] = True
oggm.cfg.PARAMS['flowline_dx'] = 1.0
oggm.cfg.PARAMS['elevation_band_flowline_binsize'] = 30  # meters
oggm.cfg.PARAMS['grid_dx_method'] = 'fixed'
oggm.cfg.PARAMS['fixed_dx'] = 50
oggm.cfg.PARAMS['border'] = 240
oggm.cfg.PARAMS['use_multiple_flowlines'] = False
oggm.cfg.PARAMS['downstream_line_shape'] = 'parabola'

#oggm.cfg.PATHS['working_dir'] = utils.gettempdir(dirname='OGGM-ebands', reset=True)

# %%

rgi_ids = list({k: v for k, v in cfg['glaciers'].items()}.keys())

# %%

oggm.cfg.PATHS['working_dir'] = Path(r'~/oggm_out/e_bands')

# Where to fetch the pre-processed directories
# todo: just until 1.6 transition is officially documented then this should be changed 
prepro_path = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/elev_bands'
base_url = 'https://cluster.klima.uni-bremen.de/data/gdirs/dems_v1/highres/'
# gdirs = workflow.init_glacier_directories(rgi_ids, reset=False,
#                                           prepro_base_url=prepro_path,
#                                           from_prepro_level=1,
#                                           prepro_rgi_version='62')
gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=2, prepro_border=160, prepro_base_url=prepro_path, reset=False, force=False)

# Shared tasks

workflow.execute_entity_task(tasks.define_glacier_region, gdirs, source=['ALASKA', 'NASADEM'])#, 'ARCTICDEM', 'COPDEM30', 'TANDEM', 'ASTER'])
workflow.execute_entity_task(tasks.process_dem, gdirs)

# Get the flowline width using elevation band flowlines
workflow.execute_entity_task(tasks.simple_glacier_masks, gdirs, write_hypsometry=True)
workflow.execute_entity_task(tasks.elevation_band_flowline, gdirs)
workflow.execute_entity_task(tasks.fixed_dx_elevation_band_flowline, gdirs)
workflow.execute_entity_task(tasks.compute_downstream_line, gdirs)
workflow.execute_entity_task(tasks.compute_downstream_bedshape, gdirs)
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

workflow.execute_entity_task(tasks.prepare_for_inversion, gdirs, invert_all_rectangular=True)
list_tasks = [
    tasks.mass_conservation_inversion,
    tasks.filter_inversion_output,
    tasks.gridded_attributes,
    # tasks.gridded_mb_attributes  # only works with centerlines
]
for task in list_tasks:
    workflow.execute_entity_task(task, gdirs)


# %% init model
workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)


# %%
for i, rgiid in enumerate(rgi_ids):
    gdir = gdirs[i]
    gdir_objs = ['inversion_flowlines', 'inversion_output', 'downstream_line', 'elevation_band_flowline']
    objs = []
    for gdir_obj in gdir_objs:
        fp = gdir.get_filepath(gdir_obj)
        if gdir_obj == 'elevation_band_flowline':
            shutil.copyfile(fp, Path(f'/mnt/c/sandbox/glacier-attribution/data/interim/oggm_flowlines/e_bands/{rgiid}.{gdir_obj}.csv'))
        
        else:
            with gzip.open(fp, "rb") as openfile:
                while True:
                    try:
                        obj = pickle.load(openfile)
                        objs.append(obj)
                    except EOFError:
                        break
            # rewrite to other location
            fp = Path(f'/mnt/c/sandbox/glacier-attribution/data/interim/oggm_flowlines/e_bands/{rgiid}.{gdir_obj}.pickle') 
            with gzip.open(fp, "wb") as f:
                # hopefully removes dependencie on oggm/salem
                objs = flatten(objs)
                #objs = [obj.__dict__ for obj in objs]
                pickle.dump(objs, f)
        
# 
# 
# #%% Get the flowline length/bed profile using geometrical centerlines
# 
# oggm.cfg.PATHS['working_dir'] = Path(r'~/oggm_out/centerlines')
# 
# # Where to fetch the pre-processed directories
# # todo: just until 1.6 transition is officially documented then this should be changed 
# prepro_path = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L1-L2_files/elev_bands'
# base_url = 'https://cluster.klima.uni-bremen.de/data/gdirs/dems_v1/highres/'
# # gdirs = workflow.init_glacier_directories(rgi_ids, reset=False,
# #                                           prepro_base_url=prepro_path,
# #                                           from_prepro_level=1,
# #                                           prepro_rgi_version='62')
# gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=2, prepro_border=160, prepro_base_url=prepro_path, reset=False, force=False)
# 
# # Shared tasks
# oggm.cfg.PARAMS['elevation_band_flowline_binsize'] = 10  # meters
# workflow.execute_entity_task(tasks.define_glacier_region, gdirs, source=['ALASKA', 'NASADEM'])#, 'ARCTICDEM', 'COPDEM30', 'TANDEM', 'ASTER'])
# workflow.execute_entity_task(tasks.process_dem, gdirs)
# 
# workflow.execute_entity_task(tasks.glacier_masks, gdirs)
# workflow.execute_entity_task(tasks.compute_centerlines, gdirs)
# workflow.execute_entity_task(tasks.initialize_flowlines, gdirs)
# workflow.execute_entity_task(tasks.compute_downstream_line, gdirs)
# 
# list_talks = [
#          tasks.catchment_area,
#          tasks.catchment_width_geom,
#          tasks.catchment_width_correction,
#          tasks.compute_downstream_bedshape
#          ]
# for task in list_talks:
#     # The order matters!
#     workflow.execute_entity_task(task, gdirs)
# 
# base_url = r'https://cluster.klima.uni-bremen.de/~oggm/ref_mb_params/oggm_v1.4/RGIV62/CRU/elev_bands/qc3/pcp2.5/'
# workflow.download_ref_tstars(base_url=base_url)
# list_tasks = [
#     tasks.process_climate_data,
#     tasks.local_t_star,
#     tasks.mu_star_calibration
# ]
# for task in list_tasks:
#     workflow.execute_entity_task(task, gdirs)
# 
# 
# workflow.execute_entity_task(tasks.prepare_for_inversion, gdirs, invert_all_rectangular=True)
# list_tasks = [
#     tasks.mass_conservation_inversion,
#     tasks.filter_inversion_output,
#     tasks.gridded_attributes,
#     # tasks.gridded_mb_attributes  # only works with centerlines
# ]
# for task in list_tasks:
#     workflow.execute_entity_task(task, gdirs)
# 
# # %% init model
# 
# workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)
# 
# # %%
# 
# 
# for i, rgiid in enumerate(rgi_ids):
#     gdir = gdirs[i]
#     gdir_objs = ['model_flowlines', 'downstream_line']
#     objs = []
#     for gdir_obj in gdir_objs:
#         fp = gdir.get_filepath(gdir_obj)
#         with gzip.open(fp, "rb") as openfile:
#             while True:
#                 try:
#                     obj = pickle.load(openfile)
#                     objs.append(obj)
#                 except EOFError:
#                     break
#         # rewrite to other location
#         fp = Path(f'/mnt/c/sandbox/glacier-attribution/data/interim/oggm_flowlines/geometrical_centerlines/{rgiid}.{gdir_obj}.pickle')
#         with gzip.open(fp, "wb") as f:
#             # hopefully removes dependencie on oggm/salem
#             objs = flatten(objs)
#             #objs = [obj.__dict__ for obj in objs]
#             pickle.dump(objs, f)
