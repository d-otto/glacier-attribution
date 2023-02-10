# -*- coding: utf-8 -*-
"""
plot_oggm_geom.py

Description.

Author: drotto
Created: 1/19/2023 @ 12:02 PM
Project: glacier-attribution
"""


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from itertools import cycle, zip_longest, product
import xarray as xr
from datetime import datetime, timezone
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import pickle
import gzip
import rioxarray
from pyproj import Transformer
import salem
from shapely.geometry import LineString, Point

import oggm
from oggm import workflow
from oggm import tasks, utils
from oggm import graphics
from oggm.shop import bedtopo

import src.data as data
from src.flowline import get_flowline_geom

import config
from config import cfg, ROOT

mpl.use('qtagg')

#%%

rgiids = list({k: v for k, v in cfg['glaciers'].items()}.keys())
for rgiid in rgiids:
    
    #%%
    
    #rgi = data.get_rgi(rgiid, from_sqllite=True).iloc[0]
    geom = get_flowline_geom(rgiid)
    
    #%%
    
    #oggm.cfg.initialize(file=r'\\wsl.localhost\Ubuntu\home\drotto\.oggm_config')
    oggm.cfg.initialize_minimal()
    oggm.cfg.PATHS['dl_cache_dir'] = Path(r'~/OGGM/download_cache')
    oggm.cfg.PATHS['rgi_dir'] = Path(r'~/OGGM/rgi')
    oggm.cfg.PATHS['test_dir'] = Path(r'~/OGGM/tests')
    oggm.cfg.PATHS['tmp_dir'] = Path(r'~/OGGM/tmp')
    oggm.cfg.PATHS['working_dir'] = Path(r'~/oggm_plot')
    oggm.cfg.PARAMS['prcp_scaling_factor'] = 2.5
    oggm.cfg.PARAMS['climate_qc_months'] = 3
    oggm.cfg.PARAMS['use_winter_prcp_factor'] = False
    oggm.cfg.PARAMS['min_mu_star'] = 50
    oggm.cfg.PARAMS['max_mu_star'] = 1000
    oggm.cfg.PARAMS['use_multiprocessing'] = False
    oggm.cfg.PARAMS['check_calib_params'] = False
    oggm.cfg.PARAMS['use_rgi_area'] = True
    oggm.cfg.PARAMS['use_tstar_calibration'] = True
    
    rgi_ids = oggm.utils.get_rgi_glacier_entities([rgiid])
    #gdirs = oggm.workflow.init_glacier_directories(rgi_ids)
    prepro_path = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/centerlines/w5e5/qc0/pcpwin/match_geod_pergla/'
    gdirs = oggm.workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_border=160, prepro_base_url=prepro_path)  # 160 is max border available rn
    gdir = gdirs[0]
    
    
    #%%
    
    fig, ax = plt.subplots(1,1, dpi=200)
    graphics.plot_centerlines(gdir, use_flowlines=True, add_downstream=True, ax=ax)
    fig.suptitle(cfg['glaciers'][rgiid]['name'])
    plt.savefig(Path('/mnt/c/sandbox/glacier-attribution/plots/case_study/glacier_geom/', f'{rgiid}_centerlines.png'))
    
    #%%
    
    fig, ax = plt.subplots(1,1, dpi=200)
    graphics.plot_catchment_areas(gdir, ax=ax)
    fig.suptitle(cfg['glaciers'][rgiid]['name'])
    plt.savefig(Path('/mnt/c/sandbox/glacier-attribution/plots/case_study/glacier_geom/', f'{rgiid}_catchment_areas.png'))
    
    #%%
    
    fig, ax = plt.subplots(1,1, dpi=200)
    graphics.plot_googlemap(gdir, ax=ax)
    plt.savefig(Path('/mnt/c/sandbox/glacier-attribution/plots/case_study/glacier_geom/', f'{rgiid}_satellite.png'))
    
    #%%
    
    
    workflow.execute_entity_task(bedtopo.add_consensus_thickness, gdir)
    
    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        ds = ds.load()
    
    fig, ax = plt.subplots(1,1, dpi=200)
    
    # plot the salem map background, make countries in grey
    smap = ds.salem.get_map(countries=False)
    smap.set_shapefile(gdir.read_shapefile('outlines'))
    smap.set_topography(ds.topo.data);
    
    
    smap.set_data(ds.consensus_ice_thickness)
    smap.set_cmap('Blues')
    smap.plot(ax=ax)
    smap.append_colorbar(ax=ax, label='ice thickness (m)')
    fig.suptitle(cfg['glaciers'][rgiid]['name'])
    plt.savefig(Path('/mnt/c/sandbox/glacier-attribution/plots/case_study/glacier_geom/', f'{rgiid}_farinotti_thickness.png'))
    
    
    #%%
    
    fig, ax = plt.subplots(1,1, dpi=200)
    graphics.plot_inversion(gdir, ax=ax)
    fig.suptitle(cfg['glaciers'][rgiid]['name'])
    plt.savefig(Path('/mnt/c/sandbox/glacier-attribution/plots/case_study/glacier_geom/', f'{rgiid}_inverted_thickness.png'))