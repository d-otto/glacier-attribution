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
    
    #%%

    from oggm import graphics
    
    dirname = 'data/interim/oggm_flowlines'
    gname = 'Argentiere'
    fp = dirname / f'e_bands/{rgiid}.downstream_line.pickle'
    with gzip.open(fp, "rb") as openfile:
        eband_fl, eband_dsl = pickle.load(openfile)
    eband_fl = eband_fl.__dict__
    fp = dirname / f'geometrical_centerlines/{rgiid}.downstream_line.pickle'
    with gzip.open(fp, "rb") as openfile:
        geom_fl = pickle.load(openfile)
    geom_dsl = geom_fl[-1]
    geom_fl = geom_fl[-2].__dict__

    fp = dirname / f'e_bands/{rgiid}.model_flowlines.pickle'
    with gzip.open(fp, "rb") as openfile:
        ebfl = pickle.load(openfile)
    fp = dirname / f'geometrical_centerlines/{rgiid}.model_flowlines.pickle'
    with gzip.open(fp, "rb") as openfile:
        gcfl = pickle.load(openfile)

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14), sharex=True, sharey=True)
    graphics.plot_modeloutput_section(gcfl, ax=ax1)
    ax1.set_title('Geometrical centerline')
    graphics.plot_modeloutput_section(ebfl, ax=ax2)
    ax2.set_title('Elevation band flowline')
    f.suptitle(f'{rgiid}: {gname}')
    f.show()

    # fig = plt.figure()
    # graphics.plot_modeloutput_section_withtrib(gcfl, fig=fig)

    # pull variables for the model
    # this results in a slight error in the x spacing because idx is not an exact continuation of geom_fl
    # it can be minimized by using a small bin size for the elevation band
    idx = np.where(eband_fl['bed_h'] - geom_fl['bed_h'][0] > 0)[0]
    if idx.size > 0:
        len0 = len(geom_fl['bed_h'])
        len1 = len0 + len(idx)  # length of the final profile
        zb = np.full(len1, fill_value=np.nan)
        zb[idx[-1] + 1:] = geom_fl['bed_h']
        zb[idx] = eband_fl['bed_h'][idx]
    else:
        zb = geom_fl['bed_h']
        len1 = len(geom_fl['bed_h'])

    # extra x's in the eb fl, prepend these to the geom fl
    xmax = max(
        geom_fl['dis_on_line'] * geom_fl['map_dx'])  # x from the geom flowline, treating this as the "true length"
    x = np.linspace(0, xmax, len1)

    # interp eband thickness to proper length
    eband_x = eband_fl['dis_on_line'] * eband_fl['map_dx']
    h = np.interp(x, eband_x, eband_fl['_thick'])
    eband_l = np.where(h > 0)[-1] * eband_fl['map_dx']

    # interp thickness to proper length after filling in the ds line
    w = eband_fl['_w0_m'].copy()  # w from the eband flowline, has nans that need to be filled after terminus
    idx_term = np.argmax(np.isnan(w)) - 1  # where is the terminus
    nan_len = len(w) - idx_term
    ds_len = len(eband_dsl['w0s'])  # length of the the data to fill in the nan's in w
    w[idx_term:idx_term + ds_len] = eband_dsl['w0s']  # fill the nan's in w
    if any(np.isnan(w)):  # not all nan's get filled, so ffill the remaining
        idx_nan = np.argmax(np.isnan(w))
        w[idx_nan:] = w[idx_nan - 1]
    w = np.interp(x, eband_x, w)

    # calculate length to check
    ebfl_l = np.where(ebfl[0].thick > 0)[0][-1] * ebfl[0].map_dx
    gcfl_l = np.where(gcfl[0].thick > 0)[0][-1] * gcfl[0].map_dx

    # calculate 3d length
    ebfl_3dl = np.sqrt(np.diff(ebfl[0].surface_h, 1)**2 + np.diff(ebfl[0].dis_on_line * ebfl[0].map_dx, 1)**2)
    ebfl_3dl = np.where(ebfl[0].thick[1:] > 0, ebfl_3dl, 0).sum()
    gcfl_3dl = np.sqrt(np.diff(gcfl[0].surface_h, 1)**2 + np.diff(gcfl[0].dis_on_line * gcfl[0].map_dx, 1)**2)
    gcfl_3dl = np.where(gcfl[0].thick[1:] > 0, gcfl_3dl, 0).sum()

    # calculate area to check
    ebfl_a = ebfl[0].widths_m * ebfl[0].dx_meter * 1e-6
    ebfl_a = np.where(ebfl[0].thick > 0, ebfl_a, 0)
    ebfl_a = ebfl_a.sum()
    gcfl_a = gcfl[0].widths_m * gcfl[0].dx_meter * 1e-6
    gcfl_a = np.where(gcfl[0].thick > 0, gcfl_a, 0)
    gcfl_a = gcfl_a.sum()

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, zb + h, c='blue')
    ax.plot(x, zb, c='black')
    ax.set_aspect('equal')
    ax.plot(x, w, c='red')
    ax.grid()
    fig.show()
