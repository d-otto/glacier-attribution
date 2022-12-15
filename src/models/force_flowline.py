# -*- coding: utf-8 -*-
"""
calibrate_flowline.py

Description.

Author: drotto
Created: 11/8/2022 @ 5:47 PM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import gm
import pickle
import gzip
from pathlib import Path
import importlib
from src.data import get_gwi, get_cmip6, sample_glaciers_from_gcms, get_rgi, get_cmip6_run_attrs, get_cmip6_lm
import src.data as data
from src.flowline import get_flowline_geom
from src.climate import detrend_gcm, extend_ts, prepend_ts
import xarray as xr
from src.util import dict_key_from_value

#%%
import config
importlib.reload(config)
from config import cfg, ROOT

def temp_stable(yr):
    return 0

rgiid = list({k: v for k, v in cfg['glaciers'].items() if v['name'] == 'Argentiere'}.keys())[0]
params = cfg['glaciers'][rgiid]['flowline_params']

#%%
rgi = get_rgi(rgiid, from_sqllite=True)
geom = get_flowline_geom(rgiid)
gcm = data.get_glacier_gcm(rgiid, variable='tas', freq='jjas', mip=6)
gcm0 = gcm.sel(experiment='past1000', model='MRI-ESM2-0', variant='r1i1p1f1', mip=6).isel(rgiid=0)
gcm1 = gcm.sel(experiment='historical', model='MRI-ESM2-0', variant='r1i1000p1f1', mip=6).isel(rgiid=0)
ds = xr.concat([gcm0, gcm1], dim='time')['tas'].dropna(how='any', dim='time')
df = xr.concat([gcm0, gcm1], dim='time').to_dataframe().reset_index().dropna(how='any')
df = df.set_index(df.time.astype(int))['tas']
gwi = get_gwi(freq='year')


#%%

for gcm_name, da_comb in ds.groupby('model'):
    da_comb = da_comb.isel(rgiid=0).squeeze()
    #gcm_name = da_comb.model.values.item()
    for scenario in ('nat', 'comb'):
        
        detrended = detrend_gcm(da_comb.copy(), trend=gwi)  # is copy needed here?
        random_climate = pd.read_csv(r"C:\Users\drotto\Documents\USGS\glacier-diseq\features\ClimateRandom.csv")
        Prand = random_climate['Prand']
        Tnat = detrended.y_nat.to_pandas()
        Tref = Tnat.loc[1850:1900].mean()
        #Tref = Tnat.loc[:1850].mean()
        Tnat = Tnat - Tref
        Tnat = extend_ts(Tnat, win=30, n=1000)
        Tnat = prepend_ts(Tnat, n=250)
        Tcomb = da_comb.copy()
        Tcomb = Tcomb.to_pandas()
        Tcomb = Tcomb - Tref
        Tcomb = extend_ts(Tcomb, win=30, n=1000)
        Tcomb = prepend_ts(Tcomb, n=250)
        forcing = {'nat': Tnat, 'comb': Tcomb}
        
        ######################
        T = forcing[scenario]
        ######################
        
        
        fp = Path(ROOT, params['ss_noise_profile'])
        with open(fp, 'rb') as f:
            last_run = pickle.load(f)
        h_init = last_run.h.mean(axis=0)
        x_init = last_run.x
        
        model = gm.flowline2d(x_gr=geom.x, zb_gr=geom.zb, x_geom=geom.x, w_geom=geom.w, x_init=x_init, h_init=h_init, xmx=geom.x.max(),
                              temp=temp_stable, sigT=1, sigP=0, P0=params['P0'],
                              delt=0.0125/16, delx=50,
                              ts=int(T.index[0]), tf=int(T.index[-1]), T0=params['T0'],
                              T=T, P=Prand,
                              rt_plot=False, dt_plot=20)
        model.Tref = Tref  # additional variable to save for later analysis
        
        fig = model.plot()
        fig.show()
        fig_output_name = Path(ROOT, f'flowline2d_{rgiid}.{gcm_name}.past1000.gwi.{scenario}.png')
        fig.savefig(fig_output_name)
        
        file_output_name = Path(ROOT, f'flowline2d_{rgiid}.{gcm_name}.past1000.gwi.{scenario}.pickle')
        model.to_pickle(file_output_name)