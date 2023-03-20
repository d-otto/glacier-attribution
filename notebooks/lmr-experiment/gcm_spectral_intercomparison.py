# -*- coding: utf-8 -*-
"""
lmr_spectral.py

Description.

Author: drotto
Created: 3/6/2023 @ 1:18 PM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator, ScalarFormatter, FixedFormatter, FixedLocator, FormatStrFormatter, NullFormatter
import gm
import dill
import gzip
from pathlib import Path
import importlib
from src.data import get_gwi, get_cmip6, sample_glaciers_from_gcms, get_rgi, get_glacier_gcm, get_lmr, sample_glaciers
import src.data as data
from src.flowline import get_flowline_geom
from src.climate import detrend_gcm, extend_ts, prepend_ts
import xarray as xr
import multiprocess as mp
from itertools import product
from src.util import dict_key_from_value
from functools import partial
import xrft
import scipy as sci

#%%

import config

importlib.reload(config)
from config import cfg, ROOT


def temp_stable(yr):
    return 0


rgiids = list(cfg['glaciers'].keys())
rgi = get_rgi(rgiids, from_sqllite=True)

#%%

gcm = get_glacier_gcm(rgiids, 'tas', freq='jjas', mip=5)
gcm = gcm.sel(collection='lm', mip=5)
gcm = gcm.sel(experiment='past1000')
for dimension in ['model', 'r', 'i', 'f', 'p']:
    is_data = (gcm['tas'].count(dim=[dim for dim in gcm.dims.keys() if dim != dimension]) > 0).compute()
    gcm = gcm.sel({dimension: is_data})
gcm = gcm.sel(p=[1, 121])
gcm = gcm.mean(dim='p')
gcm = gcm.mean(dim=['r', 'i', 'f'])

#%%

T = gcm['tas'].sel(time=slice(850,1849))
T = T.sel(model=[model for model in T.coords['model'].values if model != 'FGOALS-gl'])
P = gcm['pr'].sel(time=slice(850,1849))
P = P.sel(model=[model for model in P.coords['model'].values if model != 'FGOALS-gl'])
t = T.time.values


#%%

T = xr.apply_ufunc(sci.signal.detrend, T, dask="parallelized")
T = T/T.std(dim=['model', 'rgiid']).compute()
P = xr.apply_ufunc(sci.signal.detrend, P, dask="parallelized")
P = P/P.std(dim=['model', 'rgiid']).compute()

#%%

def welch(d, **kwargs):
    print(d.shape)
    f, Pxx = sci.signal.welch(d, **kwargs)
    Pxx = Pxx/Pxx.mean()
    print(f.shape)
    print(Pxx.shape)
    return Pxx

pcrit = 0.95
m_window = 128
nfrac = 1
ifx = int(m_window/2*nfrac)
f = np.fft.rfftfreq(m_window)
per = 1/f
Pxx = xr.apply_ufunc(welch, T, input_core_dims=[['time']], output_core_dims=[['freq']], exclude_dims={'time'}, dask='parallelized',
                      dask_gufunc_kwargs=dict(allow_rechunk=True), kwargs=dict(fs=1, nperseg=m_window, noverlap=m_window/2, detrend='linear'),
                      output_dtypes=[float], output_sizes=dict(freq=m_window/2+1))
Pxx['freq'] = f
Pxx = Pxx.compute()

Pyy = xr.apply_ufunc(welch, P, input_core_dims=[['time']], output_core_dims=[['freq']], exclude_dims={'time'}, dask='parallelized',
                      dask_gufunc_kwargs=dict(allow_rechunk=True), kwargs=dict(fs=1, nperseg=m_window, noverlap=m_window/2, detrend='linear'),
                      output_dtypes=[float], output_sizes=dict(freq=m_window/2+1))
Pyy['freq'] = f
Pyy = Pyy.compute()



#%% Plot each model for each glacier (many plots)

for rgiid, d in Pxx.groupby('rgiid'):
    fig, ax = plt.subplots(1,1, dpi=200)
    for model, da in d.groupby('model'):
        ax.plot(da.freq[1:], da[1:], label=model, lw=0.75)
    ax.plot(da.freq[1:], d.mean(dim='model')[1:], label='Gmean', lw=2, color='black')
    mean = d.mean(dim='model')[1:]
    std = d.std(dim='model')[1:]
    ax.fill_between(da.freq[1:], mean-std, mean+std, color='black', ec=None, alpha=0.125, label=r'$\pm 1 \sigma$')
    
    xticks = [0.01, 0.05, 0.1, 0.5]
    xlabels = [f'{1 / x:.0f}' for x in xticks]
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xticks(xticks, labels=xlabels)
    ax.set_xlabel('Period (years)')
    ax.grid(which='both', axis='both', ls=':')
    ax.legend()
    ax.set_title(rgiid)
    fig.show()
   
    
#%% Mean of all models for each glacier (1 plot)

fig, ax = plt.subplots(1, 1, dpi=200)
for rgiid, d in Pxx.mean(dim='model').groupby('rgiid'):
    ax.plot(d.freq[1:], d[1:], label=rgiid, lw=0.75)
mean = Pxx.mean(dim='model').mean(dim='rgiid')[1:]
std = Pxx.mean(dim='model').std(dim=['rgiid'])[1:]
ax.plot(d.freq[1:], mean, label='Gmean', lw=2, color='black')
ax.fill_between(d.freq[1:], mean - std, mean + std, color='black', ec=None, alpha=0.125, label=r'$\pm 1 \sigma$')

xticks = [0.01, 0.05, 0.1, 0.5]
xlabels = [f'{1 / x:.0f}' for x in xticks]
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_ylim(0.3, 3)
ax.set_xticks(xticks, labels=xlabels)
ax.set_xlabel('Period (years)')
ax.grid(which='both', axis='both', ls=':')
ax.legend()
ax.set_title('Case-study glaciers')
fig.show()


#%% Mean of all glaciers for each model (1 plot)

fig, ax = plt.subplots(1, 1, dpi=200)
for rgiid, d in Pxx.mean(dim='rgiid').groupby('model'):
    ax.plot(d.freq[1:], d[1:], label=rgiid, lw=0.75)
mean = Pxx.mean(dim=['rgiid', 'model'])[1:]
std = Pxx.mean(dim='rgiid').std(dim=['model'])[1:]
ax.plot(d.freq[1:], mean, label='Gmean', lw=2, color='black')
ax.fill_between(d.freq[1:], mean - std, mean + std, color='black', ec=None, alpha=0.125, label=r'$\pm 1 \sigma$')

xticks = [0.01, 0.05, 0.1, 0.5]
xlabels = [f'{1 / x:.0f}' for x in xticks]
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_ylim(0.3, 3)
ax.set_xticks(xticks, labels=xlabels)
ax.set_xlabel('Period (years)')
ax.grid(which='both', axis='both', ls=':')
ax.legend()
ax.set_title('GCM Gmeans for case-study glaciers')
fig.show()


# %% Plot each model for each glacier (many plots)

for rgiid, d in Pyy.groupby('rgiid'):
    fig, ax = plt.subplots(1, 1, dpi=200)
    for model, da in d.groupby('model'):
        ax.plot(da.freq[1:], da[1:], label=model, lw=0.75)
    ax.plot(da.freq[1:], d.mean(dim='model')[1:], label='Gmean', lw=2, color='black')
    mean = d.mean(dim='model')[1:]
    std = d.std(dim='model')[1:]
    ax.fill_between(da.freq[1:], mean - std, mean + std, color='black', ec=None, alpha=0.125, label=r'$\pm 1 \sigma$')

    xticks = [0.01, 0.05, 0.1, 0.5]
    xlabels = [f'{1 / x:.0f}' for x in xticks]
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xticks(xticks, labels=xlabels)
    ax.set_xlabel('Period (years)')
    ax.grid(which='both', axis='both', ls=':')
    ax.legend()
    ax.set_title(rgiid)
    fig.show()

# %% Mean of all models for each glacier (1 plot)

fig, ax = plt.subplots(1, 1, dpi=200)
for rgiid, d in Pyy.mean(dim='model').groupby('rgiid'):
    ax.plot(d.freq[1:], d[1:], label=rgiid, lw=0.75)
mean = Pyy.mean(dim='model').mean(dim='rgiid')[1:]
std = Pyy.mean(dim='model').std(dim=['rgiid'])[1:]
ax.plot(d.freq[1:], mean, label='Gmean', lw=2, color='black')
ax.fill_between(d.freq[1:], mean - std, mean + std, color='black', ec=None, alpha=0.125, label=r'$\pm 1 \sigma$')

xticks = [0.01, 0.05, 0.1, 0.5]
xlabels = [f'{1 / x:.0f}' for x in xticks]
ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_ylim(0.3, 3)
ax.set_xticks(xticks, labels=xlabels)
ax.set_xlabel('Period (years)')
ax.grid(which='both', axis='both', ls=':')
ax.legend()
ax.set_title('Case-study glaciers')
fig.show()

# %% Mean of all glaciers for each model (1 plot)

fig, ax = plt.subplots(1, 1, dpi=200)
for rgiid, d in Pyy.mean(dim='rgiid').groupby('model'):
    ax.plot(d.freq[1:], d[1:], label=rgiid, lw=0.75)
mean = Pyy.mean(dim=['rgiid', 'model'])[1:]
std = Pyy.mean(dim='rgiid').std(dim=['model'])[1:]
ax.plot(d.freq[1:], mean, label='Gmean', lw=2, color='black')
ax.fill_between(d.freq[1:], mean - std, mean + std, color='black', ec=None, alpha=0.125, label=r'$\pm 1 \sigma$')

xticks = [0.01, 0.05, 0.1, 0.5]
xlabels = [f'{1 / x:.0f}' for x in xticks]
ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_ylim(0.3, 3)
ax.set_xticks(xticks, labels=xlabels)
ax.set_xlabel('Period (years)')
ax.grid(which='both', axis='both', ls=':')
ax.legend()
ax.set_title('GCM Gmeans for case-study glaciers')
fig.show()

