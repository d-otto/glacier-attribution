# -*- coding: utf-8 -*-
"""
fig3.py

Description.

Author: drotto
Created: 3/10/2023 @ 1:20 PM
Project: glacier-attribution
"""

from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np
import scipy as sci
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt

from src.data import get_rgi, get_glacier_gcm, sample_glaciers, get_lmr
from config import cfg, ROOT


rng = np.random.default_rng()

#%%

rgiids = ['RGI60-11.03638', 'RGI60-01.09162']
rgi = get_rgi(rgiids, from_sqllite=True).set_index('RGIId')

gnames = ['French Alps (Argentiere Glacier)', 'Maritime Southcentral Alaska (Wolverine Glacier)']

#%%
i = 0
rgiid = rgiids[i]
gname = gnames[i]

# lmr = get_lmr()
# lmr = lmr.mean(dim='MCrun')
# lmr = lmr.sel(time=slice(850, 1849))
# lmr_rgi = sample_glaciers(lmr, rgi, as_dim=True)
# lmr_rgi = lmr_rgi.sel(rgiid=rgiid)
# lmr_T = lmr_rgi.air.to_numpy()
# lmr_P = lmr_rgi.prate.to_numpy()
# t = np.arange(851,1849,1)

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

T = gcm['tas'].sel(time=slice(850, 1849))
T = T.sel(model=[model for model in T.coords['model'].values if model != 'FGOALS-gl'])
P = gcm['pr'].sel(time=slice(850, 1849))
P = P.sel(model=[model for model in P.coords['model'].values if model != 'FGOALS-gl'])
t = T.time.values

# %%

T = xr.apply_ufunc(sci.signal.detrend, T, dask="parallelized")
T = T / T.std(dim=['model', 'rgiid']).compute()
P = xr.apply_ufunc(sci.signal.detrend, P, dask="parallelized")
P = P / P.std(dim=['model', 'rgiid']).compute()


# %%

def welch(d, **kwargs):
    print(d.shape)
    f, Pxx = sci.signal.welch(d, **kwargs)
    Pxx = Pxx / Pxx.mean()
    print(f.shape)
    print(Pxx.shape)
    return Pxx


pcrit = 0.95
m_window = 256
nfrac = 1
ifx = int(m_window / 2 * nfrac)
f = np.fft.rfftfreq(m_window)
per = 1 / f
Pxx = xr.apply_ufunc(welch, T, input_core_dims=[['time']], output_core_dims=[['freq']], exclude_dims={'time'},
                     dask='parallelized',
                     dask_gufunc_kwargs=dict(allow_rechunk=True),
                     kwargs=dict(fs=1, nperseg=m_window, noverlap=m_window / 2, detrend='linear'),
                     output_dtypes=[float], output_sizes=dict(freq=m_window / 2 + 1))
Pxx['freq'] = f
Pxx = Pxx.compute()

Pyy = xr.apply_ufunc(welch, P, input_core_dims=[['time']], output_core_dims=[['freq']], exclude_dims={'time'},
                     dask='parallelized',
                     dask_gufunc_kwargs=dict(allow_rechunk=True),
                     kwargs=dict(fs=1, nperseg=m_window, noverlap=m_window / 2, detrend='linear'),
                     output_dtypes=[float], output_sizes=dict(freq=m_window / 2 + 1))
Pyy['freq'] = f
Pyy = Pyy.compute()

# %% Plot each model for each glacier (many plots)

def smooth(d, n):
    return sci.ndimage.uniform_filter1d(d, n)

for i in range(len(rgiids)):
    rgiid = rgiids[i]
    gname = gnames[i]

    fig, ax = plt.subplots(1, 2, dpi=100, sharey=True, figsize=(10, 4), layout='constrained')
    
    d = Pxx.sel(rgiid=rgiid)
    for model, da in d.groupby('model'):
        ax[0].plot(da.freq[1:], smooth(da[1:], 10), label=model, lw=0.75)
    ax[0].plot(da.freq[1:], d.mean(dim='model')[1:], label='Gmean', lw=3, color='black')
    mean = d.mean(dim='model')[1:]
    std = d.std(dim='model')[1:]
    ax[0].fill_between(da.freq[1:], mean - std, mean + std, color='black', ec=None, alpha=0.125, label=r'$\pm 1 \sigma$')
    xticks = [0.01, 0.05, 0.1, 0.5]
    xlabels = [f'{1 / x:.0f}' for x in xticks]
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_ylim(0.3, 3)
    ax[0].set_xticks(xticks, labels=xlabels)
    ax[0].tick_params(axis='both', which='both', direction='in')
    ax[0].set_xlabel('Period (years)')
    ax[0].set_ylabel('Power spectral density')
    ax[0].grid(which='both', axis='both', ls=':')
    ax[0].set_title('Temperature', fontsize='medium')
    ax[0].legend()

    d = Pyy.sel(rgiid=rgiid)
    for model, da in d.groupby('model'):
        ax[1].plot(da.freq[1:], smooth(da[1:], 10), label=model, lw=0.75)
    ax[1].plot(da.freq[1:], d.mean(dim='model')[1:], label='Gmean', lw=3, color='black')
    mean = d.mean(dim='model')[1:]
    std = d.std(dim='model')[1:]
    ax[1].fill_between(da.freq[1:], mean - std, mean + std, color='black', ec=None, alpha=0.125, label=r'$\pm 1 \sigma$')
    xticks = [0.01, 0.05, 0.1, 0.5]
    xlabels = [f'{1 / x:.0f}' for x in xticks]
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_ylim(0.3, 3)
    ax[1].set_xticks(xticks, labels=xlabels)
    ax[1].tick_params(axis='both', which='both', direction='in')
    ax[1].set_xlabel('Period (years)')
    # ax[1].set_ylabel('Power spectral density')
    ax[1].grid(which='both', axis='both', ls=':')
    ax[1].set_title('Precipitation', fontsize='medium')
    
    fig.show()
    plt.savefig(Path(ROOT, f'notebooks/lmr-experiment/figures/fig3_{rgiid}.svg'))


