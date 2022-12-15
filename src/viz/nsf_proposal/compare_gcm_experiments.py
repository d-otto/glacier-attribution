# -*- coding: utf-8 -*-
"""
detrend_gcm_pandas.py

Description.

Author: drotto
Created: 11/9/2022 @ 3:52 PM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import statsmodels.api as sm
import xarray as xr
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d

from config import ROOT, cfg
from src.data import get_gwi, get_cmip6, sample_glaciers_from_gcms, get_rgi, get_cmip6_run_attrs, get_cmip6_lm
import src.data as data
from src.util import unwrap_coords, dict_key_from_value
from src.climate import detrend_gcm

# %%

rgiids = {'hintereisferner': 'RGI60-11.00897',
          'argentiere'     : 'RGI60-11.03638',
          'south cascade'  : 'RGI60-02.18778',
          'wolverine'      : 'RGI60-01.09162'}
rgi = get_rgi(list(rgiids.values()), from_sqllite=True)
rgi = rgi.set_index('RGIId')

#%%

rgiid = list({k: v for k, v in cfg['glaciers'].items() if v['name'] == 'Argentiere'}.keys())[0]
d = data.get_glacier_gcm(rgiid, variable='tas', freq='jjas', mip=5)


# %%

# todo: put this back in later once I find which historic runs are spawned from p1k
# ds_p1k = d.sel(experiment=['past1000'])
# ds_hist = d.sel(experiment=['historical'])
# ds = xr.merge([ds_p1k, ds_hist], compat='minimal')
ds = d.sel(experiment=['past1000'])
ds = ds.sel(time=(ds.time <= 2014))
ds = ds.groupby('model').mean(dim='variant')
# todo: change this back to 1850 & 1900 once the whole time series is stitched together
mask = (ds.time > 1800) & (ds.time <= 1849)
Tref = ds.sel(time=mask).groupby('model').mean(dim=['time'])
ds = ds - Tref
df = ds['tas'].to_dataframe()
df = df.reset_index()

#%%

sns.set_theme(style='whitegrid', palette='bright', color_codes=True)
fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)

#for key, dfg in df.groupby(['model', 'experiment', 'variant']):
for key, dfg in df.groupby(['model', 'experiment']):
    #model, experiment, variant = key
    model, experiment = key
    #model, variant = key
    #model = key
    dfg = dfg.dropna(how='any')
    if dfg.tas.any():
        sm = uniform_filter1d(dfg.tas.values, 30, mode='mirror')
        #ax.plot(dfg.time, sm, label=f'{model}, {variant}', c='#b0b0b0', lw=1)
        ax.plot(dfg.time, sm, lw=0.5, alpha=0.75)

# legend item for model data
ax.plot([0], [0], lw=0.5, alpha=0.75, label='Surface temperature (JJAS)')

# mean line
ts = df.time.unique()
sm = uniform_filter1d(df.tas.groupby(df.time).mean().values, 30, mode='mirror')
# ax.plot(ts, sm, label=f"Multi-model mean (N = {len(df['model'].unique())})", c='tab:red', lw=1.5)
ax.plot(ts, sm, label=f"Multi-model mean (N = {len(df['model'].unique())})", c='black', lw=1)

# std dev band
std = uniform_filter1d(df.tas.groupby(df.time).std().values, 30, mode='mirror')
ax.fill_between(ts, y1=sm - std, y2=sm + std, fc='grey', ec='none', alpha=0.25, label='2 std. dev.')




ax.annotate(r'$T_{ref}$ = 1800-1849',
            xy=(0, 0), xytext=(12, 12), xycoords='axes fraction', textcoords='offset points',
            fontsize=12, fontweight='bold')

#ax.set_ylim(-2, 2)
ax.set_ylabel(r'$T_{anom}$ ($\degree C$)')
ax.set_xlabel('Year')

ax.set_axisbelow(True)
ax.set_xticks(np.arange(800, 2100, 100), major=True)
ax.set_xticks(np.arange(800, 2100, 25), minor=True)
ax.grid(which='minor', axis='both', c='#f4f4f4')
ax.grid(which='major', axis='both', c='#c9c9c9')
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
ax.legend(title=cfg['glaciers'][rgiid]['name'], title_fontproperties=dict(weight='bold'))
ax.set_xlim(850, 2014)
sns.move_legend(ax, loc='upper left')
plt.tight_layout()
fig.savefig(Path(ROOT, f'plots/nsf_proposal/p1k-spaghetti_cmip5_{rgiid}.png'))
fig.show()