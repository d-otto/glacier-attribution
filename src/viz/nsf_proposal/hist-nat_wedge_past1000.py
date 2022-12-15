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

# %%

rgiid = rgi.index[0]
d = data.get_glacier_gcm(rgiid, variable='tas', freq='jjas', mip=6)

# %%

ds_nat = d.sel(experiment=['hist-nat'])
ds_anth_p1k = d.sel(experiment=['past1000'])
ds_anth_hist = d.sel(experiment=['historical'])
ds_anth = xr.concat([ds_anth_p1k, ds_anth_hist], dim='time')
ds = xr.combine_by_coords([ds_nat, ds_anth])
ds = ds.sel(time=(ds.time <= 2014))
ds = ds.groupby('model').mean(dim='variant')

# # Scheme 1: Tref is determined independently for ds_nat and ds_anth
# concat = []
# for key, dss in ds.groupby('experiment'):
#     mask = (dss.time > 1850) & (dss.time <= 1900)
#     Tref = dss.sel(time=mask).groupby('model').mean(dim=['time'])
#     dss = dss - Tref
#     concat.append(dss)
# ds = xr.concat(concat, dim='experiment')

# Scheme 2: Tref is determined by ds_nat
mask = (ds.time > 1850) & (ds.time <= 1900)
Tref = ds.sel(time=mask, experiment='hist-nat').groupby('model').mean(dim=['time'])
ds = ds - Tref


df = ds['tas'].to_dataframe().dropna(how='any')
df = df.reset_index()

# %%

sns.set_theme(style='whitegrid', palette='bright', color_codes=True)
fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)

ts = df.time.unique()
# for key, dfg in df.groupby(['model', 'experiment', 'variant']):
for key, dfg in df.groupby(['model']):
    df_nat = dfg.loc[dfg.experiment == 'hist-nat']
    df_anth = dfg.loc[dfg.experiment == 'historical']
    if len(df_nat) > 0 and len(df_anth) > 0:
        sm_nat = uniform_filter1d(df_nat.tas.values, 30, mode='mirror')
        sm_anth = uniform_filter1d(df_anth.tas.values, 30, mode='mirror')
        ax.fill_between(ts, y1=sm_anth, y2=sm_nat, where=sm_anth > sm_nat, interpolate=True, fc='tab:red', ec='none',
                        alpha=0.25)
        ax.fill_between(ts, y1=sm_anth, y2=sm_nat, where=sm_anth < sm_nat, interpolate=True, fc='tab:blue', ec='none',
                        alpha=0.25)
        ax.plot(ts, sm_anth, c='black', lw=1, ls='dashed')
        ax.plot(ts, sm_nat, c='black', lw=1)

ax.plot([0], [0], c='black', lw=1, ls='dashed', label='Anthropogenic')
ax.plot([0], [0], c='black', lw=1, label='Natural')
ax.fill_between([0], y1=[0], y2=[0], fc='tab:red', ec='none', alpha=0.25, label='Warming')
ax.fill_between([0], y1=[0], y2=[0], fc='tab:blue', ec='none', alpha=0.25, label='Cooling')

ax.set_ylabel(r'$T_{anom}$ ($\degree C$)')
ax.set_xlabel('Year')

ax.annotate(r'$T_{ref}$ = 1850-1900',
            xy=(0, 0), xytext=(12, 12), xycoords='axes fraction', textcoords='offset points',
            fontsize=12, fontweight='bold')

ax.set_axisbelow(True)
ax.set_xticks(np.arange(800, 2100, 25), major=True)
ax.set_xticks(np.arange(800, 2100, 5), minor=True)
ax.grid(which='minor', axis='both', c='#f4f4f4')
ax.grid(which='major', axis='both', c='#c9c9c9')
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
ax.legend(title=cfg['glaciers'][rgiid]['name'], title_fontproperties=dict(weight='bold'))
ax.set_xlim(850, 2014)
# sns.move_legend(ax, loc='upper left')
plt.tight_layout()
fig.savefig(Path(ROOT, f'plots/nsf_proposal/hist-nat_wedge_past1000_{rgiid}.png'))
fig.show()