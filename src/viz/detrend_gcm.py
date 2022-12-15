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

from config import ROOT, cfg
import src.data as data
from src.data import get_gwi, get_cmip6, sample_glaciers_from_gcms, get_rgi, get_cmip6_run_attrs, get_cmip6_lm
from src.util import unwrap_coords, dict_key_from_value, dropna_coords
from src.climate import detrend_gcm

# %%

rgiids = {'hintereisferner': 'RGI60-11.00897',
          'argentiere': 'RGI60-11.03638',
          'south cascade': 'RGI60-02.18778',
          'wolverine': 'RGI60-01.09162'}
rgi = get_rgi(list(rgiids.values()), from_sqllite=True)
rgi = rgi.set_index('RGIId')

rgiid = list({k: v for k, v in cfg['glaciers'].items() if v['name'] == 'South Cascade'}.keys())[0]

#%%

# experiment = 'past1000'
# gcm = get_cmip6_lm()

experiment = 'past1000'
if experiment == 'past1000':
    gcm = data.get_glacier_gcm(rgiid, variable='tas', freq='jjas', mip=6)
    gcm = gcm.sel(experiment=['past1000', 'past2k'])
    da = dropna_coords(gcm['tas'], dim='model', how='any')
else:
    gcm = get_cmip6(experiment=[experiment], variant=['r1i1p1f1'], freq='jjas')
gwi = get_gwi(freq='year')

#%%



#%%

ds = sample_glaciers_from_gcms(gcm, rgi)
ds = ds.sel(time=(ds.time <= 2014))


#%%

da_comb = ds.isel(model=0, rgiid=0).copy()
detrended = detrend_gcm(da_comb, trend=gwi)


# %%

da_nat = detrended.y_nat
diff = da_comb - da_nat
res = detrended.res
bx = (res.params[0] * gwi).iloc[-1]
Tref = da_comb.loc[(da_comb.time >= 1850) & (da_comb.time < 1900)].mean()
y_sm = da_comb.rolling(dict(time=20)).mean()
y_nat_sm = da_nat.rolling(dict(time=20)).mean()

rgiid = da_comb.rgiid.item()
model = da_comb.model.item()
glacier_name = dict_key_from_value(rgiids, rgiid)

sns.set_theme(style='whitegrid', palette='bright', color_codes=True)
fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)

sns.lineplot(x=gwi.index, y=gwi, ax=ax, label='Anthropogenic contribution', color='y')
sns.lineplot(x=y_sm.time, y=y_sm - Tref, ax=ax, label='COMB (20yr MA)', color='r')
sns.lineplot(x=y_nat_sm.time, y=y_nat_sm - Tref, ax=ax, label='NAT (20yr MA)', color='b')
sns.lineplot(x=da_comb.time, y=da_comb - Tref, ax=ax, color='r', alpha=0.5, lw=0.5)
sns.lineplot(x=da_nat.time, y=da_nat - Tref, ax=ax, color='b', alpha=0.5, lw=0.5)

ax.annotate(
    r'$\Delta{T}_{end}=$' + f'{bx:.3f} degC ' + r'$\beta=$' + f'{res.params[0]:.3f} degC $SNR=$ {res.params[0] / da_comb.std():.3f}',
    xy=(0, 0), xytext=(12, 12), xycoords='axes fraction', textcoords='offset points',
    fontsize=12, fontweight='bold')
# ax.set_xlim(850, 2015)
ax.set_xlim(1850, 2015)
ax.set_ylim(-2, 2)
ax.set_title('Detrending GCM with GWI ($T_{ref}$ = 1850-1900) [JJAS]', loc='left', fontweight='bold')
ax.set_title(f'{glacier_name.title()}, {model} [{experiment}]', loc='right')
ax.set_ylabel(r'$T_{anom}$ [$\degree C$ ]')
ax.set_xlabel('Year')
sns.move_legend(ax, loc='upper left')
plt.tight_layout()
# plt.savefig(f'plots/gcm.detrended_gwi.{glacier_name}.{model}.{experiment}.png')

# ax.set_xlim(850, 2015)
# plt.savefig(f'plots/gcm_detrended.{glacier_name}.{gcm_name}.long.png')
fig.show()

# %% detrend ensemble mean
# todo: give the regression all of the data and take the mean after... easiest way to get the right confidence interval for the trends?

for glacier_name, rgiid in rgiids.items():
    for model in list(ds.model.values):
        da_comb = ds.sel(rgiid=rgiid).copy()
        da_comb = da_comb.sel(model=model)  # take this out when I do the above todo
        detrended = detrend_gcm(da_comb, trend=gwi)
    
        da_nat = detrended.y_nat
        diff = da_comb - da_nat
        res = detrended.res
        bx = (res.params[0] * gwi).iloc[-1]
        Tref = da_comb.loc[(da_comb.time >= 1850) & (da_comb.time < 1900)].mean()
        y_sm = da_comb.rolling(dict(time=20)).mean()
        y_nat_sm = da_nat.rolling(dict(time=20)).mean()
        model = da_comb.model.item()
    
        sns.set_theme(style='whitegrid', palette='bright', color_codes=True)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
    
        sns.lineplot(x=gwi.index, y=gwi, ax=ax, label='Anthropogenic contribution', color='y')
        sns.lineplot(x=y_sm.time[y_sm.time >= 1850], y=y_sm[y_sm.time >= 1850] - Tref, ax=ax, label='COMB (20yr MA)', color='r')
        sns.lineplot(x=y_nat_sm.time, y=y_nat_sm - Tref, ax=ax, label='NAT (20yr MA)', color='b')
        sns.lineplot(x=da_comb.time[da_comb.time >= 1850], y=da_comb[da_comb.time >= 1850] - Tref, ax=ax, color='r', alpha=0.5, lw=0.5)
        sns.lineplot(x=da_nat.time, y=da_nat - Tref, ax=ax, color='b', alpha=0.5, lw=0.5)
    
        ax.annotate(
            r'$\Delta{T}_{end}=$' + f'{bx:.3f} degC ' + r'$\beta=$' + f'{res.params[0]:.3f} degC $SNR=$ {res.params[0] / da_comb.std():.3f}',
            xy=(0, 0), xytext=(12, 12), xycoords='axes fraction', textcoords='offset points',
            fontsize=12, fontweight='bold')
        # ax.set_xlim(850, 2015)
        ax.set_xlim(1850, 2015)
        ax.set_ylim(-2, 2)
        ax.set_title('Detrending GCM with GWI ($T_{ref}$ = 1850-1900) [JJAS]', loc='left', fontweight='bold')
        ax.set_title(f'{glacier_name.title()}, {model} [{experiment}]', loc='right')
        ax.set_ylabel(r'$T_{anom}$ [$\degree C$ ]')
        ax.set_xlabel('Year')
        ax.xaxis.set_major_location(MultipleLocator(100))
        ax.xaxis.set_minor_location(MultipleLocator(50))
        ax.yaxis.set_minor_location(MultipleLocator(0.25))
        sns.move_legend(ax, loc='upper left')
        plt.tight_layout()
        plt.savefig(Path(ROOT, f'plots/gcm.detrended_gwi.{glacier_name}.{model}.{experiment}.png'))
        if experiment == 'past1000':
            ax.set_xlim(850, 2015)
            plt.savefig(Path(ROOT, f'plots/gcm_detrended.{glacier_name}.{model}.{experiment}.long.png'))
        
        fig.show()