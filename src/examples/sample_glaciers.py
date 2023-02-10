# -*- coding: utf-8 -*-
"""
sample_glaciers.py

Description.

Author: drotto
Created: 1/24/2023 @ 10:35 AM
Project: glacier-attribution
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
import cartopy
import cartopy.crs as ccrs

from src.data import get_rgi, sample_glaciers, get_lmr
from config import cfg, ROOT

#%%

rgiids = list({k: v for k, v in cfg['glaciers'].items()}.keys())
rgi = get_rgi(rgiids, from_sqllite=True)
rgi = rgi.set_index('RGIId')
print(rgi)


#%%

lmr = get_lmr()
#lmr = lmr.load()
lmr = lmr.mean(dim='MCrun')

#%%

fig = plt.figure()
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
lmr['air'].isel(time=0).plot(ax=ax, cmap='bwr')
rgi.plot(x='CenLon', y='CenLat', kind='scatter', color='red', ax=ax, transform=ccrs.PlateCarree())
ax.coastlines()
fig.show()


#%%

lmr_rgi = sample_glaciers(lmr, rgi, as_dim=True)

#%%

fig = plt.figure()
ax = fig.add_subplot(111)
lmr_rgi['air'].plot.line(x='time', ax=ax)
fig.show()


#%%

df = lmr_rgi['air'].to_dataframe()
df = df.reset_index(level=0)
df.index = df.index.year
fig, ax = plt.subplots(2,1, figsize=(14,10), dpi=200)
for i, tup in enumerate(df.groupby(df.lon<100)):
    region, g = tup
    for rgiid, gg in g.groupby('rgiid'):
        d = gg['air']
        d = d.rolling(30).mean()
        name = cfg['glaciers'][rgiid]['name']
        ax[i].axhline(lw=1, c='black')
        ax[i].plot(d, lw=1, label=name)
        
        
for axis in ax:
    axis.axvspan(1951, 1980, alpha=0.1, color='red')
    axis.set_axisbelow(True)
    axis.grid(which='both', axis='both')
    axis.xaxis.set_major_locator(MultipleLocator(200))
    axis.xaxis.set_minor_locator(MultipleLocator(100))
    axis.yaxis.set_major_locator(MultipleLocator(0.25))
    axis.yaxis.set_minor_locator(MultipleLocator(0.125))
    axis.set_ylabel('degC')
    axis.legend()
ax[-1].set_xlabel('Years')
fig.suptitle('Last Millennium Reconstruction (v2.1) 2m temp (ref. 1951-1980)')
plt.tight_layout()
plt.savefig(Path(ROOT, 'plots/case_study/LMR_2mT_per_glacier.png'))
plt.show()


#%%

df = lmr_rgi['prate'].to_dataframe()
df = df.reset_index(level=0)
df.index = df.index.year
fig, ax = plt.subplots(2, 1, figsize=(14, 10), dpi=200)
for i, tup in enumerate(df.groupby(df.lon < 100)):
    region, g = tup
    for rgiid, gg in g.groupby('rgiid'):
        d = gg['prate']
        d = d.rolling(30).mean()
        name = cfg['glaciers'][rgiid]['name']
        ax[i].axhline(lw=1, c='black')
        ax[i].plot(d, lw=1, label=name)

for axis in ax:
    axis.axvspan(1951, 1980, alpha=0.1, color='red')
    axis.set_axisbelow(True)
    axis.grid(which='both', axis='both')
    axis.xaxis.set_major_locator(MultipleLocator(200))
    axis.xaxis.set_minor_locator(MultipleLocator(100))
    axis.yaxis.set_major_locator(MultipleLocator(5e-7))
    axis.yaxis.set_minor_locator(MultipleLocator(2.5e-7))
    axis.set_ylabel('mm/s')
    axis.legend()
ax[-1].set_xlabel('Years')
fig.suptitle('Last Millennium Reconstruction (v2.1) precipitation rate (ref. 1951-1980)')
plt.tight_layout()
plt.savefig(Path(ROOT, 'plots/case_study/LMR_Prate_per_glacier.png'))
plt.show()


#%%
    
df = lmr_rgi.to_dataframe()
df = df.reset_index(level=1)
df.index = df.index.year
fig, ax = plt.subplots(2, 1, figsize=(14, 10), dpi=200)
for i, tup in enumerate(df.groupby(df.lon < 100)):
    region, g = tup
    for rgiid, gg in g.groupby('rgiid'):
        cov = gg.loc[:, ['air', 'prate']].to_numpy()
        cov = (cov - cov.mean(axis=0))/cov.std(axis=0)
        cov = cov.T
        cov = cov[0] * cov[1]
        
        cov = uniform_filter1d(cov, 30)
        name = cfg['glaciers'][rgiid]['name']
        ax[i].axhline(lw=1, c='black')
        ax[i].plot(cov, lw=1, label=name)

for axis in ax:
    axis.axvspan(1951, 1980, alpha=0.1, color='red')
    axis.set_axisbelow(True)
    axis.grid(which='both', axis='both')
    axis.xaxis.set_major_locator(MultipleLocator(200))
    axis.xaxis.set_minor_locator(MultipleLocator(100))
    # axis.yaxis.set_major_locator(MultipleLocator(5e-7))
    # axis.yaxis.set_minor_locator(MultipleLocator(2.5e-7))
    axis.set_ylabel('mm/s')
    axis.legend()
ax[-1].set_xlabel('Years')
ax[0].set_title('Last Millennium Reconstruction (v2.1), (2mT x Prate)', loc='left')
ax[0].set_title('Normalized, smoothed (M30)', loc='right')
plt.tight_layout()
plt.savefig(Path(ROOT, 'plots/case_study/LMR_2mT.prate.cov_per_glacier.png'))
plt.show()