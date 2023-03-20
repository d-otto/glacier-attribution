# -*- coding: utf-8 -*-
"""
viz_ice_profiles.py

Description.

Author: drotto
Created: 3/17/2023 @ 5:17 PM
Project: glacier-attribution
"""

import dill
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import xarray as xr
from matplotlib import gridspec, cm
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize, CenteredNorm
from config import ROOT, cfg
from src.climate import temp_stable
from src.data import get_rgi
import pickle

# ylim0, ylim1 = ax[3, 1].get_ylim()
# diff1 = ylim1-ylim0
# ylim0, ylim1 = ax[2, 1].get_ylim()
# diff0 = ylim1-ylim0
# ratio = diff1/diff0
# ax[3, 1].set_ylim(ylim0*ratio, ylim1*ratio)


# %%
glaciers = ["Hintereisferner"]
rgiids = list({k: v for k, v in cfg["glaciers"].items() if v["name"] in glaciers}.keys())
rgi = get_rgi(rgiids, from_sqllite=True)
rgiid = rgiids[0]
params = cfg['glaciers'][rgiid]['flowline_params']

#%%

ensemble_dir = Path(r"C:\sandbox\glacier-attribution\models\ensembles\lmr")
# ensemble_files = ensemble_dir.glob(f'*{rgiids[0]}*Pnoise*.pickle')
ensemble_files = ensemble_dir.glob(f"*{rgiids[0]}*.lmr.[0-9].pickle")
# ensemble_files = ensemble_dir.glob(f'*{rgiids[0]}*.lmr.[0-9].early_ref.pickle')
runs = []
for p in ensemble_files:
    with open(p, "rb") as f:
        d = dill.load(f)
        runs.append(d)
runs = [run.to_xarray() for run in runs]
ds = xr.concat(runs, dim="nrun")

# calc specific mass balance for convenience
ds["sp_mb"] = ds.gwb / ds.area

# %%
sm_window = 30


def smooth(d, window=sm_window, axis=0):
    return scipy.ndimage.uniform_filter1d(d, window, mode="mirror", axis=axis)


dm = ds.mean(dim="nrun")
name = cfg['glaciers'][rgiid]['name']
rgi_geom = rgi.loc[rgi.index == rgiid].iloc[0]
t0 = 1850
t1 = 1999
trgi = int(rgi_geom['EndDate'][0:4])
ds0 = dm.sel(time=t0)
ds1 = dm.sel(time=t1)

# cmap = plt.cm.plasma(np.linspace(0, 0.75, len(ds.nrun)))
# cmap = plt.cm.brg(np.linspace(0, 1, len(ds.nrun)))

pad = 50
pedge = int(dm.edge_idx[-100:].mean()) + pad
x = dm.x[:pedge]
zb = dm.zb[:pedge]
z0 = zb + ds0.h[:pedge]
z1 = zb + ds1.h[:pedge]
cmap = plt.cm.tab20(np.linspace(0, 1, len(ds.nrun)))
fig, ax = plt.subplots(3,1, figsize=(8, 8), dpi=200, layout="constrained", height_ratios=[1, 0.5, 0.5])
ax2b = ax[2].twinx()

ax[0].plot(x / 1000, zb, color='black')
poly1 = ax[0].fill_between(
            x / 1000,
            zb,
            z0,
            fc="none",
            ec="lightblue",
            label=f"{t0} ensemble mean",
            hatch='....'
        )

poly1 = ax[0].fill_between(
            x / 1000,
            zb,
            z1,
            fc="lightblue",
            ec="lightblue",
            label=f"{t1} ensemble mean",
            
        )
ax[0].axvline(rgi_geom.Lmax/1000, c='red', label=f"{trgi} extent (RGI)")
ax[0].axvline(params['L0']['value']/1000, c='gold', label=f"{params['L0']['year']} extent")
ax[0].set_xlabel('Length (km)')

ax[1].plot(
    dm.time,
    scipy.ndimage.uniform_filter1d(dm.edge / 1000, sm_window, mode="mirror"),
    color='black',
    lw=1,
    alpha=0.75,
    label='Ensemble mean length',
)
ax[1].scatter(trgi, rgi_geom.Lmax/1000, c='gold')
ax[1].scatter(params['L0']['year'], params['L0']['value']/1000, c='red')
ax[1].axvline(trgi, c='gold')
ax[1].axvline(params['L0']['year'], c='red')
ax[1].set_xlabel('Year')
ax[1].set_ylabel('Length (km)')
ax[1].set_xlim(dm.time[0], None)

ax[2].plot(
    dm.time,
    scipy.ndimage.uniform_filter1d(dm.T, sm_window, mode="mirror"),
    color='black',
    lw=1,
    alpha=0.75,
    label='Temperature',
)
ax[2].fill_between(
    dm.time,
    scipy.ndimage.uniform_filter1d(dm.T, sm_window, mode="mirror"),
    scipy.ndimage.uniform_filter1d(dm.T, sm_window, mode="mirror"),
    color='red',
    alpha=0.5,
    label='Specific mass balance',
    visible=False
)
ax2b.fill_between(
    dm.time,
    0,
    scipy.ndimage.uniform_filter1d(dm.sp_mb, sm_window, mode="mirror"),
    color='red',
    lw=1,
    alpha=0.5,
    label='Specific mass balance',
)
ax2b.invert_yaxis()
ax[2].set_xlabel('Year')
ax[2].set_xlim(dm.time[0], dm.time[-1])
ax[2].set_ylabel('Temperature')
ax2b.set_ylabel('Specific mass balance (m/yr)')



for axis in ax.ravel():
    axis.grid(which='both', axis='both', ls=':')
    axis.legend()
    axis.tick_params(axis='both', which='both', direction='in')
ax[0].set_title(f"{rgiid}: {name}")
fig.show()
plt.savefig(Path(ROOT, f"plots/case_study/lmr_ensemble/{rgiids[0]}.profile.lmr.png"))