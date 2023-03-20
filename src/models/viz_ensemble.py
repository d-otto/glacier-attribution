# -*- coding: utf-8 -*-
"""
compare_flowline.py

Description.

Author: drotto
Created: 12/1/2022 @ 2:06 PM
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

ds_pi = ds.sel(time=slice(None, 1850))
ds_i = ds.sel(time=slice(1850, None))
# cmap = plt.cm.plasma(np.linspace(0, 0.75, len(ds.nrun)))
# cmap = plt.cm.brg(np.linspace(0, 1, len(ds.nrun)))
cmap = plt.cm.tab20(np.linspace(0, 1, len(ds.nrun)))
fig = plt.figure(figsize=(12, 8), dpi=200, layout="constrained")
gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=(1, 1, 1))
ax = np.empty((3, 2), dtype="object")
ax[0, 0] = fig.add_subplot(gs[0, 0:3])
ax[1, 0] = fig.add_subplot(gs[1, 0:3])
ax[2, 0] = fig.add_subplot(gs[2, 0:3])
ax[0, 1] = fig.add_subplot(gs[0, 3])
ax[1, 1] = fig.add_subplot(gs[1, 3])
ax[2, 1] = fig.add_subplot(gs[2, 3])
for nrun, d in ds.groupby("nrun"):
    if nrun == 0:
        label = f"MA-{sm_window}"
    else:
        label = None
    ax[0, 0].plot(
        d.time,
        scipy.ndimage.uniform_filter1d(d.edge / 1000, sm_window, mode="mirror"),
        color=cmap[nrun],
        lw=1,
        alpha=0.75,
        label=label,
    )
    ax[1, 0].plot(
        d.time,
        scipy.ndimage.uniform_filter1d(d.gwb / d.area, sm_window, mode="mirror"),
        color=cmap[nrun],
        lw=1,
        alpha=0.75,
        label=label,
    )
    ax[2, 0].plot(d.time, smooth(d.T), color=cmap[nrun], lw=1, alpha=0.75, label=label)


ax[0, 0].plot(
    dm.time,
    scipy.ndimage.uniform_filter1d(dm.edge / 1000, sm_window, mode="mirror"),
    color="black",
    lw=2,
    label="Ensemble mean",
)
ax[0, 0].fill_between(
    dm.time,
    smooth(dm.edge / 1000 + 2 * (ds.edge / 1000).std(dim="nrun")),
    smooth(dm.edge / 1000 - 2 * (ds.edge / 1000).std(dim="nrun")),
    color="#c0c0c0",
    alpha=0.5,
    label="+/- 2$\sigma$",
)
ax[0, 0].set_ylabel("Length (km)")
dl_pi = (
    ds_pi.edge.sel(time=slice(1200, None, 150)).to_numpy().ravel() / 1000
    - ds_pi.edge.sel(time=slice(1050, 1700, 150)).to_numpy().ravel() / 1000
)
dl_i = ds_i.edge.isel(time=-1).to_numpy().ravel() / 1000 - ds_pi.edge.isel(time=-1).to_numpy().ravel() / 1000
bins = np.histogram(np.hstack((dl_pi, dl_i)), bins=20)[1]  # get the bin edges
ax[0, 1].hist(dl_pi, bins=bins, density=True, color="green", label="Pre-1850", alpha=0.5)
ax[0, 1].hist(dl_i, bins=bins, density=True, color="red", label="Post-1850", alpha=0.5)
ax[0, 1].set_xlabel("Length Change (km)")


ax[1, 0].plot(
    dm.time,
    scipy.ndimage.uniform_filter1d(dm.gwb / dm.area, sm_window, mode="mirror"),
    color="black",
    lw=2,
    label="Ensemble mean",
)
ax[1, 0].fill_between(
    dm.time,
    smooth(dm.gwb / dm.area + 2 * (ds.gwb / ds.area).std(dim="nrun")),
    smooth(dm.gwb / dm.area - 2 * (ds.gwb / ds.area).std(dim="nrun")),
    color="#c0c0c0",
    alpha=0.5,
    label="+/- 2$\sigma$",
)
ax[1, 0].set_ylabel("Specific mass balance (m/yr)")
sp_mb_pi = ds_pi.sp_mb.coarsen(time=150, side="right", boundary="trim").mean().to_numpy().ravel()
sp_mb_i = ds_i.sp_mb.coarsen(time=150, side="right", boundary="trim").mean().to_numpy().ravel()
bins = np.histogram(np.hstack((sp_mb_pi, sp_mb_i)), bins=20)[1]  # get the bin edges
ax[1, 1].hist(
    sp_mb_pi,
    bins=bins,
    density=True,
    color="green",
    label="Pre-1850",
    alpha=0.5,
)
ax[1, 1].hist(
    sp_mb_i,
    bins=bins,
    density=True,
    color="red",
    label="Post-1850",
    alpha=0.5,
)
ax[1, 1].set_xlabel("Specific mass balance (m/yr)")

ax[2, 0].plot(dm.time, smooth(dm.T), c="black", lw=2, label="Ensemble mean")
ax[2, 0].fill_between(
    dm.time,
    smooth(dm.T + 2 * (ds.T).std(dim="nrun")),
    smooth(dm.T - 2 * (ds.T).std(dim="nrun")),
    color="#c0c0c0",
    alpha=0.5,
    label="+/- 2$\sigma$",
)

ax[2, 0].set_ylabel("Temperature ($\degree C$)")
T_pi = ds_pi.T.coarsen(time=150, side="right", boundary="trim").mean().to_numpy().ravel()
T_i = ds_i.T.coarsen(time=150, side="right", boundary="trim").mean().to_numpy().ravel()
bins = np.histogram(np.hstack((T_pi, T_i)), bins=20)[1]  # get the bin edges
ax[2, 1].hist(
    T_pi,
    bins=bins,
    density=True,
    color="green",
    label="Pre-1850",
    alpha=0.5,
)
ax[2, 1].hist(
    T_i,
    bins=bins,
    density=True,
    color="red",
    label="Post-1850",
    alpha=0.5,
)
ax[2, 1].set_xlabel("Temperature ($\degree C$)")

for j in np.arange(0, ax.shape[-1], 1):
    for i in np.arange(0, ax.shape[0], 1):
        if j == 0:
            ax[i, j].set_xlim(dm.time[j], dm.time[-1])
            ax[i, j].set_xlabel("Year")
            ax[i, j].xaxis.set_major_locator(MultipleLocator(200))
            ax[i, j].xaxis.set_minor_locator(MultipleLocator(50))
            ax[i, j].axvspan(ds.ref_period[0], ds.ref_period[1], color="red", alpha=0.1, ec=None)
        if i == 0:
            ax[i, j].legend(loc="upper left")
        ax[i, j].grid(which="major", axis="both", color="#c0c0c0")
        ax[i, j].grid(which="minor", axis="both", color="#f0f0f0")
        ax[i, j].set_axisbelow(True)
fig.suptitle(f"{rgiids[0]} ({glaciers[0]}) T=LMR, P=noise; Tref={ds.ref_period[0]} - {ds.ref_period[1]}")
fig.show()
plt.savefig(Path(ROOT, f"plots/case_study/lmr_ensemble/{rgiids[0]}.lmr.png"))


#%%
#
# fig = plt.figure(figsize=(10, 8), dpi=200, layout="constrained")
# gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=(1, 1, 1, 1))
# ax = np.empty((4, 2), dtype="object")
# for nrun, d in ds.groupby('nrun'):
#         pad = 10
#         xlims = [850, 2200]
#         xlims_idx0 = int(xlims[0] - nat.t[0])
#         xlims_idx1 = int(xlims[-1] - nat.t[0])
#         xlims_slice = slice(xlims_idx0, xlims_idx1)
#         attr_xlims = [1800, 3015]
#         t1_window = 200
#         sm_window = 30
#
#         area_sm = scipy.ndimage.uniform_filter1d(d.area, sm_window, mode="mirror")
#         T_sm = scipy.ndimage.uniform_filter1d(d.T, sm_window, mode="mirror")
#         edge_sm = scipy.ndimage.uniform_filter1d(d.edge/1000, sm_window, mode="mirror")
#         bal_sm = scipy.ndimage.uniform_filter1d(d.gwb / d.area, sm_window, mode="mirror")
#         cumbal_sm = scipy.ndimage.uniform_filter1d(
#             np.cumsum(d.gwb / d.area), sm_window, mode="mirror"
#         )
#         ela_sm = scipy.ndimage.uniform_filter1d(d.ela, sm_window, mode="mirror")
#
#         # plotting commands
#
#
#         # Ax 0, 0
#         ax[0, 0] = fig.add_subplot(gs[0:2, 0])
#         ax[0, 0].set_xlabel("Distance (km)")
#         ax[0, 0].set_ylabel("Elevation (m)")
#         pedge = int(d.edge[-t1_window:].mean() / d.delx + pad)
#         x1 = d.x[:pedge]
#         z0 = d.zb[:pedge]
#         z1 = z0 + d.h1[:pedge]
#         poly1 = ax[0, 0].fill_between(x1 / 1000, z0, z1, fc="lightblue", ec="lightblue")
#         ax[0, 0].plot(x1 / 1000, z0, c="black", lw=2)
#
#         poly2 = ax[0, 0].fill_between(x2 / 1000, z1, z2, fc="none", ec="lightblue")
#         if len(x1) > len(x2):
#             poly1.set_hatch("....")
#             poly1.set_facecolor("none")
#             poly2.set_facecolor("lightblue")
#             poly1.set_label('NAT profile (2015)')
#             poly2.set_label('Equilibrated ANTH profile')
#         else:
#             poly1.set_facecolor("lightblue")
#             poly2.set_hatch("....")
#             poly1.set_label('Equilibrated ANTH profile')
#             poly2.set_label('NAT profile (2015)')
#         ax[0, 0].legend()
#         ax[0, 0].xaxis.set_major_locator(MultipleLocator(1))
#         ax[0, 0].xaxis.set_minor_locator(MultipleLocator(0.25))
#         ax[0, 0].yaxis.set_minor_locator(MultipleLocator(50))
#
#         # Ax 0, 1
#         ax[0, 1] = fig.add_subplot(gs[0, 1])
#         ax[0, 1].set_ylabel("Attribution %")
#         ax[0, 1].plot(nat.t, attr, c="black", lw=2)
#         poly3 = ax[0, 1].fill_between(nat.t, attr, ec='none', fc='grey', alpha=0.25)
#         ax[0, 1].set_xlim(attr_xlims)
#         ax[0, 1].set_ylim([0, 300])
#         ax[0, 1].yaxis.set_minor_locator(MultipleLocator(25))
#
#         # Ax 1, 1
#         ax[1, 1] = fig.add_subplot(gs[1, 1])
#         ax[1, 1].set_ylabel("T ($^o$C)")
#         ax[1, 1].plot(nat.t, nat.T, c="blue", lw=0.25, alpha=0.5)
#         ax[1, 1].plot(nat.t, nat.T_sm, c="blue", lw=1)
#         ax[1, 1].plot(anth.t[anth_x0:], anth.T[anth_x0:], c="red", lw=0.25, alpha=0.5)
#         ax[1, 1].plot(anth.t[anth_x0:], anth.T_sm[anth_x0:], c="red", lw=1)
#         ax[1, 1].set_xlim(xlims)
#         ax[1, 1].yaxis.set_minor_locator(MultipleLocator(1))
#
#         # Ax 2, 0
#         ax[2, 0] = fig.add_subplot(gs[2, 0])
#         ax[2, 0].set_ylabel("Length (m)")
#         ax[2, 0].set_xlabel("Width (m)")
#         sm = cm.ScalarMappable(cmap='coolwarm_r', norm=CenteredNorm(vcenter=0, halfrange=1))
#         ax[2, 0].barh(y=anth.x, width=anth.w, height=50, lw=0, left=-0.5 * nat.w + 1.5 * anth.w.max(),
#                       color=sm.to_rgba(anth.b[-t1_window:].mean(axis=0)))
#         ax[2, 0].barh(y=nat.x, width=nat.w, height=50, lw=0, left=-0.5 * anth.w + 0.5 * anth.w.max(),
#                       color=sm.to_rgba(nat.b[-t1_window:].mean(axis=0)))
#         ax[2, 0].set_xlim(0, 2 * anth.w.max())
#         ax[2, 0].set_ylim(nat.edge.max(), 0)
#         plt.colorbar(sm, ax=ax[2, 0], location='left', orientation='vertical', fraction=0.05, pad=0.2,
#                      label='Annual mass balance (m)')
#
#         # Ax 2, 1
#         ax[2, 1] = fig.add_subplot(gs[2, 1])
#         ax[2, 1].set_ylabel("L (km)")
#         ax[2, 1].plot(nat.t[xlims_slice], nat.edge_sm[xlims_slice], c="blue", lw=1)
#         ax[2, 1].plot(anth.t[anth_x0:xlims_idx1], anth.edge_sm[anth_x0:xlims_idx1], c="red", lw=1)
#         ax[2, 1].set_xlim(xlims)
#         ax[2, 1].yaxis.set_major_locator(MultipleLocator(1))
#         ax[2, 1].yaxis.set_minor_locator(MultipleLocator(0.5))
#
#         # Ax 3, 0
#         ax[3, 0] = fig.add_subplot(gs[3, 0])
#         ax[3, 0].set_ylabel("Bal (m $yr^{-1}$)")
#         ax[3, 0].set_xlabel("Time (years)")
#         ax[3, 0].plot(nat.t, nat.gwb / nat.area, c="blue", lw=0.25, alpha=0.5)
#         ax[3, 0].plot(nat.t, nat.bal_sm, c="blue", lw=1, )
#         ax[3, 0].plot(anth.t[anth_x0:], (anth.gwb / anth.area)[anth_x0:], c="red", lw=0.25, alpha=0.5)
#         ax[3, 0].plot(anth.t[anth_x0:], anth.bal_sm[anth_x0:], c="red", lw=1)
#         ax[3, 0].set_xlim(xlims)
#         ax[3, 0].yaxis.set_minor_locator(MultipleLocator(0.5))
#
#         # Ax 3, 1
#         ax[3, 1] = fig.add_subplot(gs[3, 1])
#         ax[3, 1].set_xlabel("Time (years)")
#         ax[3, 1].set_ylabel("Glacier Area ($km^2$)")
#         ax[3, 1].plot(nat.t[xlims_slice], nat.area_sm[xlims_slice] / 1e6, c="blue", lw=1)
#         ax[3, 1].plot(anth.t[anth_x0:xlims_idx1], anth.area_sm[anth_x0:xlims_idx1] / 1e6, c="red", lw=1)
#         ax[3, 1].set_xlim(xlims)
#         ax[3, 1].yaxis.set_major_locator(MultipleLocator(1))
#         ax[3, 1].yaxis.set_minor_locator(MultipleLocator(0.5))
#
#         for row in range(0, ax.shape[0]):
#             for col in range(0, ax.shape[1]):
#                 axis = ax[row, col]
#                 if axis is not None:  # this handles gridspec col/rowspans > 1
#                     axis.grid(axis="both", alpha=0.5, lw=1)
#                     axis.set_axisbelow(True)
#                     axis.tick_params(direction='in', which='both')
#                     axis.tick_params(which='minor', length=3)
#                     axis.tick_params(which='major', length=6)
#                     axis.grid(axis="both", which='minor', alpha=0.25, lw=0.5)
#                     if (row, col) not in [(0, 0), (2, 0)]:
#                         axis.xaxis.set_major_locator(MultipleLocator(200))
#                         axis.xaxis.set_minor_locator(MultipleLocator(50))
#                         axis.axvline(1850, c='black', ls='dashed', lw=1)
#
#         plt.subplots_adjust(hspace=0.2, wspace=0.2)
#         plt.tight_layout()
#         fig.show()
#         fig_output_name = Path(ROOT, f'flowline2d_{rgiid}.{model}.past1000.gwi.comparison.png')
#         fig.savefig(fig_output_name)
