# -*- coding: utf-8 -*-
"""
compare_flowline.py

Description.

Author: drotto
Created: 12/1/2022 @ 2:06 PM
Project: glacier-attribution
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
from config import ROOT, cfg
from src.data import get_rgi

# %%

model = "MRI-ESM2-0"
glacier = "Wolverine"
rgiid = list({k: v for k, v in cfg["glaciers"].items() if v["name"] == glacier}.keys())[
    0
]
rgi = get_rgi(rgiid, from_sqllite=True)

# %%

fp = Path(ROOT, f"models/flowline2d_{rgiid}.{model}.past1000.gwi.nat.pickle")
with open(fp, "rb") as f:
    nat = pickle.load(f)
nat.h1 = nat.h[-1]
# h_init = last_run.h[-1, :]
# x_init = last_run.x
compare_fp = Path(ROOT, f"models/flowline2d_{rgiid}.{model}.past1000.gwi.comb.pickle")
with open(compare_fp, "rb") as f:
    anth = pickle.load(f)
# anth = pd.DataFrame().from_dict(anth)
anth.h1 = anth.h[-1]

# %%

# xlims = [1850, 3015]
# xlims = [1400, 2015]
# xlims = [1850, 2015]
xlims = [850, 2015]

# def plot_comparison
anth_x0 = 1850 - anth.ts
pad = 10

nat.area_sm = scipy.ndimage.uniform_filter1d(nat.area, 20, mode="mirror")
nat.T_sm = scipy.ndimage.uniform_filter1d(nat.T, 20, mode="mirror")
nat.edge_sm = scipy.ndimage.uniform_filter1d(nat.edge, 20, mode="mirror") / 1000
nat.bal_sm = scipy.ndimage.uniform_filter1d(nat.gwb / nat.area, 20, mode="mirror")
nat.cumbal_sm = scipy.ndimage.uniform_filter1d(
    np.cumsum(nat.gwb / nat.area), 20, mode="mirror"
)
nat.ela_sm = scipy.ndimage.uniform_filter1d(nat.ela, 20, mode="mirror")
anth.area_sm = scipy.ndimage.uniform_filter1d(anth.area, 20, mode="mirror")
anth.T_sm = scipy.ndimage.uniform_filter1d(anth.T, 20, mode="mirror")
anth.edge_sm = scipy.ndimage.uniform_filter1d(anth.edge, 20, mode="mirror") / 1000
anth.bal_sm = scipy.ndimage.uniform_filter1d(anth.gwb / anth.area, 20, mode="mirror")
anth.cumbal_sm = scipy.ndimage.uniform_filter1d(
    np.cumsum(anth.gwb / anth.area), 20, mode="mirror"
)
anth.ela_sm = scipy.ndimage.uniform_filter1d(anth.ela, 20, mode="mirror")

fig = plt.figure(figsize=(12, 10), dpi=200, layout="tight")
gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=(1, 1, 1, 1), hspace=0.3, wspace=0.2)
ax = np.empty((4, 2), dtype="object")

# Ax 0, 0
ax[0, 0] = fig.add_subplot(gs[0, 0])
ax[0, 0].set_xlabel("Time (years)")
ax[0, 0].set_ylabel("Glacier Area ($km^2$)")
ax[0, 0].plot(nat.t, nat.area_sm / 1e6, c="blue", lw=1)
ax[0, 0].plot(anth.t[anth_x0:], anth.area_sm[anth_x0:] / 1e6, c="red", lw=1)
ax[0, 0].set_xlim(xlims)

# Ax 0, 1
ax[0, 1] = fig.add_subplot(gs[0:2, 1])
ax[0, 1].set_xlabel("Distance (km)")
ax[0, 1].set_ylabel("Elevation (m)")
pedge = int(nat.edge[-1] / nat.delx + pad)
x1 = nat.x[:pedge]
z0 = nat.zb[:pedge]
z1 = z0 + nat.h1[:pedge]
poly1 = ax[0, 1].fill_between(x1 / 1000, z0, z1, fc="lightblue", ec="lightblue")
ax[0, 1].plot(x1 / 1000, z0, c="black", lw=2)

pedge = int(anth.edge[-1] / nat.delx + pad)
x2 = nat.x[:pedge]
z1 = nat.zb[:pedge]
z2 = z1 + anth.h1[:pedge]
poly2 = ax[0, 1].fill_between(x2 / 1000, z1, z2, fc="none", ec="lightblue")
if len(x1) > len(x2):
    poly1.set_hatch("....")
    poly1.set_facecolor("none")
    poly2.set_facecolor("lightblue")
else:
    poly1.set_facecolor("lightblue")
    poly2.set_hatch("....")

# Ax 1, 0
ax[1, 0] = fig.add_subplot(gs[1, 0])
ax[1, 0].set_xlabel("Time (years)")
ax[1, 0].set_ylabel("Equilibrium Line Altitude (m)")
ax[1, 0].plot(nat.t, nat.ela_sm, c="blue", lw=1)
ax[1, 0].plot(anth.t[anth_x0:], anth.ela_sm[anth_x0:], c="red", lw=1)
ax[1, 0].set_xlim(xlims)

# Ax 2, 0
ax[2, 0] = fig.add_subplot(gs[2, 0])
ax[2, 0].set_ylabel("T ($^o$C)")
ax[2, 0].plot(nat.t, nat.T, c="blue", lw=0.25, alpha=0.5)
ax[2, 0].plot(nat.t, nat.T_sm, c="blue", lw=1)
ax[2, 0].plot(anth.t[anth_x0:], anth.T[anth_x0:], c="red", lw=0.25, alpha=0.5)
ax[2, 0].plot(anth.t[anth_x0:], anth.T_sm[anth_x0:], c="red", lw=1)
ax[2, 0].set_xlim(xlims)

# Ax 2, 1
ax[2, 1] = fig.add_subplot(gs[2, 1])
ax[2, 1].set_ylabel("L (km)")
ax[2, 1].set_xlim(0, x1.max() / 1000 * 1.1)
ax[2, 1].plot(nat.t, nat.edge_sm, c="blue", lw=1)
ax[2, 1].plot(anth.t[anth_x0:], anth.edge_sm[anth_x0:], c="red", lw=1)
ax[2, 1].set_xlim(xlims)

# Ax 3, 0
ax[3, 0] = fig.add_subplot(gs[3, 0])
ax[3, 0].set_ylabel("Bal (m $yr^{-1}$)")
ax[3, 0].set_xlabel("Time (years)")
ax[3, 0].plot(nat.t, nat.gwb / nat.area, c="blue", lw=0.25, alpha=0.5)
ax[3, 0].plot(nat.t, nat.bal_sm, c="blue", lw=1, )
ax[3, 0].plot(anth.t[anth_x0:], (anth.gwb / anth.area)[anth_x0:], c="red", lw=0.25, alpha=0.5)
ax[3, 0].plot(anth.t[anth_x0:], anth.bal_sm[anth_x0:], c="red", lw=1)
ax[3, 0].set_xlim(xlims)

ax[3, 1] = fig.add_subplot(gs[3, 1])
ax[3, 1].set_xlabel("Time (years)")
ax[3, 1].set_ylabel("Cum. bal. (m)")
ax[3, 1].plot(nat.t, nat.cumbal_sm, c="blue", lw=1)
ax[3, 1].plot(anth.t[anth_x0:], anth.cumbal_sm[anth_x0:], c="red", lw=1)
ax[3, 1].set_xlim(xlims)

for row in range(0, ax.shape[0]):
    for col in range(0, ax.shape[1]):
        axis = ax[row, col]
        if axis is not None:  # this handles gridspec col/rowspans > 1
            axis.grid(axis="both", alpha=0.5, lw=1)
            axis.set_axisbelow(True)
            axis.tick_params(direction='in', which='both')
            axis.tick_params(which='minor', length=3)
            axis.tick_params(which='major', length=6)
            if (row != 0) | (col != 1):
                axis.xaxis.set_major_locator(MultipleLocator(200))
                axis.xaxis.set_minor_locator(MultipleLocator(50))
                axis.grid(axis="both", which='minor', alpha=0.25, lw=0.5)
                axis.axvline(1850, c='black', ls='dashed', lw=1)

fig.show()
