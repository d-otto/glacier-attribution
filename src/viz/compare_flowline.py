# -*- coding: utf-8 -*-
"""
compare_flowline.py

Description.

Author: drotto
Created: 12/1/2022 @ 2:06 PM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import pickle
import scipy
from src.climate import temp_stable

#%%

model = 'MRI-ESM2-0'
xlim0 = None
nat_fp = None
nat_fp = None
nat = None
fp = Path(f'models/flowline2d_RGI60-02.18778.{model}.past1000.gwi.nat.pickle')
with open(fp, 'rb') as f:
    nat = pickle.load(f)
# h_init = last_run.h[-1, :]
# x_init = last_run.x
    with open(compare_fp, "rb") as f:
        anth = pickle.load(f)
    # anth = pd.DataFrame().from_dict(anth)
    anth_h = anth.h[-1]

compare_fp = f'models/flowline2d_RGI60-02.18778.{model}.past1000.gwi.comb.pickle'


if xlim0 is None:
    xlim0 = nat.ts

pad = 10
pedge = int(nat.edge[-1] / nat.delx + pad)
nat.pedge = pedge
x1 = nat.x[:pedge]
z0 = nat.zb[:pedge]
z1 = z0 + nat.h[-1, :pedge]

fig = plt.figure(figsize=(12, 10), dpi=200)
gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=(1, 1, 1, 1))
ax = np.empty((4, 2), dtype="object")
plt.show(block=False)

ax[0, 0] = fig.add_subplot(gs[0, 0])
ax[0, 0].set_xlabel("Time (years)")
ax[0, 0].set_ylabel("Glacier Area ($km^2$)")

ax[1, 0] = fig.add_subplot(gs[1, 0])
ax[1, 0].set_xlabel("Time (years)")
ax[1, 0].set_ylabel("Equilibrium Line Altitude (m)")

ax[0, 1] = fig.add_subplot(gs[0:2, 1])
ax[0, 1].set_xlabel("Distance (km)")
ax[0, 1].set_ylabel("Elevation (m)")

ax[2, 0] = fig.add_subplot(gs[2, 0])
ax[2, 0].set_ylabel("T ($^o$C)")

ax[2, 1] = fig.add_subplot(gs[2, 1])
ax[2, 1].set_ylabel("L (km)")

ax[3, 0] = fig.add_subplot(gs[3, 0])
ax[3, 0].set_ylabel("Bal (m $yr^{-1}$)")
ax[3, 0].set_xlabel("Time (years)")

ax[3, 1] = fig.add_subplot(gs[3, 1])
ax[3, 1].set_xlabel("Time (years)")
ax[3, 1].set_ylabel("Cum. bal. (m)")

for axis in ax.ravel():
    if axis is not None:  # this handles gridspec col/rowspans > 1
        axis.grid(axis="both", alpha=0.5)
        axis.set_axisbelow(True)
plt.tight_layout()
poly1 = ax[0, 1].fill_between(x1 / 1000, z0, z1, fc="lightblue", ec="lightblue")
ax[0, 1].plot(
    x1 / 1000,
    z0,
    c="black",
    lw=2,
)
ax[0, 0].plot(
    nat.t,
    scipy.ndimage.uniform_filter1d(nat.area / 1e6, 20, mode="mirror"),
    c="black",
)
ax[0, 0].set_xlim(xlim0, nat.tf)
ax[1, 0].plot(
    nat.t,
    scipy.ndimage.uniform_filter1d(nat.ela, 20, mode="mirror"),
    c="blue",
)
ax[1, 0].set_xlim(xlim0, nat.tf)
ax[2, 1].set_xlim(0, x1.max() / 1000 * 1.1)
ax[2, 0].plot(nat.t, nat.T, c="blue", lw=0.25)
ax[2, 0].plot(
    nat.t,
    scipy.ndimage.uniform_filter1d(nat.T, 20, mode="mirror"),
    c="blue",
    lw=2,
)
ax[2, 0].set_xlim(xlim0, nat.tf)
ax[2, 1].plot(
    nat.t,
    scipy.ndimage.uniform_filter1d(nat.edge, 20, mode="mirror") / 1000,
    c="black",
    lw=2,
)
ax[2, 1].set_xlim(xlim0, nat.tf)
ax[3, 0].plot(nat.t, nat.bal / nat.area, c="blue", lw=0.25)
ax[3, 0].plot(
    nat.t,
    scipy.ndimage.uniform_filter1d(nat.bal / nat.area, 20, mode="mirror"),
    c="blue",
    lw=2,
)
ax[3, 0].set_xlim(xlim0, nat.tf)
ax[3, 1].plot(
    nat.t,
    scipy.ndimage.uniform_filter1d(
        np.cumsum(nat.bal / nat.area), 20, mode="mirror"
    ),
    c="blue",
    lw=2,
)
ax[3, 1].plot(
    nat.t,
    scipy.ndimage.uniform_filter1d(
        np.cumsum(nat.bal / nat.area), 20, mode="mirror"
    ),
    c="blue",
    lw=2,
)
ax[3, 1].set_xlim(xlim0, nat.tf)

# plot extras


pad = 10
pedge = int(anth.edge[-1] / nat.delx + pad)
nat.pedge = pedge
x2 = nat.x[:pedge]
z1 = nat.zb[:pedge]
z2 = z1 + anth_h[:pedge]

poly2 = ax[0, 1].fill_between(x2 / 1000, z1, z2, fc="none", ec="lightblue")
if len(x1) > len(x2):
    poly1.set_hatch("....")
    poly1.set_facecolor("none")
    poly2.set_facecolor("lightblue")
else:
    poly1.set_facecolor("lightblue")
    poly2.set_hatch("....")

anth.T_sm = scipy.ndimage.uniform_filter1d(anth.T, 20, mode="mirror")
anth.edge_sm = (
        scipy.ndimage.uniform_filter1d(anth.edge, 20, mode="mirror") / 1000
)
anth.bal_sm = scipy.ndimage.uniform_filter1d(
    anth.bal / anth.area, 20, mode="mirror"
)
anth.cumbal_sm = scipy.ndimage.uniform_filter1d(
    np.cumsum(anth.bal / anth.area), 20, mode="mirror"
)
anth.ela_sm = scipy.ndimage.uniform_filter1d(
    anth.ela, 20, mode="mirror"
)
# anth = anth.iloc[1850:]
ax[0, 0].plot(anth.t, anth.area / 1e6, c="black", lw=2, ls="dashed")
ax[1, 0].plot(
    anth.t,
    anth.ela_sm,
    c="red",
    lw=2,
)
ax[2, 0].plot(anth.t, anth.T_sm, c="red", lw=2)
ax[2, 0].plot(anth.t, anth.T, c="red", lw=0.25)
ax[2, 1].plot(anth.t, anth.edge_sm, c="black", lw=2, ls="dashed")
ax[3, 0].plot(anth.t, anth.bal_sm, c="red", lw=2)
ax[3, 0].plot(anth.t, anth.bal / anth.area, c="red", lw=0.25)
ax[3, 1].plot(anth.t, anth.cumbal_sm, c="red", lw=2)
fig.show()
