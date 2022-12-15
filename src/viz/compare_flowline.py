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
from matplotlib import gridspec, cm
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize, CenteredNorm
from config import ROOT, cfg
from src.climate import temp_stable
from src.data import get_rgi

# ylim0, ylim1 = ax[3, 1].get_ylim()
# diff1 = ylim1-ylim0
# ylim0, ylim1 = ax[2, 1].get_ylim()
# diff0 = ylim1-ylim0
# ratio = diff1/diff0
# ax[3, 1].set_ylim(ylim0*ratio, ylim1*ratio)


# %%

models = ["MRI-ESM2-0", 'MIROC-ES2L']
glaciers = ["Hintereisferner", 'Wolverine', 'South Cascade']
rgiids = list({k: v for k, v in cfg["glaciers"].items() if v["name"] in glaciers}.keys())
rgi = get_rgi(rgiids, from_sqllite=True)

# %%
for rgiid in rgiids:
    for model in models:
        fp = Path(ROOT, f"models/flowline2d_{rgiid}.{model}.past1000.gwi.nat.pickle")
        with open(fp, "rb") as f:
            nat = dill.load(f)
        nat.h1 = nat.h[-1]
        # h_init = last_run.h[-1, :]
        # x_init = last_run.x
        compare_fp = Path(ROOT, f"models/flowline2d_{rgiid}.{model}.past1000.gwi.comb.pickle")
        with open(compare_fp, "rb") as f:
            anth = dill.load(f)
        # anth = pd.DataFrame().from_dict(anth)
        anth.h1 = anth.h[-1]
        
        df = pd.concat(dict(anth=anth.to_pandas(), nat=nat.to_pandas()), names=['scenario', 'year'])
        ds = df.to_xarray()
        
        # xlims = [1850, 3015]
        # xlims = [1400, 2015]
        # xlims = [1850, 2015]
        # xlims = [850, 2015]
        # xlims = [1850, 2100]
        xlims = [850, 2200]
        xlims_idx0 = int(xlims[0] - nat.t[0])
        xlims_idx1 = int(xlims[-1] - nat.t[0])
        xlims_slice = slice(xlims_idx0, xlims_idx1)
        attr_xlims = [1800, 3015]
        t1_window = 200
        sm_window = 30
        
        
        # def plot_comparison
        # calculating/smoothing parameters
        anth_x0 = 1850 - anth.ts
        pad = 10
        
        nat.area_sm = scipy.ndimage.uniform_filter1d(nat.area, sm_window, mode="mirror")
        nat.T_sm = scipy.ndimage.uniform_filter1d(nat.T, sm_window, mode="mirror")
        nat.edge_sm = scipy.ndimage.uniform_filter1d(nat.edge, sm_window, mode="mirror") / 1000
        nat.bal_sm = scipy.ndimage.uniform_filter1d(nat.gwb / nat.area, sm_window, mode="mirror")
        nat.cumbal_sm = scipy.ndimage.uniform_filter1d(
            np.cumsum(nat.gwb / nat.area), sm_window, mode="mirror"
        )
        nat.ela_sm = scipy.ndimage.uniform_filter1d(nat.ela, sm_window, mode="mirror")
        anth.area_sm = scipy.ndimage.uniform_filter1d(anth.area, sm_window, mode="mirror")
        anth.T_sm = scipy.ndimage.uniform_filter1d(anth.T, sm_window, mode="mirror")
        anth.edge_sm = scipy.ndimage.uniform_filter1d(anth.edge, sm_window, mode="mirror") / 1000
        anth.bal_sm = scipy.ndimage.uniform_filter1d(anth.gwb / anth.area, sm_window, mode="mirror")
        anth.cumbal_sm = scipy.ndimage.uniform_filter1d(
            np.cumsum(anth.gwb / anth.area), sm_window, mode="mirror"
        )
        anth.ela_sm = scipy.ndimage.uniform_filter1d(anth.ela, sm_window, mode="mirror")
        
        ds['cumbal'] = ds['bal'].cumsum(dim='year')
        ds['cumbal_sm'] = ds['cumbal'].rolling(year=20).mean()
        attr = (ds.sel(scenario='anth')['cumbal_sm']/ds.sel(scenario='nat')['cumbal_sm'] - 1) * 100
        
        # plotting commands
        fig = plt.figure(figsize=(10, 8), dpi=200, layout="tight")
        gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=(1, 1, 1, 1))
        ax = np.empty((4, 2), dtype="object")
        
        # Ax 0, 0
        ax[0, 0] = fig.add_subplot(gs[0:2, 0])
        ax[0, 0].set_xlabel("Distance (km)")
        ax[0, 0].set_ylabel("Elevation (m)")
        pedge = int(nat.edge[-t1_window:].mean() / nat.delx + pad)
        x1 = nat.x[:pedge]
        z0 = nat.zb[:pedge]
        z1 = z0 + nat.h1[:pedge]
        poly1 = ax[0, 0].fill_between(x1 / 1000, z0, z1, fc="lightblue", ec="lightblue")
        ax[0, 0].plot(x1 / 1000, z0, c="black", lw=2)
        
        pedge = int(anth.edge[-t1_window:].mean() / nat.delx + pad)
        x2 = nat.x[:pedge]
        z1 = nat.zb[:pedge]
        z2 = z1 + anth.h1[:pedge]
        poly2 = ax[0, 0].fill_between(x2 / 1000, z1, z2, fc="none", ec="lightblue")
        if len(x1) > len(x2):
            poly1.set_hatch("....")
            poly1.set_facecolor("none")
            poly2.set_facecolor("lightblue")
            poly1.set_label('NAT profile (2015)')
            poly2.set_label('Equilibrated ANTH profile')
        else:
            poly1.set_facecolor("lightblue")
            poly2.set_hatch("....")
            poly1.set_label('Equilibrated ANTH profile')
            poly2.set_label('NAT profile (2015)')
        ax[0, 0].legend()
        ax[0, 0].xaxis.set_major_locator(MultipleLocator(1))
        ax[0, 0].xaxis.set_minor_locator(MultipleLocator(0.25))
        ax[0, 0].yaxis.set_minor_locator(MultipleLocator(50))
        
        # Ax 0, 1
        ax[0, 1] = fig.add_subplot(gs[0, 1])
        ax[0, 1].set_ylabel("Attribution %")
        ax[0, 1].plot(nat.t, attr, c="black", lw=2)
        poly3 = ax[0, 1].fill_between(nat.t, attr, ec='none', fc='grey', alpha=0.25)
        ax[0, 1].set_xlim(attr_xlims)
        ax[0, 1].set_ylim([0, 300])
        ax[0, 1].yaxis.set_minor_locator(MultipleLocator(25))
        
        # Ax 1, 1
        ax[1, 1] = fig.add_subplot(gs[1, 1])
        ax[1, 1].set_ylabel("T ($^o$C)")
        ax[1, 1].plot(nat.t, nat.T, c="blue", lw=0.25, alpha=0.5)
        ax[1, 1].plot(nat.t, nat.T_sm, c="blue", lw=1)
        ax[1, 1].plot(anth.t[anth_x0:], anth.T[anth_x0:], c="red", lw=0.25, alpha=0.5)
        ax[1, 1].plot(anth.t[anth_x0:], anth.T_sm[anth_x0:], c="red", lw=1)
        ax[1, 1].set_xlim(xlims)
        ax[1, 1].yaxis.set_minor_locator(MultipleLocator(1))
        
        
        # Ax 2, 0
        ax[2, 0] = fig.add_subplot(gs[2, 0])
        ax[2, 0].set_ylabel("Length (m)")
        ax[2, 0].set_xlabel("Width (m)")
        sm = cm.ScalarMappable(cmap='coolwarm_r', norm=CenteredNorm(vcenter=0, halfrange=1))
        ax[2, 0].barh(y=anth.x, width=anth.w, height=50, lw=0, left=-0.5*nat.w + 1.5*anth.w.max(), color=sm.to_rgba(anth.b[-t1_window:].mean(axis=0)))
        ax[2, 0].barh(y=nat.x, width=nat.w, height=50, lw=0, left=-0.5*anth.w + 0.5*anth.w.max(), color=sm.to_rgba(nat.b[-t1_window:].mean(axis=0)))
        ax[2, 0].set_xlim(0, 2*anth.w.max())
        ax[2, 0].set_ylim(nat.edge.max(), 0)
        plt.colorbar(sm, ax=ax[2,0], location='left', orientation='vertical', fraction=0.05, pad=0.2,  label='Annual mass balance (m)')
        
        
        # Ax 2, 1
        ax[2, 1] = fig.add_subplot(gs[2, 1])
        ax[2, 1].set_ylabel("L (km)")
        ax[2, 1].plot(nat.t[xlims_slice], nat.edge_sm[xlims_slice], c="blue", lw=1)
        ax[2, 1].plot(anth.t[anth_x0:xlims_idx1], anth.edge_sm[anth_x0:xlims_idx1], c="red", lw=1)
        ax[2, 1].set_xlim(xlims)
        ax[2, 1].yaxis.set_major_locator(MultipleLocator(1))
        ax[2, 1].yaxis.set_minor_locator(MultipleLocator(0.5))
        
        # Ax 3, 0
        ax[3, 0] = fig.add_subplot(gs[3, 0])
        ax[3, 0].set_ylabel("Bal (m $yr^{-1}$)")
        ax[3, 0].set_xlabel("Time (years)")
        ax[3, 0].plot(nat.t, nat.gwb / nat.area, c="blue", lw=0.25, alpha=0.5)
        ax[3, 0].plot(nat.t, nat.bal_sm, c="blue", lw=1,)
        ax[3, 0].plot(anth.t[anth_x0:], (anth.gwb / anth.area)[anth_x0:], c="red", lw=0.25, alpha=0.5)
        ax[3, 0].plot(anth.t[anth_x0:], anth.bal_sm[anth_x0:], c="red", lw=1)
        ax[3, 0].set_xlim(xlims)
        ax[3, 0].yaxis.set_minor_locator(MultipleLocator(0.5))
        
        # Ax 3, 1
        ax[3, 1] = fig.add_subplot(gs[3, 1])
        ax[3, 1].set_xlabel("Time (years)")
        ax[3, 1].set_ylabel("Glacier Area ($km^2$)")
        ax[3, 1].plot(nat.t[xlims_slice], nat.area_sm[xlims_slice] / 1e6, c="blue", lw=1)
        ax[3, 1].plot(anth.t[anth_x0:xlims_idx1], anth.area_sm[anth_x0:xlims_idx1] / 1e6, c="red", lw=1)
        ax[3, 1].set_xlim(xlims)
        ax[3, 1].yaxis.set_major_locator(MultipleLocator(1))
        ax[3, 1].yaxis.set_minor_locator(MultipleLocator(0.5))
        
        
        
        for row in range(0, ax.shape[0]):
            for col in range(0, ax.shape[1]):
                axis = ax[row, col]
                if axis is not None:  # this handles gridspec col/rowspans > 1
                    axis.grid(axis="both", alpha=0.5, lw=1)
                    axis.set_axisbelow(True)
                    axis.tick_params(direction='in', which='both')
                    axis.tick_params(which='minor', length=3)
                    axis.tick_params(which='major', length=6)
                    axis.grid(axis="both", which='minor', alpha=0.25, lw=0.5)
                    if (row, col) not in [(0, 0), (2, 0)]:
                        axis.xaxis.set_major_locator(MultipleLocator(200))
                        axis.xaxis.set_minor_locator(MultipleLocator(50))
                        axis.axvline(1850, c='black', ls='dashed', lw=1)
                        
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        plt.tight_layout()
        fig.show()
        fig_output_name = Path(ROOT, f'flowline2d_{rgiid}.{model}.past1000.gwi.comparison.png')
        fig.savefig(fig_output_name)
