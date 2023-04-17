# -*- coding: utf-8 -*-
"""
flowline.py

Description.

Author: drotto
Created: 12/1/2022 @ 6:19 AM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import scipy as sci
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import gzip
import pickle

from src.data import get_rgi
from config import ROOT, cfg

rgiids = {k: v['name'] for k, v in cfg['glaciers'].items()}
rgi = get_rgi(list(rgiids.keys()), from_sqllite=True)

#%%

class get_flowline_geom:
    def __init__(self, rgiid, dirname=Path(ROOT, 'data/interim/oggm_flowlines'), gname=None):
        '''Combine oggm geometrical centerlines with elevation bands to get a length-accurate bulk 2d representation of the glacier geometry.
        
        There are four sources of information. Elevation bands (eb) and geometrical centerlines (gc) are the two types of models.
        For each, there is a glacier flowline (fl; using the primary flowline for geometrical centerlines) and downstream lines (dl).
        We need to obtain the following variables from the combination of these: x, zb, w, and h.
        Where x is horizontal length, zb is bed elevation, w is glacier width, and h is the initial ice surface thickness.
        
        Using the main flowline of the GC, we do not include ice from tributaries that comes from higher elevations.
        However, we take the length of this flowline as the "true" length of the glacier. 
        
        Parameters
        ----------
        rgiid : 
        dirname : 
        '''
        #%%
        
            
        #%%
        fp = dirname / f'e_bands/{rgiid}.downstream_line.pickle'
        with gzip.open(fp, "rb") as openfile:
            fl, inv, dsl = pickle.load(openfile)

        fp = dirname / f'e_bands/{rgiid}.model_flowlines.pickle'
        with gzip.open(fp, "rb") as openfile:
            mfl = pickle.load(openfile)
            mfl = mfl[0]
            
        line_len = len(dsl['surface_h'])
        glacier_len = len(fl.surface_h)
        x = np.arange(0, line_len + glacier_len) * fl.map_dx
        h0 = np.zeros(line_len + glacier_len)
        h0[:glacier_len] = inv['thick']
        zb = np.concatenate([inv['hgt'] - inv['thick'], dsl['surface_h']])

        w_ds = sci.ndimage.gaussian_filter1d(dsl['w0s'], 5)
        toe_zone = glacier_len - int(glacier_len * 0.1)  # approximately the very end of the glacier
        w = np.concatenate([fl.widths_m, w_ds])
        w[toe_zone:glacier_len] = np.where(w[toe_zone:glacier_len] > w_ds[0], w[toe_zone:glacier_len], w_ds[0])

        p = Path(ROOT, rf'data\interim\oggm_flowlines\flowline_geom_{rgiid}_raw.csv')
        df = pd.DataFrame(dict(x=x, zb=zb, w=w, h0=h0))
        df.to_csv(p, index=False)
        if rgiid == "RGI60-11.00897":  # Hintereisferner
            df['zb'].iloc[133:145] = np.nan
            df['zb'] = df['zb'].interpolate(method='spline', order=3)
            df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
            df['zb'].iloc[0:5] = np.linspace(3570, 3550, 5)
            df['h0'] = df['h0'] - (df['zb'] - zb[0:len(df['zb'])])
        elif rgiid == 'RGI60-11.03638':  # argentiere
            df = df.iloc[:400]
            df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
            df['h0'] = df['h0'] - (df['zb'] - zb[0:len(df['zb'])])
        elif rgiid == 'RGI60-02.18778':  # south cascade
            df = df.iloc[:200]
            df['zb'].iloc[44:71] = np.nan
            df['zb'].iloc[57] = 1585
            df['zb'].iloc[0:2] = [2110, 2100]
            df['zb'] = df['zb'].interpolate(method='spline', order=3)
            df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=10, polyorder=4)
            
            df['h0'] = df['h0'] - (df['zb'] - zb[0:len(df['zb'])])
        elif rgiid == "RGI60-01.09162":  # wolverine
            df = df.iloc[:300]
            df['zb'].iloc[0:10] = np.linspace(1505, 1496, 10)
            
            df['zb'].iloc[143:159] = np.nan
            df['zb'].iloc[156] = 425
            df['zb'] = df['zb'].interpolate(method='spline', order=3)
            df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
            df['h0'] = df['h0'] - (df['zb'] - zb[0:len(df['zb'])])
            df['h0'].iloc[0:10] = 105
        elif rgiid == "RGI60-01.00570":  # Gulkana
            df = df.iloc[:300]
            df['zb'].iloc[0:4] = np.linspace(2350, 2310, 4)
            df['zb'].iloc[135:173] = np.nan
            df['zb'] = df['zb'].interpolate(method='spline', order=3)
            df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
            df['h0'] = df['h0'] - (df['zb'] - zb[0:len(df['zb'])])
            df['h0'].iloc[0:4] = 70
        elif rgiid == "RGI60-11.01450":  # Aletsch
            df['zb'].iloc[0:3] = np.linspace(4000, 3988, 3)
            df['h0'].iloc[0:2] = 60
            df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=30, polyorder=4)
            df['h0'] = df['h0'] - (df['zb'] - zb[0:len(df['zb'])])
            df['h0'].iloc[0:9] = 90
        elif rgiid == "RGI60-11.01346":  # Unterer Grindelwald
            df = df.iloc[:400]
            df['zb'].iloc[160:183] = np.nan
            df['zb'] = df['zb'].interpolate(method='spline', order=3)
            df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
            df['zb'].iloc[0:4] = np.linspace(3920, 3909, 4)
            df['h0'] = df['h0'] - (df['zb'] - zb[0:len(df['zb'])])
            df['h0'].iloc[0:2] = 80
        elif rgiid == "RGI60-11.01238":  # rhone    
            df = df.iloc[:400]
            df['zb'].iloc[0:12] = np.linspace(3460, 3450, 12)
            df['zb'].iloc[175:197] = np.nan
            df['zb'].iloc[180] = 2200
            df['zb'].iloc[188] = 2205
            df['zb'].iloc[195] = 2225
            df['zb'] = df['zb'].interpolate(method='spline', order=3)
            df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
            df['h0'] = df['h0'] - (df['zb'] - zb[0:len(df['zb'])])
            df['h0'].iloc[0:9] = 90
        elif rgiid == "RGI60-11.03646":  # Bossons
            df = df.iloc[:300]
            df['zb'].iloc[0:11] = np.linspace(4700, 4664, 11)
            df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
            df['h0'] = df['h0'] - (df['zb'] - zb[0:len(df['zb'])])
            df['h0'].iloc[0:11] = 50
        elif rgiid == "RGI60-11.03643":  # mer de glace
            df = df.iloc[:500]
            df['zb'].iloc[256:263] = np.nan
            df['zb'] = df['zb'].interpolate(method='spline', order=3)
            df['zb'] = sci.signal.savgol_filter(df['zb'], window_length=20, polyorder=4)
            df['h0'] = df['h0'] - (df['zb'] - zb[0:len(df['zb'])])

        df['h0'].iloc[np.argmin(h0):] = 0  # zero thickness after terminus
        p = Path(ROOT, rf'data\interim\oggm_flowlines\flowline_geom_{rgiid}.csv')
        df.to_csv(p, index=False)

        # QC plots
        name = cfg['glaciers'][rgiid]['name']
        rgi_geom = rgi.loc[rgi.index == rgiid].iloc[0]
        
        fig, ax = plt.subplots(1, 1, dpi=200)
        ax.plot(x, zb, color='black', label='Original zb')
        ax.plot(df['x'], df['zb'], color='red', label='New zb')
        ax.plot(df['x'], df['zb'] + df['h0'], color='red', label='New h0')
        ax.plot(x, zb + h0, color='black', ls='--', label='Original h0')
        ax.grid()
        ax.legend()
        ax.set_title(f'{rgiid} {name}')
        p = Path(ROOT, rf'data\interim\oggm_flowlines\flowline_geom_QC_{rgiid}_{name.lower().replace(" ", "")}.png')
        plt.savefig(p)
        fig.show()
        


        fig, ax = plt.subplots(2, 1, sharex=True, dpi=200)
        idx_term = np.where(df.h0 == 0)[0][0]
        idx = idx_term + 60
        ax[0].plot(df.x[:idx], df.zb[:idx] + df.h0[:idx], c='blue', label='zb + h0')
        ax[0].plot(df.x[:idx], df.zb[:idx], c='black', label='zb')
        ax[0].axvline(rgi_geom.Lmax, c='red', label='RGI Lmax')

        ax[1].plot(df.x[:idx], df.w[:idx] / 1000, c='black', label='w')
        ax[1].plot(df.x[:idx_term], np.cumsum(df.w[:idx_term] * 50) / 1e6, c='grey', label='L * w')
        ax[1].axhline(rgi_geom.Area, c='red', ls='--', label='RGI Area')
        ax[1].axvline(rgi_geom.Lmax, c='red', label='RGI Lmax')

        for axis in ax.ravel():
            axis.grid()
            axis.legend()
        ax[0].set_title(f'{rgiid} {name}')
        p = Path(ROOT, rf'data\interim\oggm_flowlines\flowline_geom_{rgiid}_{name.lower().replace(" ", "")}.png')
        plt.savefig(p)
        fig.show()
        
            
        #%%
        
        self.x = df.x
        self.zb = df.zb
        self.w = df.w
        self.h0 = df.h0
        
        
    def to_pandas(self):
        return pd.DataFrame(dict(x=self.x, zb=self.zb, w=self.w, h0=self.h0))