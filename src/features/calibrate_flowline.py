# -*- coding: utf-8 -*-
"""
calibrate_flowline.py

Description.

Author: drotto
Created: 11/8/2022 @ 5:47 PM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import gm
import pickle
import gzip
from pathlib import Path
import importlib
from src.data import get_rgi
from src.flowline import get_flowline_geom

#%%

import config
importlib.reload(config)
from config import cfg

rgiid = list({k: v for k, v in cfg['glaciers'].items() if v['name'] == 'Hintereisferner'}.keys())[0]
params = cfg['glaciers'][rgiid]['flowline_params']

#%% RGIID of interest

rgi = get_rgi(rgiid, from_sqllite=True)
geom = get_flowline_geom(rgiid)

#%%

def temp_comb(yr, LIA_cooling=True, ANTH_warming=True):
    Tdot = 0.0
    if (yr >= 999) & LIA_cooling:
        Tdot = Tdot -0.25 * (yr - 1000) / 1000
    if (yr >= 1850) & ANTH_warming:
        Tdot = Tdot + 1.3 * (yr - 1850) / 150
    return Tdot
def temp_nat(yr):
    Tdot = 0.0
    if yr >= 999:
        Tdot = Tdot -0.25 * (yr - 1000) / 1000
    return Tdot
def temp_stable(yr):
    return 0


#%%

climate = pd.read_csv(r"C:\Users\drotto\Documents\USGS\glacier-diseq\features\ClimateRandom.csv")
Trand = climate['Trand']
Prand = climate['Prand']


#%%

try:
    with open(f'flowline2d_{rgiid}.ss.noise.pickle', 'rb') as f:
        last_run = pickle.load(f)
    h_init = last_run.h[-1, :]
    x_init = last_run.x
except:
    print('Using default profile')
    h_init = geom.h0
    x_init = geom.x
# h_init = np.concatenate((np.tile(50, 100), np.tile(0, 280)), axis=0)

model = gm.flowline2d(x_gr=geom.x, zb_gr=geom.zb, x_geom=geom.x, w_geom=geom.w, x_init=x_init, h_init=h_init, xmx=x_init.max(),
                       temp=temp_stable, sigT=1, sigP=0, P0=params['P0'],
                       delt=0.0125/8, delx=50,
                       ts=0, tf=5000, T0=22.25,
                       Trand=Trand, Prand=Prand,
                       rt_plot=False, dt_plot=250)

fig = model.plot(xlim0=0)
fig.show()
fig_output_name = f'flowline2d_{rgiid}.ss.noise.png'
plt.savefig(fig_output_name)

#%%
file_output_name = f'flowline2d_{rgiid}.ss.noise.pickle'
model.to_pickle(file_output_name)
