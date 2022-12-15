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
from numpy.random import default_rng
import matplotlib as mpl
import matplotlib.pyplot as plt
import gm
import dill
import gzip
from pathlib import Path
import importlib
from src.data import get_rgi
from src.climate import temp_stable
from src.flowline import get_flowline_geom
from config import ROOT

#%%

import config
importlib.reload(config)
from config import cfg

rgiid = list({k: v for k, v in cfg['glaciers'].items() if v['name'] == 'Argentiere'}.keys())[0]
params = cfg['glaciers'][rgiid]['flowline_params']

#%% RGIID of interest

rgi = get_rgi(rgiid, from_sqllite=True)
geom = get_flowline_geom(rgiid)

#%%

# climate = pd.read_csv(r"C:\Users\drotto\Documents\USGS\glacier-diseq\features\ClimateRandom.csv")
# Trand = climate['Trand']
# Prand = climate['Prand']

#%%

rng = default_rng()
Trand = rng.standard_normal(10000)
Prand = rng.standard_normal(10000)

try:
    with open(Path(ROOT, f'flowline2d_{rgiid}.ss.noise.pickle'), 'rb') as f:
        last_run = dill.load(f)
    h_init = last_run.h.mean(axis=0)
    x_init = last_run.x
except:
    print('Using default profile')
    h_init = geom.h0
    x_init = geom.x

model = gm.flowline2d(x_gr=geom.x, zb_gr=geom.zb, x_geom=geom.x, w_geom=geom.w, x_init=x_init, h_init=h_init, xmx=x_init.max(),
                      temp=temp_stable, sigT=1, sigP=0, P0=params['P0'],
                      delt=0.0125/16, delx=50,
                      ts=0, tf=10000, T0=22.27,
                      T=Trand, P=Prand,
                      rt_plot=False, dt_plot=250)

fig = model.plot(xlim0=0)
fig.show()
fig_output_name = Path(ROOT, f'flowline2d_{rgiid}.ss.noise.png')
plt.savefig(fig_output_name)

#%%
file_output_name = Path(ROOT, f'flowline2d_{rgiid}.ss.noise.pickle')
model.to_pickle(file_output_name)
