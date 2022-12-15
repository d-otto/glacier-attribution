# -*- coding: utf-8 -*-
"""
example_flowline2d.py

Description.

Author: drotto
Created: 10/27/2022 @ 11:34 AM
Project: glacier-diseq
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import lgm
import pickle
from functools import partial

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
def temp_step(yr, step=0.0, Tdot=0.0):
    if yr >= 100:
        Tdot = Tdot + step
    return Tdot

#%%

geom = pd.read_csv(r'C:\Users\drotto\Documents\USGS\glacier-diseq\matlab\Wolverine\wolv_geom.csv')
x_gr = geom['length'].to_numpy()
zb_gr = geom['bed_h'].to_numpy()
zb_gr[51:57] = np.linspace(zb_gr[51], zb_gr[57], len(zb_gr[51:57]))
w_geom = geom['widths_m'].to_numpy()
w_geom[60:] = np.linspace(700, 2000, 86)  # width = wide
# w_geom[60:] = np.linspace(700, 700, 86)  # width = narrow
x_geom = geom['length']

climate = pd.read_csv(r"C:\Users\drotto\Documents\USGS\glacier-diseq\features\ClimateRandom.csv")
Trand = climate['Trand']
Prand = climate['Prand']

x_init = np.arange(0, 19000, 50)
#h_init = np.concatenate((np.tile(50, 100), np.tile(0, 280)), axis=0)
with open('flowline2d_wolv_widthexp_narrow-step-m05.pickle', 'rb') as f:
    last_run = pickle.load(f)
h_init = last_run['h'][-1, :]

model = lgm.flowline2d(x_gr=x_gr, zb_gr=zb_gr, x_geom=x_geom, w_geom=w_geom, x_init=x_init, h_init=h_init, xmx=19000,
                       temp=partial(temp_step, step=0.5, Tdot=-0.5), sigT=0, sigP=0, P0=3.5,
                       delt=0.0125/8,
                       ts=0, tf=1500, T0=12.51, t_stab=500,
                       T=Trand, P=Prand, gamma=6.5e-3,
                       rt_plot=True, dt_plot=100)

fig = model.plot(xlim0=0, compare_fp='flowline2d_wolv_widthexp_wide-step-p05.pickle')
fig.show()


fname = 'flowline2d_wolv_widthexp_comparison-step-p05'
fig_output_name = fname + '.png'
plt.savefig(fig_output_name)
# 
# file_output_name = fname + '.pickle'
# model.to_pickle(file_output_name)