# -*- coding: utf-8 -*-
"""
climate.py

Description.

Author: drotto
Created: 11/28/2022 @ 1:37 PM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm

from src import data

#%%


class detrend_gcm:
    # todo : something is still fucky w/ this. 
    def __init__(self, gcm, trend, t0=1850):
        '''take a dataarray give a dataarray'''
        
        trend = trend.loc[t0-1:]
        trend = pd.Series(np.interp(gcm.time.loc[t0:], trend.index, trend), index=gcm.time.loc[t0:])  # match the trend indexes and lengths to gcm
        
        X = trend.copy()
        X = sm.add_constant(X)
        idx = dict(time=slice(t0, None))
        y = gcm.loc[idx]
        res = sm.OLS(y, X).fit()
        print(res.summary())

        y_nat = gcm.copy()
        y_nat.loc[idx] = res.resid + res.params['const']
        
        self.trend = res.params[0]*trend
        self.res = res
        self.y_nat = y_nat

def extend_ts(ts, win, n=500):
    random_climate = pd.read_csv(r"C:\Users\drotto\Documents\USGS\glacier-diseq\features\ClimateRandom.csv")
    random_climate = random_climate['Trand'][2015:].reset_index(drop=True)
    idx_end = ts.index[-1] + 1
    window_mean = ts.iloc[-win:].mean()
    filler = random_climate[:n] * ts.std() + window_mean
    filler.index = np.arange(idx_end, idx_end+n, 1)
    return pd.concat([ts, filler])
def prepend_ts(ts, n=100):
    '''Add a flatline period to the beginning of the ts'''
    idx1 = ts.index[0] 
    idx0 = idx1 - n
    prepend = pd.Series(np.zeros(n), index=np.arange(idx0, idx1, 1))
    ts = pd.concat([prepend, ts])
    return ts
    
        
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