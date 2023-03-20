# -*- coding: utf-8 -*-
"""
lmr_spectral.py

Description.

Author: drotto
Created: 3/6/2023 @ 1:18 PM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator, ScalarFormatter, FixedFormatter, FixedLocator, FormatStrFormatter, NullFormatter
import gm
import dill
import gzip
from pathlib import Path
import importlib
from src.data import get_gwi, get_cmip6, sample_glaciers_from_gcms, get_rgi, get_cmip6_run_attrs, get_cmip6_lm, sample_glaciers, get_lmr
import src.data as data
from src.flowline import get_flowline_geom
from src.climate import detrend_gcm, extend_ts, prepend_ts
import xarray as xr
import multiprocess as mp
from itertools import product
from src.util import dict_key_from_value
from functools import partial
import xrft
import scipy as sci

#%%

import config

importlib.reload(config)
from config import cfg, ROOT


def temp_stable(yr):
    return 0


rgiid = list(cfg['glaciers'].keys())
rgi = get_rgi(rgiid, from_sqllite=True)

#%%

lmr = get_lmr()
lmr = lmr.mean(dim='MCrun')
lmr = lmr.sel(time=lmr.time>850)
lmr_rgi = sample_glaciers(lmr, rgi, as_dim=True)

#%%

T = lmr.air
P = lmr.prate
t = np.arange(851,2001,1)


#%%
#todo: sci.signal.detrend
power = xrft.power_spectrum(T, dim='time', detrend='constant')

#%%

dap = power.sel(latitude=60, longitude=270)
fig, ax = plt.subplots(1,1)
ax.semilogx(dap.freq_time, dap)
fig.show()


#%%

T = lmr_rgi.air.isel(rgiid=0)
T = T.to_numpy()
T = sci.signal.detrend(T, type='linear')

P = lmr_rgi.prate.isel(rgiid=0)
P = P.to_numpy()
P = sci.signal.detrend(P, type='linear')

#%%
# Note: there is a large peak in precip ~3 years with a chunk size of 32
# 128 is pretty good for the reconstruction. 64 is good for looking at the shape
# 512 is better for reconstruction, but bad for everything else!!

pcrit = 0.95
m_window = 512
nfrac = 1
ifx = int(m_window/2*nfrac)
f, Pxx = sci.signal.welch(T, fs=1, nperseg=m_window, noverlap=m_window/2, detrend='linear')
Pxx = Pxx/Pxx.mean()
f, Pyy = sci.signal.welch(P, fs=1, nperseg=m_window, noverlap=m_window/2, detrend='linear')
Pyy = Pyy/Pyy.mean()

# fig, ax = plt.subplots(1,1)
# ax.plot(f, Pxx)
# fig.show()


#%% compute rednoise null hypothesis

def autocorr(x, t=1, mean=None, var=None):
    '''This is definitely right! Hartmann 6.1.1.'''
    mean = x.mean()
    var = x.var()
    x -= mean
    return (x[:x.size-t] * x[t:]).mean() / var

def red_noise(a, x, t, dt, ep):
    return a * x * (t - dt) + (1 - a**2)**0.5 * ep

def red_shape(f, ac, A, Pxx=None, scale=True):
    #rs = A*(1.0-ac**2)/(1. -(2.0*ac*np.cos(f*2.0*np.pi))+ac**2)
    rs = A * (1 - ac**2) / (1 - (2 * ac * np.cos(f * 2 * np.pi)) + ac**2)
    if scale:
        rs = rs * Pxx[1:].sum()/rs[1:].sum()  # scale the variance of rs to Pxx 
    return rs

# compute the Gilman et al shape for red noise
acx = autocorr(T)
acy = autocorr(P)
acT = -1/np.log(acx)  # e-folding time of AR1

rs_x = red_shape(f, acx, 1.0, Pxx)
rs_y = red_shape(f, acy, 1.0, Pyy)

#%%

def dof1(T, m_window):
    return 2*T / m_window
def dof2(T, acx):
    return T * 1 / (2 * -1 / np.log(acx))  # Hartmann eq 6.9
def dof3(T, acx):
    return T * (1 - acx**2) / (1 + acx**2)  # alternatively, Hartmann eq 6.13b

# # dof comparison 
# fig, ax = plt.subplots(1,1)
# ax.plot(np.linspace(0, 1, 100), dof1(np.arange(1, 101, 1), m_window)/np.arange(1, 101, 1), label='dof1')
# ax.plot(np.linspace(0, 1, 100), dof2(1000, np.linspace(0, 1, 100))/1000, label='dof2')
# ax.plot(np.linspace(0, 1, 100), dof3(1000, np.linspace(0, 1, 100))/1000, label='dof3')
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.legend()
# fig.show()

#%%

dofx = 2*len(T) / m_window  # conservative estimate
dofy = 2*len(P) / m_window
dof2 = len(T) * 1/ (2 * -1/np.log(acx))  # Hartmann eq 6.9
dof3 = len(T) * (1 - acx**2)/(1 + acx**2)  # alternatively, Hartmann eq 6.13b
dof_null = len(T) / 2

fcrit_x = sci.stats.f.ppf(pcrit, dofx, dof_null)
fcrit_x2 = sci.stats.f.ppf(pcrit, dof2, dof_null)
fcrit_x3 = sci.stats.f.ppf(pcrit, dof3, dof_null)
red0_x = rs_x * fcrit_x  # computing the line we need to exceed Red Noise Theory
red0_x2 = rs_x * fcrit_x2  # computing the line we need to exceed Red Noise Theory
red0_x3 = rs_x * fcrit_x3  # computing the line we need to exceed Red Noise Theory


#%% Red noise and h0 against the data

fig, ax = plt.subplots(1, 1)
ax.loglog(f[1:ifx], Pxx[1:ifx])
ax.loglog(f[1:ifx], red0_x[1:ifx], label='red h0')
ax.loglog(f[1:ifx], red0_x2[1:ifx], label='red h0 #2')
ax.loglog(f[1:ifx], red0_x3[1:ifx], label='red h0 #3')
ax.loglog(f[1:ifx], rs_x[1:ifx], label='red shape')
ax.legend()
ax.grid(which='both', axis='both')
fig.show()


#%%

def h0_psd(f, acx, Pxx):
    rs_x = red_shape(f, acx, 1.0, Pxx)
    rs_x = rs_x * np.sum(Pxx[1:]) / np.sum(rs_x[1:])  # scale the variance of rs to Pxx 
    dof = len(T) * (1 - acx**2)/(1 + acx**2)  # alternatively, Hartmann eq 6.13b
    dof_null = len(T) / 2
    fcrit_x = sci.stats.f.ppf(pcrit, dof, dof_null)
    red0_x = rs_x * fcrit_x  # computing the line we need to exceed Red Noise Theory
    return red0_x

def fit_red(f, Pxx):
    params, cov = sci.optimize.curve_fit(partial(red_shape, Pxx=Pxx), f, Pxx, p0=(0.5, 1))
    return params

cmap = plt.cm.cividis(np.linspace(0, 1, 10))
fig, ax = plt.subplots(1, 1)
for i, ac in enumerate(np.linspace(0, 1, 10)):
    ax.loglog(f[1:ifx], h0_psd(f, ac, Pxx)[1:ifx], color=cmap[i], label=f'red ac={ac:.2f}')
ax.loglog(f[1:ifx], Pxx[1:ifx], c='red')
ax.legend()
ax.grid(which='both', axis='both')
fig.show()

#%% Fitted red noise
# And, if we only fit it to be accurate for low frequencies, what does it look like for higher frequencies where the
# data PSD might differ from red noise?

fig, ax = plt.subplots(1, 1)
offset = len(f)//2
nlines = 10
cmap = plt.cm.cool(np.linspace(0, 1, nlines))
for i, idx in enumerate(np.floor(np.linspace(-offset, -1, nlines)).astype(int)):
    red_fitted = red_shape(f, *fit_red(f[:idx], Pxx[:idx]), Pxx=Pxx)
    ax.loglog(f[1:ifx], red_fitted[1:ifx], c=cmap[i], alpha=1)
ax.loglog(f[1:ifx], Pxx[1:ifx], c='black', label='original T')
ax.loglog(f[1:ifx], h0_psd(f, acx, Pxx)[1:ifx], color='red', label=f'red h0 ac={acx:.2f}')
ax.loglog(f[1:ifx], h0_psd(f, 0, Pxx)[1:ifx], color='grey', label=f'red h0 ac=0')
ax.legend()

xticks = [0.01, 0.05, 0.1, 0.5]
xlabels = [f'{1/x:.0f}' for x in xticks]
ax.set_xticks(xticks, labels=xlabels)
ax.set_xlabel('Period (years)')
ax.grid(which='both', axis='both')
fig.show()


fig, ax = plt.subplots(1, 1)
offset = len(f)//2
nlines = 20
cmap = plt.cm.cool(np.linspace(0, 1, nlines))
for i, idx in enumerate(np.floor(np.linspace(-offset, -1, nlines)).astype(int)):
    red_fitted = red_shape(f, *fit_red(f[:idx], Pyy[:idx]), Pxx=Pyy)
    ax.loglog(f[1:ifx], red_fitted[1:ifx], c=cmap[i], alpha=0.5)
ax.loglog(f[1:ifx], Pyy[1:ifx], c='blue', label='original P')
ax.loglog(f[1:ifx], h0_psd(f, acy, Pyy)[1:ifx], color='orange', label=f'red h0 ac={acy:.2f}')
ax.loglog(f[1:ifx], h0_psd(f, 0, Pyy)[1:ifx], color='grey', label=f'red h0 ac=0')
ax.legend()

xticks = [0.01, 0.05, 0.1, 0.5]
xlabels = [f'{1/x:.0f}' for x in xticks]
ax.set_xticks(xticks, labels=xlabels)
ax.set_xlabel('Period (years)')
ax.grid(which='both', axis='both')

fig.show()

#%% look at the data again

fig, ax = plt.subplots(2,1)
ax[0].plot(t, sci.ndimage.uniform_filter1d(T, 30))
ax[1].plot(t, sci.ndimage.uniform_filter1d(P, 30))
fig.show()


#%%

def poly(f, a, b, c, d, Pxx):
    p = np.polynomial.Polynomial([a, b, c, d], domain=[f[0], f[-1]])(f)
    p = p * Pxx[1:].sum() / p[1:].sum()
    return p
def fit_poly(f, Pxx):
    params, cov = sci.optimize.curve_fit(partial(poly, Pxx=Pxx), f, Pxx)
    return params
test = fit_poly(f, Pxx)

fig, ax = plt.subplots(1,1)
ax.loglog(f[1:ifx], Pxx[1:ifx])
ax.loglog(f[1:ifx], poly(f, *test, Pxx)[1:ifx])
fig.show()

#%%

idx = np.argmin(abs(1/f - 5)) // 2 * 2
Pxx_hf = Pxx[idx:]
Pxx_lf = Pxx[:idx]
fx_hf = f[idx:]
fx_lf = f[:idx]
params = fit_red(fx_hf, Pxx_hf)
rfx_hf = red_shape(fx_hf, *params, Pxx=Pxx_hf)

fig, ax = plt.subplots(1,1)
ax.loglog(f[1:ifx], Pxx[1:ifx])
ax.loglog(fx_hf, rfx_hf)
fig.show()


#%% Savgol filter

Px_filt = Pxx.copy()
for i in range(4):
    Px_filt = 10**(sci.signal.savgol_filter(np.log10(Px_filt), 30, polyorder=4, mode='mirror'))
Px_filt = Px_filt * Pxx.sum() / Px_filt.sum()

fig, ax = plt.subplots(1,1)
ax.loglog(f[1:ifx], Pxx[1:ifx])
ax.loglog(f[1:ifx], Px_filt[1:ifx])
fig.show()

Py_filt = Pyy.copy()
for i in range(4):
    Py_filt = 10**(sci.signal.savgol_filter(np.log10(Py_filt), 30, polyorder=4, mode='mirror'))
Py_filt = Py_filt * Pxx.sum() / Py_filt.sum()

fig, ax = plt.subplots(1,1)
ax.loglog(f[1:ifx], Pxx[1:ifx])
ax.loglog(f[1:ifx], Py_filt[1:ifx])
fig.show()


#%% Un-fft Px_filt

rng = np.random.default_rng()
X = []
i = 0
npts = 1000000
while len(X) * (len(Px_filt) * 2 - 1) < npts:
    Ak = np.sqrt(Px_filt / (2*len(Px_filt))) * np.exp(1j * rng.uniform(0, 2*np.pi, size=len(Px_filt)))
    Ak = np.concatenate([Ak, Ak[:0:-1]])
    X.append(np.fft.ifft(Ak))
    i += 1
X = np.concatenate(X).real

# See if the PSD comes out right
fX, PXX = sci.signal.welch(X, fs=1, nperseg=m_window, noverlap=m_window/2)
PXX = PXX/PXX.mean()
PXX = PXX * Px_filt[2:-1].sum()/PXX[2:-1].sum()
fig, ax = plt.subplots(1,1)
ax.plot(fX[2:-1], PXX[2:-1], label='Reconstructed')
ax.plot(f[1:], Pxx[1:], label='Original')
ax.plot(f[1:], Px_filt[1:], label='Original (filtered)')
ax.legend()
fig.show()

# Visual inspection of the data
fig, ax = plt.subplots(1,1)
ax.plot(sci.ndimage.uniform_filter1d(T, 20), label='Original')
ax.plot(sci.ndimage.uniform_filter1d(X[:len(T)]/X.std() * T.std(), 20), label='Reconstructed')
ax.legend()
fig.show()


#%% Un-fft Py_filt


rng = np.random.default_rng()
X = []
i = 0
npts = 1000000
while len(X) * (len(Py_filt) * 2 - 1) < npts:
    Ak = np.sqrt(Py_filt / (2*len(Py_filt))) * np.exp(1j * rng.uniform(0, 2*np.pi, size=len(Py_filt)))
    Ak = np.concatenate([Ak, Ak[:0:-1]])
    X.append(np.fft.ifft(Ak))
    i += 1
X = np.concatenate(X).real

# See if the PSD comes out right
fX, PYY = sci.signal.welch(X, fs=1, nperseg=m_window, noverlap=m_window/2)
PYY = PYY/PYY.mean()
PYY = PYY * Py_filt[2:-1].sum()/PYY[2:-1].sum()
fig, ax = plt.subplots(1,1)
ax.plot(fX[2:-1], PYY[2:-1], label='Reconstructed')
ax.plot(f[1:], Pxx[1:], label='Original')
ax.plot(f[1:], Py_filt[1:], label='Original (filtered)')
ax.legend()
fig.show()

# Visual inspection of the data
fig, ax = plt.subplots(1,1)
ax.plot(sci.ndimage.uniform_filter1d(T, 20), label='Original')
ax.plot(sci.ndimage.uniform_filter1d(X[:len(T)]/X.std() * T.std(), 20), label='Reconstructed')
ax.legend()
fig.show()


#%% A function to tie it together




#%% Try a high pass filter and see if its volcanoes

