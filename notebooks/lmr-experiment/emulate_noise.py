# -*- coding: utf-8 -*-
"""
emulate_noise.py

Taking everything from lmr_spectral, putting it into a function, and doing it for GCM data

Author: drotto
Created: 3/9/2023 @ 4:06 PM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import scipy as sci

import matplotlib as mpl
import matplotlib.pyplot as plt

from src.data import get_rgi, get_glacier_gcm, sample_glaciers, get_lmr
from config import cfg, ROOT

rng = np.random.default_rng()
rand = rng.uniform(0, 2 * np.pi, size=10000)

#%%

rgiids = list(cfg['glaciers'].keys())
rgi = get_rgi(rgiids, from_sqllite=True).set_index('RGIId')
rgiid = 'RGI60-11.03638'

#%%

lmr = get_lmr()
lmr = lmr.mean(dim='MCrun')
lmr = lmr.sel(time=slice(850, 1849))
lmr_rgi = sample_glaciers(lmr, rgi, as_dim=True)
lmr_rgi = lmr_rgi.sel(rgiid=rgiid)
lmr_T = lmr_rgi.air.to_numpy()
lmr_P = lmr_rgi.prate.to_numpy()
t = np.arange(851,1849,1)

#%%

gcm = get_glacier_gcm(rgiids, 'tas', freq='jjas', mip=5)
gcm = gcm.sel(collection='lm', mip=5)
gcm = gcm.sel(experiment='past1000')
for dimension in ['model', 'r', 'i', 'f', 'p']:
    is_data = (gcm['tas'].count(dim=[dim for dim in gcm.dims.keys() if dim != dimension]) > 0).compute()
    gcm = gcm.sel({dimension: is_data})
gcm = gcm.sel(p=[1, 121])
gcm = gcm.mean(dim='p')
gcm = gcm.mean(dim=['r', 'i', 'f'])

gcm_T = gcm['tas'].sel(time=slice(850,1849))
gcm_T = gcm_T.sel(model=[model for model in gcm_T.coords['model'].values if model != 'FGOALS-gl'])
gcm_T = gcm_T.sel(rgiid=rgiid).mean(dim='model')
gcm_T = gcm_T.to_numpy()
gcm_P = gcm['pr'].sel(time=slice(850,1849))
gcm_P = gcm_P.sel(model=[model for model in gcm_P.coords['model'].values if model != 'FGOALS-gl'])
gcm_P = gcm_P.sel(rgiid=rgiid).mean(dim='model')
gcm_P = gcm_P.to_numpy()

#%%
# 
# def emulate_noise(d, window, npts, show_plots=False, savgol=True):
#     x = d.copy()
#     x = sci.signal.detrend(x, type='linear')
#     f, Pxx = sci.signal.welch(x, fs=1, nperseg=window, noverlap=window/2, detrend='linear')
#     Pxx = Pxx/Pxx.mean()
#     
#     # Savgol filter
#     Px_filt = Pxx.copy()
#     if savgol:
#         for i in range(4):
#             Px_filt = 10**(sci.signal.savgol_filter(np.log10(Px_filt), 30, polyorder=4, mode='mirror'))
#     Px_filt = Px_filt * Pxx.sum() / Px_filt.sum()
#     
#     if show_plots:
#         fig, ax = plt.subplots(1,1)
#         ax.loglog(f[1:], Pxx[1:])
#         ax.loglog(f[1:], Px_filt[1:])
#         fig.show()
#     
#     # Un-fft Px_filt
#     X = []
#     while len(X) * (len(Px_filt) - 1) < npts:
#         Ak = np.sqrt(Px_filt / (2*len(Px_filt))) * np.exp(1j * rng.uniform(0, 2*np.pi, size=len(Px_filt)))
#         X.append(np.fft.ifft(Ak))
#     X = np.concatenate(X).real
#     
#     
#     if show_plots:
#         # See if the PSD comes out right
#         fX, PXX = sci.signal.welch(X, fs=1, nperseg=window, noverlap=window/2)
#         PXX = PXX/PXX.mean()
#         PXX = PXX * Px_filt[2:-1].sum()/PXX[2:-1].sum()
#         fig, ax = plt.subplots(1,1)
#         ax.plot(fX[2:-1], PXX[2:-1], label='Reconstructed')
#         ax.plot(f[1:], Pxx[1:], label='Original')
#         ax.plot(f[1:], Px_filt[1:], label='Original (filtered)')
#         ax.legend()
#         fig.show()
#         
#         # Visual inspection of the data
#         fig, ax = plt.subplots(1,1)
#         ax.plot(sci.ndimage.uniform_filter1d(x, 20), label='Original')
#         ax.plot(sci.ndimage.uniform_filter1d(X[:len(x)]/X.std() * x.std(), 20), label='Reconstructed')
#         ax.legend()
#         fig.show()
# 
#     X = X[:npts] / X.std() * x.std()
#     return X
# 
# 
# def emulate_noise_fft(d, npts, show_plots=False):    
#     x = d.copy()
#     x = sci.signal.detrend(x, type='linear')
#     f, Pxx = sci.signal.periodogram(x, detrend='linear', return_onesided=True)
#     Pxx = Pxx / Pxx.mean()
# 
#     # Un-fft Px_filt
#     psd = Pxx.copy()
#     X = []
#     while len(X) * (len(psd) - 1) < npts:
#         Ak = np.sqrt(psd / (2 * len(psd))) * np.exp(1j * rng.uniform(0, 2 * np.pi, size=len(psd)))
#         X.append(np.fft.ifft(Ak))
#     X = np.concatenate(X).real
# 
#     if show_plots:
#         # See if the PSD comes out right
#         fX, PXX = sci.signal.periodogram(x, detrend='linear', return_onesided=True)
#         PXX = PXX / PXX.mean()
#         PXX = PXX * psd[2:-1].sum() / PXX[2:-1].sum()
#         fig, ax = plt.subplots(1, 1)
#         ax.plot(fX[2:-1], PXX[2:-1], label='Reconstructed')
#         ax.plot(f[1:], Pxx[1:], label='Original')
#         ax.legend()
#         fig.show()
# 
#         # Visual inspection of the data
#         fig, ax = plt.subplots(1, 1)
#         ax.plot(sci.ndimage.uniform_filter1d(x, 20), label='Original')
#         ax.plot(sci.ndimage.uniform_filter1d(X[:len(x)] / X.std() * x.std(), 20), label='Reconstructed')
#         ax.legend()
#         fig.show()
# 
#     X = X[:npts] / X.std() * x.std()
#     return X

def emulate_noise(d, window, npts, show_plots=False, savgol=True, rng=None):
    if not rng:
        rng = np.random.default_rng()    
    rand = rng.uniform(0, 2 * np.pi, size=10000)

    x = d.copy()
    x = sci.signal.detrend(x, type='linear')
    f, Pxx = sci.signal.welch(x, fs=1, nperseg=window, noverlap=window / 2, detrend='linear')
    Pxx = Pxx / Pxx.mean()

    # Savgol filter
    Px_filt = Pxx.copy()
    if savgol:
        for i in range(4):
            Px_filt = 10**(sci.signal.savgol_filter(np.log10(Px_filt), 30, polyorder=4, mode='mirror'))
    Px_filt = Px_filt * Pxx.sum() / Px_filt.sum()

    if show_plots:
        fig, ax = plt.subplots(1, 1)
        ax.loglog(f[1:], Pxx[1:])
        ax.loglog(f[1:], Px_filt[1:])
        fig.show()

    # Un-fft Px_filt
    X = []
    i = 0
    while len(X) * (window / 2 - 1) < npts:
        Ak = np.sqrt(Px_filt / (2 * len(Px_filt))) * np.exp(1j * rand[i * len(Px_filt):((i + 1) * len(Px_filt))])
        Ak = np.concatenate([Ak, Ak[:0:-1]])
        iAk = np.fft.ifft(Ak)
        X.append(iAk[1:len(iAk) // 2])
        i += 1
    X = np.concatenate(X).real

    if show_plots:
        # See if the PSD comes out right
        fX, PXX = sci.signal.welch(X, fs=1, nperseg=window, noverlap=window / 2)
        PXX = PXX / PXX.mean()
        PXX = PXX * Px_filt[1:].sum() / PXX[1:].sum()
        fig, ax = plt.subplots(1, 1)
        ax.plot(fX[1:], PXX[1:], label='Reconstructed')
        ax.plot(f[1:], Pxx[1:], label='Original')
        ax.plot(f[1:], Px_filt[1:], label='Original (filtered)')
        ax.legend()
        fig.show()

        # Visual inspection of the data
        fig, ax = plt.subplots(1, 1)
        ax.plot(sci.ndimage.uniform_filter1d(x, 20), label='Original')
        ax.plot(sci.ndimage.uniform_filter1d(X[:len(x)] / X.std() * x.std(), 20), label='Reconstructed')
        ax.legend()
        fig.show()

    X = X[:npts] / X.std() * x.std()
    return X


def emulate_noise_fft(d, npts, show_plots=False, rng=None):
    if not rng:
        rng = np.random.default_rng()    
    rand = rng.uniform(0, 2 * np.pi, size=10000)
    
    x = d.copy()
    x = sci.signal.detrend(x, type='linear')
    f, Pxx = sci.signal.periodogram(x, detrend='linear', return_onesided=True)
    Pxx = Pxx / Pxx.mean()

    # Un-fft Px_filt
    psd = Pxx.copy()
    X = []
    i = 0
    while len(X) * (len(psd) - 1) < npts:
        Ak = np.sqrt(psd / (2 * len(psd))) * np.exp(1j * rand[i * len(psd):(i + 1) * len(psd)])
        Ak = np.concatenate([Ak, Ak[:0:-1]])
        iAk = np.fft.ifft(Ak)
        X.append(iAk[1:len(iAk) // 2])
        i += 1
    X = np.concatenate(X).real

    if show_plots:
        # See if the PSD comes out right
        fX, PXX = sci.signal.periodogram(x, detrend='linear', return_onesided=True)
        PXX = PXX / PXX.mean()
        PXX = PXX * psd[2:-1].sum() / PXX[2:-1].sum()
        fig, ax = plt.subplots(1, 1)
        ax.plot(fX[2:-1], PXX[2:-1], label='Reconstructed')
        ax.plot(f[1:], Pxx[1:], label='Original')
        ax.legend()
        fig.show()

        # Visual inspection of the data
        fig, ax = plt.subplots(1, 1)
        ax.plot(sci.ndimage.uniform_filter1d(x, 20), label='Original')
        ax.plot(sci.ndimage.uniform_filter1d(X[:len(x)] / X.std() * x.std(), 20), label='Reconstructed')
        ax.legend()
        fig.show()

    X = X[:npts] / X.std() * x.std()
    return X

#%%

x = emulate_noise(lmr_T, 256, 1000, show_plots=True)

#%%
lmr_x = emulate_noise_fft(lmr_T, len(lmr_T), show_plots=True)
gcm_x = emulate_noise_fft(gcm_T, len(gcm_T), show_plots=True)