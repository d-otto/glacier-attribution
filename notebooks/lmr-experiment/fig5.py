# -*- coding: utf-8 -*-
"""
fig5.py

Description.

Author: drotto
Created: 3/10/2023 @ 2:08 PM
Project: glacier-attribution
"""

from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np
import scipy as sci
import xarray as xr

import matplotlib as mpl
import matplotlib.pyplot as plt

from src.data import get_rgi, get_glacier_gcm, sample_glaciers, get_lmr
from config import cfg, ROOT

rng = np.random.default_rng()
rand = rng.uniform(0, 2 * np.pi, size=10000)

# %%

rgiids = ['RGI60-11.03638', 'RGI60-01.09162']
rgi = get_rgi(rgiids, from_sqllite=True).set_index('RGIId')

gnames = ['French Alps (Argentiere Glacier)', 'Maritime Southcentral Alaska (Wolverine Glacier)']

#%%

def emulate_noise(d, window, npts, show_plots=False, savgol=True):
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
    while len(X) * (window/2-1) < npts:
        Ak = np.sqrt(Px_filt / (2 * len(Px_filt))) * np.exp(1j * rand[i*len(Px_filt):((i+1)*len(Px_filt))])
        Ak = np.concatenate([Ak, Ak[:0:-1]])
        iAk = np.fft.ifft(Ak)
        X.append(iAk[1:len(iAk)//2])
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

def emulate_noise_fft(d, npts, show_plots=False):    
    x = d.copy()
    x = sci.signal.detrend(x, type='linear')
    f, Pxx = sci.signal.periodogram(x, detrend='linear', return_onesided=True)
    Pxx = Pxx / Pxx.mean()

    # Un-fft Px_filt
    psd = Pxx.copy()
    X = []
    i = 0
    while len(X) * (len(psd) - 1) < npts:
        Ak = np.sqrt(psd / (2 * len(psd))) * np.exp(1j * rand[i*len(psd):(i+1)*len(psd)])
        Ak = np.concatenate([Ak, Ak[:0:-1]])
        iAk = np.fft.ifft(Ak)
        X.append(iAk[1:len(iAk)//2])
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

# %%

lmr = get_lmr()
lmr = lmr.mean(dim='MCrun')
lmr = lmr.sel(time=slice(850, 1849))
lmr = sample_glaciers(lmr, rgi, as_dim=True)


def smooth(d, n=10):
    return sci.ndimage.uniform_filter1d(d, n)


#%%
for i in range(len(rgiids)):
    rgiid = rgiids[i]
    gname = gnames[i]

    lmr_rgi = lmr.sel(rgiid=rgiid)
    lmr_T = lmr_rgi.air.to_numpy()
    lmr_P = lmr_rgi.prate.to_numpy()
    t = np.arange(851, 1849, 1)
    T = sci.signal.detrend(lmr_T, type='linear')
    P = sci.signal.detrend(lmr_P, type='linear')
    colors = 'red'
    fig, ax = plt.subplots(4, 1, dpi=100, sharey='col', sharex='col', figsize=(8, 8), layout='constrained')
    j = 0
    d = T

    #offset = np.abs(d0).max()/2
    offset = 0
    
    dn = emulate_noise(d, 64, len(d))
    ax[0].plot(smooth(d), lw=1, color='black', ls='--')
    ax[0].plot(smooth(dn - offset), lw=1, color='tab:blue')
    ax[0].set_ylabel('Chunk=64')
    
    dn = emulate_noise(d, 512, len(d))
    ax[1].plot(smooth(d), lw=1, color='black', ls='--')
    ax[1].plot(smooth(dn - offset), lw=1, color='tab:orange')
    ax[1].set_ylabel('Chunk=512')

    dn = emulate_noise(d, 512, len(d), savgol=False)
    ax[2].plot(smooth(d), lw=1, color='black', ls='--')
    ax[2].plot(smooth(dn - offset), lw=1, color='tab:green')
    ax[2].set_ylabel('Chunk=512, unfiltered PSD')

    dn = emulate_noise_fft(d, len(d))
    ax[3].plot(smooth(d), lw=1, color='black', ls='--')
    ax[3].plot(smooth(dn - offset), lw=1, color='tab:red')
    ax[3].set_ylabel('Periodogram')
    
    for axis in ax.ravel(): 
        axis.tick_params(axis='both', which='both', direction='in')
        axis.grid(which='both', axis='both', ls=':')
        axis.set_xlim(0, 1000)
        axis.xaxis.set_major_locator(mpl.ticker.MultipleLocator(200))
        axis.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(50))
    ax[-1].set_xlabel('Period (years)')
    
        # d = Pyy.sel(rgiid=rgiid)
        # for model, da in d.groupby('model'):
        #     ax[1].plot(da.freq[1:], smooth(da[1:], 10), label=model, lw=0.75)
        # ax[1].plot(da.freq[1:], d.mean(dim='model')[1:], label='Gmean', lw=3, color='black')
        # mean = d.mean(dim='model')[1:]
        # std = d.std(dim='model')[1:]
        # ax[1].fill_between(da.freq[1:], mean - std, mean + std, color='black', ec=None, alpha=0.125,
        #                    label=r'$\pm 1 \sigma$')
        # xticks = [0.01, 0.05, 0.1, 0.5]
        # xlabels = [f'{1 / x:.0f}' for x in xticks]
        # ax[1].set_xscale('log')
        # ax[1].set_yscale('log')
        # ax[1].set_ylim(0.3, 3)
        # ax[1].set_xticks(xticks, labels=xlabels)
        # ax[1].tick_params(axis='both', which='both', direction='in')
        # ax[1].set_xlabel('Period (years)')
        # # ax[1].set_ylabel('Power spectral density')
        # ax[1].grid(which='both', axis='both', ls=':')
        # ax[1].set_title('Precipitation', fontsize='medium')
        # 
    fig.show()
    plt.savefig(Path(ROOT, f'notebooks/lmr-experiment/figures/fig5_{rgiid}.svg'))
