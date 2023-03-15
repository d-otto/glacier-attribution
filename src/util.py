# -*- coding: utf-8 -*-
"""
util.py

Description.

Author: drotto
Created: 11/9/2022 @ 4:57 PM
Project: glacier-attribution
"""

import numpy as np
import xarray as xr
import scipy as sci
import matplotlib.pyplot as plt

# %%


def trunc(values, decs=0):
    return np.trunc(values * 10**decs) / (10**decs)


def flatten(items, seqtypes=(list, tuple)):
    try:
        for i, x in enumerate(items):
            while isinstance(x, seqtypes):
                items[i : i + 1] = x
                x = items[i]
    except IndexError:
        pass
    return items


def unwrap_coords(lon, lat):
    if lon < 0:
        lon = 360 + lon
    return lon, lat

def unwrap_coord(l):
    if l < 0:
        l += 360
    return l

def dict_key_from_value(d, v):
    return list(d.keys())[list(d.values()).index(v)]

def dropna_coords(da, dim, how='any'):
    '''Can only be used on dims'''
    other_coords = [c for c in da.dims if c != dim]
    da_notnull = da.copy().stack(notnull=other_coords).notnull()
    if how=='any':
        da_notnull = da_notnull.any(dim='notnull')
    elif how=='all':
        da_notnull = da_notnull.all(dim='notnull')
    notnull_coords = da.coords[dim].where(da_notnull).to_pandas()
    notnull_coords = notnull_coords.dropna().values
    return da.sel({dim:notnull_coords})


def emulate_noise(d, window, npts, show_plots=False, savgol=True, rng=None):
    if not rng:
        rng = np.random.default_rng()
    rand = rng.uniform(0, 2 * np.pi, size=npts*2)

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