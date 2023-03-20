# -*- coding: utf-8 -*-
"""
fig4.py

Description.

Author: drotto
Created: 3/10/2023 @ 4:51 PM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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
rand = rng.uniform(0, 2 * np.pi, size=2000000)

# %%

rgiids = ['RGI60-11.03638', 'RGI60-01.09162']
rgi = get_rgi(rgiids, from_sqllite=True).set_index('RGIId')

gnames = ['French Alps (Argentiere Glacier)', 'Maritime Southcentral Alaska (Wolverine Glacier)']


# %%

lmr = get_lmr()
lmr = lmr.mean(dim='MCrun')
lmr = lmr.sel(time=slice(850, 1849))
lmr = sample_glaciers(lmr, rgi, as_dim=True)


#%% def emulate_noise(d, window, npts, show_plots=False, savgol=True):

lmr_T = lmr.air.sel(rgiid=rgiids[0]).to_numpy()
T = sci.signal.detrend(lmr_T, type='linear')
t = np.arange(850, 1850, 1)

def smooth(d, n=10):
    return sci.ndimage.uniform_filter1d(d, n)

fig, ax = plt.subplots(1, 2, dpi=100, figsize=(8, 4), layout='constrained', sharey=True)

for i, window in enumerate([128, 512]):
    print(i)
    print(window)
    
    x = T.copy()
    npts = len(x)
    savgol=True
    show_plots=True
    
    x = sci.signal.detrend(x, type='linear')
    f, Pxx = sci.signal.welch(x, fs=1, nperseg=window, noverlap=window / 2, detrend='linear')
    Pxx = Pxx / Pxx.mean()
    
    
    # Savgol filter
    Px_filt = Pxx.copy()
    if savgol:
        for j in range(4):
            Px_filt = 10**(sci.signal.savgol_filter(np.log10(Px_filt), window//8, polyorder=4, mode='mirror'))
    Px_filt = Px_filt * Pxx.sum() / Px_filt.sum()
    
    # Un-fft Px_filt
    # 1,000 pts
    npts = 1000
    X1k = []
    j = 0
    while len(X1k) * (window / 2 - 1) < npts:
        Ak = np.sqrt(Px_filt / (2 * len(Px_filt))) * np.exp(1j * rand[j * len(Px_filt):((j + 1) * len(Px_filt))])
        Ak = np.concatenate([Ak, Ak[:0:-1]])
        iAk = np.fft.ifft(Ak)
        X1k.append(iAk[1:len(iAk) // 2])
        j += 1
    X1k = np.concatenate(X1k).real
    X1k = X1k[:npts] / X1k.std() * x.std()
    
    # 1,000,000 pts
    npts = 1000000
    X1mm = []
    j = 0
    while len(X1mm) * (window / 2 - 1) < npts:
        Ak = np.sqrt(Px_filt / (2 * len(Px_filt))) * np.exp(1j * rand[j * len(Px_filt):((j + 1) * len(Px_filt))])
        Ak = np.concatenate([Ak, Ak[:0:-1]])
        iAk = np.fft.ifft(Ak)
        X1mm.append(iAk[1:len(iAk) // 2])
        j += 1
    X1mm = np.concatenate(X1mm).real
    X1mm = X1mm[:npts] / X1mm.std() * x.std()
    
    # See if the PSD comes out right
    fXk, PXXk = sci.signal.welch(X1k, fs=1, nperseg=window, noverlap=window / 2)
    PXXk = PXXk / PXXk.mean()
    PXXk = PXXk * Px_filt[1:].sum() / PXXk[1:].sum()
    fXmm, PXXmm = sci.signal.welch(X1mm, fs=1, nperseg=window, noverlap=window/2)
    PXXmm = PXXmm / PXXmm.mean()
    PXXmm = PXXmm * Px_filt[1:].sum() / PXXmm[1:].sum()
    
    ax[i].plot(f[1:], Pxx[1:], label='Original', color='black')
    ax[i].plot(fXk[1:], PXXk[1:], label='Reconstructed (n=1e3)', lw=2, color='tab:blue', ls='--')
    ax[i].plot(f[1:], Px_filt[1:], label='Original (filtered)', lw=2, color='red')
    ax[i].plot(fXmm[1:], PXXmm[1:], label='Reconstructed (n=1e6)', lw=2, color='tab:orange', ls='--')
    ax[i].set_title(f'Chunk length = {window}')

for axis in ax.ravel():
    xticks = [0.01, 0.05, 0.1, 0.5]
    xlabels = [f'{1 / x:.0f}' for x in xticks]
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_xticks(xticks, labels=xlabels)
    axis.tick_params(axis='both', which='both', direction='in')
    axis.set_xlabel('Period (years)')
    
    axis.grid(which='both', axis='both', ls=':')
ax[0].legend(loc='lower left')
fig.show()
plt.savefig(Path(ROOT, f'notebooks/lmr-experiment/figures/fig4.svg'))
