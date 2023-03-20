# -*- coding: utf-8 -*-
"""
fig2.py

Description.

Author: drotto
Created: 3/10/2023 @ 12:04 PM
Project: glacier-attribution
"""

from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np
import scipy as sci

import matplotlib as mpl
import matplotlib.pyplot as plt

from src.data import get_rgi, get_glacier_gcm, sample_glaciers, get_lmr
from config import cfg, ROOT


rng = np.random.default_rng()

#%%

rgiids = ['RGI60-11.03638', 'RGI60-01.09162']
rgi = get_rgi(rgiids, from_sqllite=True).set_index('RGIId')

gnames = ['French Alps (Argentiere Glacier)', 'Maritime Southcentral Alaska (Wolverine Glacier)']

#%%
lmr = get_lmr()
lmr = lmr.mean(dim='MCrun')
lmr = lmr.sel(time=slice(850, 1849))
lmr = sample_glaciers(lmr, rgi, as_dim=True)

#%%

i = 0
for i in range(len(rgiids)):
    rgiid = rgiids[i]
    gname = gnames[i]
    
    
    lmr_rgi = lmr.sel(rgiid=rgiid)
    lmr_T = lmr_rgi.air.to_numpy()
    lmr_P = lmr_rgi.prate.to_numpy()
    t = np.arange(851,1849,1)
    

    
    T = sci.signal.detrend(lmr_T, type='linear')
    P = sci.signal.detrend(lmr_P, type='linear')
    
    
    pcrit = 0.95
    m_window = 64
    nfrac = 1
    ifx = int(m_window/2*nfrac)
    f, Pxx = sci.signal.welch(T, fs=1, nperseg=m_window, noverlap=m_window/2, detrend='linear')
    Pxx = Pxx/Pxx.mean()
    f, Pyy = sci.signal.welch(P, fs=1, nperseg=m_window, noverlap=m_window/2, detrend='linear')
    Pyy = Pyy/Pyy.mean()
    
    # compute rednoise null hypothesis
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
    
    # Red noise and h0 against the data
    
    fig, ax = plt.subplots(1, 1)
    ax.loglog(f[1:ifx], rs_x[1:ifx], label='Fitted red noise', color='black')
    ax.loglog(f[1:ifx], rs_y[1:ifx], color='black')
    ax.loglog(f[1:ifx], Pxx[1:ifx], label='Temperature', color='tab:red', lw=2, solid_joinstyle='round')
    ax.loglog(f[1:ifx], Pyy[1:ifx], label='Precipitation', color='tab:cyan', lw=2, solid_joinstyle='round')
    
    xticks = [0.01, 0.05, 0.1, 0.5]
    xlabels = [f'{1 / x:.0f}' for x in xticks]
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xticks(xticks, labels=xlabels)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.set_xlabel('Period (years)')
    ax.set_ylabel('Power spectral density')
    ax.grid(which='both', axis='both', ls=':')
    ax.legend()
    fig.show()

    
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
    # 
    # cmap = plt.cm.plasma(np.linspace(0, 1, 10))
    # fig, ax = plt.subplots(1, 1)
    # for i, ac in enumerate(np.linspace(0, 1, 10)):
    #     ax.loglog(f[1:ifx], h0_psd(f, ac, Pxx)[1:ifx], color=cmap[i])
    # ax.loglog(f[1:ifx], Pxx[1:ifx], c='black', label='Temperature', lw=3)
    # norm = mpl.colors.Normalize(vmin=0, vmax=1)
    # sm = mpl.cm.ScalarMappable(cmap='plasma', norm=norm)
    # plt.colorbar(sm, ax=ax, label='Autocorrelation')
    # xticks = [0.01, 0.05, 0.1, 0.5]
    # xlabels = [f'{1 / x:.0f}' for x in xticks]
    # ax.set_xscale('log')
    # #ax.set_yscale('log')
    # ax.set_xticks(xticks, labels=xlabels)
    # ax.tick_params(axis='both', which='both', direction='in')
    # ax.set_xlabel('Period (years)')
    # ax.set_ylabel('Power spectral density')
    # ax.grid(which='both', axis='both', ls=':')
    # ax.legend()
    # fig.show()
    
    # Fitted red noise
    # And, if we only fit it to be accurate for low frequencies, what does it look like for higher frequencies where the
    # data PSD might differ from red noise?
    
    fig, ax = plt.subplots(1, 2, layout='constrained', sharey=True, dpi=100, figsize=(10, 4))
    offset = len(f)//2
    nlines = 20
    cmap = plt.cm.plasma(np.linspace(0, 1, nlines))
    for i, idx in enumerate(np.floor(np.linspace(-offset, -1, nlines)).astype(int)):
        red_fitted = red_shape(f, *fit_red(f[:idx], Pxx[:idx]), Pxx=Pxx)
        ax[0].loglog(f[1:ifx], red_fitted[1:ifx], c=cmap[i], alpha=1, lw=1)
    ax[0].loglog(f[1:ifx], Pxx[1:ifx], c='tab:red', label='Temperature', lw=3)
    ax[0].loglog(f[1:ifx], h0_psd(f, acx, Pxx)[1:ifx], color='black', label=f'red h0 ac={acx:.2f}', lw=2, ls='--')
    xticks = [0.01, 0.05, 0.1, 0.5]
    xlabels = [f'{1 / x:.0f}' for x in xticks]
    ax[0].set_xscale('log')
    #ax[0].set_yscale('log')
    ax[0].set_xticks(xticks, labels=xlabels)
    ax[0].tick_params(axis='both', which='both', direction='in')
    ax[0].set_xlabel('Period (years)')
    ax[0].set_ylabel('Power spectral density')
    ax[0].grid(which='both', axis='both', ls=':')
    ax[0].legend()
    
    for i, idx in enumerate(np.floor(np.linspace(-offset, -1, nlines)).astype(int)):
        red_fitted = red_shape(f, *fit_red(f[:idx], Pyy[:idx]), Pxx=Pyy)
        ax[1].loglog(f[1:ifx], red_fitted[1:ifx], c=cmap[i], alpha=1, lw=1)
    ax[1].loglog(f[1:ifx], Pyy[1:ifx], c='tab:cyan', label='Precipitation', lw=3)
    ax[1].loglog(f[1:ifx], h0_psd(f, acy, Pyy)[1:ifx], color='black', label=f'red h0 ac={acy:.2f}', lw=2, ls='--')
    norm = mpl.colors.Normalize(vmin=1, vmax=offset)
    sm = mpl.cm.ScalarMappable(cmap='plasma_r', norm=norm)
    plt.colorbar(sm, ax=ax[1], label='Removed frequencies')
    xticks = [0.01, 0.05, 0.1, 0.5]
    xlabels = [f'{1 / x:.0f}' for x in xticks]
    ax[1].set_xscale('log')
    #ax[1].set_yscale('log')
    ax[1].set_xticks(xticks, labels=xlabels)
    ax[1].tick_params(axis='both', which='both', direction='in')
    ax[1].set_xlabel('Period (years)')
    ax[1].grid(which='both', axis='both', ls=':')
    ax[1].legend()
    fig.show()
    plt.savefig(Path(ROOT, f'notebooks/lmr-experiment/figures/fig2_{rgiid}.svg'))
