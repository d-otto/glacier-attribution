# -*- coding: utf-8 -*-
"""
detrend_gcm_pandas.py

Description.

Author: drotto
Created: 11/9/2022 @ 3:52 PM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from src.data import get_gwi, get_cmip6
from src.util import trunc
from src import data


#%%

gwi = get_gwi(freq='year')
gcms = data.get_cmip6(by='model', experiment=['historical'], freq='jjas')

concat = []
for model_name, ds in gcms.items():
    gcm = pd.DataFrame()
    gcm['wolverine'] = ds.interp(lat=60.423013, lon=360-148.904257, method='linear')['tas'].to_pandas()
    gcm['south_cascade'] = ds.interp(lat=48.359400, lon=360-121.058315, method='linear')['tas'].to_pandas()
    gcm['hintereisferner'] = ds.interp(lat=46.80048417779455, lon=10.766752530692314, method='linear')['tas'].to_pandas()
    gcm['argentiere'] = ds.interp(lat=45.95162293590404, lon=6.992154149538188, method='linear')['tas'].to_pandas()
    gcm['gcm'] = model_name
    gcm = gcm.reset_index()
    gcm = gcm.rename(columns={'time': 'year'})
    concat.append(gcm)
gcm = pd.concat(concat, ignore_index=True)


#%%

# def detrend_gcm(gcm, trend):
glacier_names = ['wolverine', 'south_cascade', 'hintereisferner', 'argentiere']
for glacier_name in glacier_names:
    d = gcm.groupby('gcm')
    for gcm_name, group in d:
        group = group.set_index('year')
        trend = gwi.copy()
        trend = pd.Series(np.interp(group.index, trend.index, trend), index=group.index)  # match the indexes and lengths
        group = group[glacier_name]
        
        fitting_yr0 = 1750
        group_trimmed = group.loc[fitting_yr0:]
        trend_trimmed = trend.loc[fitting_yr0:]
        
        X = trend_trimmed.copy()
        X.name = 'gwi'  # needed to get plot_regress_exog to work
        X = sm.add_constant(X)
        y = group_trimmed.copy()
        res = sm.OLS(y, X).fit()
        print(res.summary())
        
        
        # b * gwi = temp trend to remove from gcm
        bx = res.params[-1] * X.gwi
        #y_nat = y - bx
        y_nat = group
        y_nat[X.index] = y_nat[X.index] - res.predict(X) + res.params[0]
        
        
        y_yr = y.groupby(np.floor).mean()
        ref_T = y.loc[(y.index >= 1850) & (y.index < 1900)].mean()
        y_sm = y.rolling(20, center=True).mean()
        y_nat_sm = y_nat.rolling(20, center=True).mean()
        
        sns.set_theme(style='whitegrid', palette='bright', color_codes=True)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
        
        sns.lineplot(x=X.index, y=X.gwi, ax=ax, label='Anthropogenic contribution', color='y')
        sns.lineplot(x=y_sm.index, y=y_sm-ref_T, ax=ax, label='COMB (20yr MA)', color='r')
        sns.lineplot(x=y_nat_sm.index, y=y_nat_sm-ref_T, ax=ax, label='NAT (20yr MA)', color='b')
        sns.lineplot(x=y.index, y=y - ref_T, ax=ax, color='r', alpha=0.5, lw=0.25)
        sns.lineplot(x=y_nat.index, y=y_nat - ref_T, ax=ax, color='b', alpha=0.5, lw=0.25)
        
        ax.annotate(r'$\Delta{T}_{max}=$' + f'{bx.iloc[-1]:.3f} degC ' + r'$\beta=$' + f'{res.params[-1]:.3f} degC $SNR=$ {res.params[-1]/y.std():.3f}',
                    xy=(0, 0), xytext=(12, 12), xycoords='axes fraction', textcoords='offset points',
                    fontsize=12, fontweight='bold')
        #ax.set_xlim(850, 2015)
        ax.set_xlim(1850, 2015)
        ax.set_ylim(-2, 2)
        ax.set_title('Detrending GCM with GWI ($T_{ref}$ = 1850-1900) [JJAS]', loc='left', fontweight='bold')
        ax.set_title(f'{glacier_name.title()}, {gcm_name}', loc='right')
        ax.set_ylabel(r'$T_{anom}$ [$\degree C$ ]')
        ax.set_xlabel('Year')
        sns.move_legend(ax, loc='upper left')
        plt.tight_layout()
        plt.savefig(f'plots/gcm.detrended_gwi.{glacier_name}.{gcm_name}.png')

        #ax.set_xlim(850, 2015)
        #plt.savefig(f'plots/gcm_detrended.{glacier_name}.{gcm_name}.long.png')
        fig.show()


#%% detrend ensemble mean
# todo: give the regression all of the data and take the mean after... easiest way to get the right confidence interval for the trends?

for glacier_name in glacier_names:
    d = gcm.groupby('year').mean()

    
    trend = gwi.copy()
    trend = pd.Series(np.interp(d.index, trend.index, trend),
                      index=d.index)  # match the indexes and lengths
    d = d[glacier_name]

    fitting_yr0 = 1750
    d_trimmed = d.loc[fitting_yr0:]
    trend_trimmed = trend.loc[fitting_yr0:]

    X = trend_trimmed.copy()
    X.name = 'gwi'  # needed to get plot_regress_exog to work
    X = sm.add_constant(X)
    y = d_trimmed.copy()
    res = sm.OLS(y, X).fit()
    print(res.summary())

    # b * gwi = temp trend to remove from gcm
    bx = res.params[-1] * X.gwi
    # y_nat = y - bx
    y_nat = d
    y_nat[X.index] = y_nat[X.index] - res.predict(X) + res.params[0]

    y_yr = y.groupby(np.floor).mean()
    ref_T = y.loc[(y.index >= 1850) & (y.index < 1900)].mean()
    y_sm = y.rolling(20, center=True).mean()
    y_nat_sm = y_nat.rolling(20, center=True).mean()

    sns.set_theme(style='whitegrid', palette='bright', color_codes=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)

    sns.lineplot(x=X.index, y=X.gwi, ax=ax, label='Anthropogenic contribution', color='y')
    sns.lineplot(x=y_sm.index, y=y_sm - ref_T, ax=ax, label='COMB (20yr MA)', color='r')
    sns.lineplot(x=y_nat_sm.index, y=y_nat_sm - ref_T, ax=ax, label='NAT (20yr MA)', color='b')
    sns.lineplot(x=y.index, y=y - ref_T, ax=ax, color='r', alpha=0.5, lw=0.25)
    sns.lineplot(x=y_nat.index, y=y_nat - ref_T, ax=ax, color='b', alpha=0.5, lw=0.25)

    ax.annotate(
        r'$\Delta{T}_{max}=$' + f'{bx.iloc[-1]:.3f} degC ' + r'$\beta=$' + f'{res.params[-1]:.3f} degC $SNR=$ {res.params[-1] / y.std():.3f}',
        xy=(0, 0), xytext=(12, 12), xycoords='axes fraction', textcoords='offset points',
        fontsize=12, fontweight='bold')
    # ax.set_xlim(850, 2015)
    ax.set_xlim(1850, 2015)
    ax.set_ylim(-2, 2)
    ax.set_title('Detrending GCM with GWI ($T_{ref}$ = 1850-1900) [JJAS]', loc='left', fontweight='bold')
    ax.set_title(f'{glacier_name.title()}, Ensemble', loc='right')
    ax.set_ylabel(r'$T_{anom}$ [$\degree C$ ]')
    ax.set_xlabel('Year')
    sns.move_legend(ax, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/gcm.detrended_gwi.{glacier_name}.ensemble.png')

    # ax.set_xlim(850, 2015)
    # plt.savefig(f'plots/gcm_detrended.{glacier_name}.{gcm_name}.long.png')
    fig.show()