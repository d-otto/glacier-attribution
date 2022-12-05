# -*- coding: utf-8 -*-
"""
explore_gcm.py

Description.

Author: drotto
Created: 11/22/2022 @ 10:44 AM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from src import data


#%%

gwi = data.get_gwi(freq='year')
gcms = data.get_cmip6(by='model', experiment=['past1000'], freq='jjas')
gcms_historical = data.get_cmip6(by='model', experiment=['historical'], freq='jjas')


#%%
concat = []
for model_name, ds in gcms_historical.items():
    gcm = pd.DataFrame()
    gcm['wolverine'] = ds.interp(lat=60.423013, lon=360-148.904257, method='linear')['tas'].to_pandas()
    gcm['south_cascade'] = ds.interp(lat=48.359400, lon=360-121.058315, method='linear')['tas'].to_pandas()
    gcm['hintereisferner'] = ds.interp(lat=46.80048417779455, lon=10.766752530692314, method='linear')['tas'].to_pandas()
    gcm['argentiere'] = ds.interp(lat=45.95162293590404, lon=6.992154149538188, method='linear')['tas'].to_pandas()
    gcm['gcm'] = model_name
    gcm = gcm.reset_index()
    gcm = gcm.rename(columns={'time': 'year'})
    concat.append(gcm)
gcm_historical = pd.concat(concat, ignore_index=True)


#%%

dp = pd.melt(gcm_historical, id_vars=['year', 'gcm'], value_name='tas', var_name='glacier')
dp['tas'] = dp['tas'] - 273.15

for glacier, g in dp.groupby('glacier'):
    sns.set_theme(style='whitegrid', palette='bright', color_codes=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
    ax.axhline(y=0, color='#555', lw=2)
    
    
    ref_mean = g.loc[(g.year >= 1850) & (g.year < 1900), 'tas'].mean()
    g['tas'] = g['tas'] - ref_mean
    
    sns.lineplot(g, x='year', y='tas', lw=1, c='#999', units='gcm', estimator=None, alpha=0.5)
    # sns.lineplot(g, x='year', y='tas', lw=0,
    #              errorbar=('sd', 2), err_style='band', err_kws=dict(fc='red', alpha=0.15, ec='none'))
        
    
    sample = g.groupby('year')
    sm = sample.mean()
    sstd = sample.std()
    err_max = sm + sstd*2
    err_min = sm - sstd*2
    sns.lineplot(sm, x='year', y=sm.tas, label='Ensemble mean', c='black', lw=2)
    ax.fill_between(sm.index, err_min.tas, err_max.tas, fc='red', alpha=0.15, ec='none', label='2x standard deviation')
    
    ax.set_xlim(1850, 2014)
    ax.set_ylim()
    ax.set_title('Model spread $T_{ref}=$' + f'{ref_mean:.2f}' + '$\degree C$ (1850-1900, [JJAS]', loc='left', fontweight='bold')
    ax.set_title(f'{glacier.title()}', loc='right')
    ax.set_ylabel(r'$T_{anom} \degree C$')
    ax.set_xlabel('Year')
    ax.legend()
    sns.move_legend(ax, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/gcm.{glacier}.model_spread.unadjusted.png')
    
    #fig.show()
    
    
#%% same plot but with each gcm corrected to ref temp

dp = pd.melt(gcm_historical, id_vars=['year', 'gcm'], value_name='tas', var_name='glacier')
dp['tas'] = dp['tas'] - 273.15

for glacier, g in dp.groupby('glacier'):
    sns.set_theme(style='whitegrid', palette='bright', color_codes=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
    ax.axhline(y=0, color='#555', lw=2)

    ref_mean = g.loc[(g.year >= 1850) & (g.year < 1900), 'tas'].mean()
    concat = []
    for _, gg in g.groupby('gcm'):
        gg['tas'] = gg['tas'] - gg.loc[(gg.year >= 1850) & (gg.year < 1900), 'tas'].mean()
        concat.append(gg)
    g = pd.concat(concat, ignore_index=True)
    
    sns.lineplot(g, x='year', y='tas', lw=1, c='#999', units='gcm', estimator=None, alpha=0.5)

    sample = g.groupby('year')
    sm = sample.mean()
    sstd = sample.std()
    err_max = sm + sstd * 2
    err_min = sm - sstd * 2
    sns.lineplot(sm, x='year', y=sm.tas, label='Ensemble mean', c='black', lw=2)
    ax.fill_between(sm.index, err_min.tas, err_max.tas, fc='red', alpha=0.15, ec='none', label='2x standard deviation')
    
    
    ax.set_xlim(1850, 2014)
    ax.set_ylim()
    ax.set_title('Model spread relative to model $T_{ref}$ (1850-1900), [JJAS]', loc='left',
                 fontweight='bold')
    ax.set_title(f'{glacier.title()}', loc='right')
    ax.set_ylabel(r'$T_{anom} \degree C$')
    ax.set_xlabel('Year')
    ax.legend()
    sns.move_legend(ax, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/gcm.{glacier}.model_spread.Tref.png')

    #fig.show()



# %% attempt to show models color coded

dp = pd.melt(gcm_historical, id_vars=['year', 'gcm'], value_name='tas', var_name='glacier')
dp['tas'] = dp['tas'] - 273.15

for glacier, g in dp.groupby('glacier'):
    
    ref_mean = g.loc[(g.year >= 1850) & (g.year < 1900), 'tas'].mean()
    gcm_tref = g.groupby('gcm').apply(lambda gg: gg.loc[(gg.year >= 1850) & (gg.year < 1900), 'tas'].mean())
    
    concat = []
    for _, gg in g.groupby('gcm'):
        gg['tas'] = gg['tas'] - gg.loc[(gg.year >= 1850) & (gg.year < 1900), 'tas'].mean()
        concat.append(gg)
    g = pd.concat(concat, ignore_index=True)

    sample = g.groupby('year')
    sm = sample.mean()
    sstd = sample.std()
    err_max = sm + sstd * 2
    err_min = sm - sstd * 2
    
    for gcm, gg in g.groupby('gcm'):
        sns.set_theme(style='whitegrid', palette='bright', color_codes=True)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
        ax.axhline(y=0, color='#555', lw=2)
    
        sns.lineplot(g, x='year', y='tas', lw=1, c='#999', units='gcm', estimator=None, alpha=0.25)
        sns.lineplot(sm, x='year', y=sm.tas, label='Ensemble mean', c='black', lw=2, alpha=0.75)
        sns.lineplot(gg, x='year', y=gg.tas, label=gcm, c='tab:red', lw=1, alpha=0.5)
        sns.lineplot(gg, x='year', y=gg['tas'].rolling(20, center=True).mean(), label=f'20yr MA', c='tab:red', lw=2)
    
        ax.set_xlim(1850, 2014)
        ax.set_ylim()
        ax.set_title(f'Relative model spread ' + ' $T_{ref}=$' + f'{gcm_tref[gcm]:.2f} (1850-1900), [JJAS]', loc='left',
                     fontweight='bold')
        ax.set_title(f'{glacier.title()}', loc='right')
        ax.set_ylabel(r'$T_{anom} \degree C$')
        ax.set_xlabel('Year')
        ax.legend()
        sns.move_legend(ax, loc='upper left')
        plt.tight_layout()
        plt.savefig(f'plots/gcm.{glacier}.{gcm}.png')
    
        #fig.show()
        
        

# %% compare the moving average of all the models

sns.set_theme(style='whitegrid')

dp = pd.melt(gcm_historical, id_vars=['year', 'gcm'], value_name='tas', var_name='glacier')
dp['tas'] = dp['tas'] - 273.15
dp = dp.loc[(dp.year >= 1850) & (dp.year < 2015), :]

for glacier, g in dp.groupby('glacier'):
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=200)
    ax.axhline(y=0, color='#555', lw=2)

    ref_mean = g.loc[(g.year >= 1850) & (g.year < 1900), 'tas'].mean()
    concat = []
    for _, gg in g.groupby('gcm'):
        gg['tas'] = gg['tas'] - gg.loc[(gg.year >= 1850) & (gg.year < 1900), 'tas'].mean()
        gg['tas'] = gg['tas'].rolling(20, center=True).mean()
        concat.append(gg)
    g = pd.concat(concat, ignore_index=True)

    sns.lineplot(g, x='year', y='tas', lw=1, hue='gcm', palette='tab20', estimator=None, alpha=0.75)

    sample = g.groupby('year')
    sm = sample.mean()
    sstd = sample.std()
    err_max = sm + sstd * 2
    err_min = sm - sstd * 2
    sns.lineplot(sm, x='year', y=sm.tas, label='Ensemble mean', c='black', lw=2)
    #ax.fill_between(sm.index, err_min.tas, err_max.tas, fc='red', alpha=0.15, ec='none', label='2x standard deviation')

    ax.set_xlim(1860, 2005)
    ax.set_ylim()
    ax.set_title('Model spread relative to model $T_{ref}$ (1850-1900), [20yr MA, JJAS]', loc='left',
                 fontweight='bold')
    ax.set_title(f'{glacier.title()}', loc='right')
    ax.set_ylabel(r'$T_{anom} \degree C$')
    ax.set_xlabel('Year')
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1.015))
    plt.tight_layout()
    plt.savefig(f'plots/gcm.{glacier}.model_spread.ma.png')

    #fig.show()
