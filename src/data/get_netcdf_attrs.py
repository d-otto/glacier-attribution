# -*- coding: utf-8 -*-
"""
get_netcdf_attrs.py

Description.

Author: drotto
Created: 12/20/2022 @ 10:22 AM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from pathlib import Path
from tqdm import tqdm

#%%
concat = []
mips = [5, 6]
for mip in mips:
    p = Path(f'C:\sandbox\glacier-attribution\data\external\gcm\cmip{mip}').glob('**/*.nc')
    ps = list(p)
    lm = []
    nat = []
    anth = []
    #pexp_ids = []
    for fp in tqdm(ps):
        with nc.Dataset(fp, "r", format="NETCDF4") as ds:
            attrs = ds.__dict__.copy()
            try:
                # print(fp)
                # print(ds.parent_experiment_id, ds.parent_experiment_rip)
                # print(ds.experiment_id, ds.experiment, '\n')
                # pexp_ids.append(ds.parent_experiment_id)
                if (attrs.get('parent_experiment_id', False) not in ['esmControl', 'piControl']) or (attrs.get('experiment_id', False) in ['past1000', 'past2k']):
            
                    attrs['collection'] = 'lm'
                    lm.append(attrs)
                elif (attrs.get('experiment_id') in ['hist-nat', 'historicalNat']):
            
                    attrs['collection'] = 'nat'
                    nat.append(attrs)
                else:
            
                    attrs['collection'] = 'anth'
                    anth.append(attrs)
            except:
                raise BaseException
       
    # pexp_ids = list(set(pexp_ids))
    # print(pexp_ids)
    
    df = pd.concat([pd.DataFrame(grp) for grp in [lm, nat, anth]], ignore_index=True)
    attr_cols = [col for col in df.columns if col not in ['tracking_id', 'creation_date', 'history']]
    df = df.drop_duplicates(subset=attr_cols)
    df.to_csv(Path(rf'C:\sandbox\glacier-attribution\data\external\gcm\cmip{mip}_catalog_raw.csv'), index=False)

