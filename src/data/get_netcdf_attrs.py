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
import re

#%%
concat = []
mips = [5, 6]
for mip in mips:
    p = Path(f'C:\sandbox\glacier-attribution\data\external\gcm\cmip{mip}').glob('**/*.nc')
    ps = list(p)
    concat = []
    for fp in tqdm(ps, ascii=True):
        with nc.Dataset(fp, "r", format="NETCDF4") as ds:
            attrs = ds.__dict__.copy()

            
            
            if mip==6:
                res = re.findall("([A-Za-z]+)(\d+)", attrs["variant_label"])
                variant_components = dict(res)
                
                # sort into collections
                if attrs["experiment_id"] == "hist-nat":
                    attrs['collection'] = 'nat'
                elif attrs["experiment_id"] in ["past1000", "past2k"]:
                    attrs['collection'] = 'lm'
                    
                elif (attrs["experiment_id"] == "historical") & (
                        variant_components["i"] in ["1000", "2000"]
                ):
                    attrs['collection'] = 'lm'
                else:
                    attrs['collection'] = 'anth'
                concat.append(attrs)
                
                    
            elif mip==5:
                # sort into collections
                if attrs["experiment_id"] == "historicalNat":
                    attrs['collection'] = 'nat'
                elif attrs["experiment_id"] in ["past1000", "past2k"]:
                    attrs['collection'] = 'lm'
                else:
                    attrs['collection'] = 'anth'
                concat.append(attrs)
            
       
    # pexp_ids = list(set(pexp_ids))
    # print(pexp_ids)
    if mip == 5:
        cols = ['model_id', 'experiment_id', 'table_id', 'initialization_method', 'physics_version', 'realization']
    elif mip == 6:
        cols = ['source_id', 'experiment_id', 'table_id', 'variable_id', 'variant_label']
    
    df = pd.DataFrame(concat)
    df = df.drop_duplicates(subset=cols, ignore_index=True)
    df.to_csv(Path(rf'C:\sandbox\glacier-attribution\data\external\gcm\cmip{mip}_catalog_raw.csv'), index=False)

