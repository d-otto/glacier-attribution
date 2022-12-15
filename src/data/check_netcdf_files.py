# -*- coding: utf-8 -*-
"""
check_netcdf_files.py

Description.

Author: drotto
Created: 12/13/2022 @ 2:29 PM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import netCDF4 as nc
import re
from pathlib import Path

from config import ROOT

#%%

dirname = Path(r'C:\Users\drotto\OneDrive - UW\gcm\cmip5')
fps = list(dirname.iterdir())
fps = [fp for fp in fps if fp.suffix == '.nc']

for fp in fps:
    print(fp)
    try:
        xr.open_dataset(fp)
    except:
        try:
            xr.open_dataset(fp, engine='netcdf4')
        except:
            print('failed with backend netcdf4')
            try:
                xr.open_dataset(fp, engine='h5netcdf')
            except:
                print('failed with backend h5netcdf')
                try:
                    nc.Dataset(fp, 'r', format='NETCDF4')
                except:
                    print('failed with format NETCDF4')
                    try:
                        nc.Dataset(fp, 'r', format='NETCDF4_CLASSIC')
                    except:
                        print('failed with format NETCDF4_CLASSIC')
                        try:
                            nc.Dataset(fp, 'r', format='NETCDF3_64BIT_OFFSET')
                        except:
                            print('failed with format NETCDF3_64BIT_OFFSET')
                            try:
                                nc.Dataset(fp, 'r', format='NETCDF3_64BIT_DATA')
                            except:
                                print('failed with format NETCDF3_64BIT_DATA')
                                try:
                                    nc.Dataset(fp, 'r', format='NETCDF3_CLASSIC')
                                except:
                                    print('failed with format NETCDF3_CLASSIC')
                                    pass