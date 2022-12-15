# -*- coding: utf-8 -*-
"""
concat_cmip_points.py

Description.

Author: drotto
Created: 12/12/2022 @ 3:08 PM
Project: glacier-attribution
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocess as mp
from itertools import product

from config import ROOT

#%%

mip=6

def mp_loop(rgiid):
    pattern = f'^{rgiid}'
    fps = [fp for fp in ps if re.match(pattern, fp.name)]
    identifiers = fps[0].name.split('_')[1:3]
    ds = xr.open_mfdataset(fps)
    p = Path(ROOT, f'features/gcm/cmip{mip}_{rgiid}_{"_".join(identifiers)}.nc')
    ds.to_netcdf(p)
    print(p.name)

p = Path(ROOT, f'data/interim/gcm/cmip{mip}')
ps = list(p.iterdir())
fnames = [fp.name for fp in ps]
rgiids = list(set([fname.split('_')[0] for fname in fnames]))

if __name__ == '__main__':
    with mp.Pool(processes=mp.cpu_count() - 2) as pool:
        pool.starmap(mp_loop, product(rgiids))
