# -*- coding: utf-8 -*-
"""
concat_cmip_points.py

Description.

Author: drotto
Created: 12/12/2022 @ 3:08 PM
Project: glacier-attribution
"""

import logging
import re
from itertools import product
from pathlib import Path

import multiprocess as mp
import xarray as xr

from config import ROOT

logging.basicConfig(level=logging.DEBUG)

# %%

use_mp = False
mip = 5
# processes = mp.cpu_count() - 2
processes = 4


# %%


def mp_loop(rgiid):
    logging.debug(f"Working on {rgiid}")
    pattern = f"^{rgiid}"
    fps = [fp for fp in ps if re.match(pattern, fp.name)]
    identifiers = fps[0].name.split("_")[1:3]
    ds = xr.open_mfdataset(
        fps,
        compat='no_conflicts',
        engine='h5netcdf',
        coords="minimal",
        combine='nested',
        concat_dim='time',
        data_vars="minimal",
        chunks=dict(variant=1, model=1),
        parallel=True,
        decode_cf=False,
        use_cftime=True,
    )
    logging.debug("Successfully opened dataset.")
    p = Path(ROOT, f'features/gcm/cmip{mip}_{rgiid}_{"_".join(identifiers)}.nc')
    ds.to_netcdf(p, format='netcdf4', engine='h5netcdf')


# def mp_loop(rgiid):
#     pattern = f'^{rgiid}'
#     fps = [fp for fp in ps if re.match(pattern, fp.name)]
#     identifiers = fps[0].name.split('_')[1:3]
#     concat=[]
#     # for fp in fps:
#     #     concat.append(xr.open_dataset(fp))
#     #     print(fp)
#     #ds = xr.merge(concat)
#     # ds = xr.concat(concat, dim='rgiid', compat='no_conflicts', coords='minimal', data_vars='minimal')
#     #ds = xr.combine_nested(concat, concat_dim='rgiid', compat='no_conflicts', data_vars='minimal', coords='minimal')
#     print('Meeeeeeeeeeeeeeeerrrrrrrrrrrgggeeedd.')
#     p = Path(ROOT, f'features/gcm/cmip{mip}_{rgiid}_{"_".join(identifiers)}.nc')
#     ds.to_netcdf(p)
#     print(p.name)


# %%
p = Path(ROOT, f"data/interim/gcm/cmip{mip}")
ps = list(p.iterdir())
fnames = [fp.name for fp in ps]
rgiids = list(set([fname.split("_")[0] for fname in fnames]))

if __name__ == "__main__":
    if use_mp:
        with mp.Pool(processes=processes) as pool:
            pool.starmap(mp_loop, product(rgiids))
    else:
        for rgiid in rgiids:
            mp_loop(rgiid)

# %% For debugging
# 
# rgiid = rgiids[0]
# concat = []
# logging.debug(f"Working on {rgiid}")
# pattern = f"^{rgiid}"
# fps = [fp for fp in ps if re.match(pattern, fp.name)]
# for fp in fps[0:10]:
#     ds = xr.open_dataset(fp,
#         engine='h5netcdf',
#         chunks=dict(variant=1, model=1),
#         decode_cf=False,
#         use_cftime=True,)
#     concat.append(ds)
# 
# test = xr.combine_by_coords(concat, compat='no_conflicts', coords="minimal",
#         data_vars="minimal", combine_attrs='drop_conflicts')
# # test = xr.concat(concat, dim='time', coords='all')
