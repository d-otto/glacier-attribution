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
from tqdm import tqdm

import multiprocess as mp
import xarray as xr

from config import ROOT

logging.basicConfig(level=logging.DEBUG)

# %%

use_mp = False
mip = 6
#processes = mp.cpu_count() - 2
processes = 4


# %%


def mp_loop(rgiid):
    logging.debug(f"Working on {rgiid}")
    pattern = f"^{rgiid}"
    fps = [fp for fp in ps if re.match(pattern, fp.name)]
    identifiers = fps[0].name.split("_")[1:3]
    # note: this is very picky!
    ds = xr.open_mfdataset(
        fps,
        #engine='h5netcdf',
        compat='no_conflicts',
        coords="minimal",
        combine='nested',
        concat_dim=None,
        data_vars="minimal",
        chunks=dict(variant=1, model=1, time=175),
        parallel=True,
        decode_cf=False,
    )
    logging.debug("Successfully opened dataset.")
    p = Path(ROOT, f'features/gcm/cmip{mip}_{rgiid}_{"_".join(identifiers)}.nc')
    write_job = ds.to_netcdf(p, compute=False)
    write_job.compute()
    # ds.to_netcdf(p, format='netcdf4', engine='h5netcdf', unlimited_dims='time')
    logging.debug(f"Successfully wrote to {p.name}")
    ds.close()
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

def alternate_mp_loop(rgiid):
    logging.debug(f"Working on {rgiid}")
    pattern = f"^{rgiid}"
    fps = [fp for fp in ps if re.match(pattern, fp.name)]
    identifiers = fps[0].name.split("_")[1:3]
    # note: this is very picky!
    d = xr.open_dataset(fps[0], decode_cf=False, chunks=dict(variant=1, model=1, time=175))
    for fp in tqdm(fps[1:], ascii=True):
        ds = xr.open_dataset(fp, decode_cf=False, chunks=dict(variant=1, model=1, time=175))
        d = xr.merge([d, ds], compat='no_conflicts', combine_attrs='drop_conflicts')
    logging.debug("Successfully opened dataset.")
    p = Path(ROOT, f'features/gcm/cmip{mip}_{rgiid}_{"_".join(identifiers)}.nc')
    write_job = d.to_netcdf(p, compute=False)
    write_job.compute()
    # ds.to_netcdf(p, format='netcdf4', engine='h5netcdf', unlimited_dims='time')
    logging.debug(f"Successfully wrote to {p.name}")
    
def alternate_mp_loop2(rgiid):
    logging.debug(f"Working on {rgiid}")
    pattern = f"^{rgiid}"
    fps = [fp for fp in ps if re.match(pattern, fp.name)]
    identifiers = fps[0].name.split("_")[1:3]
    # note: this is very picky!
    d = fps.copy()
    i = 0
    with tqdm(total=(len(d)-1), ascii=True) as pbar:
        while len(d) > 1:    
            ds0 = d.pop(0)
            ds1 = d.pop(0)
            # print(i)
            # print(ds0)
            if isinstance(ds0, Path):
                ds0 = xr.open_dataset(ds0, decode_cf=False, chunks=dict(variant=1, model=1, time=175))
            if isinstance(ds1, Path):
                ds1 = xr.open_dataset(ds1, decode_cf=False, chunks=dict(variant=1, model=1, time=175))
            ds = xr.merge([ds0, ds1], compat='no_conflicts', combine_attrs='drop_conflicts')
            d.append(ds)
            i += 1
            pbar.update(1)
    d = d[0]
    logging.debug("Successfully opened dataset.")
    p = Path(ROOT, f'features/gcm/cmip{mip}_{rgiid}_{"_".join(identifiers)}.nc')
    write_job = d.to_netcdf(p, compute=False)
    write_job.compute()
    # ds.to_netcdf(p, format='netcdf4', engine='h5netcdf', unlimited_dims='time')
    logging.debug(f"Successfully wrote to {p.name}")

# %%
p = Path(ROOT, f"data/interim/gcm/cmip{mip}")
ps = list(p.iterdir())
fnames = [fp.name for fp in ps]
rgiids = list(set([fname.split("_")[0] for fname in fnames]))

if __name__ == "__main__":
    if use_mp:
        with mp.Pool(processes=processes) as pool:
            pool.starmap(alternate_mp_loop2, product(rgiids))
    else:
        for rgiid in rgiids:
            alternate_mp_loop2(rgiid)

# %% For debugging

# rgiid = rgiids[0]
# concat = []
# logging.debug(f"Working on {rgiid}")
# pattern = f"^{rgiid}"
# fps = [fp for fp in ps if re.match(pattern, fp.name)]
# for fp in fps:
#     ds = xr.open_dataset(fp,
#         engine='h5netcdf',
#         chunks=dict(variant=1, model=1),
#         decode_cf=False,
#         use_cftime=True,)
#     concat.append(ds)
# 
# #test = xr.concat(concat, dim='time', coords='all')
# 
# 
# # works!!!
# test = xr.combine_nested(concat, concat_dim=None, compat='no_conflicts')
# 
# # test2 = xr.combine_by_coords(concat, compat='no_conflicts', coords="all",
# #          data_vars="minimal", combine_attrs='drop_conflicts', join='override')