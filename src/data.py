# -*- coding: utf-8 -*-
"""
data.py

Description.

Author: drotto
Created: 11/9/2022 @ 3:48 PM
Project: glacier-attribution
"""

import logging
import os
import re
import sqlite3 as sq
from pathlib import Path
from itertools import groupby

import cftime
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr

from config import ROOT
from src.util import unwrap_coords, unwrap_coord


################################################################################
def get_gwi(freq="year"):
    fpath = os.path.abspath(
        r"C:\sandbox\glacier-attribution\data\external\gwi\gwi_main.csv"
    )
    data = pd.read_csv(fpath, skiprows=5)
    data = data.dropna(how="all", axis=1)
    data = data.iloc[1:]
    data = data.apply(pd.to_numeric)
    data = data.rename(columns={data.columns[-1]: "predicted_warming"})
    data = data.set_index("Date")
    data = data.rename(columns={"Ant warm": "gwi"})
    gwi = data["gwi"]

    if freq == "year":
        gwi = gwi.groupby(np.floor).mean()
        gwi.index = gwi.index.astype(int)
    if freq == "month":
        pass

    return gwi


################################################################################
def read_gcm(
    dirname=r"data/gcm_casestudy",
    colnames=("year", "wolverine", "hintereisferner", "south_cascade", "argentiere"),
    freq="year",
):
    fnames = os.listdir(dirname)
    fpaths = [os.path.join(dirname, fname) for fname in fnames]

    data = []
    for i, fpath in enumerate(fpaths):
        # colnames = np.arange(0,5)
        d = pd.read_csv(fpath, delimiter="\t", names=colnames)
        # d['year'] = abs(d['year'] - 2021)
        d["year"] = d["year"] + 1021

        if freq == "year":
            d = d.set_index("year")
            d = d.groupby(np.floor).mean()
            d.index = d.index.astype(int)
            d = d.reset_index()
        if freq == "month":
            pass

        d["gcm"] = fnames[i].split(sep=".")[0]
        data.append(d)
    data = pd.concat(data, ignore_index=True)

    return data


################################################################################
def list_to_regex_or(l):
    if isinstance(l, list):
        l = "|".join(l)
    return f"({l})"


def get_cmip_facets(dirname, mip):
    if mip == 6:
        variant_facets = ['realization', 'initialization', 'forcing', 'physics']
    elif mip == 5:
        variant_facets = ['realization', 'initialization', 'physics']
    ps = dirname.glob("**/*.nc")
    
    ps = [cmip_fname_to_dict(p) for p in ps]
    if len(ps) == 0:
        raise ValueError('No files returned. Is the drive plugged in?')
    df = pd.DataFrame().from_dict(ps)
    variant_label = df['variant'].str.findall('\d+')
    variant_label = pd.DataFrame(variant_label.to_list(), columns=variant_facets).astype('int')
    df = pd.concat([df, variant_label], axis=1)

    return df


def get_cmip_paths(
    dirname,
    mip,
    model=None,
    variant=None,
    experiment=None,
    grouped=False,
    group_order=None,
):
    df = get_cmip_facets(dirname, mip)

    # mask1 = mask2 = mask3 = pd.DataFrame(False, index=df.index, columns=df.columns)
    # if model:
    #     if not isinstance(model, list):
    #         model = [model]
    #     mask1 = mask1.where(df.model.isin(model), True)
    # if variant:
    #     if not isinstance(variant, list):
    #         variant = [variant]
    #     mask2 = mask2.where(df.variant.isin(variant), True)
    # if experiment:
    #     if not isinstance(experiment, list):
    #         experiment = [experiment]
    #     mask3 = mask3.where(df.experiment.isin(experiment), True)
    #
    # df = df[mask1 & mask2 & mask3]

    if model:
        if not isinstance(model, list):
            model = [model]
        df = df.loc[df.model.isin(model), :]
    if variant:
        if not isinstance(variant, list):
            variant = [variant]
        df = df.loc[df.variant.isin(variant), :]
    if experiment:
        if not isinstance(experiment, list):
            experiment = [experiment]
        df = df.loc[df.experiment.isin(experiment), :]

    if grouped:
        if group_order is None:
            group_order = ["experiment", "model", "variant"]
        out = df.groupby(group_order)["local_path"].apply(list).tolist()
    else:
        out = df["local_path"].to_list()

    return out


################################################################################
def get_cmip6_run_attrs(dirname, by=None, model=None, variant=None, experiment=None):
    """

    Parameters
    ----------
    dirname
    by : str
        "model", "variant", "experiment"
    model : list
    variant : list
    experiment : list

    Returns
    -------
    list

    """

    segments = {"model": model, "experiment": experiment, "variant": variant}
    segments = {
        k: list_to_regex_or(v) if v is not None else ".+" for k, v in segments.items()
    }

    segment_map = {"model": 2, "experiment": 3, "variant": 4}

    pattern = re.compile(
        rf"tas_Amon_{segments['model']}_{segments['experiment']}_{segments['variant']}_.+\.nc"
    )

    p = Path(ROOT, dirname).glob("**/*.nc")
    fps = list(p)
    fps = [fp for fp in fps if re.match(pattern, fp.name)]
    if by is not None:
        segment_num = segment_map[by]
        res = [fp.name.split("_")[segment_num] for fp in fps]
        res = list(set(res))  # get unique
    else:
        res = list(set(fps))  # get unique

    if len(res) == 0:
        raise ValueError(f"No runs found with given pattern:\n{segments}\n{pattern}")

    return res


################################################################################
def extract_cmip_points(ds, glaciers, variable, freq, mip, use_cache=True):
    """
    da should already have all the identifying coordinates it needs. All it gets here is one for the RGIID
    """
    # todo : move file saving out of the function
    model = ds.model.item()
    experiment = ds.experiment.item()
    
    variant = ds.attrs['variant_label']
#    ds = ds.assign_attrs(dict(variant_label=ds.attrs.variant_label))
    for rgiid, glacier in glaciers.iterrows():
        p = Path(
            ROOT,
            f"data/interim/gcm/cmip{mip}/{rgiid}_{variable}_{freq}_{model}_{experiment}_{variant}.nc",
        )
        xy = glacier.geometry.centroid
        lon = xy.x
        lat = xy.y
        lon, lat = unwrap_coords(lon, lat)
        pt = ds.interp(lat=lat, lon=lon, method="linear")
        pt = pt.expand_dims("rgiid")
        pt = pt.assign_coords(dict(rgiid=("rgiid", [rgiid])))
        #pt.to_netcdf(p, format="netcdf4", engine="h5netcdf", unlimited_dims='time')
        pt.to_netcdf(p, format="netcdf4", engine="h5netcdf")
        
    return None


################################################################################
def read_cmip_model(fps=None, freq="jjas", mip=6):
    # todo: refactor to read_raw_cmip()
    """

    Parameters
    ----------
    dirname
    model list or regex pattern string
    experiment list or regex pattern string
    variant : list or regex pattern string
    freq : 'month' or 'year'

    Returns
    -------

    """

    # todo: could make this into a cleaning function then use open_mfdataset for loading
    def clean_ds(ds):
        ds = ds.drop_vars(
            ["time_bnds", "lat_bnds", "lon_bnds", "height", "lon_bounds", "lat_bounds", "average_DT"], errors="ignore"
        )
        ds = ds.drop_dims("bnds", errors="ignore")
        
        
        ds["time"] = xr.apply_ufunc(
            cftime.date2num,
            ds.time,
            kwargs={
                "units": "common_years since 0000-01-01",
                "calendar": "365_day",
                "has_year_zero": True,
            },
        )
        ds = ds.drop_duplicates("time")

        ds = ds.expand_dims('mip')
        ds = ds.expand_dims('model')
        ds = ds.expand_dims('experiment')
        ds = ds.expand_dims('collection')
        if mip == 6:
            ds = ds.assign_coords(dict(mip=("mip", [6])))
            ds = ds.assign_coords(dict(model=("model", [ds.attrs["source_id"]])))
            ds = ds.assign_coords(
                dict(experiment=("experiment", [ds.attrs["experiment_id"]]))
            )

            ds = ds.expand_dims('r')
            ds = ds.expand_dims('i')
            ds = ds.expand_dims('f')
            ds = ds.expand_dims('p')
            res = re.findall("([A-Za-z]+)(\d+)", ds.attrs["variant_label"])
            variant_components = dict(res)
            ds = ds.assign_coords(
                dict(r=("r", [variant_components['r']]),
                     i=("i", [variant_components['i']]),
                     f=("f", [variant_components['f']]),
                     p=("p", [variant_components['p']]),)
            )
            
            # sort into collections
            if ds.attrs["experiment_id"] == "hist-nat":
                ds = ds.assign_coords(dict(collection=("collection", ["nat"])))
            elif ds.attrs["experiment_id"] in ["past1000", "past2k"]:
                ds = ds.assign_coords(dict(collection=("collection", ["lm"])))
            elif (ds.attrs["experiment_id"] == "historical") & (
                variant_components["i"] in ["1000", "2000"]
            ):
                ds = ds.assign_coords(dict(collection=("collection", ["lm"])))
            else:
                ds = ds.assign_coords(dict(collection=("collection", ["anth"])))

        elif mip == 5:
            ds = ds.assign_coords(dict(mip=("mip", [5])))
            ds = ds.assign_coords(dict(model=("model", [ds.attrs["model_id"]])))
            ds = ds.assign_coords(
                dict(experiment=("experiment", [ds.attrs["experiment_id"]]))
            )

            ds = ds.expand_dims('r')
            ds = ds.expand_dims('i')
            ds = ds.expand_dims('f')
            ds = ds.expand_dims('p')
            variant_components = dict(r=("r", [ds.attrs['realization']]),
                 i=("i", [ds.attrs['initialization_method']]),
                 f=("f", [1]),
                 p=("p", [ds.attrs['physics_version']]), )
            ds = ds.assign_coords(variant_components)
            ds.attrs['variant_label'] = f"r{ds.attrs['realization']}i{ds.attrs['initialization_method']}f1p{ds.attrs['physics_version']}"

            # sort into collections
            if ds.attrs["experiment_id"] == "historicalNat":
                ds = ds.assign_coords(dict(collection=("collection", ["nat"])))
            elif ds.attrs["experiment_id"] in ["past1000", "past2k"]:
                ds = ds.assign_coords(dict(collection=("collection", ["lm"])))
            else:
                ds = ds.assign_coords(dict(collection=("collection", ["anth"])))

        try:
            ds = ds.rename({"latitude": "lat", "longitude": "lon"})
        except:
            pass
        
        return ds

    if isinstance(fps, list):
        # please don't touch this anymore
        d = xr.open_mfdataset(
            fps,
            compat="no_conflicts",
            coords="all",
            combine="nested",
            concat_dim=None,
            data_vars="minimal",
            #chunks=dict(lon=10, lat=10),
            chunks=dict(lon=180, lat=90),
            preprocess=clean_ds,
            parallel=False,  # using true tends to lead to too overflow of RAM w/ files being loaded in
            decode_cf=True,
            use_cftime=True,
        )
    else:
        try:
            logging.debug(fps)
            d = xr.open_dataset(
                fps,
                chunks=dict(lon=180, lat=90),
                #chunks=dict(lon=10, lat=10),
                decode_cf=True,
                use_cftime=True,
            )
            d = clean_ds(d)
        except:
            raise ValueError(f"Failed on {fps}")
        # .drop_vars(['time_bnds', 'lat_bnds', 'lon_bnds'])

    if freq == "year":
        # ds = ds.interp(time=np.arange(np.ceil(ds.time.min()), ds.time.max().round(), 1))
        d = d.groupby(np.floor(d.time)).mean()
    elif freq == "mon":
        pass
    elif freq == "jjas":
        mask = (d.time % 1 > 6 / 12) & (
            d.time % 1 < 9.75 / 12
        )  # hopefully this captures models which have their times in the center of the month
        d = d.where(mask)
        d = d.groupby(np.floor(d.time)).mean()

    d = d.drop_duplicates("time")  # redundant?
    return d


########################################################################################
def get_cmip6_lm():
    """
    Convenience function to pull past1k & p1k-initialized historical runs and concatenate them

    Returns
    -------

    """

    gcm_p1k = get_cmip6(by="model", experiment=["past1000", "past2k"], freq="jjas")
    p1k_models = gcm_p1k.model
    gcm_hist = get_cmip6(
        by="model", experiment=["historical"], model=p1k_models, freq="jjas"
    )
    gcm = {
        model: xr.concat([ds_p1k, gcm_hist[model]], dim="time")
        for model, ds_p1k in gcm_p1k.items()
        if model in gcm_hist.keys()
    }

    return gcm


################################################################################
def get_cmip6(
    dirname="data/external/gcm/cmip6",
    by=None,
    model=None,
    experiment=None,
    variant=None,
    pattern=None,
    freq="jjas",
    merged=True,
):
    """
    Wrapper for various retrival functions. This has variable output type depending on arguments.
    Parameters
    ----------
    dirname :
    by :
    model :
    experiment :
    variant :
    pattern :
    freq :
    merged :

    Returns
    -------

    """
    # This returns a dict instead of a merged xarray to preserve the attrs of each dataset.
    p = Path(ROOT, dirname)
    fps = list(p.iterdir())
    if len(fps) == 0:
        raise ValueError(f"No files to get in {dirname}")

    if merged:
        if pattern is None:
            pattern = _cmip6_pattern_from_attrs(model, experiment, variant)
        fps = [fp for fp in fps if re.match(pattern, fp.name)]
        # d = xr.open_mfdataset(fps, chunks={'lon': 2, 'lat': 2}, data_vars='minimal', parallel=True, combine_attrs='drop_conflicts', preprocess=partial(_clean_cmip6_file, freq=freq), engine='netcdf4', use_cftime=True, combine='by_coords', coords='minimal')
        d = read_cmip_model(fps, freq=freq)

        return d

    else:
        return _get_cmip6_by_attrs(dirname, freq, by, model, experiment, variant)


########################################################################################
def cmip_fname_to_dict(p):
    segment_map = ["variable", "frequency", "model", "experiment", "variant"]
    segments = p.name.split("_")
    facets = {segment_map[i]: segments[i] for i in range(0, 5)}
    facets["local_path"] = p

    return facets


################################################################################
def _get_cmip6_by_attrs(
    dirname, freq, by=None, model=None, experiment=None, variant=None
):
    """
    Split off for convenience from get_cmip6()

    Parameters
    ----------
    dirname :
    freq :
    by :
    model :
    experiment :
    variant :

    Returns
    -------

    """
    try:
        groups = get_cmip6_run_attrs(
            dirname, by=by, model=model, experiment=experiment, variant=variant
        )
    except:
        raise ValueError(
            f"No groups found with model={model}, experiment={experiment}, variant={variant}"
        )

    d = dict()
    for group in groups:
        segments = {"model": model, "experiment": experiment, "variant": variant}
        segments[by] = group
        pattern = _cmip6_pattern_from_attrs(**segments)
        fps = [fp for fp in fps if re.match(pattern, fp.name)]
        d[group] = read_cmip_model(fps=fps, freq=freq)

    return d


################################################################################
def _cmip6_pattern_from_attrs(model, experiment, variant):
    """
    Take model, expeirment, and variant and convert it to a regex string to get file names

    Parameters
    ----------
    model :
    experiment :
    variant :

    Returns
    -------

    """
    segments = {"model": model, "experiment": experiment, "variant": variant}
    segments = {
        k: list_to_regex_or(v) if v is not None else ".+" for k, v in segments.items()
    }
    pattern = re.compile(
        rf"tas_Amon_{segments['model']}_{segments['experiment']}_{segments['variant']}_.+\.nc"
    )

    return pattern


#################################################################################
def get_glacier_gcm(rgiids, variable, freq, mip=6, dirname=Path(ROOT, "features/gcm"), compute=False):
    fps = list(dirname.iterdir())

    if isinstance(rgiids, list):
        pattern = f"^cmip{mip}_{list_to_regex_or(rgiids)}_{variable}_{freq}"
        fps = [fp for fp in fps if re.match(pattern, fp.name)]
        ds = xr.open_mfdataset(fps)
        if compute:
            ds = ds.compute()
    else:
        rgiid = rgiids
        pattern = f"^cmip{mip}_{rgiid}_{variable}_{freq}"
        fp = [fp for fp in fps if re.match(pattern, fp.name)][
            0
        ]  # should only be one file
        ds = xr.open_dataset(fp)
    return ds


################################################################################
def sample_glaciers_from_gcms(gcms, glaciers):
    """

    sorry the inputs are convoluted

    Parameters
    ----------
    gcms : dict
        dict of the form 'model name': xr.Dataset
    glaciers : GeoSeries
        GeoSeries containing a subset of the full rgi dataset. Requirements for a row are the RGIId as index and the geometry.

    Returns
    -------
    xr.Dataset

    """

    concat = []
    for rgiid, glacier in glaciers.iterrows():
        xy = glacier.geometry.centroid
        lon = xy.x
        lat = xy.y
        lon, lat = unwrap_coords(lon, lat)
        concat_models = []
        for model_name, gcm in gcms.items():
            gcm = gcm.expand_dims("model")
            gcm = gcm.assign_coords(dict(model=("model", [model_name])))
            da = gcm.interp(lat=lat, lon=lon, method="linear")["tas"]

            concat_models.append(da)
        ds = xr.concat(concat_models, dim="model", coords="minimal")
        ds = ds.expand_dims("rgiid")
        ds = ds.assign_coords(dict(rgiid=("rgiid", [rgiid])))
        concat.append(ds)
    ds = xr.concat(concat, dim="rgiid")

    return ds




# def read_cmip6(dirname=r'data/', freq='year'):
#     p = Path(dirname)
#     fps = list(p.iterdir())
#
#     # open MIROC
#     fps_hist = [fp for fp in fps if fp.stem.startswith('tas_Amon_MIROC-ES2L_historical_r1i1000p1f2_gn_')]
#     fps_lm = [fp for fp in fps if fp.stem.startswith('tas_Amon_MIROC-ES2L_past1000_r1i1p1f2_gn_')]
#     fps = fps_hist + fps_lm
#     ds = []
#     for fp in fps:
#         d = xr.open_dataset(fp, use_cftime=True)
#         ds.append(d)
#     ds = xr.concat(ds, dim='time')
#     ds['time'] = xr.apply_ufunc(cftime.date2num, ds.time,
#                    kwargs={'units':'common_years since 0000-01-01', 'calendar':'365_day', 'has_year_zero':True})
#     ds = ds.drop_duplicates('time')
#     if freq=='year':
#         da = ds['tas'].interp(time=np.arange(np.ceil(ds.time.min()), ds.time.max().round(), 1))
#     else:
#         da = ds['tas']
#
#     return da

################################################################################

def sample_glaciers(ds, glaciers, as_dim=True, interp_method='linear'):
    # this could be way faster by passing the list of coords to ds.interp (if as_dim = false), or passing a dim: (value, coords) or dataarray (if as_dim=True)
    xys = glaciers.geometry.centroid
    if (ds.longitude < 0).any():
        x = xys.x
        y = xys.y
    else:
        x = [unwrap_coord(xx) for xx in xys.x]
        y = [unwrap_coord(yy) for yy in xys.y]

    lon = xr.DataArray(data=x,
                       coords=dict(rgiid=glaciers.index.values),
                       dims='rgiid')
    lat = xr.DataArray(data=y,
                       coords=dict(rgiid=glaciers.index.values),
                       dims='rgiid')
    pts = ds.interp(latitude=lat, longitude=lon, method=interp_method)
    if as_dim is False:
        # todo: should this be swap dims?
        pts = pts.expand_dims(['latitude', 'longitude'])  # restore dims after being squeezed by interp
    return pts

def sample_glaciers_old(ds, glaciers, as_dim=True, interp_method='linear'):
    # this could be way faster by passing the list of coords to ds.interp (if as_dim = false), or passing a dim: (value, coords) or dataarray (if as_dim=True)
    # could try clipping the ds then doing the interpolation
    concat = []
    for rgiid, glacier in glaciers.iterrows():
        xy = glacier.geometry.centroid
        lon = xy.x
        lat = xy.y
        pt = ds.interp(latitude=lat, longitude=lon, method=interp_method)
        pt = pt.assign_coords(dict(rgiid=("rgiid", [rgiid])))
        concat.append(pt)
    if as_dim:
        pts = xr.concat(concat, dim="rgiid")
    else:
        pts = xr.merge(concat, compat='no_conflicts')  # shouldn't be any duplicates
    if as_dim is False:
        pt = pt.expand_dims(['latitude', 'longitude'])  # restore dims after being squeezed by interp
    return pts

################################################################################
def get_hadcrut():
    """
    coords should be a dict of {rgiid: (lat, lon)}
    """

    fp = Path(ROOT, r'data\external\HadCRUT.5.0.1.0.analysis.summary_series.global.monthly.nc')
    ds = xr.open_dataset(fp, decode_coords="all", decode_cf=True)  # recommended by rio
    
    return ds
    # concat = []
    # for name, lonlat in coords.items():
    #     lon, lat = lonlat
    #     d = ds.interp(longitude=lon, latitude=lat)
    #     d = d["temperature"].to_dataframe()
    #     d["rgiid"] = name
    #     d = d.reset_index()
    #     concat.append(d)
    # df = pd.concat(concat, ignore_index=True)
    # return df


################################################################################
def get_berkearth(coords=None):
    """
    coords should be a dict of {rgiid: (lat, lon)}
    """

    fp = os.path.abspath(
        r"C:\Users\drotto\Documents\USGS\glacier_data\berkeley_earth\Complete_TAVG_LatLong1.nc"
    )
    ds = xr.open_dataset(fp, decode_coords="all")  # recommended by rio
    ds = ds.rio.write_crs('epsg:4326')

    # properly split time into months and years
    # https://stackoverflow.com/questions/43015638/xarray-reshape-data-split-dimension
    time = np.round(ds['time'], decimals=4)
    year = np.floor(time)
    month = time - year
    ind = pd.MultiIndex.from_arrays((year.values, month.values), names=('year', 'month'))
    ds = ds.assign_coords(time=ind).unstack('time')
    
    if coords:
        concat = []
        for name, lonlat in coords.items():
            lon, lat = lonlat
            d = ds.interp(longitude=lon, latitude=lat)
            d = d["temperature"].to_dataframe()
            d["rgiid"] = name
            d = d.reset_index()
            concat.append(d)
        df = pd.concat(concat, ignore_index=True)
        return df
    else:
        return ds

########################################################################################
def get_lmr():
    ps = [Path(ROOT, r'data\external\lmr\prate_MCruns_ensemble_mean_LMRv2.1.nc'),
          Path(ROOT, r'data\external\lmr\air_MCruns_ensemble_mean_LMRv2.1.nc')]
    ds = [xr.open_dataset(p, chunks=None, use_cftime=True) for p in ps]
    lmr = xr.merge(ds)
    lmr = lmr.rename(dict(lat='latitude', lon='longitude'))

    lmr["time"] = xr.apply_ufunc(
        cftime.date2num,
        lmr.time,
        kwargs={
            "units"        : "common_years since 0000-01-01",
            "calendar"     : "365_day",
            "has_year_zero": True,
        },
    )
    
    # 
    # # properly split time into months and years
    # # https://stackoverflow.com/questions/43015638/xarray-reshape-data-split-dimension
    # time = np.round(lmr['time'], decimals=4)
    # year = np.floor(time)
    # month = time - year
    # ind = pd.MultiIndex.from_arrays((year.values, month.values), names=('year', 'month'))
    # lmr = lmr.assign_coords(time=ind).unstack('time')

    return lmr


################################################################################
def get_rgi(rgiids, from_sqllite=False):
    if not isinstance(rgiids, list):
        rgiids = [rgiids]

    if from_sqllite:
        con = sq.connect(Path(ROOT, "data/interim/rgi.sqllite"))
        with con:  # context manager bullshit (https://blog.rtwilson.com/a-python-sqlite3-context-manager-gotcha/)
            query = (
                f"SELECT * FROM rgi WHERE RGIId in ({','.join(['?'] * len(rgiids))})"
            )
            rgi = pd.read_sql_query(query, con, params=rgiids)
        con.close()

    else:
        p = Path(r"C:\Users\drotto\Documents\USGS\glacier_data\rgi60")
        p = [x for x in p.iterdir() if x.is_dir()]
        p = [x for x in p if not x.name.startswith("00")]
        concat = []
        for pp in p:
            rgi = gpd.read_file(pp)
            rgi = rgi.loc[rgi.RGIId.isin(rgiids)]
            concat.append(rgi)
        rgi = pd.concat(concat, ignore_index=True)

    rgi["geometry"] = rgi.apply(lambda x: shapely.wkt.loads(x.geometry), axis=1)
    rgi = gpd.GeoDataFrame(rgi, crs='EPSG:4326')

    return rgi


########################################################################################
def rgi_to_sqllite():
    # warning, this is slow

    p = Path(r"C:\Users\drotto\Documents\USGS\glacier_data\rgi60")
    p = [x for x in p.iterdir() if x.is_dir()]
    p = [x for x in p if not x.name.startswith("00")]
    concat = []
    for pp in p:
        gdf = gpd.read_file(pp)
        concat.append(gdf)
    gdf = pd.concat(concat, ignore_index=True)
    # convert geometry object to well-known text (wkt)
    gdf["geometry"] = gdf.apply(lambda x: shapely.wkt.dumps(x.geometry), axis=1)

    conn = sq.connect(Path("data/interim/rgi.sqllite"))
    with conn:  # context manager bullshit (https://blog.rtwilson.com/a-python-sqlite3-context-manager-gotcha/)
        gdf.to_sql("rgi", conn, if_exists="replace", index=False)
    conn.close()


########################################################################################
# glacier length
def get_glacier_lengths():
    p = Path(ROOT, r"data\external\WGMS-FoG-2020-08-C-FRONT-VARIATION.csv")
    df = pd.read_csv(p)
    df["FRONT_VARIATION_cumulative"] = (
        df.sort_values(by="Year").groupby("NAME")["FRONT_VARIATION"].cumsum()
    )


