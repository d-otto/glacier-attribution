# -*- coding: utf-8 -*-
"""
glacier_map_view.py

Description.

Author: drotto
Created: 12/22/2022 @ 3:01 PM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from itertools import cycle, zip_longest, product
import xarray as xr
from datetime import datetime, timezone
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import pickle
import gzip
import rioxarray
from pyproj import Transformer
import salem
from shapely.geometry import LineString, Point

import oggm
from oggm import workflow
from oggm import tasks, utils
from oggm import graphics

import src.data as data
from src.flowline import get_flowline_geom

import config
from config import cfg, ROOT

#%%

rgiid = list({k: v for k, v in cfg['glaciers'].items() if v['name'] == 'South Cascade'}.keys())[0]
params = cfg['glaciers'][rgiid]['flowline_params']

#%%

rgi = data.get_rgi(rgiid, from_sqllite=True).iloc[0]
geom = get_flowline_geom(rgiid)

#%%

#oggm.cfg.initialize(file=r'\\wsl.localhost\Ubuntu\home\drotto\.oggm_config')
oggm.cfg.initialize_minimal()
oggm.cfg.PATHS['dl_cache_dir'] = Path(r'~/OGGM/download_cache')
oggm.cfg.PATHS['rgi_dir'] = Path(r'~/OGGM/rgi')
oggm.cfg.PATHS['test_dir'] = Path(r'~/OGGM/tests')
oggm.cfg.PATHS['tmp_dir'] = Path(r'~/OGGM/tmp')
oggm.cfg.PATHS['working_dir'] = Path(r'~/oggm_plot')
oggm.cfg.PARAMS['prcp_scaling_factor'] = 2.5
oggm.cfg.PARAMS['climate_qc_months'] = 3
oggm.cfg.PARAMS['use_winter_prcp_factor'] = False
oggm.cfg.PARAMS['min_mu_star'] = 50
oggm.cfg.PARAMS['use_multiprocessing'] = False
oggm.cfg.PARAMS['check_calib_params'] = False
oggm.cfg.PARAMS['use_rgi_area'] = True
oggm.cfg.PARAMS['use_tstar_calibration'] = True

rgi_ids = oggm.utils.get_rgi_glacier_entities(['RGI60-01.09162'])
#gdirs = oggm.workflow.init_glacier_directories(rgi_ids)
prepro_path = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/centerlines/w5e5/qc0/pcpwin/match_geod_pergla/'
gdirs = oggm.workflow.init_glacier_directories(rgi_ids, from_prepro_level=3, prepro_border=160, prepro_base_url=prepro_path)  # 160 is max border available rn
gdir = gdirs[0]

#%%

# workflow.execute_entity_task(tasks.glacier_masks, gdirs)  # needed to write geometries.pkl for plotting
# workflow.execute_entity_task(tasks.compute_centerlines, gdirs)  # needed to write geometries.pkl for plotting
# workflow.execute_entity_task(tasks.initialize_flowlines, gdirs)  # needed to write geometries.pkl for plotting
# workflow.execute_entity_task(tasks.catchment_area, gdirs)  # needed to write geometries.pkl for plotting
# workflow.execute_entity_task(tasks.catchment_width_geom, gdirs)  # needed to write geometries.pkl for plotting
# workflow.execute_entity_task(tasks.catchment_width_correction, gdirs)  # needed to write geometries.pkl for plotting
# workflow.execute_entity_task(tasks.compute_downstream_bedshape, gdirs)  # needed to write geometries.pkl for plotting

#%%

# outline = gdir.read_shapefile('outlines')
#             
# fp = gdir.get_filepath('inversion_flowlines')
# inversion_flowlines = []
# with gzip.open(fp, "rb") as openfile:
#     while True:
#         try:
#             inversion_flowlines.append(pickle.load(openfile))
#         except EOFError:
#             break
# inversion_flowlines = inversion_flowlines[0]
# lines = [flowline.line for flowline in inversion_flowlines]
# 
# 
# fp = gdir.get_filepath('downstream_line')
# downstream_line = []
# with gzip.open(fp, "rb") as openfile:
#     while True:
#         try:
#             downstream_line.append(pickle.load(openfile))
#         except EOFError:
#             break
# inversion_flowlines = inversion_flowlines[0]
# lines = [flowline.line for flowline in inversion_flowlines]

#%%
ds = salem.open_xr_dataset(gdir.get_filepath('gridded_data'))
topo = ds.topo

def cut(line, distance):
    if distance <= 0.0 :#line.length:
        return [None, LineString(line)]
    elif distance >= 1.0:
        return [LineString(line), None]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p), normalized=True)
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance, normalized=True)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]


def cut_piece(line,distance1, distance2):
    """ From a linestring, this cuts a piece of length lgth at distance.
    Needs cut(line,distance) func from above ;-) 
    """
    l1 = cut(line, distance1)[1]
    l2 = cut(line, distance2)[0]
    result = l1.intersection(l2)
    return result

#topo = topo.salem.subset(margin=-50)
#graphics.plot_domain(gdir, figsize=(8, 7))
fig, ax = plt.subplots(1, 1, figsize=(7, 4.66), dpi=600)
#grid = salem.mercator_grid(center_ll=(rgi.CenLon, rgi.CenLat), extent=(18000, 14000))
#grid = ds.salem.grid
#smap = salem.Map(grid, extent=)
#smap = ds.salem.get_map()


# custom plot_centerlines
# Plots the centerlines of a glacier directory.

add_downstream = True
use_flowlines = True
use_model_flowlines = False
lines_cmap = 'Set1'
add_line_index=False
from matplotlib import cm as colormap
OGGM_CMAPS = dict()
OGGM_CMAPS['terrain'] = colormap.terrain
OGGM_CMAPS['section_thickness'] = plt.cm.get_cmap('YlGnBu')
OGGM_CMAPS['glacier_thickness'] = plt.get_cmap('viridis')
OGGM_CMAPS['ice_velocity'] = plt.cm.get_cmap('Reds')

if add_downstream and not use_flowlines:
    raise ValueError('Downstream lines can be plotted with flowlines only')

# Files
filename = 'centerlines'
if use_model_flowlines:
    filename = 'model_flowlines'
elif use_flowlines:
    filename = 'inversion_flowlines'

gdir = gdirs[0]
# with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
#     topo = nc.variables['topo'][:]
    
topo = salem.open_xr_dataset(gdir.get_filepath('gridded_data'))['topo'].salem.subset(margin=-100)
smap = topo.salem.get_map()

cm = graphics.truncate_colormap(OGGM_CMAPS['terrain'], minval=0.25, maxval=1.0)
smap.set_plot_params(cmap=cm)
smap.set_data(topo)
smap.set_topography(topo=topo, relief_factor=1.2)

# Change the country borders
smap.set_shapefile(countries=True, color='C3', linewidths=2)

# Add oceans and lakes

#smap.set_shapefile(oceans=True)
smap.set_shapefile(rivers=True)
smap.set_shapefile(lakes=True, facecolor='blue', edgecolor='blue')
smap.set_lonlat_contours(xinterval=0.1, yinterval=0.05, colors='black', linestyles='solid', linewidths=1, alpha=0.5, zorder=0.5)
smap.set_scale_bar(location=(0.875, 0.025), length=2500, text_kwargs=dict(fontweight='bold'), add_bbox=False, bbox_dx=1.2, bbox_dy=0.75,
                   text_delta=(0.01, 0.015))
for gdir in gdirs:
    crs = gdir.grid.center_grid
    #crs = topo.salem.grid
    geom = gdir.read_pickle('geometries')

    # Plot boundaries
    poly_pix = geom['polygon_pix']

    smap.set_geometry(poly_pix, crs=crs, fc='white',
                      alpha=0.3, zorder=2, linewidth=.2)
    poly_pix = utils.tolist(poly_pix)
    for _poly in poly_pix:
        for l in _poly.interiors:
            smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

    # plot Centerlines
    cls = gdir.read_pickle(filename)

    # Go in reverse order for red always being the longest
    cls = cls[::-1]
    nl = len(cls)
    color = graphics.gencolor(len(cls) + 1, cmap=lines_cmap)
    for i, (l, c) in enumerate(zip(cls, color)):
        if i==0:
            if add_downstream and not gdir.is_tidewater and l is cls[0]:
                line = gdir.read_pickle('downstream_line')['full_line']
            else:
                line = l.line
            
            #line = cut_piece(line, 0, 1000)
            line = LineString(line.coords[:75])
    
            smap.set_geometry(line, crs=crs, color='red',
                              linewidth=2.5, zorder=50)
    
            text = '{}'.format(nl - i - 1) if add_line_index else None
            smap.set_geometry(l.head, crs=gdir.grid, marker='o',
                              markersize=60, alpha=0.8, color='red', zorder=99,
                              text=text)
    
            # for j in l.inflow_points:
            #     smap.set_geometry(j, crs=crs, marker='o',
            #                       markersize=40, edgecolor='k', alpha=0.8,
            #                       zorder=99, facecolor='none')


out = smap.plot(ax)
ax.set_aspect(1/1.5)
plt.tight_layout()
plt.show()
plt.savefig('test.svg')
            
#%%



# Glacier outline raster
extent = ds.glacier_ext
    
# topo.rio.write_crs(4326, inplace=True)
# 
topo = topo.rio.write_crs(ds.pyproj_srs)

# transformer = Transformer.from_crs(ds.pyproj_srs, "EPSG:4326", always_xy=True)
# lon, lat = transformer.transform(ds.x, ds.y)
# da.coords["lon"] = (("y", "x"), lon)
# da.coords["lat"] = (("y", "x"), lat)

#%%

cmap = "cubehelix_r"
cmax = 0.3
cmin = 0.0001

# norm = colors.Normalize(vmin=cmin, vmax=cmax)
clevels = np.linspace(cmin, cmax, 50)
# norm = colors.SymLogNorm(linthresh=0.001, linscale=0.001, vmin=cmin, vmax=cmax, base=2, clip=True)
# norm = colors.CenteredNorm(clip=True)
norm = colors.Normalize(vmin=cmin, vmax=cmax)
fig = plt.figure(figsize=(12, 12), dpi=100)
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
#ax.set_extent([-152, -147, 58, 65], crs=ccrs.PlateCarree())
ax.coastlines()

#ax.set_extent(topo.rio.bounds(), crs=ccrs.TransverseMercator())
#topo.plot(ax=ax, transform=ccrs.TransverseMercator())
# 

for line in lines:
    ax.add_geometries([line], crs=gdir.grid.center_grid, lw=10)

fig.show()

# #%%
# ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
# ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5, edgecolor='black')
# ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1, edgecolor='black')
# plt1 = ax.contourf(
#         bvf['longitude'], bvf['latitude'], bvf,
#         levels=clevels,
#         cmap=cmap,
#         norm=norm,
#         extend='both',
#         transform=ccrs.PlateCarree(),
#     )
# plt1.cmap.set_bad('red')
# plt1.cmap.set_over('black')
# 
# ax.clabel(
#     plt1,
#     inline=True,
#     inline_spacing=10,
#     fmt='%d',
#     fontsize=14
# )
# 
# barbs = ax.barbs(
#     x=wind['longitude'],
#     y=wind['latitude'],
#     u=u.data,
#     v=v.data,
#     pivot='middle',
#     barbcolor='black',
#     transform=ccrs.PlateCarree(),
#     length=6,
#     alpha=0.75
# )
# 
# cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal', fraction=0.1,
#                   aspect=30, pad=0.075, shrink=0.8, extend='both')
# cb.set_ticks(
#     np.round(np.linspace(cmin, cmax, 15), decimals=1))
# 
# dt = datetime.fromtimestamp(bvf['time'].values.astype('int64') * 1e-9)
# ax.set_title(
#     f'Brunt Vaisala Frequency, {level} hPa',
#     fontweight='bold', fontsize=14, loc='left')
# ax.set_title(datetime.strftime(dt, '%H UTC %d %b %Y'), fontsize=14, loc='right')
# 
# # # Format the gridlines (owet_bulbional)
# gl = ax.gridlines(
#     crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False,
#     y_inline=False, linewidth=1, color='k', linestyle=':')
# gl.xlocator = mticker.FixedLocator([-145, -140, -135, -130, -125, -120, -115, -110, -105, -100])
# gl.ylocator = mticker.FixedLocator([25, 30, 35, 40, 45, 50, 55, 60])
# gl.top_labels = False
# gl.right_labels = False
# gl.xlabel_style = {'size': 16, 'rotation': 20}
# gl.ylabel_style = {'size': 16}
# 
# fig.tight_layout()
# # fp = rf'C:\sandbox\atmos\atms502\plots\brunt-vaisala-freq_{level}hPa_{abs(extent[0])}{abs(extent[1])}{abs(extent[2])}{abs(extent[3])}_{dt:%Y%m%d-%HUTC}.png'
# # plt.savefig(fp, transparent=False)