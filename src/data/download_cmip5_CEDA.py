# -*- coding: utf-8 -*-
"""
download_cmip5_CEDA.py

Description.

Author: drotto
Created: 12/14/2022 @ 11:21 AM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

from lib.cmip5_download.CEDA_download import *
from config import ROOT
import logging

logging.basicConfig(level=logging.INFO)

#%%

username, password = get_credentials()

group_models = [
    ("BCC", "bcc-csm1-1"),
    ("BCC", "bcc-csm1-1-m"),
    ("BNU", "BNU-ESM"),
    ("CCCma", "CanAM4"),
    ("CCCma", "CanCM4"),
    ("CCCma", "CanESM2"),
    ("CMCC", "CMCC-CESM"),
    ("CMCC", "CMCC-CM"),
    ("CMCC", "CMCC-CMS"),
    ("CNRM-CERFACS", "CNRM-CM5"),
    ("CNRM-CERFACS", "CNRM-CM5-2"),
    ("COLA-CFS", "CFSv2-2011"),
    ("CSIRO-BOM", "ACCESS1-0"),
    ("CSIRO-BOM", "ACCESS1-3"),
    ("CSIRO-QCCCE", "CSIRO-Mk3-6-0"),
    ("FIO", "FIO-ESM"),
    ("ICHEC", "EC-EARTH"),
    ("INM", "inmcm4"),
    ("IPSL", "IPSL-CM5A-LR"),
    ("IPSL", "IPSL-CM5A-MR"),
    ("IPSL", "IPSL-CM5B-LR"),
    ("LASG-CESS", "FGOALS-g2"),
    ("LASG-IAP", "FGOALS-gl"),
    ("LASG-IAP", "FGOALS-s2"),
    ("MIROC", "MIROC-ESM"),
    ("MIROC", "MIROC-ESM-CHEM"),
    ("MIROC", "MIROC4h"),
    ("MIROC", "MIROC5"),
    ("MOHC", "HadGEM3"),
    ("MOHC", "HadGEM2-A"),
    ("MOHC", "HadGEM2-CC"),
    ("MOHC", "HadGEM2-ES"),
    ("MPI-M", "MPI-ESM-LR"),
    ("MPI-M", "MPI-ESM-MR"),
    ("MPI-M", "MPI-ESM-P"),
    ("MRI", "MRI-AGCM3-2H"),
    ("MRI", "MRI-AGCM3-2S"),
    ("MRI", "MRI-CGCM3"),
    ("MRI", "MRI-ESM1"),
    ("NASA-GISS", "GISS-E2-H"),
    ("NASA-GISS", "GISS-E2-H-CC"),
    ("NASA-GISS", "GISS-E2-R"),
    ("NASA-GISS", "GISS-E2-R-CC"),
    ("NASA-GMAO", "GEOS-5"),
    ("NCAR", "CCSM4"),
    ("NCC", "NorESM1-M"),
    ("NCC", "NorESM1-ME"),
    ("NICAM", "NICAM-09"),
    ("NIMR-KMA", "HADGEM2-AO"),
    ("NOAA-GFDL", "GFDL-CM2p1"),
    ("NOAA-GFDL", "GFDL-CM3"),
    ("NOAA-GFDL", "GFDL-ESM2G"),
    ("NOAA-GFDL", "GFDL-ESM2M"),
    ("NOAA-GFDL", "GFDL-HIRAM-C180"),
    ("NOAA-GFDL", "GFDL-HIRAM-C360"),
    ("NOAA-NCEP", "CFSv2-2011"),
    ("NSF-DOE-NCAR", "CESM1-BGC"),
    ("NSF-DOE-NCAR", "CESM1-CAM5"),
    ("NSF-DOE-NCAR", "CESM1-CAM5-1-FV2"),
    ("NSF-DOE-NCAR", "CESM1-FASTCHEM"),
    ("NSF-DOE-NCAR", "CESM1-WACCM"),
]

experiments = ["historical", "past1000", "historicalNat", ]
freqs = ["mon", ]
ensembles = ["r1i1p1", 'r1i1p121', 'r1i1p122', 'r1i1p1221', 'r1i1p123', 'r1i1p124', 'r1i1p125', 'r1i1p126', 'r1i1p127', 'r1i1p128']

# Set 1 - cloud properties
realms = ["atmos", ]
out_realms = ["Amon", ]
variables = ["tas"]

logging.info("Set 1 - tas")
datasets = get_datasets(group_models, experiments, freqs, realms,
                        out_realms, ensembles, variables)
download_batch(datasets, Path(ROOT, "data/external/gcm/cmip5"),
               username, password, overwrite=False)