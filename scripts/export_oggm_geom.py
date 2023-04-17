# -*- coding: utf-8 -*-
"""
export_oggm_geom.py

Description.

Author: drotto
Created: 3/2/2023 @ 3:40 PM
Project: glacier-attribution
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data import get_rgi
from src.flowline import get_flowline_geom
from config import cfg, ROOT

#%%

rgiids = {k: v['name'] for k, v in cfg['glaciers'].items()}
rgi = get_rgi(list(rgiids.keys()), from_sqllite=True)

#%%

for rgiid, name in rgiids.items():
    geom = get_flowline_geom(rgiid)
    