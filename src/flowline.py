# -*- coding: utf-8 -*-
"""
flowline.py

Description.

Author: drotto
Created: 12/1/2022 @ 6:19 AM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import gzip
import pickle
import oggm

from config import ROOT

#%%

class get_flowline_geom:
    def __init__(self, rgiid, dirname=Path(ROOT, 'data/interim/oggm_flowlines')):
        fp = dirname / f'{rgiid}.downstream_line.pickle'
        with gzip.open(fp, "rb") as openfile:
            flowline_geom, downstream_geom = pickle.load(openfile)
        flowline_geom = flowline_geom.__dict__
    
        x = flowline_geom['dis_on_line'] * flowline_geom['dx_meter']
        zb = flowline_geom['bed_h']
        # w = np.concatenate([flowline_geom['_w0_m'][~np.isnan(flowline_geom['_w0_m'])], downstream_geom['w0s']])
        w = flowline_geom['_w0_m'].copy()
        idx_term = np.argmax(np.isnan(w))
        ds_len = len(downstream_geom['w0s'])
        h0 = flowline_geom['_thick']
        w[idx_term:idx_term + ds_len] = downstream_geom['w0s']
        if any(np.isnan(w)):
            idx_nan = np.argmax(np.isnan(w))
            w[idx_nan:] = w[idx_nan - 1]
            
        self.x = x
        self.zb = zb
        self.w = w
        self.h0 = h0
        