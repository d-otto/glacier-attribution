# -*- coding: utf-8 -*-
"""
download_from_json.py

Description.

Author: drotto
Created: 12/13/2022 @ 11:36 AM
Project: glacier-attribution
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import json


#%%

p = Path('C:/Users/drotto/OneDrive - UW/gcm/cmip5/tas_Amon_past1000.json')
with open(p) as f:
    d = json.load(f)
    
d = d['response']['docs']
