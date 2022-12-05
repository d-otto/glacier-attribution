# -*- coding: utf-8 -*-
"""
config.py

Description.

Author: dotto
Created: 7/6/2021 @ 1:21 PM
Project: glacier-diseq
"""

import os
import yaml

ROOT = os.path.dirname(__file__)


#%%

with open(os.path.abspath(os.path.join(ROOT, 'config.yaml'))) as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
