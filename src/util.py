# -*- coding: utf-8 -*-
"""
util.py

Description.

Author: drotto
Created: 11/9/2022 @ 4:57 PM
Project: glacier-attribution
"""

import numpy as np


# %%


def trunc(values, decs=0):
    return np.trunc(values * 10**decs) / (10**decs)


def flatten(items, seqtypes=(list, tuple)):
    try:
        for i, x in enumerate(items):
            while isinstance(x, seqtypes):
                items[i : i + 1] = x
                x = items[i]
    except IndexError:
        pass
    return items


def unwrap_coords(lon, lat):
    if lon < 0:
        lon = 360 + lon
    return lon, lat

def dict_key_from_value(d, v):
    return list(d.keys())[list(d.values()).index(v)]

