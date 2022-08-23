import random
import numpy as np
import pandas as pd
import json
import zipfile
import gzip
import pickle
import torch
import string
import os

import geopandas
from skmob.tessellation import tilers

from math import sqrt, sin, cos, pi, asin
from importlib.machinery import SourceFileLoader

path = './models/deepgravity.py'
ffnn = SourceFileLoader('ffnn', path).load_module()

def load_model(fname, oa2centroid, oa2features, oa2pop, device, dim_s=1, \
               distances=None, dim_hidden=256, lr=5e-6, momentum=0.9, dropout_p=0.0, verbose=True):
    loc_id = list(oa2centroid.keys())[0]

    model = ffnn.NN_MultinomialRegression(dim_s, dim_hidden, 'deepgravity',  dropout_p=dropout_p, device=device)
    checkpoint = torch.load(fname, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def instantiate_model(oa2centroid, oa2features, dim_input, device=torch.device("cpu"), dim_hidden=256, lr=5e-6, momentum=0.9, dropout_p=0.0, verbose=False):
    model = ffnn.NN_MultinomialRegression(dim_input, dim_hidden,  'deepgravity', dropout_p=dropout_p, device=device)
    return model

def earth_distance(lat_lng1, lat_lng2):
    lat1, lng1 = [l*pi/180 for l in lat_lng1]
    lat2, lng2 = [l*pi/180 for l in lat_lng2]
    dlat, dlng = lat1-lat2, lng1-lng2
    ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng/2.0) ** 2))
    return 6371.01 * ds  # spherical earth...