"""
Assuming 

!pip install numerapi

has been already done.
"""

import numerapi
import numpy as np
import pandas as pd
import os
import sys
import pathlib
from loguru import logger

# -----------------------------
# Config
# -----------------------------

NUMERAI_DATA = 'https://numerai-public-datasets.s3-us-west-2.amazonaws.com/'
INPUT_DIR = '../input/'

# -----------------------------
# Function
# -----------------------------
def download_data(data='train', save=True):
    if data == 'train':
        fname = 'latest_numerai_training_data.csv.xz'
    elif data in ['test', 'tournament']:
        fname = 'latest_numerai_tournament_data.csv.xz'
    df = pd.read_csv(pathlib.Path(NUMERAI_DATA + fname)) 
    feature_cols = df.columns[df.columns.str.startswith('feature')]
    mapping = {0.0 : 0, 0.25 : 1, 0.5 : 2, 0.75 : 3, 1.0 : 4}
    for c in feature_cols:
        df[c] = df[c].map(mapping).astype(np.uint8)
    logger.debug('Downloaded {} data shape: {}'.format(data, df.shape))
    if save:
        save_path = pathlib.Path(INPUT_DIR + data + '.feather')
        df.to_feather(save_path)
        logger.debug('Downloaded {} data saved to {}'.format(data, save_path))
    return df

def get_int(x):
    try:
        return int(x[3:])
    except:
        return 1000

def load_data():
    # load data
    train = pd.read_feather(pathlib.Path(INPUT_DIR + 'train.feather'))
    try:
        tournament = pd.read_feather(pathlib.Path(INPUT_DIR + 'test.feather'))
    except:
        tournament = pd.read_feather(pathlib.Path(INPUT_DIR + 'tournament.feather'))

    # split valid and test
    valid = tournament[tournament["data_type"] == "validation"].reset_index(drop = True)
    
    # drop data type
    train.drop(columns=['data_type'], inplace=True)
    valid.drop(columns=['data_type'], inplace=True)
    tournament.drop(columns=['data_type'], inplace=True)
    
    # era int
    train["era"] = train["era"].apply(get_int)
    valid["era"] = valid["era"].apply(get_int)
    tournament["era"] = tournament["era"].apply(get_int)

    return train, valid, tournament