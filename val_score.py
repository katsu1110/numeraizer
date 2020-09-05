
import numpy as np
import pandas as pd
import os
import sys
import pathlib
from scipy.stats import spearmanr
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from loguru import logger

# -----------------------------
# Config
# -----------------------------

OUTPUT_DIR = ''

# -----------------------------
# Function
# -----------------------------
def compute_val_corr(valid : pd.DataFrame, pred : np.ndarray, save : bool):
    # pandas for valid
    valid_df = valid["id"].to_frame()
    valid_df['prediction_kazutsugi'] = pred

    # save
    if save == True:
        valid_df.to_csv(pathlib.Path(OUTPUT_DIR + 'valid.csv'), index=False)

    # -----------------------------
    # compute score
    # -----------------------------
    # all validation
    ranked_prediction = valid_df["prediction_kazutsugi"].rank(pct=True, method="first")
    correlation = np.corrcoef(valid["target_kazutsugi"], ranked_prediction)[0, 1]
    logger.debug("ALL VALID: rank corr = {}".format(correlation))

    # first valid eras
    idx = np.where(valid["valid2"] == False)[0]
    correlation = np.corrcoef(valid["target_kazutsugi"].iloc[idx], ranked_prediction.iloc[idx])[0, 1]
    logger.debug("VALID 1: rank corr = {}".format(correlation))

    # second valid eras
    idx = np.where(valid["valid2"] == True)[0]
    correlation = np.corrcoef(valid["target_kazutsugi"].iloc[idx], ranked_prediction.iloc[idx])[0, 1]
    logger.debug("VALID 2: rank corr = {}".format(correlation))

TOURNAMENT_NAME = "kazutsugi"
PREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"

def feature_exposures(df):
    feature_names = [f for f in df.columns
                     if f.startswith("feature")]
    exposures = []
    for f in feature_names:
        fe = spearmanr(df[PREDICTION_NAME], df[f])[0]
        exposures.append(fe)
    return np.array(exposures)

def max_feature_exposure(df):
    return np.max(np.abs(feature_exposures(df)))

def feature_exposure(df):
    return np.sqrt(np.mean(np.square(feature_exposures(df))))
