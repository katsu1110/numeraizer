
import numpy as np
import pandas as pd
import os
import sys
import pathlib
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from loguru import logger

# -----------------------------
# Config
# -----------------------------

OUTPUT_DIR = ''

# -----------------------------
# Function
# -----------------------------
def compute_val_score(valid : pd.DataFrame, pred : np.ndarray):
    valid_df = valid["id"].to_frame()
    valid_df['prediction_kazutsugi'] = pred

    # save
    valid_df.to_csv(pathlib.Path(OUTPUT_DIR + 'valid.csv'), index=False)

    # compute score
    ranked_prediction = valid_df["prediction_kazutsugi"].rank(pct=True, method="first")
    correlation = np.corrcoef(valid["target_kazutsugi"], ranked_prediction)[0, 1]
    logger.debug("ALL VALID: rank corr = {}".format(correlation))
    idx = np.where(valid["valid2"] == False)[0]
    correlation = np.corrcoef(valid["target_kazutsugi"].iloc[idx], ranked_prediction.iloc[idx])[0, 1]
    logger.debug("VALID 1: rank corr = {}".format(correlation))
    idx = np.where(valid["valid2"] == True)[0]
    correlation = np.corrcoef(valid["target_kazutsugi"].iloc[idx], ranked_prediction.iloc[idx])[0, 1]
    logger.debug("VALID 2: rank corr = {}".format(correlation))