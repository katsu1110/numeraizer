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
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from loguru import logger

# -----------------------------
# Config
# -----------------------------

OUTPUT_DIR = ''

public_id = 'ABCDEF' # replace with yours
secret_key = 'ABCDEF' # replace with yours
PREDICTION_NAME = "prediction_kazutsugi" # will be 'prediction_nomi' in the future

# -----------------------------
# Function
# -----------------------------
def submit(tournament : pd.DataFrame, pred : np.ndarray, model_id='abcde'):
    predictions_df = tournament["id"].to_frame()
    predictions_df[PREDICTION_NAME] = pred
    
    # to rank
    predictions_df[PREDICTION_NAME] = predictions_df[PREDICTION_NAME].rank(pct=True, method="first")
    
    # save
    predictions_df.to_csv(pathlib.Path(OUTPUT_DIR + f"predictions_{model_id}.csv"), index=False)
    
    # Upload your predictions using API
    napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)
    submission_id = napi.upload_predictions(pathlib.Path(OUTPUT_DIR + f"predictions_{model_id}.csv"), model_id=model_id)
    logger.debug('submitted to {model_id}', model_id=model_id)
    
    return predictions_df