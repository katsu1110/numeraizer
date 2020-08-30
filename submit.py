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

public_id = 'ABCDEF'
secret_key = 'ABCDEF'

# -----------------------------
# Function
# -----------------------------
def submit(tournament : pd.DataFrame, pred : np.ndarray, model_id='abcde'):
    predictions_df = tournament["id"].to_frame()
    predictions_df["prediction_kazutsugi"] = pred
    
    # scaling
    predictions_df["prediction_kazutsugi"] = (predictions_df["prediction_kazutsugi"] - predictions_df["prediction_kazutsugi"].min()) / (predictions_df["prediction_kazutsugi"].max() - predictions_df["prediction_kazutsugi"].min())
        
    # use API
    napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)
    
    # Upload your predictions
    predictions_df.to_csv(pathlib.Path(OUTPUT_DIR + f"predictions_{model_id}.csv"), index=False)
    submission_id = napi.upload_predictions(pathlib.Path(OUTPUT_DIR + f"predictions_{model_id}.csv"), model_id=model_id)
    logger.debug('submitted to {model_id}', model_id=model_id)