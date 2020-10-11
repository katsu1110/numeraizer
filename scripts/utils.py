import numpy as np
import pandas as pd
import os, sys
import gc
import pathlib
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, QuantileTransformer
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, f1_score
from scipy.stats import spearmanr
from loguru import logger

# ---------------------------
# Config
# ---------------------------
INPUT_DIR = '../input/'
INPUT_TRAIN_DIR = '../input/numerai-train-to-feather-noeda/'
INPUT_TEST_DIR = '../input/numerai-test-to-feather-noeda/'
OUTPUT_DIR = ''

# naming conventions
PREDICTION_NAME = 'prediction'
TARGET_NAME = 'target_kazutsugi' # will be 'target_nomi' in the future
EXAMPLE_PRED = 'example_prediction'

# ---------------------------
# Functions
# ---------------------------
def valid4score(valid : pd.DataFrame, pred : np.ndarray, load_example: bool=False, save : bool=False) -> pd.DataFrame:
    """
    Generate new valid pandas dataframe for computing scores
    
    :INPUT:
    - valid : pd.DataFrame extracted from tournament data (data_type='validation')
    
    """
    valid_df = valid.copy()
    valid_df['prediction'] = pd.Series(pred).rank(pct=True, method="first")
    
    if load_example:
        valid_df[EXAMPLE_PRED] = pd.read_csv(INPUT_DIR + 'numerai-example/valid_df.csv')['prediction'].values
    
    if save==True:
        valid_df.to_csv(OUTPUT_DIR + 'valid_df.csv', index=False)
        logger.debug('Validation dataframe saved!')
    
    return valid_df

def compute_corr(valid_df : pd.DataFrame):
    """
    Compute rank correlation
    
    :INPUT:
    - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist
    
    """
    
    return np.corrcoef(valid_df[TARGET_NAME], valid_df['prediction'])[0, 1]

def compute_max_drawdown(validation_correlations : pd.Series):
    """
    Compute max drawdown
    
    :INPUT:
    - validation_correaltions : pd.Series
    """
    
    rolling_max = (validation_correlations + 1).cumprod().rolling(window=100, min_periods=1).max()
    daily_value = (validation_correlations + 1).cumprod()
    max_drawdown = -(rolling_max - daily_value).max()
    
    return max_drawdown

def compute_val_corr(valid_df : pd.DataFrame):
    """
    Compute rank correlation for valid periods
    
    :INPUT:
    - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist
    """
    
    # all validation
    correlation = compute_corr(valid_df)
    logger.debug("ALL VALID: rank corr = {:.4f}".format(correlation))

    # first valid eras
    if 'valid2' in valid_df.columns.values.tolist():
        idx = np.where(valid_df["valid2"] == False)[0]
        correlation = compute_corr(valid_df.iloc[idx])
        logger.debug("VALID 1: rank corr = {:.4f}".format(correlation))

        # second valid eras
        idx = np.where(valid_df["valid2"] == True)[0]
        correlation = compute_corr(valid_df.iloc[idx])
        logger.debug("VALID 2: rank corr = {:.4f}".format(correlation))
    
def compute_val_sharpe(valid_df : pd.DataFrame):
    """
    Compute sharpe ratio for valid periods
    
    :INPUT:
    - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist
    """
    # all validation
    d = valid_df.groupby('era')[[TARGET_NAME, 'prediction']].corr().iloc[0::2,-1].reset_index()
    me = d['prediction'].mean()
    sd = d['prediction'].std()
    max_drawdown = compute_max_drawdown(d['prediction'])
    logger.debug('ALL VALID: sharpe ratio = {:.4f}, corr mean = {:.4f}, corr std = {:.4f}, max drawdown = {:.4f}'.format(me / sd, me, sd, max_drawdown))
    
    if "valid2" in valid_df.columns.values.tolist():
        # first valid eras
        idx = np.where(valid_df["valid2"] == False)[0]
        d = valid_df.iloc[idx].groupby('era')[[TARGET_NAME, 'prediction']].corr().iloc[0::2,-1].reset_index()
        me = d['prediction'].mean()
        sd = d['prediction'].std()
        max_drawdown = compute_max_drawdown(d['prediction'])
        logger.debug('VALID 1: sharpe ratio = {:.4f}, corr mean = {:.4f}, corr std = {:.4f}, max drawdown = {:.4f}'.format(me / sd, me, sd, max_drawdown))
        
        # second valid eras
        idx = np.where(valid_df["valid2"] == True)[0]
        d = valid_df.iloc[idx].groupby('era')[[TARGET_NAME, 'prediction']].corr().iloc[0::2,-1].reset_index()
        me = d['prediction'].mean()
        sd = d['prediction'].std()
        max_drawdown = compute_max_drawdown(d['prediction'])
        logger.debug('VALID 2: sharpe ratio = {:.4f}, corr mean = {:.4f}, corr std = {:.4f}, max drawdown = {:.4f}'.format(me / sd, me, sd, max_drawdown))
    
def feature_exposures(valid_df : pd.DataFrame):
    """
    Compute feature exposure
    
    :INPUT:
    - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist
    """
    feature_names = [f for f in valid_df.columns if f.startswith("feature")]
    exposures = []
    for f in feature_names:
        fe = spearmanr(valid_df['prediction'], valid_df[f])[0]
        exposures.append(fe)
    return np.array(exposures)

def max_feature_exposure(fe : np.ndarray):
    return np.max(np.abs(fe))

def feature_exposure(fe : np.ndarray):
    return np.sqrt(np.mean(np.square(fe)))

def compute_val_feature_exposure(valid_df : pd.DataFrame):
    """
    Compute feature exposure for valid periods
    
    :INPUT:
    - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist
    """
    # all validation
    fe = feature_exposures(valid_df)
    logger.debug('ALL VALID: feature exposure = {:.4f}, max feature exposure = {:.4f}'.format(feature_exposure(fe), max_feature_exposure(fe)))
    
    if "valid2" in valid_df.columns.values.tolist():
        # first valid eras
        idx = np.where(valid_df["valid2"] == False)[0]
        fe = feature_exposures(valid_df.iloc[idx])
        logger.debug('VALID 1: feature exposure = {:.4f}, max feature exposure = {:.4f}'.format(feature_exposure(fe), max_feature_exposure(fe)))
        
        # second valid eras
        idx = np.where(valid_df["valid2"] == True)[0]
        fe = feature_exposures(valid_df.iloc[idx])
        logger.debug('VALID 2: feature exposure = {:.4f}, max feature exposure = {:.4f}'.format(feature_exposure(fe), max_feature_exposure(fe)))
    
# to neutralize a column in a df by many other columns
def neutralize(df, columns, by, proportion=1.0):
    scores = df.loc[:, columns]
    exposures = df[by].values

    # constant column to make sure the series is completely neutral to exposures
    exposures = np.hstack(
        (exposures,
         np.asarray(np.mean(scores)) * np.ones(len(exposures)).reshape(-1, 1)))

    scores = scores - proportion * exposures.dot(
        np.linalg.pinv(exposures).dot(scores))
    return scores / scores.std()

# to neutralize any series by any other series
def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures, np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized

def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)

def get_feature_neutral_mean(df):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [PREDICTION_NAME], feature_cols)[PREDICTION_NAME]
    scores = df.groupby("era").apply(
        lambda x: np.corrcoef(x["neutral_sub"].rank(pct=True, method="first"), x[TARGET_NAME])).mean()
    return np.mean(scores)

def compute_val_mmc(valid_df : pd.DataFrame):    
    # MMC over validation
    mmc_scores = []
    corr_scores = []
    for _, x in valid_df.groupby("era"):
        series = neutralize_series(pd.Series(unif(x[PREDICTION_NAME])),
                                   pd.Series(unif(x[EXAMPLE_PRED])))
        mmc_scores.append(np.cov(series, x[TARGET_NAME])[0, 1] / (0.29 ** 2))
        corr_scores.append(np.corrcoef(unif(x[PREDICTION_NAME]).rank(pct=True, method="first"), x[TARGET_NAME]))

    val_mmc_mean = np.mean(mmc_scores)
    val_mmc_std = np.std(mmc_scores)
    val_mmc_sharpe = val_mmc_mean / val_mmc_std
    corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
    corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)
    corr_plus_mmc_mean = np.mean(corr_plus_mmcs)

    logger.debug("MMC Mean = {:.6f}, MMC Std = {:.6f}, CORR+MMC Sharpe = {:.4f}".format(val_mmc_mean, val_mmc_std, corr_plus_mmc_sharpe))

    # Check correlation with example predictions
    corr_with_example_preds = np.corrcoef(valid_df[EXAMPLE_PRED].rank(pct=True, method="first"),
                                          valid_df[PREDICTION_NAME].rank(pct=True, method="first"))[0, 1]
    logger.debug("Corr with example preds: {:.4f}".format(corr_with_example_preds))