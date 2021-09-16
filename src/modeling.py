# libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from tqdm.auto import tqdm
import joblib
import pathlib
import glob
import json
import scipy
from scipy.stats import skew, kurtosis
from sklearn import decomposition, preprocessing, linear_model, svm
from multiprocessing import Pool, cpu_count
import time
import requests as re
import datetime
from dateutil.relativedelta import relativedelta, FR
import operator
import xgboost as xgb
import lightgbm as lgb

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib_venn import venn2, venn3
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('seaborn-colorblind')

import warnings
warnings.simplefilter('ignore')

"""Utils
"""

def make_group(feature_df, key='friday_date'):
    """group maker for rank learning
    """
    return feature_df.groupby(key).size().to_frame('size')['size'].to_numpy()

def make_dataset(feature_df, features, target='target20_d', SUFFIX='val', ):
    """Make dataset
    """
    # ---------------------------------------------------------
    # making dataset
    # ---------------------------------------------------------
    # train set
    if SUFFIX == 'val': # for validation
        train_set = {
            'X': feature_df.query('data_type == "train"')[features], 
            'y': feature_df.query('data_type == "train"')[target].astype(np.float64),
            'g': make_group(feature_df.query('data_type == "train"'))
        }

    else: # full
        data_types = ['train', 'validation']
        train_set = {
            'X': feature_df.query('data_type in @data_types')[features], 
            'y': feature_df.query('data_type in @data_types')[target].astype(np.float64),
            'g': make_group(feature_df.query('data_type in @data_types'))
        }
        assert train_set['y'].isna().sum() == 0

    # valid set 
    val_df_ = feature_df.query('data_type == "validation"').dropna(subset=[target]).copy()
    val_set = {
        'X': val_df_[features], 
        'y': val_df_[target].astype(np.float64),
        'g': make_group(val_df_)
    }
    assert train_set['y'].isna().sum() == 0
    assert val_set['y'].isna().sum() == 0

    # test set
    test_set = {
        'X': feature_df.query('data_type == "live"')[features],
        'g': make_group(feature_df.query('data_type == "live"'))
    }
    return train_set, val_set, test_set

def hypara_dispatcher(MODEL='LGB'):
    """Dispatch hyperparameter
    """

    # parameters
    if MODEL == 'LGB':
        params = {
                'n_estimators': 10000,
                'objective': 'regression',
                'boosting_type': 'gbdt',
                'max_depth': 7,
                'learning_rate': 0.01,
                'subsample': 0.72,
                'subsample_freq': 4,
                'feature_fraction': 0.1,
                'lambda_l1': 1,
                'lambda_l2': 1,
                'seed': 46,
                'verbose': -1, 
                # 'device': 'gpu'
                }    
        params["metric"] = "rmse"

    elif MODEL == 'XGB':
        params = {
            'colsample_bytree': 0.1,                 
            'learning_rate': 0.1,
            'max_depth': 4,
            'subsample': 1,
            'min_child_weight': 4,
            'gamma': 0.24,
            'alpha': 1,
            'lambda': 1,
            'seed': 46,
            'n_estimators': 10000,
            'tree_method': 'gpu_hist' # Let's use GPU for a faster experiment
        }
        # params["objective"] = 'rank:pairwise'

    elif MODEL == 'MLP':
        params = {
            'input_dropout': 0.0,
            'hidden_layers': 3,
            'hidden_units': 256,
            'embedding_out_dim': 4,
            'hidden_activation': 'relu', 
            'hidden_dropout': 0.01,
            'gauss_noise': 0.01,
            'norm_type': 'layer', # layer
            'optimizer': {'type': 'adam', 'lr': 1e-3},
            'batch_size': 1024,
            'epochs': 100
        }

    elif MODEL == 'ridge':
        params = {
            'alpha': 100
            , 'fit_intercept': True
            , 'max_iter': 10000
            , 'random_state': 46
        }

    elif MODEL == 'beyesianridge':
        params = {
            'n_iter': 10000
        }

    elif MODEL == 'lasso':
        params = {
            'alpha': 0.001
            , 'fit_intercept': True
            , 'max_iter': 10000
            , 'random_state': 46
        }

    elif MODEL == 'svm':
        params = {
            'C': 100
        }
    return params

def model_trainer(train_set, val_set, MODEL='LGB', SUFFIX='val', SAVE=False):
    """Train model
    """
    logger.info(f'Training {MODEL} in {SUFFIX} mode...')

    # get hyperparameters
    params = hypara_dispatcher(MODEL)

    # fit
    if MODEL == 'LGB':
        # train data
        dtrain_set = lgb.Dataset(
            train_set['X'].values
            , train_set['y'].values
            , feature_name=features
            )
        if SUFFIX == 'val':
            # val data
            dval_set = lgb.Dataset(
                val_set['X'].values
                , val_set['y'].values
                , feature_name=features
                )

            # train
            model = lgb.train(
                params
                , dtrain_set
                , valid_sets=[dtrain_set, dval_set]
                , early_stopping_rounds=100
                , verbose_eval=100
                )
        else:
            # train
            model = lgb.train(
                params
                , dtrain_set
                )

    elif MODEL == 'XGB':
        # model
        model = xgb.XGBRegressor(**params)
        if SUFFIX == 'val':    
            # train
            model.fit(
                train_set['X'], train_set['y'], 
                eval_set=[(val_set['X'], val_set['y'])],
                verbose=500, 
                early_stopping_rounds=100,
            )
        else:
            # train
            model.fit(
                train_set['X'], train_set['y'], 
                verbose=500, 
            )
            
    elif MODEL == 'XGBRank':
        # model
        model = xgb.XGBRanker(**params)
        if SUFFIX == 'val':
            model.fit(
                train_set['X'], train_set['y'], 
                eval_set=[(val_set['X'], val_set['y'])],
                group=train_set['g'],
                eval_group=[val_set['g']],
                verbose=100, 
                early_stopping_rounds=100,
            )
        else:
            model.fit(
                train_set['X'], train_set['y'], 
                group=train_set['g'],
                verbose=100, 
            )

    # save model
    if SAVE:
        if MODEL[-1] == 'B':
            # save via joblib
            joblib.dump(model, f'{OUTPUT_DIR}/{target}_{MODEL}_model_{SUFFIX}.pkl')
            logger.info(f'{MODEL} {SUFFIX}_model for {target} saved!')

    return model

def get_feature_importance(model, features, MODEL='LGB'):
    """Get feature importance
    """
    # feature importance
    fi_df = pd.DataFrame()
    fi_df['features'] = features
    fi_df['importance'] = np.nan
    
    # LGB
    if MODEL == 'LGB':
        fi_df['importance'] = model.feature_importance(importance_type="gain")

    # XGB
    elif 'XGB' in MODEL:
        importance = model.get_booster().get_score(importance_type='gain')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        fi_df = pd.DataFrame(importance, columns=['features', 'importance'])
        
    return fi_df

def fit_model(feature_df, features, targets=['target_20d', 'target_4d'], MODEL='LGB', ):
    """Fit model
    """
    # fit
    fi_df = pd.DataFrame()
    fi_df['features'] = features
    for target in tqdm(targets):
        logger.info(' ======================== ')
        logger.info(f'{MODEL}: predicting {target}...!')
        logger.info(' ======================== ')
        
        # make datasets
        train_set, val_set, test_set = make_dataset(feature_df, features, target, SUFFIX='val')
        
        assert train_set['y'].isna().sum() == 0
        assert val_set['y'].isna().sum() == 0

        # train with validation data
        model = model_trainer(train_set, val_set, MODEL, SUFFIX='val', SAVE=True)

        # feature importance
        fi_df_ = get_feature_importance(model, features, MODEL)
        fi_df = fi_df.merge(
            fi_df_.rename(columns={'importance': f'{target}_{MODEL}'})
            , how='left'
            , on='features'
        )

        # val model prediction
        sub_df[f'{target}_{MODEL}'] = model.predict(sub_df[features])

        # full model training
        train_set = {
            'X': feature_df.loc[feature_df['data_type'].isin(['train', 'validation']), features],
            'y': feature_df.loc[feature_df['data_type'].isin(['train', 'validation']), target].astype(np.float32)
        }
        params_full = params.copy()
        params_full['n_estimators'] = 201
        dtrain_set = lgb.Dataset(
            train_set['X'].values, train_set['y'].values
            , feature_name=features
            )
        model = lgb.train(
            params_full
            , dtrain_set
            , verbose_eval=500
            )
        
        # full model prediction
        live_pred_val = sub_df.loc[sub_df['data_type'] == 'live', f'{target}_{MODEL}'].values
        live_pred = model.predict(sub_df.loc[sub_df['data_type'] == 'live', features])
        sub_df.loc[sub_df['data_type'] == 'live', f'{target}_{MODEL}'] = 0.3*live_pred_val + 0.7*live_pred
        
        # save
        joblib.dump(model, f'{OUTPUT_DIR}/{target}_{MODEL}_model_full.pkl')
        logger.info(f'{MODEL} full-model for {target} saved!')

def plot_feature_importance(MODEL='LGB', targets=['target_20d', 'target_4d', ], top_n=25):
    """Plot feature importance
    """
    fig, ax = plt.subplots(1, len(targets), figsize=(16, 12))
    for i, target in tqdm(enumerate(targets)):
        pred_col = f'{target}_{MODEL}'
        sns.barplot(
            x=pred_col
            , y='features'
            , data=fi_df.sort_values(by=pred_col, ascending=False).iloc[:top_n]
            , ax=ax[i]
        )
        if i > 0:
            ax[i].set_ylabel('')
        if i == 1:
            ax[i].set_xlabel('importance')
        else:
            ax[i].set_xlabel('')
        ax[i].set_title(pred_col)
    plt.tight_layout()