import numpy as np
import pandas as pd
import os
import sys
import gc
import pathlib
from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, QuantileTransformer
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, f1_score
from scipy.stats import spearmanr
import joblib

# model
from sklearn import linear_model
import lightgbm as lgb
import xgboost as xgb
import operator

# visualize
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter
sns.set_context("talk")
style.use('seaborn-colorblind')

class Modeling:
    """
    modeling class

    # instantiate
    model = Modeling(
        df
        , model_name='lgb'
        , features=['feature_dexterity11', 'feature_charisma63']
        , target=['target']
        , categoricals=['feature_intelligence1']
        , out_dir='./'
        )
    
    # fit
    train_df = df.query('data_type == "train"')
    val_df = df.query('data_type == "validation"')
    model.fit(train_df, val_df, early_stop=500)

    # save model
    model.save('mymodel')

    # get feature importance
    feature_importance_df = model.get_feature_importance()

    # predict
    test_df = df.query('data_type == "live"')
    test_df['prediction'] = model.predict(test_df)
    """

    def __init__(self
        , df
        , model_name: str='lgb'
        , params: dict={}
        , features: list=['feature_dexterity11', 'feature_charisma63']
        , target: list=['target']
        , categoricals: list=[]
        , output_dir: str='./'
        ):
        self.df = df
        self.model_name = model_name
        self.features = features
        self.target = target
        self.categoricals = categoricals
        self.output_dir = output_dir

    def get_params(self):
        """
        get model hyperparameters
        """
        if len(self.params) == 0:
            print('Using default paramters...')
            if self.model_name.lower() == 'lgb':
                self.params = {
                    'n_estimators': 2000,
                    'objective': 'regression',
                    'boosting_type': 'gbdt',
                    'max_depth': 5,
                    'learning_rate': 0.01, 
                    'feature_fraction': 0.1,
                    'seed': 42
                }

            elif self.model_name.lower() == 'xgb':
                self.params = {
                    'max_depth': 5, 
                    'learning_rate': 0.01,
                    'n_estimators': 2000,
                    'n_jobs': -1,
                    'colsample_bytree': 0.1,
                    'seed': 42
                }

            elif self.model_name.lower() == 'lasso':
                self.params = {
                    'alpha': 0.001, 
                    'fit_intercept': True,
                    'max_iter': 10000, 
                    'tol': 1e-06,
                    'random_state': 42,
                }

            elif self.model_name.lower() == 'bayesianridge':
                self.params = {
                    'fit_intercept': True,
                    'n_iter': 10000, 
                    'tol': 1e-06,
                }

            elif self.model_name.lower() == 'ridge':
                self.params = {
                    'alpha': 240, 
                    'fit_intercept': True,
                    'max_iter': 10000, 
                    'tol': 1e-06,
                    'random_state': 42,
                }

    def model_dispatcher(self):
        """
        dispatch a model object
        """
        if self.model_name == 'lgb':
            return lgb.LGBMRegressor(**self.params)
            
        elif self.model_name == 'xgb':
            return xgb.XGBRegressor(**self.params)

    def save(self, file_name):
        """
        save trained model
        """
        file_name = file_name.split('.')[0]
        joblib.dump(self.trained_model, f'{self.output_dir}/{file_name}.pkl')
        print(f'{file_name} saved!')

    def get_feature_target(self):
        """
        get features, target
        """
        # features and target
        features = self.features
        categoricals = self.categoricals
        if len(self.target) == 1:
            target = self.target[0]
        else:
            target = self.target
        return features, target, categoricals

    def fit_noval(self, train_df):
        """
        fit model without validation
        """
        model = self.model_dispatcher()
        features, target, categoricals = self.get_feature_target()

        # fit by model
        if self.model_name == 'lgb':
            # fit
            model.fit(
                train_df[features], train_df[target]
                , verbose=-1
                , categorical_feature=categoricals
                )

        elif self.model_name == 'xgb':
            # fit
            model.fit(
                train_df[features], train_df[target]
                , verbose=500
                )

        self.trained_model = model

    def fit_es(self, train_df, val_df, early_stop=100):
        """
        fit model with validation
        """
        model = self.model_dispatcher()
        features, target, categoricals = self.get_feature_target()

        # fit by model
        if self.model_name == 'lgb':
            # fit
            model.fit(
                train_df[features], train_df[target]
                , eval_set=[(val_df[features], val_df[target])]
                , verbose=-1
                , early_stopping_rounds=early_stop
                , categorical_feature=categoricals
                )

        elif self.model_name == 'xgb':
            # fit
            model.fit(
                train_df[features], train_df[target]
                , eval_set=[(val_df[features], val_df[target])]
                , early_stopping_rounds=early_stop
                , verbose=500
                )

        self.trained_model = model

    def fit(self, train_df, val_df=None, early_stop=None):
        """
        model fitting
        """
        if val_df is None:
            self.fit_noval(train_df)
        else:
            if early_stop is None:
                early_stop = 100
            self.fit_es(train_df, val_df, early_stop)

    def get_feature_importance(self):
        """
        compute feature importance from the trained model
        """
        model = self.trained_model
        fi_df = pd.DataFrame()
        fi_df['features'] = self.features

        if self.model_name == 'lgb':
            fi_df['importance'] = model.booster_.feature_importance(importance_type="gain")

        elif self.model_name == 'xgb':
            fi_df = model.get_booster().get_score(importance_type='gain')
            fi_df = sorted(fi_df.items(), key=operator.itemgetter(1))
            fi_df = pd.DataFrame(fi_df, columns=['features', 'importance'])

        else:
            fi_df['importance'] = np.nan
        
        return fi_df

    def predict(self, test_df):
        features, _, _ = self.get_feature_target()
        return self.trained_model.predict(test_df[features])