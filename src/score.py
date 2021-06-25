import numpy as np
import pandas as pd
from scipy.stats import spearmanr

class TournamentScoring:
    """
    Scoring Class for Numerai Tournament

    :HOW TO USE:
    # get numerai tournament validation data
    valid_df = tournament.query('data_type == "validation"').reset_index(drop=True)

    # features
    features = valid_df.columns[valid_df.columns.str.startswith('feature')].values.tolist()

    # inference with your model
    valid_df['prediction'] = model.predict(valid_df[features])

    # compute validation scores
    scorer = TournamentScoring(
        valid_df
        , target_name='target'
        , pred_name='prediction'
        , era='era'
        , features=features
        , neut_col='feature_dexterity7'
        )
    score_df = scorer.score_summary() 
    """

    def __init__(self
        , valid_df: pd.DataFrame
        , target_name: str='target'
        , pred_name: str='prediction'
        , era: str='era'
        , features: list=['feature_charisma19', 'feature_strength34']
        , neut_col: str='feature_dexterity7'
        ):
        """
        :INPUT:
        - valid_df : validation data (era, target, pred + feature columns)
        - target_name : target name
        - pred name : prediction name
        - era : era name
        - features : features list
        - neut_col : str, used for mmc computation (idealy use example prediction)
        """
        # test
        for f in [target_name, pred_name, era, neut_col] + features:
            assert f in valid_df.columns.values.tolist()
        
        # assign
        self.valid_df = valid_df
        self.target_name = target_name
        self.pred_name = pred_name
        self.era = era
        self.features = features
        self.neut_col = neut_col

    @staticmethod
    def to_rank(df: pd.DataFrame, col: str):
        """
        rank a feature
        """
        df[col] = df[col].rank(pct=True, method="first")
        return df

    def compute_corr(self):
        """
        Compute rank correlation
        
        :INPUT:
        - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist
        
        """
        
        return np.corrcoef(
            self.valid_df[self.target_name]
            , self.valid_df[self.pred_name])[0, 1]

    @staticmethod
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

    def compute_val_corr(self):
        """
        Compute rank correlation for valid periods
        
        :INPUT:
        - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist
        """
        
        # all validation
        correlation = self.compute_corr()
        print("rank corr = {:.4f}".format(correlation))
        return correlation
        
    def compute_val_sharpe(self):
        """
        Compute sharpe ratio for valid periods
        
        :INPUT:
        - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist
        """
        # all validation
        d = self.valid_df.groupby(self.era)[
            [self.target_name, self.pred_name]
            ].corr().iloc[0::2,-1].reset_index()
        me = d[self.pred_name].mean()
        sd = d[self.pred_name].std()
        max_drawdown = self.compute_max_drawdown(d[self.pred_name])
        print(
            'sharpe ratio = {:.4f}, corr mean = {:.4f}, corr std = {:.4f}, max drawdown = {:.4f}'.format(
                me / sd, me, sd, max_drawdown
                ))
        
        return me / sd, me, sd, max_drawdown
        
    def feature_exposures(self):
        """
        Compute feature exposure
        
        :INPUT:
        - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist
        """
        feature_names = self.features
        exposures = []
        for f in feature_names:
            fe = spearmanr(self.valid_df[self.pred_name], self.valid_df[f])[0]
            exposures.append(fe)
        return np.array(exposures)

    @staticmethod
    def max_feature_exposure(fe : np.ndarray):
        return np.max(np.abs(fe))

    @staticmethod
    def feature_exposure(fe : np.ndarray):
        return np.sqrt(np.mean(np.square(fe)))

    def compute_val_feature_exposure(self):
        """
        Compute feature exposure for valid periods
        
        :INPUT:
        - valid_df : pd.DataFrame where at least 2 columns ('prediction' & 'target') exist
        """
        # all validation
        fe = self.feature_exposures()
        fe1, fe2 = self.feature_exposure(fe), self.max_feature_exposure(fe)
        print('feature exposure = {:.4f}, max feature exposure = {:.4f}'.format(fe1, fe2))
        
        return fe1, fe2

    # to neutralize a column in a df by many other columns
    @staticmethod
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
    @staticmethod
    def neutralize_series(series, by, proportion=1.0):
        scores = series.values.reshape(-1, 1)
        exposures = by.values.reshape(-1, 1)

        # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
        exposures = np.hstack(
            (exposures,
            np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

        correction = proportion * (exposures.dot(
            np.linalg.lstsq(exposures, scores, rcond=None)[0]))
        corrected_scores = scores - correction
        neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
        return neutralized

    @staticmethod
    def unif(df):
        x = (df.rank(method="first") - 0.5) / len(df)
        return pd.Series(x, index=df.index)

    def get_feature_neutral_mean(self):
        feature_cols = self.features
        self.valid_df.loc[:, "neutral_sub"] = self.neutralize(
            self.valid_df
            , [self.pred_name]
            , feature_cols)[self.pred_name]
        scores = self.valid_df.groupby(self.era).apply(
            lambda x: np.corrcoef(x["neutral_sub"].rank(pct=True, method="first")
            , x[self.target_name])).mean()
        return np.mean(scores)

    def compute_val_mmc(self):
        # MMC over validation
        mmc_scores = []
        corr_scores = []
        for _, x in self.valid_df.groupby(self.era):
            series = self.neutralize_series(pd.Series(self.unif(x[self.pred_name])),
                                    pd.Series(self.unif(x[self.neut_col])))
            mmc_scores.append(np.cov(series, x[self.target_name])[0, 1] / (0.29 ** 2))
            corr_scores.append(
                np.corrcoef(self.unif(x[self.pred_name]).rank(pct=True, method="first")
                , x[self.target_name])
                )

        val_mmc_mean = np.mean(mmc_scores)
        val_mmc_std = np.std(mmc_scores)
        val_mmc_sharpe = val_mmc_mean / val_mmc_std
        corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
        corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)
        corr_plus_mmc_mean = np.mean(corr_plus_mmcs)

        print("MMC Mean = {:.6f}, MMC Std = {:.6f}, CORR+MMC Sharpe = {:.4f}".format(
            val_mmc_mean, val_mmc_std, corr_plus_mmc_sharpe
            ))

        # Check correlation with example predictions
        corr_with_example_preds = np.corrcoef(
            self.valid_df[self.neut_col].rank(pct=True, method="first")
            , self.valid_df[self.pred_name].rank(pct=True, method="first"))[0, 1]
        print("Corr with example preds: {:.4f}".format(corr_with_example_preds))
        
        return val_mmc_mean, val_mmc_std, corr_plus_mmc_sharpe, corr_with_example_preds
        
    def calculate_fnc(self):
        """    
        compute feature neutral correlation
        https://docs.numer.ai/tournament/feature-neutral-correlation
        """
        # necessary vectors
        sub = self.valid_df[self.pred_name]
        features = self.valid_df[self.features]
        targets = self.valid_df[self.target_name]

        # Normalize submission
        sub = (sub.rank(method="first").values - 0.5) / len(sub)

        # Neutralize submission to features
        f = features.values
        sub -= f.dot(np.linalg.pinv(f).dot(sub))
        sub /= sub.std()
        
        sub = pd.Series(np.squeeze(sub)) # Convert np.ndarray to pd.Series

        # FNC: Spearman rank-order correlation of neutralized submission to target
        fnc = np.corrcoef(sub.rank(pct=True, method="first"), targets)[0, 1]

        return fnc
        
    def score_summary(self):
        """
        compute all the metrics for summary
        """
        score_df = {}
        
        try:
            score_df['correlation'] = self.compute_val_corr()
        except:
            print('ERR: computing correlation')
        try:
            score_df['corr_sharpe'], score_df['corr_mean'], score_df['corr_std'], score_df['max_drawdown'] = self.compute_val_sharpe()
        except:
            print('ERR: computing sharpe')
        try:
            score_df['feature_exposure'], score_df['max_feature_exposure'] = self.compute_val_feature_exposure()
        except:
            print('ERR: computing feature exposure')
        try:
            score_df['mmc_mean'], score_df['mmc_std'], score_df['corr_mmc_sharpe'], score_df[f'corr_with_{self.neut_col}'] = self.compute_val_mmc()
        except:
            print('ERR: computing MMC')
        try:
            score_df['fnc'] = self.calculate_fnc()
        except:
            print('ERR: FNC')
        
        score_df = pd.DataFrame.from_dict(score_df, orient='index').rename(columns={0: 'score'})
        print(score_df.to_markdown(tablefmt='grid'))
        return score_df