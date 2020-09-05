import pandas as pd
import numpy as np

def neutralize_series(series : pd.Series, by : pd.Series, proportion=0.5) -> pd.Series:
    """
    Make a neutralized prediction

    :INPUTS:
    - series : your model predictions (pd.Series)
    - by : meta-model (example XGB?) predictions (pd.Series)
    - proportion : proportion to reduce correlation
    """
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)
    exposures = np.hstack((exposures, np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))
    correction = proportion * (exposures.dot(np.linalg.lstsq(exposures, scores)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized