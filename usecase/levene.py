import numpy as np
import pandas as pd
from scipy.stats import levene as levene_test

from usecase.detect_peaks import detect_peaks


def select_levene(
    df: pd.DataFrame,
    top_n: int,
    distance: float = 10.0,
    delta: float = 0.1,
    top: float = None,
    bottom: float = None,
) -> tuple[list[int], pd.Series]:

    features = df.columns
    states = df.index.unique()
    p_values = []
    for feature in features:
        groups = [df.loc[state, feature] for state in states]
        p_value = levene_test(*groups).pvalue
        p_values.append(p_value)

    p_values = np.log(p_values)
    p_values = np.array(p_values) * -1

    if top is None:
        top = max(p_values)
    if bottom is None:
        bottom = min(p_values)

    indices = detect_peaks(p_values, top_n, distance, delta, top, bottom)
    if indices is None:
        raise ValueError("Peaks not found")
    return indices, pd.Series(p_values, index=features)
