import numpy as np
import pandas as pd

from usecase.detect_peaks import detect_peaks


def select_robust_cv(
    df: pd.DataFrame,
    top_n: int,
    distance: float = 10.0,
    delta: float = 0.1,
    top: float = None,
    bottom: float = None,
) -> tuple[list[int], pd.Series]:
    mad = np.median(np.abs(df - df.median(axis=0)), axis=0)
    median = df.median(axis=0)
    cv = mad / median

    if top is None:
        top = cv.max()
    if bottom is None:
        bottom = cv.min()

    indices = detect_peaks(cv, top_n, distance, delta, top, bottom)

    return indices, cv
