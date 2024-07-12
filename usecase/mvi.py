import numpy as np
import pandas as pd

from usecase.detect_peaks import detect_peaks


def select_mvi(
    df: pd.DataFrame,
    top_n: int,
    distance: float = 10.0,
    delta: float = 0.1,
    top: float = None,
    bottom: float = None,
) -> tuple[list[int], pd.Series]:
    # 中央値と中央絶対偏差を比較し、差が大きい特徴量を選択する
    mad = np.median(np.abs(df - df.median(axis=0)), axis=0)
    median = df.median(axis=0)
    mvi = mad - median

    if top is None:
        top = mvi.max()
    if bottom is None:
        bottom = mvi.min()

    indices = detect_peaks(mvi, top_n, distance, delta, top, bottom)
    if indices is None:
        raise ValueError("Peaks not found")
    return indices, mvi
