import pandas as pd

from usecase.detect_peaks import detect_peaks


def select_cv(
    df: pd.DataFrame,
    top_n: int,
    distance: float = 10.0,
    delta: float = 0.1,
    top: float = None,
    bottom: float = None,
) -> tuple[list[int], pd.Series]:
    _cv = df.std() / df.mean()

    if top is None:
        top = _cv.max()
    if bottom is None:
        bottom = _cv.min()

    indices = detect_peaks(
        _cv.to_numpy(),
        top_n,
        distance,
        delta,
        top,
        bottom,
    )
    if indices is None:
        raise ValueError("Peaks not found")

    return indices, _cv
