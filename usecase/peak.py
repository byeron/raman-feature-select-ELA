import pandas as pd

from usecase.detect_peaks import detect_peaks


def select_peak(
    df: pd.DataFrame,
    top_n: int,  # 選択するpeakの数
    distance: float = 10.0,  # peak間の最小距離
    delta: float = 0.1,  # peak探索のステップ
    top: float = None,  # peak探索の最大値
    bottom: float = None,  # peak探索の最小値
) -> tuple[list[int], pd.Series]:
    peak_indices = None

    if top is None:
        top = df.max().max()
    if bottom is None:
        bottom = df.min().min()

    means = df.mean(axis=0)

    peak_indices = detect_peaks(
        means.to_numpy(),
        top_n,
        distance,
        delta,
        top,
        bottom,
    )
    if peak_indices is None:
        raise ValueError("Peaks not found")

    return (peak_indices, means)
