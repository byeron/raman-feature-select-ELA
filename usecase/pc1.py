import numpy as np
import pandas as pd

from usecase.detect_peaks import detect_peaks


def select_pc1(
    df: pd.DataFrame,
    top_n: int,
    distance: float = 10.0,
    delta: float = 0.1,
    top: float = None,
    bottom: float = None,
) -> tuple[list[int], pd.Series]:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=1)
    pca.fit(df)
    weights = pca.components_[0]

    weights = np.abs(weights)  # 重みの絶対値を取る

    if top is None:
        top = weights.max()
    if bottom is None:
        bottom = weights.min()

    indices = detect_peaks(weights, top_n, distance, delta, top, bottom)
    if indices is None:
        raise ValueError("Peaks not found")

    return indices, pd.Series(weights, index=df.columns)
