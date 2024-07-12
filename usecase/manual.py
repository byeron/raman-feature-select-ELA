import pandas as pd


def select_manual(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[list[int], pd.Series]:
    indices = [df.columns.get_loc(i) for i in features]
    return indices, df.mean(axis=0)
