from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from scipy.signal import find_peaks
from scipy.stats import levene as levene_test

app = typer.Typer()
options = {"path": "", "outdir": "output", "imgdir": "img"}


@app.callback()
def callback(path: str, outdir: str = "output", imgdir: str = "img"):
    options["path"] = path
    options["outdir"] = outdir
    options["imgdir"] = imgdir


@app.command()
def run(
    outdir: str = typer.Option(
        options["outdir"],
        help="Output directory",
    )
):
    typer.echo(f"Output directory: {outdir}")


@app.command()
def manual(
    x: Optional[list[str]] = typer.Option(None, help="List of features"),
    outdir: str = typer.Option(options["outdir"], help="Output directory"),
    gtet: bool = typer.Option(
        True, help="Greater than(False) or equal(True, default) to the mean"
    ),
    negative_inactive: bool = typer.Option(
        False,
        help="Inactive features set to -1",
    ),
    viz: bool = typer.Option(False, help="Visualize the peaks"),
):
    if x is None:
        typer.echo("No features selected")
        return
    df = pd.read_csv(options["path"], index_col=0, header=0)
    indices, origin = select_manual(df, x)
    bins = binarize(
        df,
        indices,
        gtet=gtet,
        negative_inactive=negative_inactive,
    )
    typer.echo(bins)

    bins.to_csv(f"{outdir}/bins_manual.csv")

    display_data_per_pattern(bins)

    if viz:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.plot(origin)
        ax.plot(indices, origin[indices], "ro")
        ax.set_xticks(np.arange(0, len(origin), 100))
        ax.set_xticklabels(np.arange(0, len(origin), 100))
        plt.tight_layout()
        fig.savefig(f"{options['imgdir']}/manual.png")


@app.command()
def peak(
    top_n: int = typer.Option(7, help="Number of peaks"),
    remove_state: Optional[list[str]] = typer.Option(
        None,
        help="List of states to remove",
    ),
    outdir: str = typer.Option(options["outdir"], help="Output directory"),
    distance: float = typer.Option(
        10.0,
        help="Minimum distance between peaks",
    ),
    delta: float = typer.Option(0.1, help="Step for peak search"),
    top: float = typer.Option(None, help="Max value for peak search"),
    bottom: float = typer.Option(None, help="Min value for peak search"),
    gtet: bool = typer.Option(
        True, help="Greater than(False) or equal(True, default) to the mean"
    ),
    negative_inactive: bool = typer.Option(
        False,
        help="Inactive features set to -1",
    ),
    viz: bool = typer.Option(False, help="Visualize the peaks"),
):
    df = pd.read_csv(options["path"], index_col=0, header=0)

    # indexを文字列に変換
    df.index = df.index.astype(str)
    if remove_state:
        print(type(df.index.unique()[0]))
        print(f"Remove states: {remove_state}")
        # 解析対象に含めないstate(行)を削除
        df = df.drop(remove_state)

    peak_indices, origin = select_peak(
        df, top_n, distance=distance, delta=delta, top=top, bottom=bottom
    )
    bins = binarize(
        df,
        peak_indices,
        gtet=gtet,
        negative_inactive=negative_inactive,
    )
    typer.echo(bins)

    bins.to_csv(f"{outdir}/bins_peak.csv")

    # display the number of data / activity patterns
    display_data_per_pattern(bins)

    if viz:
        line_plot(peak_indices, origin, "peak", options["imgdir"])


@app.command()
def levene(
    top_n: int = typer.Option(7, help="Top nth lowest p-value features"),
    remove_state: Optional[list[str]] = typer.Option(
        None,
        help="List of states to remove",
    ),
    outdir: str = typer.Option(options["outdir"], help="Output directory"),
    distance: float = typer.Option(
        10.0,
        help="Minimum distance between peaks",
    ),
    delta: float = typer.Option(0.001, help="Step for peak search"),
    top: float = typer.Option(None, help="Max value for peak search"),
    bottom: float = typer.Option(None, help="Min value for peak search"),
    gtet: bool = typer.Option(
        True, help="Greater than(False) or equal(True, default) to the mean"
    ),
    negative_inactive: bool = typer.Option(
        False,
        help="Inactive features set to -1",
    ),
    viz: bool = typer.Option(False, help="Visualize the peaks"),
):
    # Levene検定をもとに特徴量を選択する
    typer.echo("Processing...")
    df = pd.read_csv(options["path"], index_col=0, header=0)

    # indexを文字列に変換
    df.index = df.index.astype(str)
    if remove_state:
        print(type(df.index.unique()[0]))
        print(f"Remove states: {remove_state}")
        # 解析対象に含めないstate(行)を削除
        df = df.drop(remove_state)

    _indices, origin = select_levene(df, top_n, distance, delta, top, bottom)
    bins = binarize(
        df,
        _indices,
        gtet=gtet,
        negative_inactive=negative_inactive,
    )
    typer.echo(bins)

    bins.to_csv(f"{outdir}/bins_levene.csv")

    display_data_per_pattern(bins)

    if viz:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.plot(origin)
        ax.plot(_indices, origin[_indices], "ro")
        ax.set_xticks(np.arange(0, len(origin), 100))
        ax.set_xticklabels(np.arange(0, len(origin), 100))
        # y軸を対数スケールにする
        # ax.set_yscale("log")
        ax.set_ylabel("-log(p-value)")
        plt.tight_layout()
        fig.savefig(f"{options['imgdir']}/levene.png")


@app.command()
def pc1(
    top_n: int = typer.Option(7, help="Top nth lowest p-value features"),
    remove_state: Optional[list[str]] = typer.Option(
        None,
        help="List of states to remove",
    ),
    outdir: str = typer.Option(options["outdir"], help="Output directory"),
    distance: float = typer.Option(
        10.0,
        help="Minimum distance between peaks",
    ),
    delta: float = typer.Option(0.1, help="Step for peak search"),
    top: float = typer.Option(None, help="Max value for peak search"),
    bottom: float = typer.Option(None, help="Min value for peak search"),
    gtet: bool = typer.Option(
        True, help="Greater than(False) or equal(True, default) to the mean"
    ),
    negative_inactive: bool = typer.Option(
        False,
        help="Inactive features set to -1",
    ),
    viz: bool = typer.Option(False, help="Visualize the peaks"),
):
    # PCAの第一主成分の重みの上位を選択する
    df = pd.read_csv(options["path"], index_col=0, header=0)

    # indexを文字列に変換
    df.index = df.index.astype(str)
    if remove_state:
        print(type(df.index.unique()[0]))
        print(f"Remove states: {remove_state}")
        # 解析対象に含めないstate(行)を削除
        df = df.drop(remove_state)

    indices, weights = select_pc1(df, top_n, distance, delta, top, bottom)
    bins = binarize(
        df,
        indices,
        gtet=gtet,
        negative_inactive=negative_inactive,
    )
    typer.echo(bins)

    bins.to_csv(f"{outdir}/bins_pc1.csv")

    display_data_per_pattern(bins)

    if viz:
        line_plot(indices, weights, "pc1", options["imgdir"])


def get_data_per_pattern(bins):
    n_data = bins.shape[1]
    n_bits = bins.shape[0]
    n_patterns = 2**n_bits
    data_per_pattern = n_data / n_patterns
    return data_per_pattern


def display_data_per_pattern(bins):
    n_data = bins.shape[1]
    n_bits = bins.shape[0]
    data_per_pattern = get_data_per_pattern(bins)
    print(f"data/patterns > {data_per_pattern} ({n_data} / 2^{n_bits})")
    print()


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


@app.command()
def cv(
    top_n: int = typer.Option(
        7,
        help="Top n coefficient of variation features",
    ),
    remove_state: Optional[list[str]] = typer.Option(
        None,
        help="List of states to remove",
    ),
    outdir: str = typer.Option(options["outdir"], help="Output directory"),
    distance: float = typer.Option(
        10.0,
        help="Minimum distance between peaks",
    ),
    delta: float = typer.Option(0.1, help="Step for peak search"),
    top: float = typer.Option(None, help="Max value for peak search"),
    bottom: float = typer.Option(None, help="Min value for peak search"),
    gtet: bool = typer.Option(
        True, help="Greater than(False) or equal(True, default) to the mean"
    ),
    negative_inactive: bool = typer.Option(
        False,
        help="Inactive features set to -1",
    ),
    viz: bool = typer.Option(False, help="Visualize the peaks"),
):
    # 変動係数をもとに特徴量を選択する
    df = pd.read_csv(options["path"], index_col=0, header=0)

    # indexを文字列に変換
    df.index = df.index.astype(str)
    if remove_state:
        print(type(df.index.unique()[0]))
        print(f"Remove states: {remove_state}")
        # 解析対象に含めないstate(行)を削除
        df = df.drop(remove_state)

    cv_indices, origin = select_cv(df, top_n, distance, delta, top, bottom)
    bins = binarize(
        df,
        cv_indices,
        gtet=gtet,
        negative_inactive=negative_inactive,
    )
    typer.echo(bins)

    bins.to_csv(f"{outdir}/bins_cv.csv")

    if viz:
        line_plot(cv_indices, origin, "cv", options["imgdir"])


@app.command()
def robust_cv(
    top_n: int = typer.Option(7, help="Top n coefficient of variation features"),
    outdir: str = typer.Option(options["outdir"], help="Output directory"),
    distance: float = typer.Option(
        10.0,
        help="Minimum distance between peaks",
    ),
    delta: float = typer.Option(0.1, help="Step for peak search"),
    top: float = typer.Option(None, help="Max value for peak search"),
    bottom: float = typer.Option(None, help="Min value for peak search"),
    gtet: bool = typer.Option(
        True, help="Greater than(False) or equal(True, default) to the mean"
    ),
    negative_inactive: bool = typer.Option(
        False,
        help="Inactive features set to -1",
    ),
    viz: bool = typer.Option(False, help="Visualize the peaks"),
):
    # ロバストな変動係数をもとに特徴量を選択する
    df = pd.read_csv(options["path"], index_col=0, header=0)
    cv_indices, origin = select_robust_cv(df, top_n, distance, delta, top, bottom)
    bins = binarize(df, cv_indices, gtet=gtet, negative_inactive=negative_inactive)
    typer.echo(bins)

    bins.to_csv(f"{outdir}/bins_robust_cv.csv")

    if viz:
        line_plot(cv_indices, origin, "robust_cv", options["imgdir"])


@app.command()
def mvi(  # Median Variablity Index, 中央値変動指数
    top_n: int = typer.Option(7, help="Top nth lowest p-value features"),
    remove_state: Optional[list[str]] = typer.Option(
        None,
        help="List of states to remove",
    ),
    outdir: str = typer.Option(options["outdir"], help="Output directory"),
    distance: float = typer.Option(
        10.0,
        help="Minimum distance between peaks",
    ),
    delta: float = typer.Option(0.1, help="Step for peak search"),
    top: float = typer.Option(None, help="Max value for peak search"),
    bottom: float = typer.Option(None, help="Min value for peak search"),
    gtet: bool = typer.Option(
        True, help="Greater than(False) or equal(True, default) to the mean"
    ),
    negative_inactive: bool = typer.Option(
        False,
        help="Inactive features set to -1",
    ),
    viz: bool = typer.Option(False, help="Visualize the peaks"),
):
    df = pd.read_csv(options["path"], index_col=0, header=0)

    # indexを文字列に変換
    df.index = df.index.astype(str)
    if remove_state:
        print(type(df.index.unique()[0]))
        print(f"Remove states: {remove_state}")
        # 解析対象に含めないstate(行)を削除
        df = df.drop(remove_state)

    _indices, origin = select_mvi(df, top_n, distance, delta, top, bottom)
    bins = binarize(
        df,
        _indices,
        gtet=gtet,
        negative_inactive=negative_inactive,
    )
    typer.echo(bins)

    bins.to_csv(f"{outdir}/bins_mvi.csv")

    if viz:
        line_plot(_indices, origin, "mvi", options["imgdir"])


def line_plot(indices, origin, key, outdir):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(origin)
    ax.plot(indices, origin[indices], "ro")
    ax.set_xticks(np.arange(0, len(origin), 100))
    ax.set_xticklabels(np.arange(0, len(origin), 100))
    plt.tight_layout()
    fig.savefig(f"{outdir}/{key}.png")


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


def select_manual(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[list[int], pd.Series]:
    indices = [df.columns.get_loc(i) for i in features]
    return indices, df.mean(axis=0)


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


def detect_peaks(series, top_n, distance, delta, top, bottom, reverse=False):
    if top is None or bottom is None:
        raise ValueError("top and bottom must be specified")

    _iterator = np.arange(top, bottom, -delta)
    if reverse:
        _iterator = reversed(_iterator)

    for h in _iterator:
        peaks, _ = find_peaks(series, height=h, distance=distance)

        peaks = peaks.tolist()
        print(peaks)
        if len(peaks) >= top_n:
            return peaks

    return None


def binarize(
    df,
    peak_indices,
    gtet=True,  # gtet: greater than or equal to
    negative_inactive=False,
):  # gtet: greater than or equal to
    peaks = df.iloc[:, peak_indices].copy()
    _index = peaks.index
    _columns = peaks.columns
    peaks = peaks.to_numpy()
    means = peaks.mean(axis=0)

    for i in range(len(means)):
        if gtet:
            peaks[:, i] = peaks[:, i] >= means[i]
        else:
            peaks[:, i] = peaks[:, i] > means[i]
    peaks_bin = peaks.astype(int)

    if negative_inactive:
        # 0を-1に変換
        peaks_bin = np.where(peaks_bin == 0, -1, peaks_bin)

    peaks_bin = pd.DataFrame(peaks_bin, index=_index, columns=_columns)
    return peaks_bin.T


if __name__ == "__main__":
    app()
