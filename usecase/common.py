import typer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from usecase.robust_cv import select_robust_cv
from usecase.cv import select_cv
from usecase.pc1 import select_pc1
from usecase.levene import select_levene
from usecase.mvi import select_mvi
from usecase.peak import select_peak
from elapy import elapy


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


def run_common(
    path: str,
    top_n: int,
    mode: str,
    options,
):
    preprocesses = {
        "cv": select_cv,
        "robust_cv": select_robust_cv,
        "pc1": select_pc1,
        "levene": select_levene,
        "mvi": select_mvi,
        "peak": select_peak,
    }
    # options = get_common_options(ctx)

    # ロバストな変動係数をもとに特徴量を選択する
    df = pd.read_csv(path, index_col=0, header=0)

    # 解析対象に含めないstate(行)を削除
    df.index = df.index.astype(str)  # indexを文字列に変換
    if options.remove_state:
        print(f"Remove states: {options.remove_state}")
        df = df.drop(options.remove_state)  # stateを削除

    # コマンドライン引数に応じた前処理を実行
    indices, origin = preprocesses[mode](
        df,
        top_n,
        options.distance,
        options.delta,
        options.top,
        options.bottom,
    )

    # バイナリ化
    bins = binarize(
        df,
        indices,
        gtet=options.gtet,
        negative_inactive=options.negative_inactive,
    )
    typer.echo(bins)
    bins.to_csv(f"{options.outdir}/bins_{mode}.csv")

    # Caluculate the accuracy
    h, W = elapy.fit_exact(bins)
    acc1, acc2 = elapy.calc_accuracy(h, W, bins)

    typer.echo(f"Accuracy:\t{acc1}, {acc2}")
    display_data_per_pattern(bins)

    return indices, origin


def line_plot(indices, origin, mode, outdir):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(origin)
    ax.plot(indices, origin.iloc[indices], "ro")
    ax.set_xticks(np.arange(0, len(origin), 100))
    ax.set_xticklabels(np.arange(0, len(origin), 100))
    plt.tight_layout()
    fig.savefig(f"{outdir}/{mode}.png")


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
    print(f"data/patterns:\t{data_per_pattern} ({n_data} / 2^{n_bits})")
