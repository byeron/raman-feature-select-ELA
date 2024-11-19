import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from elapy import elapy
from usecase.cv import select_cv
from usecase.levene import select_levene
from usecase.mvi import select_mvi
from usecase.pc1 import select_pc1
from usecase.peak import select_peak
from usecase.random_forest import select_random_forest
from usecase.robust_cv import select_robust_cv


def binarize(
    df,
    peak_indices,
    gtet=True,  # gtet: greater than or equal to
    negative_inactive=False,
    ignore_for_avg=[],
    robust=False,
):
    peaks = df.iloc[:, peak_indices].copy()
    if ignore_for_avg:
        # ignore_for_avgに含まれる行を除外して平均を計算
        _peaks = peaks.copy()
        _peaks = _peaks.drop(ignore_for_avg)

        if robust:
            centers = _peaks.median(axis=0)
        else:
            centers = _peaks.mean(axis=0)
    else:
        if robust:
            centers = peaks.median(axis=0)
        else:
            centers = peaks.mean(axis=0)
    print("centers:")
    print(f"{centers.iloc[:]}")
    print()

    _index = peaks.index
    _columns = peaks.columns
    peaks = peaks.to_numpy()

    for i in range(len(centers)):
        if gtet:
            peaks[:, i] = peaks[:, i] >= centers.iloc[i]
        else:
            peaks[:, i] = peaks[:, i] > centers.iloc[i]
    peaks_bin = peaks.astype(int)

    # 0と1の比率を出力する
    prop = peaks.sum(axis=0) / len(_index)
    print("1 appearance rate")
    print(pd.DataFrame(np.array([prop]).T, columns=["rate"], index=_columns))
    print()
    # print(pd.DataFrame(np.array([prop]).T, index=_index, columns=["rate(0 or 1)"]))

    if negative_inactive:
        # 0を-1に変換
        peaks_bin = np.where(peaks_bin == 0, -1, peaks_bin)

    peaks_bin = pd.DataFrame(peaks_bin, index=_index, columns=_columns)
    return peaks_bin.T, centers


def run_common_rf(
    path: str,
    top_n,
    options,
    n_estimators: int,
    random_state: int,
    criterion: str,
    max_depth: int,
    min_samples_leaf: int,
    bootstrap: bool,
):
    df = pd.read_csv(path, index_col=0, header=0)
    indices, origin = select_random_forest(
        df,
        top_n,
        n_estimators,
        random_state,
        criterion,
        max_depth,
        min_samples_leaf,
        bootstrap,
    )

    bins, thresholds = binarize(
        df,
        indices,
        gtet=options.gtet,
        negative_inactive=options.negative_inactive,
        ignore_for_avg=options.ignore_for_avg,
    )
    typer.echo(bins)
    bins.to_csv(f"{options.outdir}/bins_rf.csv")

    h, W = elapy.fit_exact(bins)
    acc1, acc2 = elapy.calc_accuracy(h, W, bins)

    typer.echo(f"Accuracy:\t{acc1}, {acc2}")
    display_data_per_pattern(bins)

    return bins, indices, origin, thresholds


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
    bins, thresholds = binarize(
        df,
        indices,
        gtet=options.gtet,
        negative_inactive=options.negative_inactive,
        ignore_for_avg=options.ignore_for_avg,
        robust=options.robust_th,
    )
    typer.echo(bins)
    bins.to_csv(f"{options.outdir}/bins_{mode}.csv")

    # Caluculate the accuracy
    h, W = elapy.fit_exact(bins)
    acc1, acc2 = elapy.calc_accuracy(h, W, bins)

    typer.echo(f"Accuracy:\t{acc1}, {acc2}")
    display_data_per_pattern(bins)

    return bins, indices, origin, thresholds


def line_plot(indices, origin, mode, outdir):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(origin)
    ax.plot(indices, origin.iloc[indices], "ro")
    ax.set_xticks(np.arange(0, len(origin), 100))
    ax.set_xticklabels(np.arange(0, len(origin), 100))
    plt.tight_layout()
    fig.savefig(f"{outdir}/{mode}.png")


def plot_hist(path, indices, thresholds, mode, outdir):
    fig = plt.figure(figsize=(9, 16))
    df = pd.read_csv(path, index_col=0)
    print("### DEBUG: plot hist")
    df = df.iloc[:, indices]  # Pickup selected features
    for n, ((columns_name, item), threshold) in enumerate(
        zip(df.items(), thresholds), start=1
    ):
        ax = fig.add_subplot(len(indices), 1, n)
        ax.hist(item, bins=30)
        ax.axvline(threshold, ymin=0, ymax=30, color="tab:red")
        ax.set_ylabel(f"frequency ({columns_name})")
    plt.tight_layout()
    fig.savefig(f"{outdir}/hist_{mode}.png")


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
