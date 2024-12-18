from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from domain.common_option import get_common_options
from elapy import elapy
from usecase.common import binarize, display_data_per_pattern, plot_hist
from usecase.ela import run_ela
from usecase.manual import select_manual

manual = typer.Typer()
mode = "manual"


def common(path: str, x: Optional[list[str]], mode: str, options):
    df = pd.read_csv(path, index_col=0, header=0)

    # indexを文字列に変換
    df.index = df.index.astype(str)
    if options.remove_state:
        print(f"Remove states: {options.remove_state}")
        # 解析対象に含めないstate(行)を削除
        df = df.drop(options.remove_state)

    indices, origin = select_manual(df, x)
    bins, thresholds = binarize(
        df,
        indices,
        gtet=options.gtet,
        negative_inactive=options.negative_inactive,
    )
    typer.echo(bins)

    bins.to_csv(f"{options.outdir}/bins_{mode}.csv")

    h, W = elapy.fit_exact(bins)
    acc1, acc2 = elapy.calc_accuracy(h, W, bins)

    typer.echo(f"Accuracy: {acc1}, {acc2}")
    display_data_per_pattern(bins)

    return bins, indices, origin, thresholds


@manual.command()
def bin(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Path to input file"),
    x: Optional[list[str]] = typer.Option(None, help="List of features"),
):
    options = get_common_options(ctx)

    if x is None:
        typer.echo("No features selected")
        return
    bins, indices, origin, thresholds = common(path, x, mode, options)

    if options.viz:

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.plot(origin)
        ax.plot(indices, origin.iloc[indices], "ro")
        ax.set_xticks(np.arange(0, len(origin), 100))
        ax.set_xticklabels(np.arange(0, len(origin), 100))
        plt.tight_layout()
        fig.savefig(f"{options.imgdir}/manual.png")

        plot_hist(path, indices, thresholds, mode, options.imgdir)


@manual.command()
def ela(
    ctx: typer.Context,
    path: str = typer.Argument(
        ...,
        help="Path to the input data",
    ),
    x: Optional[list[str]] = typer.Option(None, help="List of features"),
    energy_th: float = typer.Option(
        None,
        help="Energy threshold",
    ),
    weighted_count: bool = typer.Option(
        False,
        help="Use weighted count",
    ),
):
    options = get_common_options(ctx)
    bins, indices, origin, _ = common(path, x, mode, options)

    # ELAを計算する
    run_ela(bins, energy_threshold=energy_th, weighted_count=weighted_count)


if __name__ == "__main__":
    manual()
