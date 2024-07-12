import typer
import pandas as pd
import numpy as np

from typing import Optional
import elapy
from usecase.manual import select_manual
from usecase.common import binarize, display_data_per_pattern
from domain.common_option import get_common_options

manual = typer.Typer()


@manual.command()
def run(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Path to input file"),
    x: Optional[list[str]] = typer.Option(None, help="List of features"),
):
    options = get_common_options(ctx)

    if x is None:
        typer.echo("No features selected")
        return
    df = pd.read_csv(path, index_col=0, header=0)

    # indexを文字列に変換
    df.index = df.index.astype(str)
    if options.remove_state:
        print(f"Remove states: {options.remove_state}")
        # 解析対象に含めないstate(行)を削除
        df = df.drop(options.remove_state)

    indices, origin = select_manual(df, x)
    bins = binarize(
        df,
        indices,
        gtet=options.gtet,
        negative_inactive=options.negative_inactive,
    )
    typer.echo(bins)

    bins.to_csv(f"{options.outdir}/bins_manual.csv")

    display_data_per_pattern(bins)

    h, W = elapy.fit_exact(bins)
    acc1, acc2 = elapy.calc_accuracy(h, W, bins)

    typer.echo(f"Accuracy: {acc1}, {acc2}")

    if options.viz:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.plot(origin)
        ax.plot(indices, origin.iloc[indices], "ro")
        ax.set_xticks(np.arange(0, len(origin), 100))
        ax.set_xticklabels(np.arange(0, len(origin), 100))
        plt.tight_layout()
        fig.savefig(f"{options.imgdir}/manual.png")


if __name__ == "__main__":
    manual()
