import typer
import numpy as np
import matplotlib.pyplot as plt

from usecase.common import run_common
from domain.common_option import get_common_options
from usecase.ela import run_ela

levene = typer.Typer()
mode = "levene"


@levene.command()
def bin(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Path to the input data"),
    top_n: int = typer.Option(7, help="Top nth lowest p-value features"),
):

    # Levene検定をもとに特徴量を選択する
    typer.echo("Processing...")
    options = get_common_options(ctx)
    _, indices, origin = run_common(path, top_n, mode, options)

    if options.viz:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        ax.plot(origin)
        ax.plot(indices, origin.iloc[indices], "ro")
        ax.set_xticks(np.arange(0, len(origin), 100))
        ax.set_xticklabels(np.arange(0, len(origin), 100))
        # y軸を対数スケールにする
        # ax.set_yscale("log")
        ax.set_ylabel("-log(p-value)")
        plt.tight_layout()
        fig.savefig(f"{options.imgdir}/levene.png")


@levene.command()
def ela(
    ctx: typer.Context,
    path: str = typer.Argument(
        ...,
        help="Path to the input data",
    ),
    top_n: int = typer.Option(
        7,
        help="Top n coefficient of variation features",
    ),
    energy_th: float = typer.Option(
        None,
        help="Threshold of energy level",
    ),
):
    options = get_common_options(ctx)
    bins, _, _ = run_common(path, top_n, mode, options)

    # ELAを計算する
    run_ela(bins, energy_threshold=energy_th)


if __name__ == "__main__":
    levene()
