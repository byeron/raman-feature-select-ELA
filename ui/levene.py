import typer
import numpy as np
import matplotlib.pyplot as plt

from usecase.common import run_common
from domain.common_option import get_common_options

levene = typer.Typer()


@levene.command()
def run(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Path to the input data"),
    top_n: int = typer.Option(7, help="Top nth lowest p-value features"),
):

    # Levene検定をもとに特徴量を選択する
    typer.echo("Processing...")
    mode = "levene"
    options = get_common_options(ctx)
    indices, origin = run_common(path, top_n, mode, options)

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


if __name__ == "__main__":
    levene()
