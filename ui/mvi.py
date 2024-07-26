import typer

from usecase.common import line_plot, run_common
from domain.common_option import get_common_options
from usecase.ela import run_ela

mvi = typer.Typer()
mode = "mvi"


@mvi.command()
def bin(  # Median Variablity Index, 中央値変動指数
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Path to the input file"),
    top_n: int = typer.Option(7, help="Top nth lowest p-value features"),
):

    options = get_common_options(ctx)
    _, indices, origin = run_common(path, top_n, mode, options)

    if options.viz:
        line_plot(indices, origin, mode, options.imgdir)


@mvi.command()
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
        help="Threshold for energy",
    ),
):
    options = get_common_options(ctx)
    bins, _, _ = run_common(path, top_n, mode, options)

    # ELAを計算する
    run_ela(bins, energy_threshold=energy_th)


if __name__ == "__main__":
    mvi()
