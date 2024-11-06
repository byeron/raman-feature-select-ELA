import typer

from usecase.common import line_plot, run_common
from domain.common_option import get_common_options
from usecase.ela import run_ela

pc1 = typer.Typer()
mode = "pc1"


@pc1.command()
def bin(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Path to the input data"),
    top_n: int = typer.Option(7, help="Top nth lowest p-value features"),
):
    # PCAの第一主成分の重みの上位を選択する
    options = get_common_options(ctx)
    _, indices, weights = run_common(path, top_n, mode, options)

    if options.viz:
        line_plot(indices, weights, mode, options.imgdir)


@pc1.command()
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
        help="Threshold of energy",
    ),
    weighted_count: bool = typer.Option(
        False,
        help="Use weighted count for stackbar plot",
    ),
):
    options = get_common_options(ctx)
    bins, _, _ = run_common(path, top_n, mode, options)

    # ELAを計算する
    run_ela(bins, energy_threshold=energy_th, weighted_count=weighted_count)


if __name__ == "__main__":
    pc1()
