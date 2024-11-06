import typer
from usecase.common import line_plot, run_common
from domain.common_option import get_common_options
from usecase.ela import run_ela

peak = typer.Typer()
mode = "peak"


@peak.command()
def bin(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Path to the input file"),
    top_n: int = typer.Option(7, help="Number of peaks"),
):
    options = get_common_options(ctx)
    _, indices, origin = run_common(path, top_n, mode, options)

    if options.viz:
        line_plot(indices, origin, mode, options.imgdir)


@peak.command()
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
    weighted_count: bool = typer.Option(
        False,
        help="Use weighted count",
    )
):
    options = get_common_options(ctx)
    bins, _, _ = run_common(path, top_n, mode, options)

    # ELAを計算する
    run_ela(bins, energy_threshold=energy_th, weighted_count=weighted_count)


if __name__ == "__main__":
    peak()
