import typer

from domain.common_option import get_common_options
from usecase.common import line_plot, plot_hist, run_common
from usecase.ela import run_ela

cv = typer.Typer()
mode = "cv"


@cv.command()
def bin(
    ctx: typer.Context,
    path: str = typer.Argument(
        ...,
        help="Path to the input data",
    ),
    top_n: int = typer.Option(
        7,
        help="Top n coefficient of variation features",
    ),
):

    # 変動係数をもとに特徴量を選択する
    options = get_common_options(ctx)
    _, indices, origin, thresholds = run_common(path, top_n, mode, options)

    if options.viz:
        line_plot(indices, origin, "cv", options.imgdir)
        plot_hist(path, indices, thresholds, mode, options.imgdir)


@cv.command()
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
    bins, _, _, _ = run_common(path, top_n, mode, options)

    # ELAを計算する
    run_ela(bins, energy_threshold=energy_th, weighted_count=weighted_count)


if __name__ == "__main__":
    cv()
