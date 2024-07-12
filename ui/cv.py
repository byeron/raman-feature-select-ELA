import typer

from usecase.common import line_plot, run_common
from domain.common_option import get_common_options

cv = typer.Typer()


@cv.command()
def run(
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
    mode = "cv"
    options = get_common_options(ctx)
    indices, origin = run_common(path, top_n, mode, options)

    if options.viz:
        line_plot(indices, origin, "cv", options.imgdir)


if __name__ == "__main__":
    cv()
