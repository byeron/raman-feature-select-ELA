import typer

from usecase.common import line_plot, run_common
from domain.common_option import get_common_options

robust_cv = typer.Typer()


@robust_cv.command()
def run(
    ctx: typer.Context,
    path: str = typer.Argument(
        ...,
        help="Path to the input file",
    ),
    top_n: int = typer.Option(
        7,
        help="Top n coefficient of variation features",
    ),
):

    # ロバストな変動係数をもとに特徴量を選択する
    mode = "robust_cv"
    options = get_common_options(ctx)
    indices, origin = run_common(path, top_n, mode, options)

    if options.viz:
        line_plot(indices, origin, mode, options.imgdir)


if __name__ == "__main__":
    robust_cv()
