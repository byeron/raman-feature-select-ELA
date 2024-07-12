import typer

from usecase.common import line_plot, run_common
from domain.common_option import get_common_options

mvi = typer.Typer()


@mvi.command()
def run(  # Median Variablity Index, 中央値変動指数
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Path to the input file"),
    top_n: int = typer.Option(7, help="Top nth lowest p-value features"),
):

    mode = "mvi"
    options = get_common_options(ctx)
    indices, origin = run_common(path, top_n, mode, options)

    if options.viz:
        line_plot(indices, origin, mode, options.imgdir)


if __name__ == "__main__":
    mvi()
