import typer

from usecase.common import line_plot, run_common
from domain.common_option import get_common_options

pc1 = typer.Typer()


@pc1.command()
def run(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Path to the input data"),
    top_n: int = typer.Option(7, help="Top nth lowest p-value features"),
):
    # PCAの第一主成分の重みの上位を選択する
    mode = "pc1"
    options = get_common_options(ctx)
    indices, weights = run_common(path, top_n, mode, options)

    if options.viz:
        line_plot(indices, weights, mode, options.imgdir)


if __name__ == "__main__":
    pc1()
