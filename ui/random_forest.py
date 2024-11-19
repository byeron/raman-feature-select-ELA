import typer

from domain.common_option import get_common_options
from usecase.common import line_plot, plot_hist, run_common_rf
from usecase.ela import run_ela

random_forest = typer.Typer()
mode = "random-forest"


@random_forest.command()
def bin(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Path to the input file"),
    top_n: int = typer.Option(7, help="Number of peaks"),
    n_estimators: int = typer.Option(100, help="Number of trees in the forest"),
    random_state: int = typer.Option(0, help="Random state"),
    criterion: str = typer.Option("gini", help="Criterion"),
    max_depth: int = typer.Option(None, help="Max depth of the tree"),
    min_samples_leaf: int = typer.Option(1, help="Minimum samples in leaf node"),
    bootstrap: bool = typer.Option(True, help="Bootstrap"),
):
    options = get_common_options(ctx)
    _, indices, origin, thresholds = run_common_rf(
        path,
        top_n,
        options,
        n_estimators,
        random_state,
        criterion,
        max_depth,
        min_samples_leaf,
        bootstrap,
    )

    if options.viz:
        line_plot(indices, origin, mode, options.imgdir)
        plot_hist(path, indices, thresholds, mode, options.imgdir)


@random_forest.command()
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
    ),
    n_estimators: int = typer.Option(100, help="Number of trees in the forest"),
    random_state: int = typer.Option(0, help="Random state"),
    criterion: str = typer.Option("gini", help="Criterion"),
    max_depth: int = typer.Option(None, help="Max depth of the tree"),
    min_samples_leaf: int = typer.Option(1, help="Minimum samples in leaf node"),
    bootstrap: bool = typer.Option(True, help="Bootstrap"),
):
    options = get_common_options(ctx)
    bins, _, _, _ = run_common_rf(
        path,
        top_n,
        options,
        n_estimators,
        random_state,
        criterion,
        max_depth,
        min_samples_leaf,
        bootstrap,
    )

    # ELAを計算する
    run_ela(bins, energy_threshold=energy_th, weighted_count=weighted_count)


if __name__ == "__main__":
    random_forest()
