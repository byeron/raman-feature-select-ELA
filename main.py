import typer
from typing import Optional

from ui.manual import manual
from ui.peak import peak
from ui.levene import levene
from ui.pc1 import pc1
from ui.cv import cv
from ui.robust_cv import robust_cv
from ui.mvi import mvi
from domain.common_option import common_options_callback

app = typer.Typer()
app.add_typer(manual, name="manual")
app.add_typer(peak, name="peak")
app.add_typer(levene, name="levene")
app.add_typer(pc1, name="pc1")
app.add_typer(cv, name="cv")
app.add_typer(robust_cv, name="robust-cv")
app.add_typer(mvi, name="mvi")


@app.callback()
def callback(
    ctx: typer.Context,
    exclude_state: Optional[list[str]] = typer.Option(
        None,
        help="List of exclude states for calculation",
    ),
    outdir: str = typer.Option(
        "output",
        help="Output directory"
    ),
    distance: float = typer.Option(
        10.0,
        help="Minimum distance between peaks",
    ),
    delta: float = typer.Option(
        0.1,
        help="Step for peak search",
    ),
    top: float = typer.Option(
        None,
        help="Max value for peak search",
    ),
    bottom: float = typer.Option(
        None,
        help="Min value for peak search",
    ),
    gtet: bool = typer.Option(
        True,
        help="Greater than(False) or equal(True, default) to the mean",
    ),
    negative_inactive: bool = typer.Option(
        False,
        help="Inactive features set to -1",
    ),
    ignore_for_avg: Optional[list[str]] = typer.Option(
        None,
        help="List of ignore states for average calculation",
    ),
    viz: bool = typer.Option(False, help="Visualize the peaks"),
    imgdir: str = typer.Option("img", help="Image directory"),
):
    common_options_callback(
        ctx,
        exclude_state,
        outdir,
        distance,
        delta,
        top,
        bottom,
        gtet,
        negative_inactive,
        ignore_for_avg,
        viz,
        imgdir,
    )


if __name__ == "__main__":
    app()
