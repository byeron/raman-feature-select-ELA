import typer
from typing import Optional


class CommonOptions:
    def __init__(
        self,
        remove_state: Optional[list[str]],
        outdir: str,
        distance: float,
        delta: float,
        top: float,
        bottom: float,
        gtet: bool,
        negative_inactive: bool,
        viz: bool,
        imgdir: str,
    ):
        self.remove_state = remove_state
        self.outdir = outdir
        self.distance = distance
        self.delta = delta
        self.top = top
        self.bottom = bottom
        self.gtet = gtet
        self.negative_inactive = negative_inactive
        self.viz = viz
        self.imgdir = imgdir


def common_options_callback(
    ctx: typer.Context,
    remove_state: Optional[list[str]] = typer.Option(
        None,
        help="List of states to remove",
    ),
    outdir: str = typer.Option(
        "output",
        help="Output directory"
        ),
    distance: float = typer.Option(
        10.0,
        help="Minimum distance between peaks",
    ),
    delta: float = typer.Option(0.1, help="Step for peak search"),
    top: float = typer.Option(None, help="Max value for peak search"),
    bottom: float = typer.Option(
        None, help="Min value for peak search"
        ),
    gtet: bool = typer.Option(
        True,
        help="Greater than(False) or equal(True, default) to the mean",
    ),
    negative_inactive: bool = typer.Option(
        False,
        help="Inactive features set to -1",
    ),
    viz: bool = typer.Option(False, help="Visualize the peaks"),
    imgdir: str = typer.Option("img", help="Image directory"),
):
    ctx.obj = CommonOptions(
        remove_state=remove_state,
        outdir=outdir,
        distance=distance,
        delta=delta,
        top=top,
        bottom=bottom,
        gtet=gtet,
        negative_inactive=negative_inactive,
        viz=viz,
        imgdir=imgdir,
    )


def get_common_options(ctx: typer.Context) -> CommonOptions:
    return ctx.obj