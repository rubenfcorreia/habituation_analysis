from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def set_poster_style() -> None:
    """Apply poster-friendly defaults for saved statistics figures."""
    mpl.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "figure.titlesize": 24,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "lines.linewidth": 2.0,
            "savefig.dpi": 300,
        }
    )


def style_axes(ax, *, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None):
    if title:
        ax.set_title(title, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax.tick_params(axis="both", which="major", length=4, width=1.0)
    return ax


def save_figure(fig, stem: str, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    return svg_path, png_path
