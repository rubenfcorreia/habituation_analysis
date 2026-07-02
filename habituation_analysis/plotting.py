from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


POSTER_DPI = 300
POSTER_FONT_SIZE = 18
POSTER_LABEL_SIZE = 20
POSTER_TITLE_SIZE = 24
POSTER_LEGEND_SIZE = 16
POSTER_NOTE_SIZE = 14
SPINE_COLOR = "#888888"
GRID_ALPHA = 0.16


def set_poster_style() -> None:
    """Apply poster-ready defaults for saved statistics figures."""
    mpl.rcParams.update(
        {
            "font.size": POSTER_FONT_SIZE,
            "axes.titlesize": POSTER_TITLE_SIZE,
            "axes.labelsize": POSTER_LABEL_SIZE,
            "xtick.labelsize": POSTER_FONT_SIZE,
            "ytick.labelsize": POSTER_FONT_SIZE,
            "legend.fontsize": POSTER_LEGEND_SIZE,
            "figure.titlesize": POSTER_TITLE_SIZE,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": SPINE_COLOR,
            "axes.grid": False,
            "grid.alpha": GRID_ALPHA,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "savefig.dpi": POSTER_DPI,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.22,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def lighten_color(color: str, amount: float = 0.42) -> str:
    """Return a lighter version of *color*."""
    try:
        rgb = mcolors.to_rgb(color)
    except ValueError:
        return color
    amount = max(0.0, min(1.0, float(amount)))
    lightened = tuple(channel + (1.0 - channel) * amount for channel in rgb)
    return mcolors.to_hex(lightened)


def style_axes(
    ax,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
):
    """Style one axes."""
    ax.figure.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if title:
        ax.set_title(title, fontsize=POSTER_TITLE_SIZE, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=POSTER_LABEL_SIZE, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=POSTER_LABEL_SIZE, labelpad=6)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.tick_params(
        axis="both",
        which="major",
        length=4,
        width=1.0,
        labelsize=POSTER_FONT_SIZE,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    return ax


def _clamp_font_size(value: float, *, minimum: float, maximum: float) -> float:
    return float(max(minimum, min(maximum, float(value))))


def scale_axes_text(ax, *, scale: float) -> None:
    """Scale title, labels, tick labels, legends, and annotations for an axes."""
    try:
        scale = float(scale)
    except Exception:
        scale = 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    title_size = _clamp_font_size(POSTER_TITLE_SIZE * scale, minimum=10.0, maximum=28.0)
    label_size = _clamp_font_size(POSTER_LABEL_SIZE * scale, minimum=9.0, maximum=22.0)
    tick_size = _clamp_font_size(POSTER_FONT_SIZE * scale, minimum=8.0, maximum=18.0)
    legend_size = _clamp_font_size(POSTER_LEGEND_SIZE * scale, minimum=8.0, maximum=16.0)
    note_size = _clamp_font_size(POSTER_NOTE_SIZE * scale, minimum=7.0, maximum=14.0)
    if ax.title.get_text():
        ax.title.set_fontsize(title_size)
    if ax.xaxis.label.get_text():
        ax.xaxis.label.set_fontsize(label_size)
    if ax.yaxis.label.get_text():
        ax.yaxis.label.set_fontsize(label_size)
    for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        tick.set_fontsize(tick_size)
    for text in ax.texts:
        if text is ax.title or text is ax.xaxis.label or text is ax.yaxis.label:
            continue
        text.set_fontsize(note_size)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_size)
        title = legend.get_title()
        if title is not None and title.get_text():
            title.set_fontsize(legend_size)

def set_sparse_numeric_ticks(ax, *, xbins: int = 6, ybins: int = 5) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins, prune=None))


def style_legend_outside(ax, *, loc: str = "upper left", anchor=(1.02, 1.0)):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc=loc, bbox_to_anchor=anchor, frameon=False)


def suppress_interior_labels(axes, *, keep_left: bool = True, keep_bottom: bool = True) -> None:
    axes = np.asarray(axes, dtype=object)
    if axes.ndim != 2:
        return
    nrows, ncols = axes.shape
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if ax is None:
                continue
            if keep_left and c > 0:
                ax.set_ylabel("")
            if keep_bottom and r < nrows - 1:
                ax.set_xlabel("")


def finalize_figure(fig) -> None:
    try:
        fig.align_ylabels()
    except Exception:
        pass


def _finite_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def poster_boxplot(
    ax,
    data,
    labels,
    *,
    colors,
    title: str,
    xlabel: str,
    ylabel: str,
    sample_sizes: list[int] | None = None,
    show_violin: bool = True,
    show_points: bool = True,
    show_summary_markers: bool = True,
    show_legend: bool = True,
    seed: int = 0,
    violin_alpha: float = 0.28,
    box_alpha: float = 0.55,
    jitter_width: float = 0.11,
    point_scale: float = 1.0,
) -> None:
    """
    Sleep-state-style boxplot:
    - light violin behind
    - semi-transparent colored box
    - jittered raw points
    - mean circle + median diamond
    - n labels near top
    """
    arrays = [_finite_array(arr) for arr in data]
    n = len(arrays)
    positions = np.arange(1, n + 1, dtype=float)

    style_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_axisbelow(True)
    ax.title.set_fontweight("bold")
    if ax.xaxis.label.get_text():
        ax.xaxis.label.set_fontweight("bold")
    if ax.yaxis.label.get_text():
        ax.yaxis.label.set_fontweight("bold")

    if sample_sizes is None:
        sample_sizes = [int(arr.size) for arr in arrays]

    if not any(arr.size for arr in arrays):
        ax.text(
            0.5,
            0.5,
            "No data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=90 if len(labels) > 4 else 0)
        return

    rng = np.random.default_rng(seed)
    try:
        point_scale = float(point_scale)
    except Exception:
        point_scale = 1.0
    if not np.isfinite(point_scale) or point_scale <= 0.0:
        point_scale = 1.0
    point_scale = float(np.clip(point_scale, 0.20, 0.90))

    finite_all = np.concatenate([arr for arr in arrays if arr.size])
    y_min = float(np.nanmin(finite_all))
    y_max = float(np.nanmax(finite_all))
    y_span = y_max - y_min
    if not np.isfinite(y_span) or y_span <= 0:
        y_span = 1.0
    lower_margin = 0.08 * y_span
    upper_margin = 0.18 * y_span
    ax.set_ylim(y_min - lower_margin, y_max + upper_margin)

    if show_violin:
        for pos, arr, color in zip(positions, arrays, colors):
            if arr.size == 0:
                continue
            vp = ax.violinplot(
                [arr],
                positions=[pos],
                widths=0.80,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor(lighten_color(color, 0.16))
                body.set_edgecolor(lighten_color(color, 0.0))
                body.set_alpha(violin_alpha)
                body.set_linewidth(1.1)

    plot_data = [arr if arr.size else np.array([np.nan], dtype=float) for arr in arrays]
    bp = ax.boxplot(
        plot_data,
        positions=positions,
        widths=0.44,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.8},
        whiskerprops={"color": "#555555", "linewidth": 1.4},
        capprops={"color": "#555555", "linewidth": 1.4},
        boxprops={"edgecolor": "#555555", "linewidth": 1.4},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(box_alpha)
        patch.set_edgecolor("#555555")
        patch.set_linewidth(1.4)

    if show_points:
        for pos, arr, color in zip(positions, arrays, colors):
            if arr.size == 0:
                continue
            jitter = rng.uniform(-jitter_width, jitter_width, size=arr.size)
            ax.scatter(
                np.full(arr.size, pos, dtype=float) + jitter,
                arr,
                s=11.0 * point_scale,
                c=color,
                alpha=0.55,
                linewidths=0,
                zorder=3,
            )

    if show_summary_markers:
        for pos, arr in zip(positions, arrays):
            if arr.size == 0:
                continue
            mean_val = float(np.nanmean(arr))
            median_val = float(np.nanmedian(arr))
            ax.scatter(
                [pos - 0.02],
                [mean_val],
                s=34.0 * point_scale,
                marker="o",
                c="#111111",
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )
            ax.scatter(
                [pos + 0.02],
                [median_val],
                s=34.0 * point_scale,
                marker="D",
                c="#7a7a7a",
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )

    y_text = y_max + 0.07 * y_span
    for pos, count in zip(positions, sample_sizes):
        ax.text(
            pos,
            y_text,
            f"n={int(count)}",
            ha="center",
            va="bottom",
            fontsize=POSTER_FONT_SIZE,
            fontweight="bold",
            color="#444444",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90 if len(labels) > 4 else 0)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    if show_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="#111111",
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=7,
                label="Mean",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="None",
                markerfacecolor="#7a7a7a",
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=7,
                label="Median",
            ),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False, title=None)


def save_figure(fig, stem: str, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"

    with mpl.rc_context(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.32)

    fig.savefig(
        png_path,
        bbox_inches="tight",
        pad_inches=0.32,
        dpi=POSTER_DPI,
    )
    return svg_path, png_path