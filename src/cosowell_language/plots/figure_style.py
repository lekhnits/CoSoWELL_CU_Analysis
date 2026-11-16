"""
Figure style following Nature's figure guidelines: 89 mm (single column) or
183 mm (double column) wide at final size, Arial 5-7 pt, 0.5 pt lines,
outward ticks, no gridlines or in-figure titles (titles go in the legend),
bold lowercase panel letters, and export as a 600 dpi PNG.
Colours: the reference palette of the dataviz guidelines (colour-blind safe). Plots are drawn with seaborn
on this theme.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SINGLE_COLUMN, DOUBLE_COLUMN = 89, 183          # mm

BLUE, RED, MID, INK, MUTED = "#2a78d6", "#e34948", "#f0efec", "#0b0b0b", "#52514e"
ORANGE, AQUA, YELLOW, GREY = "#eb6834", "#1baf7a", "#eda100", "#b5b4ae"
BLUE_RAMP = ["#b7d3f6", "#6da7ec", "#2a78d6", "#184f95"]
ORANGE_RAMP = ["#fbd4c2", "#f4a07a", "#eb6834", "#b8461b"]
DIVERGING = LinearSegmentedColormap.from_list("div", [BLUE, MID, RED])
# Distinct colours for ordered groups (low -> high), for scatter plots where shades of one colour blur together.
ORDERED_GROUP_COLOURS = {2: [BLUE, RED], 3: [BLUE, YELLOW, RED], 4: [BLUE, AQUA, YELLOW, RED]}

# Seaborn's "ticks" theme, then the Nature-style sizes on top of it.
sns.set_theme(style="ticks", rc={
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "legend.frameon": False,
    "axes.linewidth": .5, "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": .5, "ytick.major.width": .5, "xtick.color": INK, "ytick.color": INK,
    "lines.linewidth": .8, "errorbar.capsize": 0,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": .02, "figure.dpi": 150,
})


def mm(width: float, height: float) -> tuple[float, float]:
    """Figure size in inches from millimetres."""
    return width / 25.4, height / 25.4


def panel_label(ax, letter: str, x: float = -0.18, y: float = 1.06) -> None:
    """Bold lowercase panel letter at the top left of an axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, path: Path) -> None:
    """Save as PNG (600 dpi)."""
    fig.savefig(Path(path))
    plt.close(fig)


def draw_estimates(ax, estimate, position, low, high, significant, colour: str) -> None:
    """
    Estimates with their 95% CIs: filled dot when significant, open dot otherwise. The CIs come from the models
    (seaborn only draws intervals it computes from raw data), so they are drawn as error bars under seaborn dots.
    """
    points = pd.DataFrame({"estimate": np.asarray(estimate, dtype=float), "position": np.asarray(position, dtype=float),
                           "low": np.asarray(low, dtype=float), "high": np.asarray(high, dtype=float),
                           "significant": np.asarray(significant, dtype=bool)}).dropna(subset=["estimate"])
    ax.errorbar(points["estimate"], points["position"],
                xerr=[points["estimate"] - points["low"], points["high"] - points["estimate"]],
                fmt="none", ecolor=colour, elinewidth=.9)
    filled, open_ = points[points["significant"]], points[~points["significant"]]
    sns.scatterplot(data=filled, x="estimate", y="position", color=colour, s=14, linewidth=0, ax=ax, zorder=3)
    sns.scatterplot(data=open_, x="estimate", y="position", facecolor="white", edgecolor=colour, s=14,
                    linewidth=.8, ax=ax, zorder=3)
    ax.set_xlabel("")
    ax.set_ylabel("")
