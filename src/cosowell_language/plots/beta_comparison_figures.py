"""
Several survey terms from the same model, side by side for each metric (beta and 95% CI):
RQ1-2 isolation vs loneliness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .figure_style import BLUE, INK, ORANGE, SINGLE_COLUMN, draw_estimates, mm, plt, save


def draw_term_comparison(est: pd.DataFrame, metrics: dict[str, str], terms: list[tuple[str, str, str]], path: Path,
                    xlabel: str = "β (s.d. per s.d. of score), 95% CI") -> None:
    """
    est: one row per (feature, term) with beta, se and `sig` (filled marker when True).
    terms: (term, legend label, colour) in display order.
    """
    est = est.set_index(["feature", "term"])
    n = len(terms)
    offsets = np.linspace(.18, -.18, n) if n > 1 else [0]
    fig, ax = plt.subplots(figsize=mm(SINGLE_COLUMN, 12 + (5 + 2.5 * n) * len(metrics)))
    y = np.arange(len(metrics))[::-1]
    ax.axvline(0, color=INK, lw=.5, ls=(0, (2, 2)))
    # The survey variables of one metric sit just above and below each other (dy).
    for (term, label, color), dy in zip(terms, offsets):
        beta = np.array([est.loc[(metric, term), "beta"] for metric in metrics])
        se = np.array([est.loc[(metric, term), "se"] for metric in metrics])
        sig = np.array([bool(est.loc[(metric, term), "sig"]) for metric in metrics])
        draw_estimates(ax, beta, y + dy, beta - 1.96 * se, beta + 1.96 * se, sig, color)
    ax.set_yticks(y, list(metrics.values()))
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(xlabel)
    handles = [Line2D([], [], marker="o", ms=3.5, color=color, ls="", label=label) for _, label, color in terms]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=.3,
              columnspacing=1.2, borderaxespad=.2)
    save(fig, path)


def screen_terms(all_tests: pd.DataFrame, finding: str, scope: str = "pooled") -> pd.DataFrame:
    """Rows of the screen for one finding and scope, with sig = FDR q < .05."""
    out = all_tests[all_tests["finding"].eq(finding) & all_tests["scope"].eq(scope)].copy()
    out["sig"] = out["q"] < .05
    return out


def draw_isolation_vs_loneliness_figure(all_tests: pd.DataFrame, metrics: dict[str, str], path: Path) -> None:
    """Figure 2 (RQ1-2): isolation and loneliness from the same model."""
    draw_term_comparison(screen_terms(all_tests, "RQ1-2 isolation vs loneliness"), metrics,
                    [("isolation_index", "Objective isolation", BLUE), ("loneliness", "Loneliness", ORANGE)], path)

