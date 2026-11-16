"""Robust metrics across narrative types: one panel per metric (double column)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import figure_labels as labels
from .figure_style import BLUE, DOUBLE_COLUMN, INK, draw_estimates, mm, plt, save


def draw_forest_panel(ax, hit, all_tests: pd.DataFrame, scopes: list[str]) -> float:
    """beta and 95% CI of one robust metric in every scope; returns the largest |beta| + CI for the x-axis."""
    same_test = ((all_tests["finding"] == hit.finding) & (all_tests["feature"] == hit.feature)
                 & (all_tests["term"] == hit.term))
    e = all_tests[same_test].set_index("scope").reindex(scopes)
    y = np.arange(len(scopes))[::-1]
    tested = e["beta"].notna().to_numpy()
    sig = (e["p"] < .05).to_numpy() & tested
    ax.axvline(0, color=INK, lw=.5, ls=(0, (2, 2)))
    half_width = 1.96 * e["se"]
    draw_estimates(ax, e["beta"][tested], y[tested], (e["beta"] - half_width)[tested], (e["beta"] + half_width)[tested],
                   sig[tested], BLUE)
    ax.set_yticks(y, [labels.scope(s) for s in scopes])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"{labels.metric(hit.feature)}\n{labels.finding_term(hit.finding, hit.term)}", loc="left",
                 fontsize=6.5, pad=3)
    return np.nanmax(np.abs(e["beta"]) + 1.96 * e["se"])


def draw_text_type_forest_figure(hits: pd.DataFrame, all_tests: pd.DataFrame, scopes: list[str], path: Path,
                                 max_panels: int = 12, ncol: int = 4, finding: str | None = None,
                                 term: str | None = None) -> None:
    """
    Figures 4 and 6: beta and 95% CI per text type for each robust metric (its strongest finding,
    or only `finding` / `term` when given); filled = p < .05.
    """
    if finding:
        hits = hits[hits["finding"].eq(finding)]
    if term:
        hits = hits[hits["term"].eq(term)]
    robust = hits[hits["verdict"].eq("robust")]
    # The strongest test of each metric, strongest metrics first.
    top = robust.sort_values("q").drop_duplicates("feature").head(max_panels)
    if not len(top):
        return
    ncol = min(ncol, len(top))
    nrow = int(np.ceil(len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=mm(DOUBLE_COLUMN, 30 * nrow + 8), squeeze=False, sharex=True,
                             sharey=True, gridspec_kw=dict(hspace=.7, wspace=.12))
    lim = 0
    for ax, hit in zip(axes.flat, top.itertuples()):
        lim = max(lim, draw_forest_panel(ax, hit, all_tests, scopes))
    for ax in axes.flat[len(top):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("β, 95% CI")
    axes[0, 0].set_xlim(-lim * 1.05, lim * 1.05)
    save(fig, path)
