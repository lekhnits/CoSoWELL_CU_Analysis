"""Heatmap of the pooled beta of every metric that is an FDR hit anywhere (double column)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from ..config import FINDINGS
from . import figure_labels as labels
from .figure_style import DIVERGING, DOUBLE_COLUMN, INK, mm, plt, save


def metric_by_variable_table(main: pd.DataFrame, column: str, metrics: list[str], variables: list) -> pd.DataFrame:
    """Rows: metrics; columns: (finding, term); values: `column` (beta or q) of the main scope."""
    table = {}
    for variable in variables:
        table[variable] = [main[column].get((metric, *variable), np.nan) for metric in metrics]
    return pd.DataFrame(table, index=metrics)


TEXT_SIZE = 7.5
# Short names for "held constant" notes under the columns.
HELD_CONSTANT_NAME = {"isolation_index": "isolation", "loneliness": "loneliness"}


def column_label(finding: str, term: str) -> str:
    """The score tested, and the other survey scores in the same regression ('held constant')."""
    spec = FINDINGS[finding]
    others = [other for other in spec["terms"] + spec.get("adjust", []) if other != term]
    if not others:
        return labels.term(term)
    return f"{labels.term(term)}\n({', '.join(HELD_CONSTANT_NAME[o] for o in others)} held constant)"


def label_research_questions(ax, cols: list[tuple[str, str]]) -> None:
    """One label per research question above its columns, with a bracket over them, and a white gap between
    research questions."""
    findings = [finding for finding, _ in cols]
    starts = [j for j in range(len(cols)) if j == 0 or findings[j] != findings[j - 1]]
    ends = starts[1:] + [len(cols)]
    for start, end in zip(starts, ends):
        if start:
            ax.axvline(start, color="white", lw=3)
        ax.plot([start + .1, end - .1], [-.3, -.3], color=INK, lw=.7, clip_on=False)
        name = labels.HYPOTHESIS.get(findings[start], findings[start].split()[0])
        ax.text((start + end) / 2, -.45, name, ha="center", va="bottom", fontsize=TEXT_SIZE + 1, fontweight="bold")


def draw_fdr_hit_heatmap(all_tests: pd.DataFrame, hits: pd.DataFrame, main_scope: str, path: Path,
                         max_rows: int = 40) -> None:
    """Figure S1: pooled beta of every metric that is an FDR hit anywhere. One column per survey score tested in
    each research question's regression, labelled with the scores held constant in it. Dot: q < .05 in the main
    scope; ring: FDR hit only within one narrative type."""
    if not len(hits):
        return
    cols = [(finding, term) for finding, spec in FINDINGS.items() for term in spec["terms"]]
    order = hits.sort_values("q").drop_duplicates("feature")["feature"].head(max_rows).tolist()
    main = all_tests[all_tests["scope"].eq(main_scope)].set_index(["feature", "finding", "term"])
    B = metric_by_variable_table(main, "beta", order, cols)
    Q = metric_by_variable_table(main, "q", order, cols)
    anywhere = {(hit.feature, hit.finding, hit.term) for hit in hits.itertuples()}
    lim = np.nanmax(np.abs(B.to_numpy()))
    fig, ax = plt.subplots(figsize=mm(DOUBLE_COLUMN * 1.15, 5 * len(order) + 75))
    sns.heatmap(B.to_numpy(), cmap=DIVERGING, vmin=-lim, vmax=lim, linewidths=.4, linecolor="white",
                xticklabels=[column_label(finding, term) for finding, term in cols],
                yticklabels=[labels.metric(f) for f in order], ax=ax,
                cbar_kws=dict(shrink=.35, aspect=15, pad=.02))
    colour_bar = ax.collections[0].colorbar
    colour_bar.set_label(f"β ({labels.scope(main_scope).lower()})", fontsize=TEXT_SIZE)
    colour_bar.ax.tick_params(labelsize=TEXT_SIZE)
    # Cell centres are at +0.5 in a seaborn heatmap.
    for i, f in enumerate(order):
        for j, c in enumerate(cols):
            if Q.iat[i, j] < .05:
                ax.plot(j + .5, i + .5, "o", ms=4, color=INK)
            elif (f, *c) in anywhere:
                ax.plot(j + .5, i + .5, "o", ms=5, mfc="none", mec=INK, mew=.8)
    symbols = [Line2D([], [], ls="none", marker="o", ms=4, color=INK, label="q < .05, all text types pooled"),
               Line2D([], [], ls="none", marker="o", ms=5, mfc="none", mec=INK, mew=.8,
                      label="FDR hit in one text type only")]
    # Above the colour bar, which is centred on the right.
    ax.legend(handles=symbols, loc="upper left", bbox_to_anchor=(1.01, 1), handletextpad=.3, borderaxespad=0,
              fontsize=TEXT_SIZE)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=50, ha="right", rotation_mode="anchor", fontsize=TEXT_SIZE)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=TEXT_SIZE)
    ax.set_xlabel("Survey score tested", fontsize=TEXT_SIZE + 1)
    ax.tick_params(length=0)
    label_research_questions(ax, cols)
    save(fig, path)
