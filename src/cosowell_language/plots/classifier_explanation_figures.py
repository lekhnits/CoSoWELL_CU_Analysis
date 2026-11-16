"""Top-ranked classifier, a closer look: which metrics it relies on."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import NARRATIVE_TYPES
from . import figure_labels as labels
from .figure_style import BLUE, DOUBLE_COLUMN, INK, ORANGE, mm, plt, save

N_METRICS_SHOWN = 10
# The colours of the rest of the report: loneliness orange, isolation blue.
TARGET_COLOURS = {"loneliness": ORANGE, "isolation_index": BLUE}


def draw_importance_bars(ax, of_panel: pd.DataFrame, colour: str) -> None:
    """
    The metrics with the largest mean AUC drop, largest at the top; error bar = SD over the outer folds. Metrics
    with no effect at all (the lasso set their weight to 0 in every fold) are left out.
    """
    used = of_panel[of_panel["auc_drop_mean"].ne(0)]
    top = used.nlargest(N_METRICS_SHOWN, "auc_drop_mean").iloc[::-1]
    position = np.arange(len(top))
    ax.barh(position, top["auc_drop_mean"], height=.7, color=colour, linewidth=0)
    ax.errorbar(top["auc_drop_mean"], position, xerr=top["auc_drop_std"], fmt="none", ecolor=INK, elinewidth=.6,
                capsize=1.2, capthick=.6)
    ax.axvline(0, color=INK, lw=.5)
    ax.set_yticks(position, [labels.metric(m) for m in top["metric"]])
    ax.tick_params(axis="y", length=0)


def draw_metric_importance_figure(importance: pd.DataFrame, model: str, target_names: dict, path: Path) -> None:
    """
    Metric importance of the top-ranked classifier: rows = text types, columns = labels. Bars, drop in test ROC AUC
    when the metric is shuffled (mean over the 5 outer folds); error bars, SD over the folds.
    """
    targets = list(target_names)
    types = [t for t in NARRATIVE_TYPES if t in set(importance["ntype"])]
    fig, axes = plt.subplots(len(types), len(targets), figsize=mm(DOUBLE_COLUMN, 10 + 44 * len(types)),
                             sharex=True, squeeze=False)
    letters = iter("abcdefgh")
    for row, ntype in zip(axes, types):
        for ax, target in zip(row, targets):
            of_panel = importance[importance["target"].eq(target) & importance["ntype"].eq(ntype)]
            draw_importance_bars(ax, of_panel, TARGET_COLOURS[target])
            # The letter goes in the title: the metric names fill the space left of the axes.
            ax.set_title(rf"$\bf{{{next(letters)}}}$   {target_names[target]}, {labels.scope(ntype)}", loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("Drop in ROC AUC when the metric is shuffled")
    fig.suptitle(model, x=.02, ha="left", fontsize=7)
    fig.tight_layout(h_pad=1.2)
    save(fig, path)

