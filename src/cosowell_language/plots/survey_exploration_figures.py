"""Exploratory survey figures: associations between the survey scores (all sessions, each session), and loneliness
against isolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

from ..config import SEED
from .figure_style import BLUE, DIVERGING, DOUBLE_COLUMN, INK, SINGLE_COLUMN, mm, plt, save

LONELINESS_JITTER = .18     # loneliness is a whole number (3-9); jitter keeps overlapping records visible


SESSIONS_PER_ROW = 2
PANEL_BACKDROP = "#f7f6f3"   # behind each session panel, so the eight matrices read as separate blocks
SIGNIFICANCE_NOTE = "Numbers shown only where Benjamini–Hochberg q < .05"
TEXT_SIZE = 10           # cell numbers and score names; larger than the house 5-7 pt so the cells stay readable


def draw_score_matrix(ax, associations: pd.DataFrame, labels: dict[str, str], show_row_labels: bool,
                      colour_bar_ax=None, square: bool = True, line_colour: str = "white") -> None:
    """
    Lower triangle of a score-by-score matrix: each cell is one pair's estimate, coloured from -1 (blue) to +1
    (red). Only significant estimates (Benjamini-Hochberg q < .05) are written out, so the eye goes to them and
    the numbers stay large enough to read. Pairs without an estimate stay blank.
    """
    names = list(labels)
    estimates = pd.DataFrame(np.nan, index=names, columns=names)
    texts = pd.DataFrame("", index=names, columns=names)
    for row in associations.itertuples():
        # score_a comes before score_b in SCORES_USED, so (score_b, score_a) is below the diagonal.
        estimates.loc[row.score_b, row.score_a] = row.estimate
        if row.q < .05:
            texts.loc[row.score_b, row.score_a] = f"{row.estimate:.2f}".replace("-", "−")
    # The first row and the last column of a lower triangle are empty.
    estimates, texts = estimates.iloc[1:, :-1], texts.iloc[1:, :-1]

    sns.heatmap(estimates, mask=estimates.isna(), annot=texts, fmt="", annot_kws=dict(fontsize=TEXT_SIZE),
                cmap=DIVERGING, vmin=-1, vmax=1, linewidths=.3, linecolor=line_colour, square=square, ax=ax,
                cbar=colour_bar_ax is not None, cbar_ax=colour_bar_ax,
                cbar_kws=dict(ticks=[-1, -.5, 0, .5, 1]))
    ax.set_xticklabels([labels[name] for name in estimates.columns], rotation=45, ha="right", fontsize=TEXT_SIZE)
    ax.set_yticklabels([labels[name] for name in estimates.index] if show_row_labels else [], rotation=0,
                       fontsize=TEXT_SIZE)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)


def draw_mixed_model_figure(associations: pd.DataFrame, labels: dict[str, str], path: Path) -> None:
    """
    Every pair of scores over all sessions: the slope of the mixed model z(a) ~ z(b) + session + (1 | person),
    on the scale of Pearson's r.
    """
    fig = plt.figure(figsize=mm(DOUBLE_COLUMN, 135))
    ax = fig.add_axes([0, 0, .88, 1])
    colour_bar_ax = fig.add_axes([.9, .35, .02, .6])
    draw_score_matrix(ax, associations, labels, show_row_labels=True, colour_bar_ax=colour_bar_ax)
    colour_bar_ax.set_ylabel("β (z-scores, mixed model)", fontsize=TEXT_SIZE)
    colour_bar_ax.tick_params(labelsize=TEXT_SIZE, length=0)
    ax.text(1, 1, SIGNIFICANCE_NOTE, transform=ax.transAxes, fontsize=TEXT_SIZE, ha="right", va="top")
    save(fig, path)


def add_panel_backdrop(fig, ax, pad: float = .008, title_space: float = .025) -> None:
    """A rounded grey box behind the matrix and its title; the axes itself is made transparent so blank cells
    (scores not asked at that session, the empty half of the triangle) show the box too."""
    ax.set_facecolor("none")
    box = ax.get_position()
    fig.add_artist(FancyBboxPatch((box.x0 - pad, box.y0 - pad), box.width + 2 * pad, box.height + pad + title_space,
                                  boxstyle="round,pad=0,rounding_size=.006", transform=fig.transFigure,
                                  facecolor=PANEL_BACKDROP, edgecolor="none", zorder=-1))


def draw_session_correlation_figure(associations: pd.DataFrame, labels: dict[str, str], path: Path) -> None:
    """One score-by-score matrix of cross-sectional Pearson r per session, in the design of the mixed-model figure."""
    sessions = sorted(associations["scope"].unique(), key=lambda session: int(session[1:]))
    rows = -(-len(sessions) // SESSIONS_PER_ROW)
    fig, axes = plt.subplots(rows, SESSIONS_PER_ROW, figsize=mm(DOUBLE_COLUMN * 1.25, 75 * rows),
                             gridspec_kw=dict(wspace=.12, hspace=.28), squeeze=False)
    colour_bar_ax = fig.add_axes([.92, .6, .012, .25])
    for i, ax in enumerate(axes.flat):
        if i >= len(sessions):
            ax.set_visible(False)
            continue
        draw_score_matrix(ax, associations[associations["scope"].eq(sessions[i])], labels,
                          show_row_labels=i % SESSIONS_PER_ROW == 0, colour_bar_ax=colour_bar_ax if i == 0 else None,
                          square=False, line_colour=PANEL_BACKDROP)
        if i < len(sessions) - SESSIONS_PER_ROW:
            ax.set_xticklabels([])
        ax.set_title(sessions[i], fontsize=TEXT_SIZE + 2, fontweight="bold", pad=3)
        add_panel_backdrop(fig, ax)
    colour_bar_ax.set_ylabel("Pearson r", fontsize=TEXT_SIZE)
    colour_bar_ax.tick_params(labelsize=TEXT_SIZE, length=0)
    fig.text(.9, .925, SIGNIFICANCE_NOTE, fontsize=TEXT_SIZE, ha="right", va="bottom")
    save(fig, path)


def draw_loneliness_vs_isolation_figure(scores: pd.DataFrame, summary: pd.Series, path: Path) -> None:
    """
    Every survey record: objective isolation index (x) against loneliness (y, jittered), with the
    mean loneliness in each tenth of isolation as a guide, and Spearman rho with its 95% CI.
    """
    pair = scores.dropna(subset=["loneliness", "isolation_index"])
    rng = np.random.default_rng(SEED)
    jitter = rng.uniform(-LONELINESS_JITTER, LONELINESS_JITTER, len(pair))

    fig, ax = plt.subplots(figsize=mm(SINGLE_COLUMN, 72))
    sns.scatterplot(x=pair["isolation_index"], y=pair["loneliness"] + jitter, s=3, color=BLUE, alpha=.3, linewidth=0,
                    ax=ax)

    tenths = pd.qcut(pair["isolation_index"], 10, duplicates="drop")
    means = pair.groupby(tenths, observed=True).agg(isolation=("isolation_index", "mean"),
                                                    loneliness=("loneliness", "mean"))
    sns.lineplot(data=means, x="isolation", y="loneliness", color=INK, lw=.9, marker="o", markersize=3, ax=ax)

    ax.text(.02, .98,
            f"Spearman ρ = {summary['spearman_rho']:.2f} (95% CI {summary['ci_low']:.2f} to {summary['ci_high']:.2f})\n"
            f"{int(summary['n_records']):,} records, {int(summary['n_people']):,} people\n"
            "Black line: mean loneliness in each tenth of isolation",
            transform=ax.transAxes, ha="left", va="top", fontsize=6, color=INK)
    ax.set_xlabel("Objective isolation index (higher = more isolated)")
    ax.set_ylabel("Loneliness (3-item sum)")
    ax.set_yticks(range(3, 10))
    ax.set_ylim(2.4, 10.9)
    save(fig, path)
