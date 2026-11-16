"""
Figures S4-S5: the robust markers of each research question by groups of its survey score (seaborn).

Up to three markers: one box plot per marker (box_plot_figure.draw_marker_panels). More than three: a PCA
of the markers, each session a dot coloured by its group, because many box plots side by side are hard to read.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns

from ..stats.marker_pca import principal_components_of_markers
from . import figure_labels as labels
from .box_plot_figure import draw_marker_panels
from .figure_style import DOUBLE_COLUMN, INK, ORDERED_GROUP_COLOURS, mm, panel_label, plt, save

MAX_MARKERS_FOR_BOX_PLOTS = 3


def marker_name(column: str) -> str:
    """'fpp_plural_rate | future' -> "'We' words (future)"."""
    feature, scope = column.split(" | ")
    texts = "all texts" if scope == "pooled" else labels.scope(scope).lower()
    return f"{labels.metric(feature)} ({texts})"


def ends_of_component(component: str, loadings: pd.DataFrame, n: int = 3) -> str:
    """The markers at the two ends of a component, so that the axis can be read in words."""
    order = loadings[component].sort_values()
    low = ", ".join(marker_name(m) for m in order.index[:n] if order[m] < 0)
    high = ", ".join(marker_name(m) for m in order.index[::-1][:n] if order[m] > 0)
    return f"{component}: low = {low or '–'}; high = {high or '–'}"


def draw_group_means(ax, scores: pd.DataFrame, group: str, levels: list, palette: list[str]) -> None:
    """Each group's mean PC1 and PC2 with its 95% CI."""
    for level, color in zip(levels, palette):
        of_group = scores.loc[scores[group] == level, ["PC1", "PC2"]]
        mean, half_width = of_group.mean(), 1.96 * of_group.sem()
        ax.errorbar(mean["PC1"], mean["PC2"], xerr=half_width["PC1"], yerr=half_width["PC2"], fmt="o", ms=7,
                    color=color, mec=INK, mew=.8, elinewidth=1, zorder=5, label=str(level))


def draw_marker_pca_figure(sessions: pd.DataFrame, marker_columns: list[str], group: str, xlabel: str,
                           levels: list, palette: list[str], path: Path) -> None:
    """
    a: where the middle 50% of each group's sessions lie (density contour) and the group's mean. The groups
       overlap heavily, so contours show this more clearly than one dot per session.
    b: the group means with their 95% CIs, zoomed in: where the groups actually differ.
    """
    scores, loadings, explained = principal_components_of_markers(sessions, marker_columns, group)
    # Clearly different colours, ordered low -> high: shades of one colour blur together.
    palette = ORDERED_GROUP_COLOURS[len(levels)]
    fig, (ax_all, ax_means) = plt.subplots(1, 2, figsize=mm(DOUBLE_COLUMN, 82), gridspec_kw=dict(wspace=.3))
    # The contour enclosing the densest 50% of each group's sessions (filled: between it and the peak).
    sns.kdeplot(data=scores, x="PC1", y="PC2", hue=group, hue_order=levels, palette=palette, levels=[.5, 1],
                fill=True, alpha=.15, common_norm=False, legend=False, ax=ax_all)
    sns.kdeplot(data=scores, x="PC1", y="PC2", hue=group, hue_order=levels, palette=palette, levels=[.5],
                linewidths=1, common_norm=False, legend=False, ax=ax_all)
    draw_group_means(ax_all, scores, group, levels, palette)
    draw_group_means(ax_means, scores, group, levels, palette)
    ax_means.legend(title=f"{xlabel}\n(a: middle 50% of sessions and mean;\nb: mean, 95% CI)", loc="upper left",
                    bbox_to_anchor=(1.0, 1.0), fontsize=5.5, title_fontsize=5.5, alignment="left")
    for ax in (ax_all, ax_means):
        ax.axhline(0, color=INK, lw=.3, ls=(0, (2, 2)))
        ax.axvline(0, color=INK, lw=.3, ls=(0, (2, 2)))
        ax.set_xlabel(f"PC1 ({explained['PC1']:.0%} of variance)")
        ax.set_ylabel(f"PC2 ({explained['PC2']:.0%} of variance)")
    ax_all.set_title(f"Where each group's sessions lie ({len(scores):,} sessions)", loc="left", fontsize=6)
    ax_means.set_title("Group means (zoomed in)", loc="left", fontsize=6)
    panel_label(ax_all, "a")
    panel_label(ax_means, "b")
    note = (f"PCA of the {len(marker_columns)} robust markers (normal scores). "
            + ends_of_component("PC1", loadings) + "\n" + ends_of_component("PC2", loadings))
    fig.text(0.02, -0.02, note, fontsize=5.5, va="top", ha="left")
    save(fig, path)


def draw_markers_of_question(sessions: pd.DataFrame, effects: pd.DataFrame, group: str, xlabel: str, levels: list,
                             colors: list[str], path: Path) -> None:
    """Box plots for up to three markers, a PCA of the markers for more."""
    if len(effects) <= MAX_MARKERS_FOR_BOX_PLOTS:
        draw_marker_panels(sessions, effects, group, xlabel, levels, colors, path)
    else:
        draw_marker_pca_figure(sessions, effects["metric"].tolist(), group, xlabel, levels, colors, path)
