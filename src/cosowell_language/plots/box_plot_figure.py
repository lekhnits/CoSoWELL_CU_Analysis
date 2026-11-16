"""Box plots of raw metric values by survey group, with effect sizes above each panel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..config import SEED
from ..stats.box_plot_effect_sizes import star_if_interval_excludes_zero
from . import figure_labels as labels
from .figure_style import INK, mm, panel_label, plt, save

LEGEND_GREY = "#c9c9c9"


def symbol_handles(with_share_bars: bool) -> list:
    handles = [Patch(facecolor=LEGEND_GREY, label="Box: middle 50% of sessions"),
               Line2D([], [], color=INK, lw=.9, label="Median"),
               Line2D([], [], marker="D", ms=2.8, mfc="white", mec=INK, mew=.5, ls="", label="Mean"),
               Line2D([], [], color=INK, lw=.5, label="Whiskers: up to 1.5 × box height")]
    if with_share_bars:
        handles.append(Patch(facecolor=LEGEND_GREY, label="Bar: % of sessions that use the words at all (95% CI)"))
    return handles


def add_symbol_legend(fig, effect_note: str, with_share_bars: bool = False) -> None:
    """What every symbol of a box-plot figure means, in one block just above the panels."""
    # Leave about 14 mm above the top panels for their titles, whatever the height of the figure.
    height_mm = fig.get_figheight() * 25.4
    fig.subplots_adjust(top=1 - 14 / height_mm)
    fig.legend(handles=symbol_handles(with_share_bars), loc="lower left", bbox_to_anchor=(0, 1.0),
               ncol=3 if with_share_bars else 4, frameon=False, fontsize=5.5, handlelength=1.4, columnspacing=1.2,
               alignment="left", title="Under each group: number of sessions. " + effect_note, title_fontsize=5.5)


def add_symbol_legend_in_corner(ax, effect_note: str, with_share_bars: bool = False) -> None:
    """The same legend for a figure with a single panel: stacked in the panel's top right corner, outside it."""
    note = "Under each group:\nnumber of sessions.\n" + effect_note.replace("; ", ";\n")
    ax.legend(handles=symbol_handles(with_share_bars), loc="upper left", bbox_to_anchor=(1.03, 1.0), ncol=1,
              frameon=False, fontsize=5.5, handlelength=1.4, alignment="left", title=note, title_fontsize=5.5)


def groups_in_long_form(groups: list[pd.Series], labels: list[str]) -> pd.DataFrame:
    """One row per session: its value and its group label, the form seaborn plots from."""
    return pd.concat([pd.DataFrame({"value": g.dropna().to_numpy(), "group": label}) for g, label in zip(groups, labels)],
                     ignore_index=True)


def label_groups_with_counts(ax, groups: list[pd.Series], labels: list[str]) -> None:
    """Below each group label: the number of sessions in the group."""
    ax.set_xticks(range(len(groups)), [f"{label}\n{g.notna().sum():,}" for label, g in zip(labels, groups)])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("")


def draw_boxes(ax, groups: list[pd.Series], labels: list[str], colors: list[str]) -> None:
    """Box (median, IQR, whiskers 1.5 IQR, outliers hidden) + white diamond for the mean (seaborn)."""
    sns.boxplot(data=groups_in_long_form(groups, labels), x="group", y="value", hue="group", order=labels,
                hue_order=labels, palette=colors, saturation=1, legend=False, width=.6, showfliers=False, showmeans=True, ax=ax,
                medianprops=dict(color=INK, lw=.9), whiskerprops=dict(color=INK, lw=.5),
                capprops=dict(color=INK, lw=.5), boxprops=dict(lw=0),
                meanprops=dict(marker="D", markersize=2.8, markerfacecolor="white", markeredgecolor=INK,
                               markeredgewidth=.5))
    ax.set_ylabel("")
    label_groups_with_counts(ax, groups, labels)


def draw_box_plots(d: pd.DataFrame, metrics: dict[str, str], rows: list[tuple[str, str, list, list[str]]],
                   path: Path, width_mm: float, labels: dict[tuple[str, str], str] | None = None,
                   letters: str = "abcdefgh") -> None:
    """
    Small multiples: one row per grouping variable, one column per metric.

    rows: (grouping column, x-axis label, group levels in order, colours)
    metrics: column -> y-axis label
    labels: optional (grouping column, metric) -> effect-size text shown above the panel
    Each row gets a panel letter. Below each group label: number of sessions.
    """
    fig, axes = plt.subplots(len(rows), len(metrics), figsize=mm(width_mm, 50 * len(rows)), squeeze=False,
                             gridspec_kw=dict(hspace=.75, wspace=.45))
    for i, (col, xlabel, levels, colors) in enumerate(rows):
        for j, (m, ylabel) in enumerate(metrics.items()):
            ax = axes[i, j]
            values_per_group = [d.loc[d[col] == level, m] for level in levels]
            draw_boxes(ax, values_per_group, [str(level) for level in levels], colors)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_ylim(bottom=0)
            if labels and (col, m) in labels:
                ax.set_title(labels[(col, m)], loc="left", fontsize=5.5, linespacing=1.25, pad=3)
        if letters:
            panel_label(axes[i, 0], letters[i], x=-.42, y=1.18)
    add_symbol_legend(fig, "Above each panel: Cliff's δ (last vs first group) and Spearman ρ (trend) [95% CI]; "
                           "* CI excludes 0.")
    save(fig, path)


def draw_share_bars(ax, groups: list[pd.Series], labels: list[str], colors: list[str]) -> None:
    """
    Percentage of sessions in each group whose value is above zero (for words most texts never use), with a
    bootstrap 95% CI (seaborn).
    """
    used = groups_in_long_form(groups, labels).assign(value=lambda d: 100 * (d["value"] > 0))
    sns.barplot(data=used, x="group", y="value", hue="group", order=labels, hue_order=labels, palette=colors,
                saturation=1, legend=False, width=.6, errorbar=("ci", 95), seed=SEED, err_kws=dict(color=INK, linewidth=.6),
                ax=ax)
    ax.set_ylabel("% of sessions using them", fontsize=5.5)
    label_groups_with_counts(ax, groups, labels)


def is_mostly_zero(values: pd.Series) -> bool:
    """True when more than half of the sessions never use the words: a box would collapse onto zero."""
    return (values.dropna() == 0).mean() > .5


def draw_marker_panels(d: pd.DataFrame, effects: pd.DataFrame, group: str, xlabel: str, levels: list,
                       colors: list[str], path: Path, max_width_mm: float = 183) -> None:
    """
    Figures S4-S5: one panel per robust marker of a research question, by groups of its survey score.
    A box plot of the raw session values, or, for words that most sessions never use, the percentage of
    sessions that use them at all. Panel title: metric, the texts it is a marker in, and Cliff's delta
    (computed on the raw values in both cases) with its 95% CI.
    """
    # Wider panels for four groups, so that the group names do not overlap.
    panel_width = 36 if len(levels) <= 3 else 58
    n = len(effects)
    ncol = min(n, int(max_width_mm // panel_width))
    nrow = -(-n // ncol)                                  # ceiling division
    fig, axes = plt.subplots(nrow, ncol, figsize=mm(panel_width * ncol + 4, 52 * nrow), squeeze=False,
                             gridspec_kw=dict(hspace=1.0, wspace=.55))
    for ax, row in zip(axes.flat, effects.itertuples()):
        feature, scope = row.metric.split(" | ")
        values_per_group = [d.loc[d[group] == level, row.metric] for level in levels]
        group_names = [str(level) for level in levels]
        if is_mostly_zero(d[row.metric]):
            draw_share_bars(ax, values_per_group, group_names, colors)
        else:
            draw_boxes(ax, values_per_group, group_names, colors)
        interval = f"[{row.delta_ci_low:+.2f}, {row.delta_ci_high:+.2f}]"
        star = star_if_interval_excludes_zero(row.delta_ci_low, row.delta_ci_high)
        delta = f"δ = {row.cliffs_delta:+.2f} {interval}{star}".replace("-", "\u2212")   # typographic minus
        texts = "All texts" if scope == "pooled" else f"{labels.scope(scope)} texts"
        ax.set_title(f"{labels.metric(feature)}\n{texts}\n{delta}", loc="left", fontsize=5.5,
                     linespacing=1.25, pad=3)
        ax.set_xlabel(xlabel, fontsize=5.5)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    any_bars = any(is_mostly_zero(d[metric]) for metric in effects["metric"])
    if n == 1:
        add_symbol_legend_in_corner(axes[0, 0], "Above the panel: Cliff's δ,\nlast vs first group [95% CI];"
                                                " * CI excludes 0.", with_share_bars=any_bars)
    else:
        add_symbol_legend(fig, "Above each panel: Cliff's δ, last vs first group [95% CI]; * CI excludes 0.",
                          with_share_bars=any_bars)
    save(fig, path)
