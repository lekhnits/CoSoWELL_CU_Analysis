"""Machine-learning classifiers: balanced accuracy and ROC AUC of the best classifiers per text type, ROC
curves of the top-ranked classifier, and the same for the two pilot groupings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import figure_labels as labels
from .figure_style import AQUA, BLUE, DOUBLE_COLUMN, GREY, INK, ORANGE, YELLOW, mm, panel_label, plt, save

MODEL_COLOURS = [BLUE, ORANGE, AQUA, YELLOW]
SCORE_PANELS = [("balanced_accuracy", "Balanced accuracy"), ("roc_auc", "ROC AUC")]
CHANCE = .5                                      # both scores, whatever the class sizes


def best_models(ranking: pd.DataFrame) -> list[str]:
    return ranking.loc[ranking["shown_in_figure"], "model"].tolist()


def draw_score_bars(ax, summary: pd.DataFrame, models: list[str], score: str) -> None:
    """One group of bars per text type, one bar per model: mean over the outer folds, error bar = SD."""
    types = summary["ntype"].unique().tolist()
    group = np.arange(len(types))
    width = .8 / len(models)
    for i, (model, colour) in enumerate(zip(models, MODEL_COLOURS)):
        of_model = summary[summary["model"].eq(model)].set_index("ntype").loc[types]
        position = group - .4 + width * (i + .5)
        ax.bar(position, of_model[f"{score}_mean"], width=width * .9, color=colour, linewidth=0)
        ax.errorbar(position, of_model[f"{score}_mean"], yerr=of_model[f"{score}_std"], fmt="none", ecolor=INK,
                    elinewidth=.6, capsize=1.2, capthick=.6)
    ax.axhline(CHANCE, color=INK, lw=.7, ls=(0, (2, 1.5)), zorder=4)
    ax.set_xticks(group, [labels.scope(t) for t in types])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, .8)


def draw_model_comparison_figure(summary: pd.DataFrame, ranking: pd.DataFrame, target_names: dict,
                                 path: Path) -> None:
    """
    Figure 5 (classifiers): rows = lonely and isolated; columns = balanced accuracy and ROC AUC of the four best models in
    each text type (mean ± SD over the 5 outer test folds). Dashed: chance (0.5).
    """
    models = best_models(ranking)
    targets = list(target_names)
    fig, axes = plt.subplots(len(targets), len(SCORE_PANELS), figsize=mm(DOUBLE_COLUMN, 16 + 52 * len(targets)))
    letters = iter("abcd")
    for row, target in zip(axes, targets):
        of_target = summary[summary["target"].eq(target)]
        for ax, (score, name) in zip(row, SCORE_PANELS):
            draw_score_bars(ax, of_target, models, score)
            ax.set_ylabel(name)
            panel_label(ax, next(letters))
        row[0].set_title(target_names[target], loc="left")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colour, label=model) for model, colour in zip(models, MODEL_COLOURS)]
    handles.append(plt.Line2D([], [], color=INK, lw=.7, ls=(0, (2, 1.5)), label="Chance"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(.5, -.04))
    fig.tight_layout(h_pad=2)
    save(fig, path)


ROC_GRID = np.linspace(0, 1, 201)             # common false-positive rates, to average curves over folds


def roc_curves_per_fold(predictions: pd.DataFrame) -> tuple[list, list, list]:
    """For each outer test fold: the ROC curve (false- and true-positive rates), and its AUC."""
    from sklearn.metrics import roc_auc_score, roc_curve

    curves, aucs = [], []
    for _, of_fold in predictions.groupby("fold"):
        false_positive, true_positive, _ = roc_curve(of_fold["label"], of_fold["probability"])
        curves.append((false_positive, true_positive))
        aucs.append(roc_auc_score(of_fold["label"], of_fold["probability"]))
    mean_curve = [np.interp(ROC_GRID, fp, tp) for fp, tp in curves]
    return curves, mean_curve, aucs


def draw_mean_roc_curve(ax, curves_on_grid: list, colour: str) -> None:
    """Mean of the fold curves (interpolated on ROC_GRID) as a thick line, with a band of ± 1 SD."""
    mean, sd = np.mean(curves_on_grid, axis=0), np.std(curves_on_grid, axis=0, ddof=1)
    ax.fill_between(ROC_GRID, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), color=colour, alpha=.15, lw=0)
    ax.plot(ROC_GRID, mean, color=colour, lw=1.3)


def draw_chance_diagonal(ax) -> None:
    ax.plot([0, 1], [0, 1], color=GREY, lw=.6, ls=(0, (2, 2)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")


def draw_roc_curves_figure(predictions: pd.DataFrame, model: str, target_names: dict, path: Path,
                           groups: pd.DataFrame | None = None, group_names: dict | None = None) -> None:
    """
    ROC curves of one model (the top-ranked) in each text type (columns) for each label (rows). Thick line, mean of
    the outer-fold curves; band, ± 1 SD; AUC, mean ± SD over the folds. Figure 6: all sessions together. Figures 7b
    and 8c (pilots): drawn separately for two groups of sessions (group_names: code -> legend label; first group
    blue, second orange), the same numbers as the bars of the matching pilot figure.
    """
    of_model = predictions[predictions["model"].eq(model)]
    if groups is None:
        of_model, group_names = of_model.assign(group="all"), {"all": "All sessions"}
    else:
        of_model = of_model.merge(groups[["id", "test_session", "group"]], on=["id", "test_session"])
    targets = list(target_names)
    types = of_model["ntype"].unique().tolist()
    fig, axes = plt.subplots(len(targets), len(types), figsize=mm(DOUBLE_COLUMN, 12 + 46 * len(targets)),
                             sharex=True, sharey=True, squeeze=False)
    letters = iter("abcdefgh")
    for row, target in zip(axes, targets):
        for ax, ntype in zip(row, types):
            of_panel = of_model[of_model["target"].eq(target) & of_model["ntype"].eq(ntype)]
            for i, (group, colour) in enumerate(zip(group_names, [BLUE, ORANGE])):
                _, curves_on_grid, aucs = roc_curves_per_fold(of_panel[of_panel["group"].eq(group)])
                draw_mean_roc_curve(ax, curves_on_grid, colour)
                ax.text(.97, .04 + .1 * (len(group_names) - 1 - i), f"AUC {np.mean(aucs):.2f} ± {np.std(aucs, ddof=1):.2f}",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color=colour)
            draw_chance_diagonal(ax)
            ax.set_title(labels.scope(ntype), loc="left")
            panel_label(ax, next(letters))
        row[0].set_ylabel(f"{target_names[target]}\nTrue-positive rate")
    for ax in axes[-1]:
        ax.set_xlabel("False-positive rate")

    handles = [plt.Line2D([], [], color=colour, lw=1.3, label=label)
               for label, colour in zip(group_names.values(), [BLUE, ORANGE])]
    handles.append(plt.Line2D([], [], color=GREY, lw=.6, ls=(0, (2, 2)), label="Chance"))
    fig.suptitle(model, x=.02, ha="left", fontsize=7)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(.5, -.03))
    fig.tight_layout(h_pad=1.5)
    save(fig, path)


def draw_scores_by_group_figure(by_group: pd.DataFrame, model: str, group_names: dict, target_names: dict,
                                path: Path) -> None:
    """
    Figures 7a and 8b (classifier pilots): balanced accuracy and ROC AUC of one model (the top-ranked) in each text type, for
    two groups of people (group_names: code -> legend label; first group blue, second orange); mean ± SD over the
    outer folds.
    """
    of_model = by_group[by_group["model"].eq(model)]
    targets = list(target_names)
    types = of_model["ntype"].unique().tolist()
    position = np.arange(len(types))
    width = .38
    fig, axes = plt.subplots(len(targets), len(SCORE_PANELS), figsize=mm(DOUBLE_COLUMN, 16 + 48 * len(targets)))
    letters = iter("abcd")
    for row, target in zip(axes, targets):
        for ax, (score, name) in zip(row, SCORE_PANELS):
            for i, (group, colour) in enumerate(zip(group_names, [BLUE, ORANGE])):
                values = of_model[of_model["target"].eq(target) & of_model["group"].eq(group)]
                values = values.set_index("ntype").loc[types]
                x = position - width / 2 + i * width
                ax.bar(x, values[f"{score}_mean"], width=width * .92, color=colour, linewidth=0)
                ax.errorbar(x, values[f"{score}_mean"], yerr=values[f"{score}_std"], fmt="none", ecolor=INK,
                            elinewidth=.6, capsize=1.2, capthick=.6)
            ax.axhline(CHANCE, color=INK, lw=.7, ls=(0, (2, 1.5)), zorder=4)
            ax.set_xticks(position, [labels.scope(t) for t in types])
            ax.tick_params(axis="x", length=0)
            ax.set_ylim(0, .8)
            ax.set_ylabel(name)
            panel_label(ax, next(letters))
        row[0].set_title(f"{target_names[target]}, {model}", loc="left")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colour, label=label)
               for label, colour in zip(group_names.values(), [BLUE, ORANGE])]
    handles.append(plt.Line2D([], [], color=INK, lw=.7, ls=(0, (2, 1.5)), label="Chance"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(.5, -.04))
    fig.tight_layout(h_pad=2)
    save(fig, path)


def draw_distance_from_line_groups_figure(sessions: pd.DataFrame, group_names: dict, path: Path) -> None:
    """
    Figure 8a (classifier pilot): which sessions are on and off the line. Each dot is one session (loneliness jittered
    vertically, as it takes only seven values) with its session's shift removed, so one line serves all sessions;
    solid line, mixed-model line of loneliness on isolation; dashed lines, ± the residual cut-off (1 SD of the
    residuals).
    """
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-.06, .06, len(sessions))
    loneliness = sessions["loneliness"] - sessions["session_shift"]
    line = sessions["fitted"] - sessions["session_shift"]
    fig, ax = plt.subplots(figsize=mm(DOUBLE_COLUMN / 2, 78))
    for group, colour in zip(group_names, [BLUE, ORANGE]):
        of_group = sessions["group"].eq(group).to_numpy()
        ax.scatter(sessions["isolation_index"][of_group], loneliness[of_group] + jitter[of_group],
                   s=3, color=colour, alpha=.55, linewidth=0,
                   label=f"{group_names[group]}, n = {of_group.sum():,}")
    order = sessions["isolation_index"].sort_values().index
    cut = sessions["cut"].iloc[0]
    ax.plot(sessions["isolation_index"][order], line[order], color=INK, lw=.9)
    for sign in (-1, 1):
        ax.plot(sessions["isolation_index"][order], line[order] + sign * cut, color=INK, lw=.6, ls=(0, (2, 1.5)))
    ax.set_xlabel("Objective isolation (z)")
    ax.set_ylabel("Loneliness (z, session shift removed)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=1, markerscale=2.5, handletextpad=.3)
    save(fig, path)
