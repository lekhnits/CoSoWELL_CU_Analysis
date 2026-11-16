"""Overview of all hypotheses: one volcano panel (beta vs -log10 p) per survey term (double column)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from ..config import FINDINGS
from . import figure_labels as labels
from .figure_style import BLUE, DOUBLE_COLUMN, GREY, INK, mm, plt, save


# Short panel titles; other survey variables use their label from figure_labels.
SHORT_TERM_NAMES = {"isolation_index": "Objective isolation"}


def draw_volcano_panel(ax, tests: pd.DataFrame, verdict: pd.Series, finding: str, term: str, n_labels: int) -> None:
    """The tests of one survey variable: beta against -log10 p, coloured by FDR and replication."""
    tests = tests.copy()
    tests["nlp"] = -np.log10(tests["p"])
    keys = list(zip(tests["finding"], tests["feature"], tests["scope"], tests["term"]))
    tests["verdict"] = [verdict.get(key, "") for key in keys]
    not_significant = tests[tests["verdict"].eq("")]
    fragile = tests[tests["verdict"].eq("fragile")]
    robust = tests[tests["verdict"].eq("robust")]
    sns.scatterplot(data=not_significant, x="beta", y="nlp", s=3, color=GREY, linewidth=0, rasterized=True, ax=ax)
    sns.scatterplot(data=fragile, x="beta", y="nlp", s=7, facecolor="white", edgecolor=BLUE, linewidth=.5, ax=ax)
    sns.scatterplot(data=robust, x="beta", y="nlp", s=7, color=BLUE, linewidth=0, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Benjamini-Hochberg has no single cut-off; the line is the largest p-value that still passed.
    passed = tests.loc[tests["q"] < .05, "p"]
    if len(passed):
        ax.axhline(-np.log10(passed.max()), color=INK, lw=.4, ls=(0, (3, 2)))
    strongest_pooled = robust[robust["scope"].eq("pooled")].nlargest(n_labels, "nlp")
    for hit in strongest_pooled.itertuples():
        ax.annotate(labels.metric(hit.feature), (hit.beta, hit.nlp), xytext=(2, 1), textcoords="offset points",
                    fontsize=5, color=INK)
    ax.axvline(0, color=INK, lw=.4)

    n_hits = int((tests["q"] < .05).sum())
    name = SHORT_TERM_NAMES.get(term, labels.term(term))
    ax.set_title(f"{labels.HYPOTHESIS[finding]}: {name}\n{n_hits} FDR hits, {len(robust)} robust",
                 loc="left", fontsize=5.5, pad=3)


def draw_volcano_figure(all_tests: pd.DataFrame, hits: pd.DataFrame, path: Path, ncol: int = 4,
                        n_labels: int = 1) -> None:
    """
    Figure 1: every hypothesis test as a dot, one panel per survey variable. Blue filled: robust hit;
    blue ring: FDR hit that did not replicate; grey: not significant after FDR. Dashed line: the
    p-value below which tests pass the Benjamini-Hochberg correction for that hypothesis.
    """
    panels = [(finding, term) for finding, spec in FINDINGS.items() for term in spec["terms"]]
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=mm(DOUBLE_COLUMN, 42 * nrow), squeeze=False, sharex=True,
                             gridspec_kw=dict(hspace=.6, wspace=.45))
    verdict = hits.set_index(["finding", "feature", "scope", "term"])["verdict"]
    x_limit = np.nanmax(np.abs(all_tests["beta"])) * 1.05            # same axes in every panel
    y_limit = np.nanmax(-np.log10(all_tests["p"])) * 1.08
    for ax, (finding, term) in zip(axes.flat, panels):
        tests = all_tests[all_tests["finding"].eq(finding) & all_tests["term"].eq(term)]
        draw_volcano_panel(ax, tests, verdict, finding, term, n_labels)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_ylim(0, y_limit)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("β")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$-\log_{10}\,p$")
    save(fig, path)
