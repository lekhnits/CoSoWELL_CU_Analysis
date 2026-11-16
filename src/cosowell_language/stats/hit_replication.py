"""
Split-half and cross-task replication of FDR hits, and test-retest stability (report Table 5, steps 10-11).

A hit (q < .05) is robust when (1) refitted in each of two random halves of
participants it has the same sign and p < .05, and (2) for pooled hits, the
estimate has the same sign in at least 3 of the narrative types.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import FINDINGS, SEED
from ..session_order import session_number
from .metric_regression_model import fit_metric_model, scope_data

ALPHA = 0.05
MIN_TYPES_SAME_SIGN = 3
HIT_COLUMNS = ["finding", "term", "feature", "scope", "beta", "ci_low", "ci_high", "q", "q_all_findings", "p_holm",
               "n_persons", "beta_half_a", "p_half_a", "beta_half_b", "p_half_b", "types_same_sign", "retest_rho",
               "verdict"]


def test_retest(text_metrics: pd.DataFrame, feature: str, ntype: str) -> float:
    """Spearman correlation of a feature between a person's first and last session (same narrative type)."""
    d = text_metrics.loc[text_metrics["ntype"].eq(ntype), ["id", "test_session", feature]].dropna()
    d = d.assign(order=session_number(d["test_session"])).sort_values(["id", "order"])
    d = d[d.groupby("id")["id"].transform("size") > 1]
    first, last = d.groupby("id")[feature].first(), d.groupby("id")[feature].last()
    return stats.spearmanr(first, last)[0] if len(first) >= 30 else np.nan


def test_retest_table(text_metrics: pd.DataFrame, used_metrics: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    return pd.DataFrame({t: {f: test_retest(text_metrics, f, t) if used_metrics.at[f, t] else np.nan
                             for f in used_metrics.index}
                         for t in types})


def random_half_of_participants(scored_texts: pd.DataFrame) -> set:
    persons = scored_texts["id"].unique()
    shuffled = np.random.default_rng(SEED).permutation(persons)
    return set(shuffled[: len(persons) // 2])


def refit_in_halves(hit, scored_texts: pd.DataFrame, half_a: set) -> list[dict]:
    """The hit's estimate refitted in half A and in half B of the participants."""
    spec = FINDINGS[hit.finding]
    d = scope_data(scored_texts, hit.scope)
    in_half_a = d["id"].isin(half_a)
    halves = []
    for half in (d[in_half_a], d[~in_half_a]):
        estimates = fit_metric_model(half, hit.feature, spec["terms"], spec.get("adjust", ()),
                                     pooled=hit.scope == "pooled")
        halves.append(next(estimate for estimate in estimates if estimate["term"] == hit.term))
    return halves


def same_test_in_each_type(all_tests: pd.DataFrame, hit, types: list[str]) -> pd.DataFrame:
    same_test = ((all_tests["finding"] == hit.finding) & (all_tests["feature"] == hit.feature)
                 & (all_tests["term"] == hit.term))
    return all_tests[same_test & all_tests["scope"].isin(types)]


def hit_row(hit, halves: list[dict], by_type: pd.DataFrame, retest: pd.DataFrame) -> dict:
    same_sign = int((np.sign(by_type["beta"]) == np.sign(hit.beta)).sum())
    halves_agree = all(np.sign(h["beta"]) == np.sign(hit.beta) and h["p"] < ALPHA for h in halves)
    types_agree = hit.scope != "pooled" or same_sign >= MIN_TYPES_SAME_SIGN
    retest_rho = retest.loc[hit.feature].mean() if hit.scope == "pooled" else retest.at[hit.feature, hit.scope]
    return {"finding": hit.finding, "term": hit.term, "feature": hit.feature, "scope": hit.scope,
            "beta": hit.beta, "ci_low": hit.beta - 1.96 * hit.se, "ci_high": hit.beta + 1.96 * hit.se,
            "q": hit.q, "q_all_findings": hit.q_all_findings, "p_holm": hit.p_holm, "n_persons": hit.n_persons,
            "beta_half_a": halves[0]["beta"], "p_half_a": halves[0]["p"],
            "beta_half_b": halves[1]["beta"], "p_half_b": halves[1]["p"],
            "types_same_sign": f"{same_sign}/{len(by_type)}", "retest_rho": retest_rho,
            "verdict": "robust" if halves_agree and types_agree else "fragile"}


def replicate_fdr_hits(all_tests: pd.DataFrame, scored_texts: pd.DataFrame, text_metrics: pd.DataFrame,
                       used_metrics: pd.DataFrame, types: list[str]) -> pd.DataFrame:
    """Every FDR hit with its split-half and cross-task checks, test-retest rho and verdict."""
    half_a = random_half_of_participants(scored_texts)
    retest = test_retest_table(text_metrics, used_metrics, types)
    rows = []
    for hit in all_tests[all_tests["q"] < ALPHA].itertuples():
        in_halves = refit_in_halves(hit, scored_texts, half_a)
        in_each_type = same_test_in_each_type(all_tests, hit, types)
        rows.append(hit_row(hit, in_halves, in_each_type, retest))
    hits = pd.DataFrame(rows, columns=HIT_COLUMNS)
    return hits.sort_values(["finding", "term", "verdict", "q"]).reset_index(drop=True)
