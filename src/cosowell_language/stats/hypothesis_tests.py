"""RQ1, RQ2 and RQ1-2: every metric x research question x text type, one regression each, with FDR and Holm corrections."""

from __future__ import annotations

import pandas as pd
from statsmodels.stats.multitest import multipletests

from ..config import FINDINGS
from .metric_regression_model import fit_metric_model, scope_data, scopes

MIN_OBS = 200          # a feature is tested in a scope only with at least this many narratives


def types_with_enough_data(scored_texts: pd.DataFrame, features: list[str], types: list[str]) -> dict[str, list[str]]:
    return {f: [t for t in types if scope_data(scored_texts, t)[f].notna().sum() >= MIN_OBS] for f in features}


def is_testable(scored_texts: pd.DataFrame, feature: str, scope: str, tested_in: dict) -> bool:
    """Enough values in the scope; pooled only when the feature is tested in at least two types
    (otherwise the pooled model would repeat the single-type test on the same data)."""
    if scope_data(scored_texts, scope)[feature].notna().sum() < MIN_OBS:
        return False
    return scope != "pooled" or len(tested_in[feature]) >= 2


def tests_of_finding(scored_texts: pd.DataFrame, name: str, features: list[str], types: list[str],
                     tested_in: dict) -> pd.DataFrame:
    """One row per metric x text type x survey variable of the finding."""
    spec = FINDINGS[name]
    terms, adjust = spec["terms"], spec.get("adjust", ())
    rows = []
    for feature in features:
        for scope in scopes(types):
            if not is_testable(scored_texts, feature, scope, tested_in):
                continue
            texts_of_scope = scope_data(scored_texts, scope)
            estimates = fit_metric_model(texts_of_scope, feature, terms, adjust, pooled=scope == "pooled")
            for estimate in estimates:
                rows.append({"finding": name, "feature": feature, "scope": scope, **estimate})
    return pd.DataFrame(rows)


def add_within_finding_corrections(tests: pd.DataFrame) -> pd.DataFrame:
    tests["q"] = multipletests(tests["p"], method="fdr_bh")[1]
    tests["p_holm"] = multipletests(tests["p"], method="holm")[1]
    return tests


def run_hypothesis_tests(scored_texts: pd.DataFrame, features: list[str], types: list[str]) -> pd.DataFrame:
    """
    Every feature x finding x scope. Benjamini-Hochberg (q) and Holm p-values
    are computed within each finding, over features x terms x scopes; q_all_findings
    applies Benjamini-Hochberg over every test of every finding together.
    """
    tested_in = types_with_enough_data(scored_texts, features, types)
    per_finding = [add_within_finding_corrections(tests_of_finding(scored_texts, name, features, types, tested_in))
                   for name in FINDINGS]
    out = pd.concat(per_finding, ignore_index=True)
    out["q_all_findings"] = multipletests(out["p"], method="fdr_bh")[1]
    return out
