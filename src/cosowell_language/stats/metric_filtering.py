"""Sparsity filter and rank-normalisation of the metrics (report Table 5, steps 5-6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import KEYS, LENGTH_COVARIATE, MIN_WORDS
from ..standardisation import rank_int, zscore

def usable(s: pd.Series) -> bool:
    """Present in >= 80% of narratives, non-zero in >= 10%, more than five distinct values.
    A metric that is almost always 0 or takes few values cannot show an association."""
    mostly_present = s.notna().mean() >= .8
    often_non_zero = s.fillna(0).ne(0).mean() >= .10
    return mostly_present and often_non_zero and s.nunique() > 5


def build_feature_matrix(raw: pd.DataFrame, types: list[str]) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Long table (one row per narrative) of rank-normalised features.

    `raw` has id, test_session, ntype and the raw metrics. Returns the
    normalised table, the feature list and a features x types availability table.
    """
    raw = raw[raw["n_words"] >= MIN_WORDS].reset_index(drop=True)
    candidates = [c for c in raw.columns if c not in KEYS + ["ntype"]]
    used_metrics = {}
    for ntype in types:
        of_type = raw[raw["ntype"].eq(ntype)]
        used_metrics[ntype] = {feature: usable(of_type[feature]) for feature in candidates}
    used_metrics = pd.DataFrame(used_metrics)
    features = used_metrics.index[used_metrics.any(axis=1)].tolist()

    # Normalised within each narrative type, because the tasks differ in what and how much people write.
    parts = []
    for ntype, of_type in raw.groupby("ntype"):
        columns = {}
        for feature in features:
            if used_metrics.at[feature, ntype]:
                columns[feature] = rank_int(of_type[feature])
            else:
                columns[feature] = pd.Series(np.nan, index=of_type.index)
        columns[LENGTH_COVARIATE] = zscore(np.log(of_type["n_words"]))
        parts.append(pd.concat([of_type[KEYS + ["ntype"]], pd.DataFrame(columns)], axis=1))
    return pd.concat(parts, ignore_index=True), features, used_metrics.loc[features]
