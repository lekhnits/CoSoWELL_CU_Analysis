"""Put metrics and survey scores on a common scale."""

from __future__ import annotations

import pandas as pd
from scipy import stats


def zscore(values: pd.Series) -> pd.Series:
    return (values - values.mean()) / values.std()


def rank_int(values: pd.Series) -> pd.Series:
    """
    Rank-based inverse normal transform; missing values stay missing.

    Each value is replaced by the normal quantile of its rank, Φ⁻¹((rank − 0.5) / n). Rates of words are
    skewed (many zeros, a few large values), so this gives every metric the same bell-shaped scale
    without letting a few extreme texts dominate a regression.
    """
    n_present = values.notna().sum()
    quantiles = (values.rank() - 0.5) / n_present
    return pd.Series(stats.norm.ppf(quantiles), index=values.index)
