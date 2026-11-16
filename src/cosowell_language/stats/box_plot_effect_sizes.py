"""
Unadjusted effect sizes for the descriptive box plots (figures 3 and 7).

    Cliff's delta   P(highest group > lowest group) - P(highest < lowest), from -1 to 1;
                    |delta| < .15 negligible, < .33 small, < .47 medium, else large (Romano et al., 2006)
    Spearman rho    rank correlation between the group order and the metric, all groups

95% confidence intervals come from a bootstrap that resamples participants
(their sessions stay together), since some people contribute two sessions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import SEED

N_BOOT = 1000


def cliffs_delta(high: np.ndarray, low: np.ndarray) -> float:
    """
    P(high > low) - P(high < low) over all pairs. The Mann-Whitney U counts the pairs with high > low
    (ties count 1/2), so U / (n_high * n_low) is P(high > low) + P(tie) / 2 and the delta is 2 * that - 1.
    """
    u = stats.mannwhitneyu(high, low).statistic
    return 2 * u / (len(high) * len(low)) - 1


def delta_size(delta: float) -> str:
    """Verbal size of Cliff's delta (Romano et al., 2006)."""
    size = abs(delta)
    if size < .147:
        return "negligible"
    if size < .33:
        return "small"
    if size < .474:
        return "medium"
    return "large"


def delta_and_rho(order: np.ndarray, value: np.ndarray, top: int) -> tuple[float, float]:
    """Cliff's delta of the top group vs the first group, and Spearman rho over all groups."""
    delta = cliffs_delta(value[order == top], value[order == 0])
    rho = stats.spearmanr(order, value)[0]
    return delta, rho


def bootstrap_by_person(sub: pd.DataFrame, order: np.ndarray, value: np.ndarray, top: int,
                        rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """(delta, rho) in resamples of participants; a person's sessions stay together."""
    rows_of = sub.groupby("id").indices
    persons = np.array(list(rows_of))
    boot = []
    for _ in range(n_boot):
        drawn_persons = rng.choice(persons, len(persons))
        rows = np.concatenate([rows_of[person] for person in drawn_persons])
        o, v = order[rows], value[rows]
        # A resample without anyone in the first or the top group has no delta.
        if (o == top).any() and (o == 0).any():
            boot.append(delta_and_rho(o, v, top))
    return np.array(boot)


def effect_of_grouping(d: pd.DataFrame, group: str, levels: list, metric: str,
                       rng: np.random.Generator, n_boot: int) -> dict:
    sub = d.loc[d[group].isin(levels), ["id", group, metric]].dropna().reset_index(drop=True)
    order = sub[group].map({lv: i for i, lv in enumerate(levels)}).to_numpy(dtype=int)
    value = sub[metric].to_numpy(dtype=float)
    top = len(levels) - 1
    low, high = value[order == 0], value[order == top]
    delta, rho = delta_and_rho(order, value, top)
    boot = bootstrap_by_person(sub, order, value, top, rng, n_boot)
    return {
        "grouping": group, "metric": metric, "low": levels[0], "high": levels[-1],
        "n_low": len(low), "n_high": len(high), "median_low": float(np.median(low)),
        "median_high": float(np.median(high)),
        "cliffs_delta": delta, "delta_ci_low": np.percentile(boot[:, 0], 2.5),
        "delta_ci_high": np.percentile(boot[:, 0], 97.5), "delta_size": delta_size(delta),
        "spearman_rho": rho, "rho_ci_low": np.percentile(boot[:, 1], 2.5),
        "rho_ci_high": np.percentile(boot[:, 1], 97.5), "n": len(sub),
    }


def group_effects(d: pd.DataFrame, group: str, levels: list, metrics: list[str],
                  n_boot: int = N_BOOT) -> pd.DataFrame:
    """Cliff's delta (last vs first level) and Spearman rho (all levels) per metric, with person-bootstrap CIs."""
    rng = np.random.default_rng(SEED)
    rows = [effect_of_grouping(d, group, levels, metric, rng, n_boot) for metric in metrics]
    return pd.DataFrame(rows)


def star_if_interval_excludes_zero(low: float, high: float) -> str:
    """'*' when the 95% CI does not include 0, i.e. the difference is significant at the 5% level."""
    return "*" if low > 0 or high < 0 else ""


def effect_label(row) -> str:
    """Short text for a plot panel."""
    delta_star = star_if_interval_excludes_zero(row.delta_ci_low, row.delta_ci_high)
    rho_star = star_if_interval_excludes_zero(row.rho_ci_low, row.rho_ci_high)
    text = (f"δ = {row.cliffs_delta:+.2f} [{row.delta_ci_low:+.2f}, {row.delta_ci_high:+.2f}]{delta_star}\n"
            f"ρ = {row.spearman_rho:+.2f} [{row.rho_ci_low:+.2f}, {row.rho_ci_high:+.2f}]{rho_star}")
    return text.replace("-", "\u2212")                   # typographic minus sign
