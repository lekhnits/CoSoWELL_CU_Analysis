"""Raw metric values per session and survey group, and their effect sizes, for the box plots (figs 3, S4-S5)."""

from __future__ import annotations

import pandas as pd

from ..config import KEYS, MIN_WORDS
from .box_plot_effect_sizes import group_effects

# The markers are strongest in the texts about the person's own day and plans.
DESCRIPTIVE_TYPES = ["yesterday", "future"]


def session_means(raw: pd.DataFrame, session_scores: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    Raw values of `metrics` averaged over a session's yesterday and future texts,
    joined with the survey scores and grouped into isolation thirds and loneliness
    bands (3 / 4-5 / 6-9).
    """
    r = raw[raw["ntype"].isin(DESCRIPTIVE_TYPES) & (raw["n_words"] >= MIN_WORDS)]
    d = r.groupby(KEYS)[metrics].mean().reset_index()
    d = d.merge(session_scores, on=KEYS)
    d["isolation_group"] = pd.qcut(d["isolation_index"], 3, labels=["Low", "Middle", "High"])
    d["loneliness_group"] = pd.cut(d["loneliness_raw"], [2, 3, 5, 9], labels=["3", "4–5", "6–9"])
    return d


# Box plots (fig 3): metrics in raw units, and the survey groups they are split by.
SOCIAL_METRICS = {"fpp_plural_rate": "'We' words (per 100 words)",
                  "fpp_singular_rate": "'I' words (per 100 words)",
                  "social_plus_we_rate": "Social words + 'we' (per 100 words)"}
BOX_GROUPS = {"isolation_group": (["Low", "Middle", "High"], SOCIAL_METRICS),
              "loneliness_group": (["3", "4–5", "6–9"], SOCIAL_METRICS)}


def box_plot_data_and_effect_sizes(raw: pd.DataFrame,
                                   session_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Session means of the box-plot metrics, and Cliff's delta / Spearman rho for every grouping."""
    sessions = session_means(raw, session_scores, list(SOCIAL_METRICS))
    effects = pd.concat([group_effects(sessions, group, levels, list(metrics))
                         for group, (levels, metrics) in BOX_GROUPS.items()], ignore_index=True)
    return sessions, effects


# Box plots of every robust marker (figures S4-S5): research question and survey variable ->
# grouping column, x-axis label and group levels in order.
MARKER_GROUPINGS = {
    ("RQ1 isolation", "isolation_index"):
        ("isolation_group", "Objective isolation (thirds)", ["Low", "Middle", "High"]),
    ("RQ2 loneliness", "loneliness"):
        ("loneliness_group", "Loneliness score", ["3", "4–5", "6–9"]),
}


def marker_column(feature: str, scope: str) -> str:
    """Column of a marker's session values: the metric in the text type where it is a robust marker."""
    return f"{feature} | {scope}"


def robust_markers_in_their_strongest_texts(hits: pd.DataFrame, finding: str, term: str) -> pd.DataFrame:
    """Each robust marker once, in the scope (pooled or one text type) of its smallest q."""
    robust = hits[hits["finding"].eq(finding) & hits["term"].eq(term) & hits["verdict"].eq("robust")]
    return robust.sort_values("q").drop_duplicates("feature")[["feature", "scope", "beta", "q"]]


def marker_session_values(raw: pd.DataFrame, session_scores: pd.DataFrame, markers: pd.DataFrame) -> pd.DataFrame:
    """
    Raw value of each marker per session: the mean over the session's texts of the marker's scope (all four
    text types for a pooled marker), joined with the isolation and loneliness groups.
    """
    texts = raw[raw["n_words"] >= MIN_WORDS]
    columns = []
    for marker in markers.drop_duplicates(["feature", "scope"]).itertuples():
        of_scope = texts if marker.scope == "pooled" else texts[texts["ntype"].eq(marker.scope)]
        values = of_scope.groupby(KEYS)[marker.feature].mean().rename(marker_column(marker.feature, marker.scope))
        columns.append(values)
    d = pd.concat(columns, axis=1).reset_index().merge(session_scores, on=KEYS)
    thirds = ["Low", "Middle", "High"]
    d["isolation_group"] = pd.qcut(d["isolation_index"], 3, labels=thirds)
    d["loneliness_group"] = pd.cut(d["loneliness_raw"], [2, 3, 5, 9], labels=["3", "4–5", "6–9"])
    return d


def marker_values_and_effect_sizes(raw: pd.DataFrame, session_scores: pd.DataFrame,
                                   hits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Session values of the robust markers of every research question, and their effect sizes
    (one row per research question x marker: Cliff's delta of the last vs first group, Spearman rho).
    """
    markers = {key: robust_markers_in_their_strongest_texts(hits, *key) for key in MARKER_GROUPINGS}
    sessions = marker_session_values(raw, session_scores, pd.concat(markers.values()))
    effects = []
    for (finding, term), of_question in markers.items():
        if of_question.empty:
            continue
        group, _, levels = MARKER_GROUPINGS[(finding, term)]
        columns = [marker_column(m.feature, m.scope) for m in of_question.itertuples()]
        found = group_effects(sessions, group, levels, columns)
        found.insert(0, "term", term)
        found.insert(0, "finding", finding)
        found["beta"] = of_question["beta"].to_numpy()
        found["q"] = of_question["q"].to_numpy()
        effects.append(found)
    return sessions, pd.concat(effects, ignore_index=True)
