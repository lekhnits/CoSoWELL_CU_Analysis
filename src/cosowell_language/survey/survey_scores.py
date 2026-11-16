"""
Survey scores and covariates (CoSoWELL release 2 codebook; columns in data_loading.SurveyRecord).

    loneliness            3-item loneliness scale (Hughes et al., 2004), sum 3-9
    isolation_index       mean z-score of six contact indicators, higher = more isolated
    support_quality       mean z-score of open up to / rely on family and friends
    health                self-rated health, 1 = poor ... 4 = excellent
    well_being_score      Flourishing Scale sum, 8-56 (t7-t8 only)
    expressive_suppression, cognitive_reappraisal
                          Emotion Regulation Questionnaire facet means, 1-7 (t7-t8 only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import COVARIATES, KEYS, MEASURES
from ..standardisation import zscore

MAX_EDUCATION_YEARS = 30   # larger values (up to 60) are implausible -> missing

# Sign so that higher = more isolated.
ISOLATION_COMPONENTS = {"social_meet_freq": -1, "professional_meet_freq": -1, "talk_social_network": -1,
                        "log_num_friends": -1, "num_household": -1, "lives_alone": +1}
# The spouse items apply only to people with a partner; confide_family/friend/colleague
# are identical columns in the release and are not used.
SUPPORT_ITEMS = ["divulge_family", "rely_family", "divulge_friend", "rely_friend"]
NUMERIC = ["age", "years_education", "health", "loneliness", "social_meet_freq", "professional_meet_freq",
           "talk_social_network", "num_friends", "num_household", *SUPPORT_ITEMS,
           "well_being_score", "cognitive_reappraisal", "expressive_suppression"]


def answers_as_numbers(survey: pd.DataFrame) -> pd.DataFrame:
    out = survey.copy()
    for col in NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_covariates(s: pd.DataFrame) -> pd.DataFrame:
    s["years_education"] = s["years_education"].where(s["years_education"] <= MAX_EDUCATION_YEARS)
    s["female"] = s["gender"].map({"female": 1.0, "male": 0.0})                     # other / not stated -> missing
    s["covid_phase"] = (s["phase"] == "cov").astype(float).where(s["phase"].notna())  # after 13 Mar 2020
    return s


def add_isolation_index(s: pd.DataFrame) -> pd.DataFrame:
    s["log_num_friends"] = np.log1p(s["num_friends"])
    s["lives_alone"] = (s["living_arrangements"] == "alone").astype(float)
    s.loc[s["living_arrangements"].isna(), "lives_alone"] = np.nan
    # z-scored so that every indicator weighs the same; a missing indicator is skipped, not counted as 0.
    oriented = pd.DataFrame({column: sign * zscore(s[column]) for column, sign in ISOLATION_COMPONENTS.items()})
    s["isolation_index"] = oriented.mean(axis=1, skipna=True)
    return s


def add_support_quality(s: pd.DataFrame) -> pd.DataFrame:
    """Mean z-score of the support items; needs at least two answered items."""
    items = s[SUPPORT_ITEMS].apply(zscore)
    enough_items = items.notna().sum(axis=1) >= 2
    s["support_quality"] = items.mean(axis=1).where(enough_items)
    return s


def add_derived(survey: pd.DataFrame) -> pd.DataFrame:
    """Covariates, objective isolation index and support quality (raw scale)."""
    s = answers_as_numbers(survey)
    s = add_covariates(s)
    s = add_isolation_index(s)
    s = add_support_quality(s)
    return s


def build_session_table(survey: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Survey scores for the sessions that have narratives, z-scored across those sessions."""
    sessions_with_texts = add_derived(survey.merge(sessions[KEYS].drop_duplicates(), on=KEYS))
    session_scores = sessions_with_texts[KEYS].copy()
    for column in dict.fromkeys(MEASURES + COVARIATES):
        session_scores[column] = zscore(sessions_with_texts[column])
    # Loneliness on its raw 3-9 scale too, for the groups of the box plots.
    session_scores["loneliness_raw"] = sessions_with_texts["loneliness"]
    return session_scores
