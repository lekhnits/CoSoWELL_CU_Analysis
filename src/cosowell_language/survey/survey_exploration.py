"""
Exploratory look at the survey scores the language models use (questionnaire only, no hypothesis tests).

Every survey record is used (2,621 records from 2,054 people). Each pair of the scores is described twice:
over all sessions with a mixed model whose random intercept per person accounts for people who answered at
several sessions (the time domain), and cross-sectionally, as Pearson's r within each session (one record
per person). The confidence interval of the loneliness-isolation correlation resamples people.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from ..config import KEYS, SEED
from .survey_scores import add_derived

# The scores and covariates of the language models, in the order shown in the correlation matrix.
SCORES_USED = {
    "loneliness": "Loneliness",
    "isolation_index": "Objective isolation",
    "support_quality": "Support quality",
    "health": "Self-rated health",
    "well_being_score": "Well-being",
    "expressive_suppression": "Expressive suppression",
    "cognitive_reappraisal": "Cognitive reappraisal",
    "age": "Age",
    "years_education": "Years of education",
    "female": "Female",
}

BOOTSTRAP_SAMPLES = 2000
MIN_RECORDS_PER_SESSION = 30


def survey_scores_used(survey: pd.DataFrame) -> pd.DataFrame:
    """One row per survey record: id, test_session and the scores of SCORES_USED on their own scale."""
    scores = add_derived(survey)
    return scores[KEYS + list(SCORES_USED)].reset_index(drop=True)


def pairs_of_scores() -> list[tuple[str, str]]:
    """Every pair of SCORES_USED once, in their listed order (loneliness-isolation, loneliness-support, ...)."""
    names = list(SCORES_USED)
    return [(first, second) for i, first in enumerate(names) for second in names[i + 1:]]


def z_score(values: pd.Series) -> pd.Series:
    """Scaled to mean 0, SD 1, so a slope between two scores is on the scale of Pearson's r."""
    return (values - values.mean()) / values.std()


def correlations_at_each_session(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional Pearson r of every pair of scores at each session, over the records with both scores
    (one record per person within a session, so records are independent). Sessions with fewer than
    MIN_RECORDS_PER_SESSION such records are left out. Benjamini-Hochberg q over all pairs and sessions together.
    """
    rows = []
    for first, second in pairs_of_scores():
        for session, records in scores.dropna(subset=[first, second]).groupby("test_session"):
            if len(records) < MIN_RECORDS_PER_SESSION:
                continue
            r, p = stats.pearsonr(records[first], records[second])
            rows.append(dict(score_a=first, score_b=second, scope=session, estimate=r, p=p,
                             n_records=len(records), n_people=len(records)))
    table = pd.DataFrame(rows)
    table["q"] = multipletests(table["p"], method="fdr_bh")[1]
    return table


def mixed_model_across_sessions(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Every pair of scores over all sessions together, as a linear mixed model:
    z(score_a) ~ z(score_b) + session + (1 | person)
    (ordinary least squares with the same terms when nobody has two records of the pair). The random intercept
    accounts for people who answered at several sessions, the session terms for differences between sessions
    (e.g. the COVID-19 period). The estimate is the slope of score_b, on the scale of Pearson's r.
    Benjamini-Hochberg q over the pairs.
    """
    rows = []
    for first, second in pairs_of_scores():
        records = scores.dropna(subset=[first, second])
        data = pd.DataFrame({"outcome": z_score(records[first]), "predictor": z_score(records[second]),
                             "session": records["test_session"], "id": records["id"]})
        # Sessions with too few records would make their session term unstable; they are dropped here too.
        data = data[data.groupby("session")["session"].transform("size") >= MIN_RECORDS_PER_SESSION]
        formula = "outcome ~ predictor" + (" + C(session)" if data["session"].nunique() > 1 else "")
        if data["id"].is_unique:
            # Nobody answered twice (e.g. well-being, asked at t7-t8 only): a person intercept cannot be told apart
            # from the residual, and the mixed model reduces to ordinary least squares.
            fit = smf.ols(formula, data).fit()
        else:
            with warnings.catch_warnings():
                # Few people answering twice can leave the person variance near 0; the fit is still valid.
                warnings.simplefilter("ignore", ConvergenceWarning)
                fit = smf.mixedlm(formula, data, groups=data["id"]).fit(reml=True)
        rows.append(dict(score_a=first, score_b=second, scope="all", estimate=fit.params["predictor"],
                         p=fit.pvalues["predictor"], n_records=len(data), n_people=data["id"].nunique()))
    table = pd.DataFrame(rows)
    table["q"] = multipletests(table["p"], method="fdr_bh")[1]
    return table


def significance_stars(q: float) -> str:
    """*** q < .001, ** q < .01, * q < .05."""
    if q < .001:
        return "***"
    if q < .01:
        return "**"
    if q < .05:
        return "*"
    return ""


def loneliness_and_isolation(scores: pd.DataFrame, samples: int = BOOTSTRAP_SAMPLES, seed: int = SEED) -> pd.Series:
    """
    Spearman rho of loneliness and objective isolation, with a 95% CI from resampling people
    (a person's records are drawn together, because records of one person are not independent).
    """
    pair = scores.dropna(subset=["loneliness", "isolation_index"])
    rho, p = stats.spearmanr(pair["loneliness"], pair["isolation_index"])

    rows_of_person = pair.groupby("id").indices
    people = np.array(list(rows_of_person))
    rng = np.random.default_rng(seed)
    resampled = []
    for _ in range(samples):
        drawn = rng.choice(people, size=len(people), replace=True)
        rows = np.concatenate([rows_of_person[person] for person in drawn])
        sample = pair.iloc[rows]
        resampled.append(stats.spearmanr(sample["loneliness"], sample["isolation_index"])[0])
    ci_low, ci_high = np.percentile(resampled, [2.5, 97.5])

    return pd.Series({"spearman_rho": rho, "ci_low": ci_low, "ci_high": ci_high, "p": p,
                      "n_records": len(pair), "n_people": pair["id"].nunique()})
