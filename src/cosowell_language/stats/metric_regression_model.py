"""
One regression of a metric on survey score(s) + covariates (step 8):

    metric ~ terms + adjust + age + education + health + female + COVID-19 period + log word count
             [+ text-type indicators when several narrative types are pooled]

OLS with standard errors clustered by participant.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from ..config import COVARIATES, LENGTH_COVARIATE, LENGTH_FEATURES


def covariates_for(feature: str, terms: list[str], adjust=()) -> list[str]:
    """Extra adjustments and the standard covariates, minus the terms; length too, unless the metric is a length."""
    covs = [c for c in dict.fromkeys([*adjust, *COVARIATES]) if c not in terms]
    return covs if feature in LENGTH_FEATURES else [*covs, LENGTH_COVARIATE]


def design_matrix(x: pd.DataFrame, terms: list[str], covs: list[str], pooled: bool) -> pd.DataFrame:
    """Terms and covariates (constant covariates dropped) plus text-type indicators when pooling."""
    X = x[[*terms, *covs]]
    X = X.loc[:, (X.std() > 0) | X.columns.isin(terms)]
    if pooled and x["ntype"].nunique() > 1:
        X = X.join(pd.get_dummies(x["ntype"], prefix="t", drop_first=True, dtype=float))
    return sm.add_constant(X, has_constant="add")


def fit_metric_model(d: pd.DataFrame, feature: str, terms: list[str], adjust=(), pooled: bool = True) -> list[dict]:
    """One row per term: beta, standard error and p (clustered by person), number of texts and people."""
    covs = covariates_for(feature, terms, adjust)
    x = d[[feature, *terms, *covs, "id", "ntype"]].dropna()
    model = sm.OLS(x[feature], design_matrix(x, terms, covs, pooled)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(x["id"])[0]})
    return [{"term": t, "beta": model.params[t], "se": model.bse[t], "p": model.pvalues[t],
             "n_obs": len(x), "n_persons": x["id"].nunique()} for t in terms]


def scopes(types: list[str]) -> list[str]:
    """Where each metric is tested: every narrative type, plus all of them pooled."""
    return ["pooled", *types] if len(types) > 1 else list(types)


def scope_data(scored_texts: pd.DataFrame, scope: str) -> pd.DataFrame:
    return scored_texts if scope == "pooled" else scored_texts[scored_texts["ntype"].eq(scope)]
