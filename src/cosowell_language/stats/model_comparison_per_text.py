"""
Machine-learning classifiers (serve RQ1, RQ2): can machine-learning models tell lonely from not lonely, and isolated from not isolated people, from the
language markers found earlier? Several classifiers are compared, separately for each text type, with nested
cross-validation:

1. Labels: lonely = loneliness score 6-9 (of 3-9); isolated = top third of the objective isolation index
   (the same groups as the box plots of the report).
2. Candidate metrics: every metric that was an FDR hit for loneliness or isolation in RQ1, RQ2 or RQ1-2
   (results/all/hits.csv), in any text type; the same pool for both labels. These hits were found with all people,
   including those in the test folds below, so the scores are somewhat optimistic.
3. Outer loop: 5 folds by person (all of a person's sessions in one fold; the same folds for every text type and
   model). Each fold is the test set once, so every model gets five test scores, whose mean and SD are reported.
4. Inner loop: on the four training folds, imputation, z-scoring, feature selection, class balancing and model
   are one pipeline whose hyperparameters are tuned by grid search with 5-fold cross-validation by person (scored
   by ROC AUC). Feature selection keeps the candidates whose difference between the two classes passes
   Benjamini-Hochberg FDR (q < 0.05) in the training data. Class balancing: SMOTE adds synthetic lonely / isolated
   texts (interpolated between near neighbours of that class) until both classes are equally large. It runs on
   training data only (imbalanced-learn pipeline); the test fold keeps its real class sizes.
5. Model choice: the four models with the highest inner cross-validated AUC are kept for the figures, so the
   choice does not look at the test folds.
6. Test scores per outer fold: balanced accuracy (chance 0.5 whatever the class sizes), ROC AUC, and, for
   reference, accuracy next to the accuracy of always guessing the larger class, and F1.

Unlike A1, the models see language only (no covariates), and each text type is used on its own, so the result
says which kind of narrative carries the information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import NARRATIVE_TYPES, SEED

TARGETS = {"loneliness": "Lonely (score 6–9)", "isolation_index": "Isolated (top third)"}
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
N_BEST_MODELS = 4
FDR_LEVEL = .05
MIN_SELECTED_METRICS = 5


def logistic_regression():
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(penalty="l2", max_iter=5000, random_state=0)
    return model, {"C": [.001, .01, .1, 1]}


def lasso_logistic_regression():
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=0)
    return model, {"C": [.01, .03, .1, .3]}


def elastic_net_logistic_regression():
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=.5, max_iter=5000,
                               random_state=0)
    return model, {"C": [.01, .03, .1]}


def support_vector_machine():
    from sklearn.svm import SVC

    model = SVC(kernel="rbf", probability=True, random_state=0)
    return model, {"C": [.1, 1], "gamma": ["scale", .001]}


def random_forest():
    from sklearn.ensemble import RandomForestClassifier

    # Large leaves and a third of the metrics per split: with a weak signal, deep trees only learn noise.
    model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    return model, {"min_samples_leaf": [10, 30], "max_features": ["sqrt", .33]}


def gradient_boosting():
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(max_iter=300, max_depth=3, l2_regularization=1.0, early_stopping=True, validation_fraction=.15, random_state=0)
    return model, {"learning_rate": [.02, .05], "min_samples_leaf": [20, 50]}


def nearest_neighbours():
    from sklearn.neighbors import KNeighborsClassifier

    return KNeighborsClassifier(weights="distance"), {"n_neighbors": [15, 35, 75]}


def naive_bayes():
    from sklearn.naive_bayes import GaussianNB

    return GaussianNB(), {"var_smoothing": [1e-9, 1e-3, 1e-1]}


# Every classifier compared, with the function that builds it and its hyperparameter grid.
MODELS = {
    "Logistic regression": logistic_regression, "Lasso logistic": lasso_logistic_regression,
    "Elastic-net logistic": elastic_net_logistic_regression, "SVM (RBF)": support_vector_machine,
    "Random forest": random_forest, "Gradient boosting": gradient_boosting,
    "k-nearest neighbours": nearest_neighbours, "Naive Bayes": naive_bayes,
}


def add_class_labels(scored_texts: pd.DataFrame, session_scores: pd.DataFrame) -> pd.DataFrame:
    """1 = lonely / isolated, 0 = not. The isolation cut-off is the top third over all sessions with texts."""
    isolation_cut = session_scores["isolation_index"].quantile(2 / 3)
    labelled = scored_texts.copy()
    labelled["lonely"] = (labelled["loneliness_raw"] >= 6).where(labelled["loneliness_raw"].notna())
    labelled["isolated"] = (labelled["isolation_index"] > isolation_cut).where(labelled["isolation_index"].notna())
    return labelled


LABEL_OF_TARGET = {"loneliness": "lonely", "isolation_index": "isolated"}
# The research questions whose FDR hits are the candidate metrics.
CANDIDATE_FINDINGS = ["RQ1 isolation", "RQ2 loneliness", "RQ1-2 isolation vs loneliness"]


def candidate_metrics(hits: pd.DataFrame) -> list[str]:
    """Every metric that was an FDR hit for loneliness or isolation, robust or not, in any text type."""
    of_loneliness_or_isolation = hits["finding"].isin(CANDIDATE_FINDINGS) & hits["term"].isin(list(TARGETS))
    return sorted(hits.loc[of_loneliness_or_isolation, "feature"].unique())


def outer_fold_of_person(people: np.ndarray, seed: int = SEED) -> dict:
    """Outer fold (0-4) of every person; the same for every text type, target and model."""
    rng = np.random.default_rng(seed)
    people = np.sort(np.unique(people))
    return dict(zip(people, rng.permutation(len(people)) % N_OUTER_FOLDS))


def fdr_selected_metrics(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Which metrics differ between the two classes: an F-test (= t-test) per metric, Benjamini-Hochberg q < 0.05.
    If fewer than MIN_SELECTED_METRICS pass, the ones with the smallest p are kept, so every model has something
    to learn from (and a fold where nothing passes is scored rather than skipped).
    """
    from scipy.stats import false_discovery_control
    from sklearn.feature_selection import f_classif

    _, p = f_classif(X, y)
    p = np.nan_to_num(p, nan=1.0)
    selected = false_discovery_control(p, method="bh") < FDR_LEVEL
    if selected.sum() < MIN_SELECTED_METRICS:
        selected = np.zeros(len(p), dtype=bool)
        selected[np.argsort(p)[:MIN_SELECTED_METRICS]] = True
    return selected


def fdr_selection_step():
    """
    fdr_selected_metrics as a pipeline step, so it is learnt from the training data of each fold only.
    scikit-learn pipelines need an object with fit/transform, which is why this one helper is a class.
    """
    from sklearn.base import BaseEstimator, TransformerMixin

    class FdrSelection(BaseEstimator, TransformerMixin):
        def fit(self, X, y):
            self.selected_ = fdr_selected_metrics(X, y)
            return self

        def transform(self, X):
            return X[:, self.selected_]

    return FdrSelection()


def model_pipeline(name: str):
    """
    Median imputation, z-scoring, FDR feature selection, SMOTE class balancing and the model, all learnt from the
    training data only. The imbalanced-learn Pipeline applies SMOTE when fitting and skips it when predicting, so
    test texts are never resampled.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    model, grid = MODELS[name]()
    pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("select", fdr_selection_step()), ("balance", SMOTE(random_state=SEED)),
                         ("model", model)])
    return pipeline, {f"model__{parameter}": values for parameter, values in grid.items()}


def tune_on_training_people(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Grid search over the model's hyperparameters, 5-fold cross-validation by person, scored by ROC AUC."""
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

    pipeline, grid = model_pipeline(name)
    folds = StratifiedGroupKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=folds, n_jobs=-1)
    return search.fit(X, y, groups=groups)


def classification_scores(y_true: np.ndarray, predicted: np.ndarray, probability: np.ndarray) -> dict:
    """
    Test-set scores. The classes come from the model's own decision (predict), not from probability >= 0.5:
    the SVM's probabilities are calibrated separately from its decision function, so the two can disagree.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    return {"accuracy": accuracy_score(y_true, predicted),
            "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
            "roc_auc": roc_auc_score(y_true, probability),
            "f1": f1_score(y_true, predicted, zero_division=0)}


def texts_of_type(labelled: pd.DataFrame, used_metrics: pd.DataFrame, candidates: list[str], ntype: str,
                  target: str) -> pd.DataFrame:
    """Every text of one type with a label for the target, and the candidate metrics used in that type."""
    metrics = [m for m in candidates if used_metrics.loc[m, ntype]]
    label = LABEL_OF_TARGET[target]
    texts = labelled[labelled["ntype"].eq(ntype) & labelled[label].notna()]
    return texts[["id", "test_session", label] + metrics].rename(columns={label: "label"})


def classify_one_text_type(texts: pd.DataFrame, fold_of_person: dict,
                           models: list[str]) -> tuple[list, list]:
    """For every outer fold: tune each model on the other folds and score it on this one."""
    metrics = [c for c in texts.columns if c not in ("id", "test_session", "label")]
    X, y, groups = texts[metrics].to_numpy(), texts["label"].astype(int).to_numpy(), texts["id"].to_numpy()
    fold = texts["id"].map(fold_of_person).to_numpy()

    fold_rows, prediction_rows = [], []
    for k in range(N_OUTER_FOLDS):
        train, test = fold != k, fold == k
        # Always guessing the larger class of the training folds: the accuracy any model has to beat.
        majority_class = int(y[train].mean() >= .5)
        baseline_accuracy = (y[test] == majority_class).mean()
        for name in models:
            search = tune_on_training_people(name, X[train], y[train], groups[train])
            predicted = search.predict(X[test])
            probability = search.predict_proba(X[test])[:, 1]
            fold_rows.append({"model": name, "fold": k, "n_train": train.sum(), "n_test": test.sum(),
                              "n_metrics": len(metrics), "share_positive_test": y[test].mean(),
                              "majority_baseline_accuracy": baseline_accuracy,
                              "n_selected_metrics": int(search.best_estimator_["select"].selected_.sum()),
                              "inner_cv_roc_auc": search.best_score_, "best_parameters": str(search.best_params_),
                              **classification_scores(y[test], predicted, probability)})
            prediction_rows.append(pd.DataFrame({"model": name, "fold": k, "id": groups[test],
                                                 "test_session": texts["test_session"].to_numpy()[test],
                                                 "label": y[test], "predicted": predicted,
                                                 "probability": probability}))
    return fold_rows, prediction_rows


def classify_per_text_type(scored_texts: pd.DataFrame, session_scores: pd.DataFrame, used_metrics: pd.DataFrame,
                           hits: pd.DataFrame, types: list[str] = NARRATIVE_TYPES, targets: list[str] = list(TARGETS),
                           models: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test scores of every model in every outer fold, for each text type and target, and the out-of-fold
    probabilities (each text predicted once, by models that never saw its writer; for the ROC curves).
    """
    models = models or list(MODELS)
    candidates = candidate_metrics(hits)
    labelled = add_class_labels(scored_texts, session_scores)
    fold_of_person = outer_fold_of_person(labelled["id"].to_numpy())
    folds, predictions = [], []
    for target in targets:
        for ntype in types:
            texts = texts_of_type(labelled, used_metrics, candidates, ntype, target)
            fold_rows, prediction_rows = classify_one_text_type(texts, fold_of_person, models)
            folds.append(pd.DataFrame(fold_rows).assign(target=target, ntype=ntype))
            predictions.append(pd.concat(prediction_rows).assign(target=target, ntype=ntype))
            print(f"{target} / {ntype}: {len(texts)} texts x {texts.shape[1] - 3} metrics, "
                  f"{texts['label'].mean():.0%} positive")
    folds = pd.concat(folds, ignore_index=True)
    first = ["target", "ntype", "model", "fold"]
    folds = folds[first + [c for c in folds.columns if c not in first]]
    return folds, pd.concat(predictions, ignore_index=True)


SCORES = ["accuracy", "majority_baseline_accuracy", "balanced_accuracy", "roc_auc", "f1", "n_selected_metrics"]


def summarise_folds(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over the outer folds of each score, per target, text type and model."""
    summary = folds.groupby(["target", "ntype", "model"], sort=False)[SCORES].agg(["mean", "std"])
    summary.columns = [f"{score}_{statistic}" for score, statistic in summary.columns]
    return summary.reset_index()


def rank_models(folds: pd.DataFrame, n_best: int = N_BEST_MODELS) -> pd.DataFrame:
    """
    Models ordered by their inner cross-validated AUC (on training folds only), averaged over outer folds, text
    types and targets; the first n_best are kept for the figures. The test folds play no part in the choice.
    """
    ranking = folds.groupby("model").agg(mean_inner_cv_roc_auc=("inner_cv_roc_auc", "mean"),
                                         mean_test_accuracy=("accuracy", "mean"),
                                         mean_test_balanced_accuracy=("balanced_accuracy", "mean"),
                                         mean_test_roc_auc=("roc_auc", "mean")).reset_index()
    ranking = ranking.sort_values("mean_inner_cv_roc_auc", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["shown_in_figure"] = ranking["rank"] <= n_best
    return ranking



# Pilot groupings of sessions: how far a person's loneliness and isolation go together.
LABEL_AGREEMENT = {"agree": "Labels agree (both or neither)", "disagree": "Labels disagree (only one of the two)"}
DISTANCE_FROM_LINE = {"on the line": "On the line (|residual| ≤ 1 SD)", "off the line": "Off the line (|residual| > 1 SD)"}
RESIDUAL_CUT_SD = 1.0


def label_agreement_groups(scored_texts: pd.DataFrame, session_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Per session: do the two labels agree (lonely and isolated, or neither) or disagree (lonely but not isolated,
    or isolated but not lonely)? Sessions without both labels are left out.
    """
    labelled = add_class_labels(scored_texts, session_scores)
    sessions = labelled[["id", "test_session", "lonely", "isolated"]].drop_duplicates().dropna()
    agree = sessions["lonely"].astype(bool) == sessions["isolated"].astype(bool)
    return sessions.assign(group=np.where(agree, "agree", "disagree"))[["id", "test_session", "group"]]


def distance_from_line_groups(session_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Per session: how far loneliness lies from what isolation predicts. Over all sessions, a linear mixed model
    z(loneliness) ~ z(isolation) + C(session) + (1 | person)
    gives the line; the person intercept accounts for people with two sessions, the session terms for shifts
    between sessions (e.g. the COVID-19 period). The residual is loneliness minus the fixed part (intercept,
    slope, session term): the person intercept is left out of it, because with about one session per person it
    would absorb most of what makes someone lonelier than their isolation predicts. Sessions whose residual is
    more than 1 SD of the residuals away are 'off the line', the others 'on the line'. Uses the continuous
    scores, unlike label_agreement_groups, which uses the two cut-offs. The z-scores, session shift, fitted values
    and residuals are returned too, for the figure of the groups.
    """
    import statsmodels.formula.api as smf

    sessions = session_scores[["id", "test_session", "loneliness", "isolation_index"]].dropna().reset_index(drop=True)
    sessions["loneliness"] = (sessions["loneliness"] - sessions["loneliness"].mean()) / sessions["loneliness"].std()
    sessions["isolation_index"] = ((sessions["isolation_index"] - sessions["isolation_index"].mean())
                                   / sessions["isolation_index"].std())
    fit = smf.mixedlm("loneliness ~ isolation_index + C(test_session)", sessions,
                      groups=sessions["id"]).fit(reml=True)

    fitted = pd.Series(fit.model.exog @ fit.fe_params.to_numpy(), index=sessions.index)
    line = fit.fe_params["Intercept"] + fit.fe_params["isolation_index"] * sessions["isolation_index"]
    residual = sessions["loneliness"] - fitted
    off_the_line = residual.abs() > RESIDUAL_CUT_SD * residual.std()
    return sessions.assign(fitted=fitted, session_shift=fitted - line, residual=residual,
                           cut=RESIDUAL_CUT_SD * residual.std(), slope=fit.fe_params["isolation_index"],
                           group=np.where(off_the_line, "off the line", "on the line"))


def scores_by_group(predictions: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """
    Pilot: balanced accuracy and ROC AUC of the out-of-fold predictions within each group of sessions; mean and
    SD over the outer folds. The models are not refitted, only scored on the groups. Where loneliness and isolation
    part ways, a model that really picked up the other score (e.g. isolation language when predicting loneliness)
    falls to chance or below.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    with_group = predictions.merge(groups[["id", "test_session", "group"]], on=["id", "test_session"])
    rows = []
    keys = ["target", "ntype", "model", "group", "fold"]
    for (target, ntype, model, group, fold), texts in with_group.groupby(keys, sort=False):
        rows.append({"target": target, "ntype": ntype, "model": model, "group": group, "fold": fold,
                     "n": len(texts), "share_positive": texts["label"].mean(),
                     "balanced_accuracy": balanced_accuracy_score(texts["label"], texts["predicted"]),
                     "roc_auc": roc_auc_score(texts["label"], texts["probability"])})
    per_fold = pd.DataFrame(rows)
    summary = per_fold.groupby(keys[:-1], sort=False)[["n", "share_positive", "balanced_accuracy", "roc_auc"]].agg(
        ["mean", "std"])
    summary.columns = [f"{score}_{statistic}" for score, statistic in summary.columns]
    return summary.reset_index()
