"""
A closer look at the top-ranked classifier (stats/model_comparison_per_text.py):

1. Classification report: precision, recall and F1 of each class, from the out-of-fold predictions, per label and
   text type (mean and SD over the outer folds), and the confusion matrix of all texts together.
2. Metric importance: how much the test ROC AUC drops when one metric is shuffled (permutation importance). The
   top model is tuned again on the training people of each outer fold, exactly as before (same folds, same
   seeds), and shuffled on the test people only, so the importance is measured on writers the model never saw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import NARRATIVE_TYPES, SEED
from .model_comparison_per_text import (N_OUTER_FOLDS, TARGETS, add_class_labels, candidate_metrics,
                                        outer_fold_of_person, texts_of_type, tune_on_training_people)

CLASS_NAMES = {"loneliness": ["Not lonely", "Lonely"], "isolation_index": ["Not isolated", "Isolated"]}
N_SHUFFLES = 20


def report_of_one_fold(of_fold: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    """scikit-learn's classification report of one outer test fold, one row per class and average."""
    from sklearn.metrics import classification_report

    report = classification_report(of_fold["label"], of_fold["predicted"], labels=[0, 1], target_names=class_names,
                                   output_dict=True, zero_division=0)
    accuracy = report.pop("accuracy")
    rows = pd.DataFrame(report).T.rename_axis("class").reset_index()
    return rows.assign(accuracy=accuracy)


def classification_report_table(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Per label, text type and class (and the macro and support-weighted averages): precision, recall and F1, mean
    and SD over the outer test folds; support = number of test texts over all folds. Recall of the positive class
    is the sensitivity, recall of the negative class the specificity; their mean is the balanced accuracy.
    """
    of_model = predictions[predictions["model"].eq(model)]
    per_fold = []
    for (target, ntype, fold), of_fold in of_model.groupby(["target", "ntype", "fold"], sort=False):
        report = report_of_one_fold(of_fold, CLASS_NAMES[target])
        per_fold.append(report.assign(target=target, ntype=ntype, fold=fold))
    per_fold = pd.concat(per_fold, ignore_index=True)

    keys = ["target", "ntype", "class"]
    scores = per_fold.groupby(keys, sort=False)[["precision", "recall", "f1-score"]].agg(["mean", "std"])
    scores.columns = [f"{score}_{statistic}" for score, statistic in scores.columns]
    support = per_fold.groupby(keys, sort=False)["support"].sum().astype(int)
    return scores.join(support).reset_index()


def confusion_matrices(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    """Counts of true / false negatives and positives of all out-of-fold texts, per label and text type."""
    of_model = predictions[predictions["model"].eq(model)]
    rows = []
    for (target, ntype), texts in of_model.groupby(["target", "ntype"], sort=False):
        actual, predicted = texts["label"].astype(bool), texts["predicted"].astype(bool)
        rows.append({"target": target, "ntype": ntype,
                     "true_negative": (~actual & ~predicted).sum(), "false_positive": (~actual & predicted).sum(),
                     "false_negative": (actual & ~predicted).sum(), "true_positive": (actual & predicted).sum()})
    return pd.DataFrame(rows)


def importance_in_one_text_type(texts: pd.DataFrame, fold_of_person: dict, model: str) -> pd.DataFrame:
    """Permutation importance of every metric in every outer test fold (drop in ROC AUC, mean of the shuffles)."""
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    metrics = [c for c in texts.columns if c not in ("id", "test_session", "label")]
    X, y, groups = texts[metrics].to_numpy(), texts["label"].astype(int).to_numpy(), texts["id"].to_numpy()
    fold = texts["id"].map(fold_of_person).to_numpy()

    rows = []
    for k in range(N_OUTER_FOLDS):
        train, test = fold != k, fold == k
        fitted = tune_on_training_people(model, X[train], y[train], groups[train]).best_estimator_
        shuffled = permutation_importance(fitted, X[test], y[test], scoring="roc_auc", n_repeats=N_SHUFFLES,
                                          random_state=SEED, n_jobs=-1)
        test_auc = roc_auc_score(y[test], fitted.predict_proba(X[test])[:, 1])
        rows.append(pd.DataFrame({"fold": k, "metric": metrics, "test_roc_auc": test_auc,
                                  "auc_drop": shuffled.importances_mean,
                                  "selected": fitted["select"].selected_}))
    return pd.concat(rows, ignore_index=True)


def metric_importance(scored_texts: pd.DataFrame, session_scores: pd.DataFrame, used_metrics: pd.DataFrame,
                      hits: pd.DataFrame, model: str, types: list[str] = NARRATIVE_TYPES) -> pd.DataFrame:
    """
    Per label, text type and metric: drop in test ROC AUC when the metric is shuffled, mean and SD over the outer
    folds, and in how many folds the FDR selection kept it (a metric not kept has importance 0 in that fold).
    """
    candidates = candidate_metrics(hits)
    labelled = add_class_labels(scored_texts, session_scores)
    fold_of_person = outer_fold_of_person(labelled["id"].to_numpy())

    per_fold = []
    for target in TARGETS:
        for ntype in types:
            texts = texts_of_type(labelled, used_metrics, candidates, ntype, target)
            importance = importance_in_one_text_type(texts, fold_of_person, model)
            per_fold.append(importance.assign(target=target, ntype=ntype))
    per_fold = pd.concat(per_fold, ignore_index=True)

    summary = per_fold.groupby(["target", "ntype", "metric"], sort=False).agg(
        auc_drop_mean=("auc_drop", "mean"), auc_drop_std=("auc_drop", "std"),
        folds_selected=("selected", "sum")).reset_index()
    return summary.sort_values(["target", "ntype", "auc_drop_mean"], ascending=[True, True, False],
                               ignore_index=True)

