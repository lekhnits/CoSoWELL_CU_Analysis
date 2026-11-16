"""Load the metrics and survey scores used by the research-question tests."""

from __future__ import annotations

import pandas as pd

from ..config import DUPLICATE_METRICS, EXTRA_FEATURES, FEATURES_DIR, KEYS, SURVEY_PATH
from ..data_loading import load_survey
from ..survey.survey_scores import build_session_table
from .metric_filtering import build_feature_matrix

def read_features(features_dir, ntype: str) -> pd.DataFrame:
    """Metrics of one narrative type: results/features/<type>.csv plus its extra sets (config.EXTRA_FEATURES)."""
    out = pd.read_csv(features_dir / f"{ntype}.csv")
    for name in EXTRA_FEATURES.get(ntype, []):
        extra = pd.read_csv(features_dir / f"{ntype}_{name}.csv")
        assert not set(extra.columns) & set(out.columns) - set(KEYS), f"duplicate metric names in {name}"
        out = out.merge(extra, on=KEYS, how="left")
    return out.drop(columns=[c for c in DUPLICATE_METRICS if c in out])


def load_analysis_data(types: list[str], features_dir=FEATURES_DIR, survey_path=SURVEY_PATH) -> dict:
    """
    raw:            raw metrics, one row per narrative (all narratives of the given types)
    text_metrics:   rank-normalised metrics, one row per narrative (normalised within each text type)
    features:       names of the metrics kept after the sparsity filter
    used_metrics:   metrics x text types, True where the metric is used in that type
    session_scores: z-scored survey scores, one row per session with narratives
    scored_texts:   text_metrics joined with session_scores (narratives of sessions with survey data)
    """
    raw = pd.concat([read_features(features_dir, t).assign(ntype=t) for t in types], ignore_index=True)
    text_metrics, features, used_metrics = build_feature_matrix(raw, types)
    session_scores = build_session_table(load_survey(survey_path), text_metrics)
    scored_texts = text_metrics.merge(session_scores, on=KEYS)
    return dict(raw=raw, text_metrics=text_metrics, features=features, used_metrics=used_metrics,
                session_scores=session_scores, scored_texts=scored_texts)
