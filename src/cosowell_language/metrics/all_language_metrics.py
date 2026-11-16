"""
Every language metric of every narrative (results/features/<type>.csv and cookie_gaze.csv).

Each metric file holds one family of metrics. A per-text family is a function
doc -> {metric name: value}; a family that runs a transformer model over all
sentences at once is a function [docs] -> table. The metric names are the columns
of results/features/, in the order listed here.
"""

from __future__ import annotations

import pandas as pd

from ..config import KEYS, TRANSFORMER_TYPES
from .cognitive_process_metrics import cognitive_process_metrics
from .coherence_metrics import coherence_metrics
from .emotion_and_illness_word_metrics import emotion_and_illness_word_metrics
from .empath_category_metrics import empath_category_metrics
from .first_person_pronoun_metrics import first_person_pronoun_metrics
from .hedging_metrics import hedging_metrics
from .language_resources import ensure_nltk_resources
from .narrative_gaze_metrics import narrative_gaze_metrics
from .other_pronoun_metrics import other_pronoun_metrics
from .picture_content_unit_metrics import picture_content_unit_metrics
from .sentence_emotion_metrics import sentence_emotion_metrics
from .sentiment_metrics import sentiment_metrics
from .social_word_metrics import social_word_metrics
from .syntax_metrics import syntax_metrics
from .tense_and_modality_metrics import tense_and_modality_metrics
from .text_length_metrics import text_length_metrics
from .text_parsing import parse_texts
from .vocabulary_metrics import vocabulary_metrics
from .writing_style_metrics import writing_style_metrics

# Computed for every narrative type, one text at a time.
FOR_EVERY_NARRATIVE = [
    text_length_metrics, first_person_pronoun_metrics, social_word_metrics, cognitive_process_metrics,
    emotion_and_illness_word_metrics, vocabulary_metrics, syntax_metrics, writing_style_metrics,
    sentiment_metrics, other_pronoun_metrics, tense_and_modality_metrics, empath_category_metrics,
]


def measure_every_text(docs: list, family) -> pd.DataFrame:
    return pd.DataFrame([family(doc) for doc in docs])


def measure_language(narratives: pd.DataFrame, ntype: str) -> pd.DataFrame:
    """Language metrics of narratives of one type (columns id, test_session, narrative), one row each, same order."""
    ensure_nltk_resources()
    docs = parse_texts(narratives["narrative"].tolist())
    tables = [measure_every_text(docs, family) for family in FOR_EVERY_NARRATIVE]
    if ntype in TRANSFORMER_TYPES:
        # Picture descriptions only: the transformer models are slow, and content units belong to the picture.
        tables.append(hedging_metrics(docs))
        tables.append(sentence_emotion_metrics(docs))
        tables.append(measure_every_text(docs, picture_content_unit_metrics))
        tables.append(coherence_metrics(docs))
    keys = narratives[KEYS].reset_index(drop=True)
    return pd.concat([keys, *tables], axis=1)


def measure_gaze(narratives: pd.DataFrame) -> pd.DataFrame:
    """Narrative-gaze metrics of picture descriptions, one row each."""
    gaze = pd.DataFrame([narrative_gaze_metrics(text) for text in narratives["narrative"]])
    keys = narratives[KEYS].reset_index(drop=True)
    return pd.concat([keys, gaze], axis=1)
