"""
Expressed emotion (picture task): the mean probability of each of 7 emotions over the sentences
of a narrative (j-hartmann/emotion-english-distilroberta-base), and the share of sentences
whose most likely emotion is sadness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .text_parsing import sentence_rows_of_each_text, sentences_of_all_texts
from .tools.pretrained_text_models import emotion_probabilities

EMOTIONS = ["joy", "sadness", "neutral", "disgust", "surprise", "anger", "fear"]
COLUMNS = [f"emo_{emotion}_mean" for emotion in EMOTIONS] + ["sadness_sentence_prop"]


def emotions_of_one_text(probabilities: pd.DataFrame) -> dict:
    """`probabilities`: one row per sentence of the text, one column per emotion."""
    if len(probabilities) == 0:
        return dict.fromkeys(COLUMNS, np.nan)
    row = {f"emo_{emotion}_mean": float(probabilities[emotion].mean()) for emotion in EMOTIONS}
    most_likely = probabilities.idxmax(axis=1)
    row["sadness_sentence_prop"] = float((most_likely == "sadness").mean())
    return row


def sentence_emotion_metrics(docs: list) -> pd.DataFrame:
    """One row per text. The classifier runs once over all sentences of all texts."""
    sentences, owner = sentences_of_all_texts(docs)
    probabilities = emotion_probabilities(sentences)
    rows = [emotions_of_one_text(probabilities.iloc[sentence_rows])
            for sentence_rows in sentence_rows_of_each_text(owner, len(docs))]
    return pd.DataFrame(rows, columns=COLUMNS)
