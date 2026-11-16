"""Sentence sentiment with VADER (Hutto & Gilbert, 2014): mean over the sentences of the narrative."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .language_resources import vader
from .text_parsing import sentence_texts


def sentiment_metrics(doc) -> dict:
    sentences = sentence_texts(doc)
    if not sentences:
        return {"vader_positive": np.nan, "vader_negative": np.nan, "vader_compound": np.nan}
    # VADER is built for short texts, so each sentence is scored on its own and the scores are averaged.
    scores = pd.DataFrame([vader().polarity_scores(sentence) for sentence in sentences])
    return {
        "vader_positive": float(scores["pos"].mean()),
        "vader_negative": float(scores["neg"].mean()),
        "vader_compound": float(scores["compound"].mean()),
    }
