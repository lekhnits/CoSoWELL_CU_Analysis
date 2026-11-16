"""
The pretrained transformer models of the picture-task metrics, run over lists of sentences.

    jeniakim/hedgehog                              BERT hedge-cue tagger
    j-hartmann/emotion-english-distilroberta-base  7-class sentence emotion classifier
    sentence-transformers/all-mpnet-base-v2        sentence embeddings
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...config import EMBEDDING_MODEL, EMOTION_MODEL, HEDGE_MODEL
from ..language_resources import torch_device


def hedge_word_counts(sentences: list[str], batch_size: int = 32) -> np.ndarray:
    """Number of words tagged as hedge cues ("maybe", "seems", "I think") in each sentence."""
    from transformers import pipeline

    tagger = pipeline("token-classification", model=HEDGE_MODEL, aggregation_strategy="first",
                      stride=64, device=torch_device())
    counts = []
    for tagged in tagger(sentences, batch_size=batch_size):
        # LABEL_0 is "not a hedge"; a tagged cue can span several words ("I think").
        hedge_cues = [cue for cue in tagged if cue["entity_group"] != "LABEL_0"]
        counts.append(sum(len(cue["word"].split()) for cue in hedge_cues))
    return np.array(counts)


def emotion_probabilities(sentences: list[str], batch_size: int = 64) -> pd.DataFrame:
    """Probability of each of 7 emotions (one column each) for each sentence."""
    from transformers import pipeline

    classifier = pipeline("text-classification", model=EMOTION_MODEL, top_k=None, truncation=True,
                          device=torch_device())
    rows = []
    for scores in classifier(sentences, batch_size=batch_size):
        rows.append({score["label"]: score["score"] for score in scores})
    return pd.DataFrame(rows)


def embed_sentences(sentences: list[str]) -> np.ndarray:
    """Unit-length sentence embeddings, one row per sentence."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device=str(torch_device()))
    return model.encode(sentences, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
