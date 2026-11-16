"""
Coherence and repetition of ideas (picture task), from unit-length sentence embeddings
(all-mpnet-base-v2). The similarity of two unit-length embeddings is their dot product (cosine).

    local coherence     mean similarity of consecutive sentences
    global coherence    mean similarity of each sentence to the narrative's own centroid
    repeated idea       a sentence nearly identical (cosine >= 0.85) to an earlier one
    off-topic sentence  less typical of the task than 95% of all sentences of all narratives
                        (typicality = similarity to the centroid of every sentence)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .text_parsing import sentence_rows_of_each_text, sentences_of_all_texts
from .tools.pretrained_text_models import embed_sentences

REPEATED_IDEA_COS = 0.85
OFF_TOPIC_PERCENTILE = 5
COLUMNS = ["local_coherence", "global_coherence", "repeated_idea_prop", "off_topic_prop"]


def unit_length(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    return vector / (length or 1.0)


def local_coherence(embeddings: np.ndarray) -> float:
    similarity_to_next = np.diag(embeddings @ embeddings.T, k=1)
    if len(similarity_to_next) == 0:
        return np.nan
    return float(similarity_to_next.mean())


def global_coherence(embeddings: np.ndarray) -> float:
    centroid = unit_length(embeddings.mean(axis=0))
    return float((embeddings @ centroid).mean())


def repeated_idea_share(embeddings: np.ndarray) -> float:
    similarity = embeddings @ embeddings.T
    repeats = [(similarity[i, :i] >= REPEATED_IDEA_COS).any() for i in range(1, len(embeddings))]
    if not repeats:
        return 0.0
    return float(np.mean(repeats))


def coherence_metrics(docs: list) -> pd.DataFrame:
    """One row per text. Off-topic needs every sentence of every text, so all texts are measured together."""
    sentences, owner = sentences_of_all_texts(docs)
    embeddings = embed_sentences(sentences)
    typicality = embeddings @ unit_length(embeddings.mean(axis=0))
    off_topic_cut = np.percentile(typicality, OFF_TOPIC_PERCENTILE)
    rows = []
    for sentence_rows in sentence_rows_of_each_text(owner, len(docs)):
        text_embeddings = embeddings[sentence_rows]
        if len(text_embeddings) == 0:
            rows.append(dict.fromkeys(COLUMNS, np.nan))
            continue
        rows.append({
            "local_coherence": local_coherence(text_embeddings),
            "global_coherence": global_coherence(text_embeddings),
            "repeated_idea_prop": repeated_idea_share(text_embeddings),
            "off_topic_prop": float(np.mean(typicality[sentence_rows] < off_topic_cut)),
        })
    return pd.DataFrame(rows, columns=COLUMNS)
