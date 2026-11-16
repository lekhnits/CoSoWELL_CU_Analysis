"""
Tentative language (picture task): words tagged as hedge cues by a BERT hedge tagger
(jeniakim/hedgehog), and all cognitive-process words together.
"""

from __future__ import annotations

import pandas as pd

from .cognitive_process_metrics import count_cognition_verbs, count_differentiation
from .text_parsing import per_100_words, sentence_rows_of_each_text, sentences_of_all_texts, words_of
from .tools.pretrained_text_models import hedge_word_counts


def hedging_metrics(docs: list) -> pd.DataFrame:
    """One row per text. The tagger runs once over all sentences of all texts."""
    sentences, owner = sentences_of_all_texts(docs)
    hedges_per_sentence = hedge_word_counts(sentences)
    rows = []
    for doc, sentence_rows in zip(docs, sentence_rows_of_each_text(owner, len(docs))):
        n_hedges = int(hedges_per_sentence[sentence_rows].sum())
        n_words = len(words_of(doc))
        # All cognitive-process words: hedge words + cognition verbs + negation and contrast.
        n_cognitive = n_hedges + count_cognition_verbs(doc) + count_differentiation(doc)
        rows.append({
            "tentative_rate": per_100_words(n_hedges, n_words),
            "cognitive_process_rate": per_100_words(n_cognitive, n_words),
        })
    return pd.DataFrame(rows, columns=["tentative_rate", "cognitive_process_rate"])
