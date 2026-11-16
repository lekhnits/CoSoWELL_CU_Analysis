"""Length of the narrative: words and sentences."""

from __future__ import annotations

from .text_parsing import words_of


def text_length_metrics(doc) -> dict:
    return {
        "n_words": len(words_of(doc)),
        "n_sentences": len(list(doc.sents)),
    }
