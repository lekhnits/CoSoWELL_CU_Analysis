"""
Information content of the picture description: Cookie Theft content units (CUs) mentioned
(tools/cookie_theft_content_units.py).
"""

from __future__ import annotations

from .text_parsing import share, words_of
from .tools.cookie_theft_content_units import count_content_units


def picture_content_unit_metrics(doc) -> dict:
    different_units, all_mentions = count_content_units(doc.text)
    n_words = len(words_of(doc))
    return {
        "cu_unique": different_units,
        "cu_all": all_mentions,
        "cu_per_word": share(all_mentions, n_words),
        "cu_efficiency": share(different_units, n_words),
    }
