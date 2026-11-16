"""All 194 Empath topic categories (Fast et al., 2016), as % of tokens: one column empath_<category> each."""

from __future__ import annotations

import numpy as np

from .language_resources import empath_lexicon


def empath_category_metrics(doc) -> dict:
    lexicon = empath_lexicon()
    # Empath returns nothing for a text without tokens; every category is then missing.
    shares = lexicon.analyze(doc.text, normalize=True) or {}
    return {f"empath_{category}": 100 * shares[category] if category in shares else np.nan
            for category in lexicon.cats}
