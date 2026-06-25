"""
Script 1: Quantitext-style Content Unit (CU) Extractor
=======================================================
Implements the content-unit extraction pipeline described in:

    Josephy-Hernandez, S., et al. (2023).
    "Automated analysis of written language in the three variants of
    primary progressive aphasia."
    Brain Communications, 5(4), fcad202.
    https://doi.org/10.1093/braincomms/fcad202

The original toolbox (developed at the MGH FTD Unit) uses Stanford Parser / Stanza,
spaCy, and R. This script re-implements the core algorithm in pure Python
using spaCy + NLTK WordNet for lemmatisation, following the paper's
description of the dictionary-lookup approach.

Related public resources:
  - spaCy:          https://github.com/explosion/spaCy
  - NLTK / WordNet: https://github.com/nltk/nltk
  - Stanza NLP:     https://github.com/stanfordnlp/stanza

"""

import re
import math
import string
from collections import Counter

import spacy

import nltk

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    import os
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

from nltk.corpus import wordnet as wn

# ---------------------------------------------------------------------------
# 1.  Pre-defined Content-Unit dictionary for the Cookie Theft picture
#     (Ahmed et al. 2010 / Josephy-Hernandez et al. 2023 content-word list;
#      23 semantic elements expected in a complete description).
#
#     Each key is a canonical lemma; synonyms / alternate surface forms are
#     listed as values. The extractor accepts any word whose lemma matches
#     a key or any of its synonyms.
# ---------------------------------------------------------------------------

COOKIE_THEFT_CU_DICT: dict[str, list[str]] = {
    # 1. Boy
    "boy": ["adolescent", "boy", "brother", "child", "guy", "he", "himself",
            "his", "kid", "male", "man", "shirt", "shoe", "sock", "son"],
    # 2. Girl
    "girl": ["child", "daughter", "dress", "girl", "kid", "sister", "skirt"],
    # 3. Woman
    "woman": ["adult", "apron", "domestic", "dress", "fifties", "heels",
              "high-heels", "homemaker", "housewife", "lady", "ladys", "mama",
              "mom", "mother", "nineteen-fifties", "wife", "woman"],
    # 4. Kitchen
    "kitchen": ["counter", "culinary", "home", "interior", "kitchen"],
    # 5. Outside
    "outside": ["background", "backyard", "bush", "day", "exterior", "flower",
                "garage", "garden", "grass", "grow", "lawn", "leaves",
                "outdoor", "outside", "path", "pathway", "scenery",
                "shrubbery", "shrub", "sidewalk", "snow", "spring", "summer",
                "sunny", "tree", "walkway", "yard"],
    # 6. Cookie
    "cookie": ["chocolatechip", "cookie", "pastry", "snack"],
    # 7. Jar
    "jar": ["canister", "container", "holder", "jar", "lid", "vessel"],
    # 8. Stool
    "stool": ["bench", "chair", "footstool", "furniture", "ladder", "perch",
              "step", "stepladder", "stepstool", "stool", "three",
              "threelegged"],
    # 9. Sink
    "sink": ["basin", "drain", "faucet", "sink"],
    # 10. Plate
    "plate": ["dish", "plate", "saucer"],
    # 11. Dishcloth
    "dishcloth": ["cloth", "dishcloth", "dishrag", "dishtowel", "handtowel",
                  "napkin", "rag", "textile", "towel", "sponge"],
    # 12. Water
    "water": ["deluge", "flood", "floor", "flow", "inundation", "liquid",
              "moisture", "puddle", "torrent", "water"],
    # 13. Window
    "window": ["casement", "glass", "pane", "window"],
    # 14. Cupboard
    "cupboard": ["cabinet", "cupboard", "door", "handle", "shelf", "storage"],
    # 15. Dishes
    "dishes": ["bowl", "cup", "dish", "plate"],
    # 16. Curtains
    "curtains": ["curtain", "drape", "dressing", "fabric", "hang", "textile",
                 "tie", "tieback", "wave", "wind", "window"],
    # 17. Boy taking / stealing
    "stealing": ["acquire", "climb", "extract", "grab", "raid", "rob",
                 "secure", "sneak", "snitch", "steal", "snatch", "take",
                 "try"],
    # 18. Boy or stool falling
    "falling": ["backwards", "balance", "collapse", "crash", "fall", "hurt",
                "overturn", "tilt", "tip", "tipping", "topple", "unstable"],
    # 19. Woman drying / washing plates / dishes
    "washing": ["clean", "dry", "rinse", "scrub", "wash", "washing", "wipe"],
    # 20. Water overflowing / spilling
    "overflowing": ["deluge", "drip", "faucet", "flood", "flow", "inundation",
                    "overfill", "overflow", "overflowing", "overrun", "pour",
                    "run", "spill", "spilling", "splash", "torrent"],
    # 21. Action performed by the girl
    "girl_action": ["ask", "finger", "gesture", "giggle", "laugh", "lip",
                    "motion", "mouth", "point", "quiet", "reach", "request",
                    "say", "shout", "shh", "signal", "speak", "tell", "warn"],
    # 22. Woman unconcerned by overflowing
    "unconcerned": ["attention", "aware", "care", "concern", "daydream",
                    "daze", "disregard", "distract", "ignorant", "ignore",
                    "neglectful", "nonchalant", "notice", "oblivious",
                    "realize", "stand", "unaware", "unconcerned"],
    # 23. Woman indifferent to the children
    "indifferent": ["apathetic", "attention", "aware", "back", "behind",
                    "clue", "disregard", "distract", "doesn’t see",
                    "focus", "ignorant", "ignore", "indifferent",
                    "nonchalant", "notice", "oblivious", "turn", "unaware",
                    "unconcern", "unconcerned"],
}

# Flat lookup: synonym → canonical CU
_SYNONYM_TO_CU: dict[str, str] = {}
for canonical, synonyms in COOKIE_THEFT_CU_DICT.items():
    _SYNONYM_TO_CU[canonical] = canonical
    for syn in synonyms:
        _SYNONYM_TO_CU[syn.lower()] = canonical


# ---------------------------------------------------------------------------
# 2.  Load spaCy model
# ---------------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise SystemExit(
        "spaCy model not found.\n"
        "Run: python -m spacy download en_core_web_sm"
    )


# ---------------------------------------------------------------------------
# 3.  Helpers
# ---------------------------------------------------------------------------

def _wordnet_lemmas(word: str) -> set[str]:
    """Return all WordNet lemma strings for a given word."""
    lemmas: set[str] = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            lemmas.add(lemma.name().replace("_", " ").lower())
    return lemmas


def _match_cu(token_lemma: str, token_text: str) -> str | None:
    """
    Return the canonical CU label if the token matches a dictionary entry,
    or None otherwise.  Tries: (1) exact lemma, (2) exact text, (3) synonym
    table, (4) WordNet overlap with any synonym.
    """
    candidates = {token_lemma.lower(), token_text.lower()}
    for cand in candidates:
        if cand in _SYNONYM_TO_CU:
            return _SYNONYM_TO_CU[cand]

    # WordNet fallback: check if the token is semantically close to any synonym
    token_wn = _wordnet_lemmas(token_lemma)
    for entry, canonical in _SYNONYM_TO_CU.items():
        if token_wn and entry in token_wn:
            return canonical
    return None


def _log_word_frequency(word: str, frequency_table: dict[str, int] | None = None) -> float:
    """
    Log word frequency proxy.  When no external frequency table is provided
    we use a simple inverse-document-frequency stand-in (log of 1/rank).
    In the original Quantitext toolbox this is replaced with SUBTLEX-US
    or similar corpus frequencies.
    """
    if frequency_table and word in frequency_table:
        return math.log1p(frequency_table[word])
    # Fallback: use length as a crude surrogate (shorter = more frequent)
    return -math.log1p(len(word))


# ---------------------------------------------------------------------------
# 4.  Main analysis function
# ---------------------------------------------------------------------------

def analyse_language_sample(
    text: str,
    cu_dict: dict[str, list[str]] | None = None,
    frequency_table: dict[str, int] | None = None,
) -> dict:
    """
    Analyse a single language sample (typed or transcribed speech).

    Parameters
    ----------
    text : str
        Raw language sample.
    cu_dict : dict | None
        Custom CU dictionary; defaults to COOKIE_THEFT_CU_DICT.
    frequency_table : dict | None
        External word-frequency table {word: count}.

    Returns
    -------
    dict with keys:
        total_words, total_sentences, mean_sentence_length,
        content_units, content_unit_count, total_units,
        cu_efficiency, pos_counts, mean_log_word_frequency,
        type_token_ratio
    """
    if cu_dict is not None:
        # Rebuild lookup from custom dict
        global _SYNONYM_TO_CU
        _SYNONYM_TO_CU = {}
        for canonical, synonyms in cu_dict.items():
            _SYNONYM_TO_CU[canonical] = canonical
            for syn in synonyms:
                _SYNONYM_TO_CU[syn.lower()] = canonical

    doc = nlp(text)

    # ---- Tokenisation -------------------------------------------------------
    tokens = [t for t in doc if not t.is_space]
    words  = [t for t in tokens if not t.is_punct]

    total_words     = len(words)
    total_sentences = len(list(doc.sents))
    mean_sentence_length = total_words / max(total_sentences, 1)

    # ---- POS counts ---------------------------------------------------------
    pos_counts: dict[str, int] = Counter(t.pos_ for t in words)

    # ---- Content-unit identification ----------------------------------------
    found_cus: list[dict] = []
    seen_canonical: set[str] = set()   # each CU counted once per sample

    for token in words:
        canonical = _match_cu(token.lemma_, token.text)
        if canonical and canonical not in seen_canonical:
            seen_canonical.add(canonical)
            found_cus.append({
                "token":     token.text,
                "lemma":     token.lemma_,
                "canonical": canonical,
                "sentence":  token.sent.text.strip(),
                "pos":       token.pos_,
            })

    content_unit_count = len(found_cus)

    # ---- Total units (all words including function words) -------------------
    total_units = total_words

    # ---- CU efficiency = CU count / total words -----------------------------
    cu_efficiency = content_unit_count / max(total_units, 1)

    # ---- Mean log word frequency --------------------------------------------
    log_freqs = [_log_word_frequency(t.lemma_, frequency_table) for t in words]
    mean_log_wf = sum(log_freqs) / max(len(log_freqs), 1)

    # ---- Type-token ratio ---------------------------------------------------
    types = {t.lemma_.lower() for t in words}
    ttr   = len(types) / max(total_words, 1)

    return {
        "total_words":           total_words,
        "total_sentences":       total_sentences,
        "mean_sentence_length":  round(mean_sentence_length, 3),
        "content_units":         found_cus,
        "content_unit_count":    content_unit_count,
        "total_units":           total_units,
        "cu_efficiency":         round(cu_efficiency, 4),
        "pos_counts":            dict(pos_counts),
        "mean_log_word_frequency": round(mean_log_wf, 4),
        "type_token_ratio":      round(ttr, 4),
    }


# ---------------------------------------------------------------------------
# 5.  CLI / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys

    # Example Cookie Theft transcript (healthy control)
    sample = (
        "The woman is standing at the sink washing dishes while the water is overflowing onto the floor. "
        "She seems unaware of this. In the background, a boy is standing on a stool trying to steal cookies "
        "from a cookie jar on the shelf. He might fall because the stool is tipping over. "
        "A girl is standing next to him with her hand outstretched, asking for a cookie. "
        "There is a window above the sink with curtains, and you can see outside."
    )

    if len(sys.argv) > 1:
        # Accept a plain-text file as argument
        with open(sys.argv[1], encoding="utf-8") as fh:
            sample = fh.read()

    results = analyse_language_sample(sample)

    print("=" * 60)
    print("Quantitext-style Language Sample Analysis")
    print("=" * 60)
    print(f"Total words            : {results['total_words']}")
    print(f"Total sentences        : {results['total_sentences']}")
    print(f"Mean sentence length   : {results['mean_sentence_length']}")
    print(f"Content unit count     : {results['content_unit_count']}")
    print(f"Total units            : {results['total_units']}")
    print(f"CU efficiency          : {results['cu_efficiency']}")
    print(f"Type-token ratio       : {results['type_token_ratio']}")
    print(f"Mean log word freq     : {results['mean_log_word_frequency']}")
    print("\nPOS distribution:", results["pos_counts"])
    print("\nIdentified content units:")
    for cu in results["content_units"]:
        print(f"  [{cu['canonical']:20s}]  '{cu['token']}'  (POS: {cu['pos']})")
    print("\nFull results (JSON):")
    print(json.dumps(results, indent=2, ensure_ascii=False))