"""
Features of written language: spelling errors (pyspellchecker), function words, bare nouns,
missing subjects and repetition.

    bare noun         singular common noun without determiner, possessive or number ("went to shop")
    missing subject   finite main or subordinate verb without a subject ("Went to the shop."), imperatives excluded
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .language_resources import spell_checker
from .text_parsing import lemma, per_100_words, share, words_of
from .vocabulary_metrics import CONTENT_POS

FUNCTION_POS = {"DET", "ADP", "AUX", "PRON", "CCONJ", "SCONJ", "PART"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
DETERMINER_DEPS = {"det", "poss", "nummod", "predet"}


def words_to_spell_check(words: list) -> list[str]:
    """Lower-case alphabetic words longer than one letter. Names are left out: the checker does not know them."""
    return [word.lower_ for word in words if word.is_alpha and word.pos_ != "PROPN" and len(word) > 1]


def is_bare_noun(token) -> bool:
    if token.tag_ != "NN" or token.dep_ in {"compound", "npadvmod"}:
        return False
    return not any(child.dep_ in DETERMINER_DEPS for child in token.children)


def is_finite_clause_verb(token) -> bool:
    heads_a_clause = token.dep_ in {"ROOT", "ccomp", "advcl", "parataxis"} and token.pos_ in {"VERB", "AUX"}
    is_finite = "Fin" in token.morph.get("VerbForm")
    is_imperative = "Imp" in token.morph.get("Mood")
    return heads_a_clause and is_finite and not is_imperative


def has_subject(verb) -> bool:
    return any(child.dep_ in SUBJECT_DEPS for child in verb.children)


def share_of_repeated_lemmas(words: list) -> float:
    """Share of content words that repeat an earlier content-word lemma."""
    lemmas = [lemma(word) for word in words if word.pos_ in CONTENT_POS]
    if not lemmas:
        return np.nan
    return 1 - len(set(lemmas)) / len(lemmas)


def share_of_repeated_trigrams(words: list) -> float:
    """Share of word trigrams that occur more than once."""
    lower = [word.lower_ for word in words]
    trigrams = list(zip(lower, lower[1:], lower[2:]))
    if not trigrams:
        return np.nan
    counts = Counter(trigrams)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(trigrams)


def writing_style_metrics(doc) -> dict:
    words = words_of(doc)
    checked = words_to_spell_check(words)
    unknown = spell_checker().unknown(checked)
    clause_verbs = [token for token in doc if is_finite_clause_verb(token)]
    return {
        "spelling_error_rate": per_100_words(sum(word in unknown for word in checked), len(words)),
        "function_word_prop": share(sum(word.pos_ in FUNCTION_POS for word in words), len(words)),
        "bare_noun_rate": per_100_words(sum(is_bare_noun(word) for word in words), len(words)),
        "missing_subject_rate": share(sum(not has_subject(verb) for verb in clause_verbs), len(clause_verbs)),
        "repeated_lemma_prop": share_of_repeated_lemmas(words),
        "repeated_trigram_prop": share_of_repeated_trigrams(words),
    }
