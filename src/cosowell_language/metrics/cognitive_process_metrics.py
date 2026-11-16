"""
Cognitive-process words.

    differentiation   negation ("not") and contrastive coordination ("but", "or"; any coordinator except "and")
    cognition verb    the first WordNet sense of the verb is a cognition verb (think, know, remember ...)
"""

from __future__ import annotations

from .language_resources import is_cognition_verb
from .text_parsing import lemma, per_100_words, words_of

ADDITIVE_COORDINATORS = {"and", "&", "plus"}


def is_negation(token) -> bool:
    return token.dep_ == "neg"


def is_contrastive_coordinator(token) -> bool:
    is_coordinator = token.pos_ == "CCONJ" and token.dep_ == "cc"
    return is_coordinator and lemma(token) not in ADDITIVE_COORDINATORS


def is_cognition_verb_token(token) -> bool:
    return token.pos_ == "VERB" and is_cognition_verb(lemma(token))


def count_differentiation(doc) -> int:
    """Counted over every token, not only words: "not" and "but" markers may be tagged as particles."""
    n_negations = sum(is_negation(token) for token in doc)
    n_contrasts = sum(is_contrastive_coordinator(token) for token in doc)
    return n_negations + n_contrasts


def count_cognition_verbs(doc) -> int:
    return sum(is_cognition_verb_token(word) for word in words_of(doc))


def cognitive_process_metrics(doc) -> dict:
    n_words = len(words_of(doc))
    return {
        "differentiation_rate": per_100_words(count_differentiation(doc), n_words),
        "cognition_verb_rate": per_100_words(count_cognition_verbs(doc), n_words),
    }
