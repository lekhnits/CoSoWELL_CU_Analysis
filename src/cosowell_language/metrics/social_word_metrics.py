"""
Relationships and social words.

    relationship noun   WordNet: the first sense of the noun is a relative, spouse, friend, neighbour ...
    social word         Empath family / friends / communication / meeting, or a relationship noun
"""

from __future__ import annotations

from .first_person_pronoun_metrics import is_we_word
from .language_resources import empath_word_sets, is_relationship_noun
from .text_parsing import in_word_set, lemma, per_100_words, words_of


def is_relationship_word(token) -> bool:
    return token.pos_ == "NOUN" and is_relationship_noun(lemma(token))


def is_social_word(token) -> bool:
    return in_word_set(token, empath_word_sets()["social"]) or is_relationship_word(token)


def social_word_metrics(doc) -> dict:
    words = words_of(doc)
    relationship_words = [word for word in words if is_relationship_word(word)]
    n_social_words = sum(is_social_word(word) for word in words)
    n_we_words = sum(is_we_word(word) for word in words)
    return {
        # Number of different relationship nouns (mother, friend, neighbour ...).
        "n_relationships_named": len({lemma(word) for word in relationship_words}),
        "relationship_word_rate": per_100_words(len(relationship_words), len(words)),
        "social_word_rate": per_100_words(n_social_words, len(words)),
        "social_plus_we_rate": per_100_words(n_social_words + n_we_words, len(words)),
    }
