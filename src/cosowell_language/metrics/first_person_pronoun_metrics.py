"""'I' and 'we': first-person singular and plural pronouns (spaCy morphology)."""

from __future__ import annotations

from .text_parsing import per_100_words, words_of


def is_first_person_pronoun(token, number: str) -> bool:
    is_pronoun = token.pos_ == "PRON"
    is_first_person = token.morph.get("Person") == ["1"]
    return is_pronoun and is_first_person and token.morph.get("Number") == [number]


def is_i_word(token) -> bool:
    return is_first_person_pronoun(token, "Sing")


def is_we_word(token) -> bool:
    return is_first_person_pronoun(token, "Plur")


def first_person_pronoun_metrics(doc) -> dict:
    words = words_of(doc)
    n_i_words = sum(is_i_word(word) for word in words)
    n_we_words = sum(is_we_word(word) for word in words)
    return {
        "fpp_singular_rate": per_100_words(n_i_words, len(words)),
        "fpp_plural_rate": per_100_words(n_we_words, len(words)),
    }
