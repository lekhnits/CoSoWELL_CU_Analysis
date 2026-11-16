"""Second- and third-person personal pronouns (spaCy morphology)."""

from __future__ import annotations

from .text_parsing import per_100_words, words_of


def is_personal_pronoun_in_person(token, person: str) -> bool:
    is_personal_pronoun = token.pos_ == "PRON" and "Prs" in token.morph.get("PronType")
    return is_personal_pronoun and token.morph.get("Person") == [person]


def is_he_or_she(token) -> bool:
    has_gender = bool({"Masc", "Fem"} & set(token.morph.get("Gender")))
    return is_personal_pronoun_in_person(token, "3") and has_gender


def is_they(token) -> bool:
    return is_personal_pronoun_in_person(token, "3") and token.morph.get("Number") == ["Plur"]


def other_pronoun_metrics(doc) -> dict:
    words = words_of(doc)
    n_you = sum(is_personal_pronoun_in_person(word, "2") for word in words)
    n_he_or_she = sum(is_he_or_she(word) for word in words)
    n_they = sum(is_they(word) for word in words)
    return {
        "second_person_rate": per_100_words(n_you, len(words)),
        "third_person_human_rate": per_100_words(n_he_or_she, len(words)),
        "third_person_plural_rate": per_100_words(n_they, len(words)),
    }
