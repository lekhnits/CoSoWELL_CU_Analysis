"""
Emotion, illness and pain words (Empath word sets; illness nouns also from WordNet).

    illness or pain word   Empath pain / health / injury / medical emergency / suffering, or a noun
                           whose first WordNet sense is an illness, pain, injury or symptom
"""

from __future__ import annotations

from .language_resources import empath_word_sets, is_sickness_noun
from .text_parsing import in_word_set, lemma, per_100_words, words_of


def in_empath_set(token, name: str) -> bool:
    return in_word_set(token, empath_word_sets()[name])


def is_illness_or_pain_word(token) -> bool:
    if in_empath_set(token, "sick_pain"):
        return True
    return token.pos_ == "NOUN" and is_sickness_noun(lemma(token))


def is_negative_or_illness_word(token) -> bool:
    return in_empath_set(token, "negative_emotion") or is_illness_or_pain_word(token)


def emotion_and_illness_word_metrics(doc) -> dict:
    words = words_of(doc)

    def rate_of(is_counted) -> float:
        return per_100_words(sum(is_counted(word) for word in words), len(words))

    return {
        "negative_emotion_rate": rate_of(lambda word: in_empath_set(word, "negative_emotion")),
        "sickness_pain_rate": rate_of(is_illness_or_pain_word),
        "neg_emo_sick_pain_rate": rate_of(is_negative_or_illness_word),
        "sadness_word_rate": rate_of(lambda word: in_empath_set(word, "sadness")),
        "positive_emotion_rate": rate_of(lambda word: in_empath_set(word, "positive_emotion")),
    }
