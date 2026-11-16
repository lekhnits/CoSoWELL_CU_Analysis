"""
Vocabulary: fillers, idea density, pronouns vs nouns, word frequency (wordfreq, Zipf scale),
lexical diversity (lexicalrichness) and vague words.

    idea density   propositions per word: verbs, adjectives, adverbs, prepositions, conjunctions (CPIDR-style)
    vague word     an indefinite pronoun (something, anyone ...), or a noun with >= 10 WordNet senses
                   that is very frequent (thing, place, way ...) or a WordNet top-level noun
"""

from __future__ import annotations

import numpy as np

from .language_resources import is_vague_noun, zipf
from .text_parsing import lemma, per_100_words, share, words_of

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}
PROPOSITION_POS = {"VERB", "ADJ", "ADV", "ADP", "CCONJ", "SCONJ"}
LOW_FREQUENCY_ZIPF = 3.0        # rarer than about 1 per million words
MATTR_WINDOW = 25               # moving-average type-token ratio over windows of 25 words
MTLD_THRESHOLD = 0.72           # the standard MTLD factor size (McCarthy & Jarvis, 2010)


def is_filler(token) -> bool:
    return token.pos_ == "INTJ" or token.dep_ == "discourse"


def is_vague_word(token) -> bool:
    is_indefinite_pronoun = token.pos_ == "PRON" and "Ind" in token.morph.get("PronType")
    if is_indefinite_pronoun:
        return True
    return token.pos_ == "NOUN" and is_vague_noun(lemma(token))


def content_word_frequencies(words: list) -> list[float]:
    """Zipf frequency of the lemma of each content word."""
    content_lemmas = [lemma(word) for word in words if word.pos_ in CONTENT_POS]
    return [zipf(content_lemma) for content_lemma in content_lemmas if content_lemma.isalpha()]


def lexical_diversity(words: list) -> dict:
    """Type-token ratio, MATTR and MTLD of the alphabetic words (lexicalrichness)."""
    from lexicalrichness import LexicalRichness

    if not words:
        return {"ttr": np.nan, "mattr": np.nan, "mtld": np.nan}
    alphabetic = [word.lower_ for word in words if word.lower_.isalpha()]
    richness = LexicalRichness(" ".join(alphabetic))
    n = richness.words
    return {
        "ttr": richness.ttr if n else np.nan,
        "mattr": richness.mattr(window_size=min(MATTR_WINDOW, n)) if n >= 2 else np.nan,
        "mtld": richness.mtld(threshold=MTLD_THRESHOLD) if n >= 2 else np.nan,
    }


def vocabulary_metrics(doc) -> dict:
    words = words_of(doc)
    n_nouns = sum(word.pos_ in {"NOUN", "PROPN"} for word in words)
    n_pronouns = sum(word.pos_ == "PRON" for word in words)
    frequencies = content_word_frequencies(words)
    diversity = lexical_diversity(words)
    return {
        "filler_rate": per_100_words(sum(is_filler(word) for word in words), len(words)),
        "idea_density": share(sum(word.pos_ in PROPOSITION_POS for word in words), len(words)),
        "pronoun_noun_ratio": share(n_pronouns, n_nouns),
        "mean_word_zipf": float(np.mean(frequencies)) if frequencies else np.nan,
        "low_freq_word_prop": share(sum(frequency < LOW_FREQUENCY_ZIPF for frequency in frequencies), len(frequencies)),
        "ttr": diversity["ttr"],
        "mattr": diversity["mattr"],
        "mtld": diversity["mtld"],
        "vague_word_rate": per_100_words(sum(is_vague_word(word) for word in words), len(words)),
    }
