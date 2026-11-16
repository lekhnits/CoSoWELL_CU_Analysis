"""
Parsing a narrative with spaCy, and the word-level definitions that every metric file shares.

A "word" is a token that is not punctuation or whitespace, not a clitic ('s, n't) and
contains a letter or digit. Most metrics are rates per 100 words.
"""

from __future__ import annotations

import numpy as np

from .language_resources import nlp


def parse_texts(texts: list[str]) -> list:
    """spaCy parse of every text. Parsing in batches is much faster than one text at a time."""
    return list(nlp().pipe(texts, batch_size=64))


def is_word(token) -> bool:
    if token.is_punct or token.is_space:
        return False
    if token.text.startswith("'") or token.lower_ == "n't":
        return False
    return any(character.isalnum() for character in token.text)


def words_of(doc) -> list:
    return [token for token in doc if is_word(token)]


def lemma(token) -> str:
    return token.lemma_.lower()


def in_word_set(token, word_set: set[str]) -> bool:
    """The word or its lemma is in the set (lexicon look-up)."""
    return token.lower_ in word_set or lemma(token) in word_set


def per_100_words(count: float, n_words: int) -> float:
    if n_words == 0:
        return np.nan
    return 100.0 * count / n_words


def share(count: float, total: int) -> float:
    if total == 0:
        return np.nan
    return count / total


def sentence_texts(doc) -> list[str]:
    """The sentences as text, empty ones left out."""
    texts = [sentence.text.strip() for sentence in doc.sents]
    return [text for text in texts if text]


def sentences_of_all_texts(docs: list) -> tuple[list[str], np.ndarray]:
    """
    Every sentence of every text in one list, and which text each sentence belongs to.

    The transformer models run over one long list of sentences (fast, in batches);
    the owner array is used afterwards to give each text its own sentences back.
    """
    per_text = [sentence_texts(doc) for doc in docs]
    all_sentences = [sentence for sentences in per_text for sentence in sentences]
    sentences_per_text = [len(sentences) for sentences in per_text]
    owner = np.repeat(np.arange(len(docs)), sentences_per_text)
    return all_sentences, owner


def sentence_rows_of_each_text(owner: np.ndarray, n_texts: int) -> list[np.ndarray]:
    """Positions of each text's sentences in the list from sentences_of_all_texts."""
    return [np.where(owner == i)[0] for i in range(n_texts)]
