"""Sentence length, clauses, parse-tree depth and sentence fragments (spaCy dependency parse)."""

from __future__ import annotations

import numpy as np

from .text_parsing import is_word

CLAUSE_DEPS = {"ROOT", "ccomp", "xcomp", "advcl", "relcl", "acl", "csubj", "csubjpass", "parataxis"}


def is_clause_head(token) -> bool:
    """A verb heading a clause, or a verb coordinated with another verb."""
    heads_a_clause = token.dep_ in CLAUSE_DEPS and token.pos_ in {"VERB", "AUX"}
    is_coordinated_verb = token.dep_ == "conj" and token.pos_ == "VERB"
    return heads_a_clause or is_coordinated_verb


def tree_depth(token) -> int:
    """Number of steps from the token up to the root of its sentence."""
    depth = 0
    # spaCy creates new Token objects on every access, so the root is found by comparing indices.
    while token.head.i != token.i:
        token = token.head
        depth += 1
    return depth


def parse_depth(sentence) -> int:
    return max((tree_depth(token) for token in sentence), default=0)


def is_fragment(sentence) -> bool:
    """No finite verb in the sentence."""
    return not any("Fin" in token.morph.get("VerbForm") for token in sentence)


def syntax_metrics(doc) -> dict:
    sentences = list(doc.sents)
    if not sentences:
        return dict.fromkeys(["mean_sentence_length", "clauses_per_sentence", "mean_parse_depth",
                              "max_parse_depth", "fragment_prop"], np.nan)
    words_per_sentence = [sum(is_word(token) for token in sentence) for sentence in sentences]
    clauses_per_sentence = [sum(is_clause_head(token) for token in sentence) for sentence in sentences]
    depths = [parse_depth(sentence) for sentence in sentences]
    return {
        "mean_sentence_length": float(np.mean(words_per_sentence)),
        "clauses_per_sentence": float(np.mean(clauses_per_sentence)),
        "mean_parse_depth": float(np.mean(depths)),
        "max_parse_depth": max(depths),
        "fragment_prop": float(np.mean([is_fragment(sentence) for sentence in sentences])),
    }
