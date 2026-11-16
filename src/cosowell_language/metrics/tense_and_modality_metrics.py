"""Tense of the finite verbs, modal verbs and negation (spaCy morphology and tags)."""

from __future__ import annotations

from .cognitive_process_metrics import is_negation
from .text_parsing import lemma, per_100_words, share, words_of


def is_modal(token) -> bool:
    return token.tag_ == "MD"


def tense_and_modality_metrics(doc) -> dict:
    words = words_of(doc)
    finite_verbs = [word for word in words if "Fin" in word.morph.get("VerbForm")]
    n_past = sum("Past" in verb.morph.get("Tense") for verb in finite_verbs)
    n_present = sum("Pres" in verb.morph.get("Tense") for verb in finite_verbs)
    n_modals = sum(is_modal(word) for word in words)
    n_will = sum(is_modal(word) and lemma(word) in {"will", "shall"} for word in words)
    # Negation is counted over every token, as in the differentiation rate (cognitive_process_metrics).
    n_negations = sum(is_negation(token) for token in doc)
    return {
        "past_tense_share": share(n_past, len(finite_verbs)),
        "present_tense_share": share(n_present, len(finite_verbs)),
        "modal_rate": per_100_words(n_modals, len(words)),
        "future_will_rate": per_100_words(n_will, len(words)),
        "negation_rate": per_100_words(n_negations, len(words)),
    }
