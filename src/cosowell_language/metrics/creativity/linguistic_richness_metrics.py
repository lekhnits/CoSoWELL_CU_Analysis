"""
Family 4 - linguistic richness and complexity: lexical diversity (HD-D), word length,
concreteness (Brysbaert et al., 2014), syntax, metaphors (FrameBERT) and lexical surprisal (distilgpt2).

    subordination rate         subordinate clauses (adverbial, complement, relative ...) per sentence
    mean dependency distance   how far apart, in words, a word and the word it depends on are
    metaphor diversity         number of different words used as metaphors
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .creativity_settings import CONCRETENESS_URL, LM_MODEL, RESOURCES
from .model_output_cache import cached_per_text
from .picture_corpus import PictureCorpus, per_100_framebert_words

SUBORDINATE_DEPS = {"advcl", "ccomp", "relcl", "acl", "csubj", "xcomp"}
HDD_DRAWS = 42           # the standard sample size of HD-D (McCarthy & Jarvis, 2007)


def concreteness_norms() -> dict:
    path = RESOURCES / "brysbaert_concreteness.txt"
    if not path.exists():
        import urllib.request
        urllib.request.urlretrieve(CONCRETENESS_URL, path)
    norms = pd.read_csv(path, sep="\t")
    return norms.set_index("Word")["Conc.M"].to_dict()


def hd_d(text: str) -> float:
    """Hypergeometric distribution diversity (lexicalrichness), NaN when the text is too short."""
    from lexicalrichness import LexicalRichness

    richness = LexicalRichness(text)
    try:
        return richness.hdd(draws=min(HDD_DRAWS, richness.words))
    except (ValueError, ZeroDivisionError):
        return np.nan


def lm_surprisal(texts: list[str], device) -> list[float]:
    """Mean surprisal (bits per token) of each text under distilgpt2: how unexpected each next word is."""
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(LM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LM_MODEL).to(device).eval()
    surprisals = []
    for text in tqdm(texts, desc="distilgpt2 surprisal"):
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(device)
        with torch.no_grad():
            # The prediction at position t is for token t + 1, so the last prediction has no target.
            log_probabilities = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
        next_tokens = ids[0, 1:, None]
        log_probability_of_next = log_probabilities.gather(1, next_tokens).squeeze(1)
        bits = -log_probability_of_next / math.log(2)
        surprisals.append(float(bits.mean()))
    return surprisals


def mean_word_length(content: list) -> float:
    """Mean number of characters of the content words."""
    if not content:
        return np.nan
    return np.mean([len(word.text) for word in content])


def mean_concreteness(content: list, norms: dict) -> float:
    if not content:
        return np.nan
    return np.nanmean([norms.get(token.lemma_.lower(), np.nan) for token in content])


def subordinate_clauses_per_sentence(doc, sentences: list) -> float:
    n_subordinate_clauses = sum(token.dep_ in SUBORDINATE_DEPS for token in doc)
    return n_subordinate_clauses / len(sentences)


def different_metaphor_words(framebert_words: list) -> int:
    return len({word.text.lower() for word in framebert_words if word.is_metaphor})


def mean_dependency_distance(words: list) -> float:
    """Mean distance in words between a word and its head (the root is left out)."""
    return np.mean([abs(token.i - token.head.i) for token in words if token.head.i != token.i])


def linguistic_richness_metrics(corpus: PictureCorpus) -> pd.DataFrame:
    norms = concreteness_norms()
    missing = corpus.framebert_missing

    def unless_framebert_missing(values: list) -> list:
        return [np.nan if is_missing else value for value, is_missing in zip(values, missing)]

    surprisal = cached_per_text("lm_surprisal", corpus.texts, lambda texts: lm_surprisal(texts, corpus.device),
                                may_compute=corpus.run_models)
    columns = {
        "hdd": [hd_d(text) for text in corpus.texts],
        "mean_word_length": [mean_word_length(content) for content in corpus.content_words],
        "concreteness": [mean_concreteness(content, norms) for content in corpus.content_words],
        "subordination_rate": [subordinate_clauses_per_sentence(doc, sentences)
                               for doc, sentences in zip(corpus.docs, corpus.sentence_spans)],
        "mean_dependency_distance": [mean_dependency_distance(words) for words in corpus.words],
        "metaphor_rate": per_100_framebert_words(corpus, lambda word: word.is_metaphor),
        "metaphor_diversity": unless_framebert_missing([different_metaphor_words(words)
                                                        for words in corpus.framebert_words]),
        "novel_metaphor_rate": per_100_framebert_words(corpus, lambda word: word.is_novel_metaphor),
        "lexical_surprisal": np.array(surprisal, dtype=float),
    }
    return pd.DataFrame(columns, index=corpus.narratives.index)
