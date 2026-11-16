"""
Family 1 - semantic originality and novelty.

DSI (Johnson et al., 2023), forward flow (Gray et al., 2019), word-chain distance and
remote association (GloVe), word / concept rarity, and originality against the texts
of other participants.

    word rarity       how rare the words are in English in general (wordfreq)
    concept rarity    how few participants use the same content-word lemmas
    unique idea       a sentence more original than UNIQUE_IDEA_PCTL % of all sentences, where the originality of
                      a sentence is 1 - its similarity to the closest sentence written by anyone else
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .creativity_settings import DSI_LAYERS, DSI_MODEL, N_NEAREST_NARRATIVES, UNIQUE_IDEA_PCTL
from .model_output_cache import cached_per_text
from .picture_corpus import PictureCorpus, glove_vector, mean_rarity, share_of_rare_items
from .vector_helpers import mean_pairwise_distance, unit_norm

MAX_ZIPF = 7          # the most frequent English words have a Zipf frequency of about 7
RARE_WORD_ZIPF = 3    # rarer than about 1 per million words


def dsi_scores(corpus: PictureCorpus, texts: list[str]) -> list[float]:
    """Divergent semantic integration: mean pairwise distance of all in-context word embeddings
    (bert-large, layers 6 and 7, averaged over the two layers)."""
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModel, AutoTokenizer

    sentences_of = dict(zip(corpus.texts, corpus.sentences))
    tokenizer = AutoTokenizer.from_pretrained(DSI_MODEL)
    model = AutoModel.from_pretrained(DSI_MODEL, output_hidden_states=True).to(corpus.device).eval()
    scores = []
    for text in tqdm(texts, desc="DSI"):
        encoded = tokenizer(sentences_of[text], padding=True, truncation=True, max_length=512,
                            return_tensors="pt", return_special_tokens_mask=True)
        is_special = encoded.pop("special_tokens_mask").bool()
        keep = encoded["attention_mask"].bool() & ~is_special
        with torch.no_grad():
            hidden = model(**encoded.to(corpus.device)).hidden_states
        per_layer = []
        for layer in DSI_LAYERS:
            token_vectors = hidden[layer][keep.to(corpus.device)].float().cpu().numpy()
            per_layer.append(mean_pairwise_distance(token_vectors))
        scores.append(float(np.nanmean(per_layer)))
    return scores


def forward_flow(vectors: np.ndarray) -> float:
    """Mean cosine distance of each unit (word or sentence) to all earlier units: how far the text moves on."""
    if len(vectors) < 2:
        return np.nan
    distance = 1 - unit_norm(vectors) @ unit_norm(vectors).T
    distance_to_earlier = [distance[i, :i].mean() for i in range(1, len(vectors))]
    return float(np.mean(distance_to_earlier))


def mean_distance_to_next(vectors: np.ndarray) -> float:
    """Word-chain distance: mean cosine distance between each word and the next one."""
    if len(vectors) < 2:
        return np.nan
    vectors = unit_norm(vectors)
    similarity_to_next = (vectors[1:] * vectors[:-1]).sum(1)
    return float(np.mean(1 - similarity_to_next))


def remote_association(corpus: PictureCorpus, content: list) -> float:
    """Mean GloVe distance between content words linked by a dependency arc."""
    positions = {token.i for token in content}
    distances = []
    for token in content:
        linked_to_content_word = token.head.i in positions and token.head.i != token.i
        if not linked_to_content_word:
            continue
        vector, head_vector = glove_vector(corpus, token), glove_vector(corpus, token.head)
        if vector is None or head_vector is None:
            continue
        distances.append(1 - float(unit_norm(vector) @ unit_norm(head_vector)))
    return float(np.mean(distances)) if distances else np.nan


def word_frequencies(words: list) -> list[float]:
    from wordfreq import zipf_frequency

    return [zipf_frequency(token.lower_, "en") for token in words]


def glove_sequence(corpus: PictureCorpus, content: list) -> np.ndarray:
    """GloVe vector of each content word that has one, in text order."""
    vectors = [glove_vector(corpus, token) for token in content]
    vectors = [vector for vector in vectors if vector is not None]
    return np.vstack(vectors) if vectors else np.zeros((0, 300))


def mean_or_nan(values: list) -> float:
    return np.mean(values) if values else np.nan


def word_rarity(frequencies: list[float]) -> float:
    """Mean of MAX_ZIPF - Zipf frequency: 0 for the most frequent English words, larger for rarer ones."""
    return mean_or_nan([MAX_ZIPF - frequency for frequency in frequencies])


def share_of_rare_words(frequencies: list[float]) -> float:
    return mean_or_nan([frequency < RARE_WORD_ZIPF for frequency in frequencies])


def originality_knn(corpus: PictureCorpus) -> np.ndarray:
    """1 - mean similarity to the N_NEAREST_NARRATIVES most similar narratives of other participants."""
    similarity = corpus.narrative_embeddings @ corpus.narrative_embeddings.T
    same_person = corpus.person[:, None] == corpus.person[None, :]
    similarity[same_person] = -np.inf
    most_similar = np.sort(similarity, axis=1)[:, -N_NEAREST_NARRATIVES:]
    return 1 - most_similar.mean(1)


def originality_metrics(corpus: PictureCorpus) -> pd.DataFrame:
    glove_sequences = [glove_sequence(corpus, content) for content in corpus.content_words]
    frequencies = [word_frequencies(content) for content in corpus.content_words]
    unique_idea_cut = np.percentile(corpus.all_sentence_originality, UNIQUE_IDEA_PCTL)
    is_unique_idea = [originality > unique_idea_cut for originality in corpus.sentence_originality]
    typical_narrative = unit_norm(corpus.narrative_embeddings.mean(0))
    dsi = cached_per_text("dsi", corpus.texts, lambda texts: dsi_scores(corpus, texts),
                          may_compute=corpus.run_models)

    columns = {
        "dsi": np.array(dsi, dtype=float),
        "forward_flow_words": [forward_flow(word_vectors) for word_vectors in glove_sequences],
        "forward_flow_sentences": [forward_flow(embeddings) for embeddings in corpus.sentence_embeddings],
        "word_chain_distance": [mean_distance_to_next(word_vectors) for word_vectors in glove_sequences],
        "remote_association": [remote_association(corpus, content) for content in corpus.content_words],
        "word_rarity": [word_rarity(word_frequencies) for word_frequencies in frequencies],
        "rare_word_prop": [share_of_rare_words(word_frequencies) for word_frequencies in frequencies],
        "concept_rarity": [mean_rarity(lemmas, corpus.people_using_lemma, corpus.n_people)
                           for lemmas in corpus.lemma_sets],
        "rare_concept_prop": [share_of_rare_items(lemmas, corpus.people_using_lemma, corpus.n_people)
                              for lemmas in corpus.lemma_sets],
        "originality_knn": originality_knn(corpus),
        # Cosine distance of the narrative from the average narrative.
        "distance_from_typical": 1 - corpus.narrative_embeddings @ typical_narrative,
        "idea_originality": [originality.mean() for originality in corpus.sentence_originality],
        "n_unique_ideas": [int(unique.sum()) for unique in is_unique_idea],
        "unique_idea_prop": [unique.mean() for unique in is_unique_idea],
        # Originality of each sentence weighted by how well it fits its own narrative.
        "appropriate_novelty": [float((originality * fit).mean()) for originality, fit
                                in zip(corpus.sentence_originality, corpus.fit_to_own_narrative)],
    }
    return pd.DataFrame(columns, index=corpus.narratives.index)
