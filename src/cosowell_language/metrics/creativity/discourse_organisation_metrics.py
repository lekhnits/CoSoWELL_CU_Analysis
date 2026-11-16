"""
Family 3 - discourse organisation and thematic development: coherence, task relevance,
semantic spread, topic switching and recurrence, LDA topic entropy.

Sentences are compared by their embeddings (cosine). A "concept" is a cluster of the narrative's sentences
about the same thing (picture_corpus.add_sentence_clusters); a "topic switch" is a move to another concept.

    large jump           a step between consecutive sentences larger than LARGE_JUMP_PCTL % of all such steps
    semantic entropy     how evenly the content words spread over WordNet supersenses (noun.person, verb.motion ...)
    topic entropy        how evenly the narrative spreads over the corpus-wide LDA topics
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .creativity_settings import COOKIE_PICTURE, LARGE_JUMP_PCTL
from .model_output_cache import cached_per_text
from .picture_corpus import PictureCorpus, encode_sentences
from .vector_helpers import entropy, mean_pairwise_distance, slope

PERSISTENCE_LAG = 3      # sentences at least this far apart count as "later in the text"


def share_of_switches(labels: list) -> float:
    """Share of consecutive sentences that move to another concept cluster."""
    switches = [labels[i] != labels[i - 1] for i in range(1, len(labels))]
    return np.array(switches).mean()


def share_of_recurrences(labels: list) -> float:
    """Share of consecutive sentences that switch back to a cluster visited earlier."""
    recurrences = [labels[i] != labels[i - 1] and labels[i] in labels[:i - 1] for i in range(1, len(labels))]
    return np.array(recurrences).mean()


def semantic_persistence(embeddings: np.ndarray) -> float:
    """Mean similarity of sentences at least PERSISTENCE_LAG apart: do later sentences return to earlier ideas?"""
    similarity = embeddings @ embeddings.T
    n = len(embeddings)
    pairs = [similarity[i, j] for i in range(n) for j in range(i + PERSISTENCE_LAG, n)]
    return float(np.mean(pairs)) if pairs else np.nan


def thematic_progression(embeddings: np.ndarray) -> float:
    """Trend of the distance from the first sentence over the text: does the narrative move away from its start?"""
    distance_from_first = 1 - embeddings @ embeddings[0]
    return slope(distance_from_first)


def distances_to_next(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance between each sentence and the next (embeddings have unit length)."""
    return 1 - (embeddings[1:] * embeddings[:-1]).sum(1)


def task_relevance(corpus: PictureCorpus) -> list[float]:
    """Mean similarity of the sentences to a reference description of the picture."""
    picture = cached_per_text("sentence_embedding", [COOKIE_PICTURE],
                              lambda sentences: encode_sentences(sentences, corpus.device), may_compute=True)[0]
    picture = np.asarray(picture)
    return [float((embeddings @ picture).mean()) for embeddings in corpus.sentence_embeddings]


def discourse_organisation_metrics(corpus: PictureCorpus) -> pd.DataFrame:
    steps = [distances_to_next(embeddings) for embeddings in corpus.sentence_embeddings]
    large_jump_cut = np.percentile(np.concatenate(steps), LARGE_JUMP_PCTL)
    concept_labels = [list(labels) for labels in corpus.concept_labels]

    columns = {
        # How unevenly the narrative moves from sentence to sentence: SD of the consecutive-sentence distances.
        "local_coherence_sd": [step.std() if len(step) > 1 else np.nan for step in steps],
        "task_relevance": task_relevance(corpus),
        "semantic_spread": [mean_pairwise_distance(embeddings) for embeddings in corpus.sentence_embeddings],
        "large_jump_prop": [(step > large_jump_cut).mean() if len(step) else np.nan for step in steps],
        "conceptual_breadth": [len(set(labels)) for labels in concept_labels],
        "semantic_entropy": [entropy(senses.values()) for senses in corpus.supersenses],
        "topic_switch_rate": [share_of_switches(labels) if len(labels) > 1 else np.nan for labels in concept_labels],
        "topic_recurrence": [share_of_recurrences(labels) if len(labels) > 2 else np.nan
                             for labels in concept_labels],
        "semantic_persistence": [semantic_persistence(embeddings) for embeddings in corpus.sentence_embeddings],
        "thematic_progression": [thematic_progression(embeddings) for embeddings in corpus.sentence_embeddings],
        "topic_entropy": [entropy(mix) for mix in corpus.topic_mix],
    }
    return pd.DataFrame(columns, index=corpus.narratives.index)
