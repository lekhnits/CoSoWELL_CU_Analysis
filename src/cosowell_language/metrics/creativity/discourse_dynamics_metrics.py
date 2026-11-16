"""
Family 7 - discourse dynamics: how unexpected the sequence of sentence topics is
(leave-participant-out Markov chain over corpus-wide sentence clusters), exploration
of new topics, and the trend of originality over the text.

    transition surprisal   how unexpected each move from one sentence topic to the next is, given how often
                           everyone else makes that move
    novelty decay          slope of sentence originality over the text (negative = later sentences less original)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .creativity_settings import N_CORPUS_TOPICS, SEED
from .picture_corpus import PictureCorpus, sentences_of_each_narrative
from .vector_helpers import slope


def sentence_topic_sequences(corpus: PictureCorpus) -> list[np.ndarray]:
    """Corpus-wide topic (one of N_CORPUS_TOPICS sentence clusters) of each sentence, per narrative."""
    from sklearn.cluster import MiniBatchKMeans

    clustering = MiniBatchKMeans(N_CORPUS_TOPICS, random_state=SEED, n_init=5, batch_size=2048)
    topic_of_sentence = clustering.fit_predict(corpus.all_sentence_embeddings)
    return sentences_of_each_narrative(corpus, topic_of_sentence)


def transition_counts(corpus: PictureCorpus, sequences: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Topic-to-topic transitions counted over all narratives, and separately for each participant."""
    everyone = np.zeros((N_CORPUS_TOPICS, N_CORPUS_TOPICS))
    per_person = np.zeros((corpus.n_people, N_CORPUS_TOPICS, N_CORPUS_TOPICS))
    for narrative, sequence in enumerate(sequences):
        person = corpus.person[narrative]
        for topic, next_topic in zip(sequence, sequence[1:]):
            everyone[topic, next_topic] += 1
            per_person[person, topic, next_topic] += 1
    return everyone, per_person


def transition_surprisal(corpus: PictureCorpus, sequences: list[np.ndarray]) -> list[float]:
    """Mean surprisal (bits) of the narrative's topic transitions under everyone else's transitions."""
    everyone, per_person = transition_counts(corpus, sequences)
    surprisals = []
    for narrative, sequence in enumerate(sequences):
        if len(sequence) < 2:
            surprisals.append(np.nan)
            continue
        # The participant's own transitions are left out, otherwise a narrative would predict itself;
        # + 1 (Laplace smoothing) keeps transitions nobody else made from having probability zero.
        counts = everyone - per_person[corpus.person[narrative]] + 1
        probability_of_next = counts / counts.sum(1, keepdims=True)
        bits = [-np.log2(probability_of_next[topic, next_topic])
                for topic, next_topic in zip(sequence, sequence[1:])]
        surprisals.append(float(np.mean(bits)))
    return surprisals


def exploration_ratio(sequence: list) -> float:
    """Share of sentences (after the first) whose topic has not appeared before in the narrative."""
    if len(sequence) < 2:
        return np.nan
    is_new_topic = [sequence[i] not in sequence[:i] for i in range(1, len(sequence))]
    return float(np.mean(is_new_topic))


def discourse_dynamics_metrics(corpus: PictureCorpus) -> pd.DataFrame:
    sequences = sentence_topic_sequences(corpus)
    columns = {
        "transition_surprisal": transition_surprisal(corpus, sequences),
        "exploration_ratio": [exploration_ratio(list(sequence)) for sequence in sequences],
        "novelty_decay": [slope(originality) for originality in corpus.sentence_originality],
    }
    return pd.DataFrame(columns, index=corpus.narratives.index)
