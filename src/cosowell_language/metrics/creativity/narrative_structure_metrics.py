"""
Family 5 - narrative novelty and structure: picture coverage, going beyond the picture,
order originality, themes, events, causal links (FrameNet), tense shifts, characters,
hedges, emotional arc and structural outliers.

    event                     a (verb, object) pair, e.g. (take, cookie); "rare" events are used by few people
    theme                     an LDA topic that makes up at least THEME_SHARE of the narrative
    emotional arc             the VADER compound score of each sentence in turn
    structure originality     how unusual the narrative's overall shape is (the STRUCTURE metrics together)
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from ..tools.cookie_theft_content_units import CIU_LABELS
from ..tools.pretrained_text_models import hedge_word_counts
from .divergent_thinking_metrics import MENTAL_SUPERSENSES
from .model_output_cache import cached_per_text
from .picture_corpus import (PictureCorpus, mean_rarity, people_using_each_item, per_100_framebert_words,
                             share_of_rare_items, supersense)
from .vector_helpers import unit_norm

CAUSAL_FRAME_NAMES = {"Causation", "Reason", "Explaining_the_facts", "Purpose", "Contingency"}
THEME_SHARE = 0.15       # an LDA topic counts as a theme of the narrative from this share on
LOF_NEIGHBOURS = 20
# Metrics describing the shape of a narrative; structure_originality is how unusual their combination is.
STRUCTURE = ["beyond_picture_prop", "unit_order_originality", "past_tense_prop", "first_person_narration",
             "n_characters", "causal_link_rate", "temporal_shift_rate", "question_prop", "arc_range",
             "arc_turns", "n_themes"]


def share_outside_picture(sentences: list, mentions: list) -> float:
    """Share of sentences that contain no picture content unit."""
    mention_starts = [mention.start for mention in mentions]

    def mentions_the_picture(sentence) -> bool:
        return any(sentence.start_char <= start < sentence.end_char for start in mention_starts)

    return np.mean([not mentions_the_picture(sentence) for sentence in sentences])


def mentions_per_10_words(mentions: list, n_words: int) -> float:
    """Picture anchoring: content-unit mentions per 10 words."""
    return 10 * len(mentions) / n_words


def n_characters(doc) -> int:
    """Different people: named persons and nouns whose first WordNet sense is a person."""
    people = {entity.text.lower() for entity in doc.ents if entity.label_ == "PERSON"}
    people |= {token.lemma_.lower() for token in doc if token.pos_ == "NOUN" and supersense(token) == "noun.person"}
    return len(people)


def events(doc) -> list[tuple[str, str]]:
    """(verb lemma, object lemma) pairs; the object is empty for verbs without one."""
    pairs = []
    for token in doc:
        if token.pos_ != "VERB":
            continue
        objects = [child.lemma_.lower() for child in token.children if child.dep_ in {"dobj", "obj", "attr"}]
        pairs.append((token.lemma_.lower(), objects[0] if objects else ""))
    return pairs


def n_arc_turns(valence: np.ndarray) -> int:
    """How often the emotional tone changes direction (up to down or down to up) over the sentences."""
    direction = np.sign(np.diff(valence))
    direction = direction[direction != 0]
    if len(direction) < 2:
        return 0
    return int((direction[1:] != direction[:-1]).sum())


def share_of_tense_shifts(finite_verbs: list) -> float:
    """Share of consecutive finite verbs whose tense differs from the verb before."""
    tenses = [verb.morph.get("Tense")[0] for verb in finite_verbs if verb.morph.get("Tense")]
    if len(tenses) < 2:
        return np.nan
    return np.mean([tense != next_tense for tense, next_tense in zip(tenses, tenses[1:])])


def share_of_past_tense(finite_verbs: list) -> float:
    if not finite_verbs:
        return np.nan
    return np.mean([verb.morph.get("Tense") == ["Past"] for verb in finite_verbs])


def share_of_first_person_subjects(doc) -> float:
    """First-person narration: share of grammatical subjects that are first person (0 when there is no subject)."""
    subjects = [token for token in doc if token.dep_ in {"nsubj", "nsubjpass"}]
    if not subjects:
        return 0
    return np.mean([subject.morph.get("Person") == ["1"] for subject in subjects])


def time_expressions_per_100_words(doc, words: list) -> float:
    """Dates and times named in the text ("yesterday", "at five o'clock") per 100 words."""
    n_time_expressions = sum(entity.label_ in {"DATE", "TIME"} for entity in doc.ents)
    return 100 * n_time_expressions / len(words)


def mental_state_verbs_per_100_words(words: list) -> float:
    """Verbs of thinking, feeling and perceiving per 100 words."""
    n_mental_verbs = sum(word.pos_ == "VERB" and supersense(word) in MENTAL_SUPERSENSES for word in words)
    return 100 * n_mental_verbs / len(words)


def share_of_questions(sentences: list[str]) -> float:
    return np.mean([sentence.endswith("?") for sentence in sentences])


def unit_order_originality(unit_orders: list[list[str]]) -> list[float]:
    """
    For each narrative: for every pair of its content units, the share of all narratives mentioning that pair in
    the opposite order, averaged over the pairs. High = the narrative goes through the picture in an unusual order.
    """
    times_before = Counter(pair for order in unit_orders for pair in combinations(order, 2))

    def originality(order: list[str]) -> float:
        pairs = list(combinations(order, 2))
        if not pairs:
            return np.nan
        opposite_share = []
        for first, second in pairs:
            n_same_order, n_opposite_order = times_before[(first, second)], times_before[(second, first)]
            opposite_share.append(n_opposite_order / (n_same_order + n_opposite_order))
        return float(np.mean(opposite_share))

    return [originality(order) for order in unit_orders]


def valence_arcs(sentences: list[list[str]]) -> list[np.ndarray]:
    """VADER compound score of each sentence, per narrative."""
    from nltk.sentiment import SentimentIntensityAnalyzer

    vader = SentimentIntensityAnalyzer()
    return [np.array([vader.polarity_scores(sentence)["compound"] for sentence in text]) for text in sentences]


def hedge_rate(corpus: PictureCorpus) -> pd.Series:
    """Hedge words (BERT hedge tagger, cached per sentence) per 100 words."""
    hedges = cached_per_text("hedges", corpus.all_sentences, lambda sentences: list(hedge_word_counts(sentences)),
                             may_compute=corpus.run_models)
    hedges = np.asarray(hedges, dtype=float)
    hedges_per_narrative = np.bincount(corpus.sentence_owner, weights=hedges, minlength=corpus.n_narratives)
    return 100 * hedges_per_narrative / corpus.narratives["n_words"]


def theme_originality(topic_mix: np.ndarray) -> np.ndarray:
    """Cosine distance of each narrative's LDA topic mix from the average topic mix of all narratives."""
    average_mix = topic_mix.mean(0)
    return 1 - unit_norm(topic_mix) @ unit_norm(average_mix)


def structure_originality(corpus: PictureCorpus, columns: dict) -> np.ndarray:
    """Local outlier factor of the narrative among all narratives, on the STRUCTURE metrics."""
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    shape = pd.DataFrame({name: columns[name] for name in STRUCTURE}, index=corpus.narratives.index)
    shape = shape.fillna(shape.median()).assign(log_sentences=np.log(corpus.n_sentences))
    standardised = StandardScaler().fit_transform(shape)
    return -LocalOutlierFactor(n_neighbors=LOF_NEIGHBOURS).fit(standardised).negative_outlier_factor_


def narrative_structure_metrics(corpus: PictureCorpus) -> pd.DataFrame:
    # Each narrative's content units in the order they are first mentioned.
    unit_orders = [list(dict.fromkeys(mention.label for mention in mentions)) for mentions in corpus.content_units]
    n_units = [len(order) for order in unit_orders]
    event_sets = [set(events(doc)) for doc in corpus.docs]
    people_using_event = people_using_each_item(corpus, event_sets)
    arcs = valence_arcs(corpus.sentences)
    finite_verbs = [[token for token in doc if token.morph.get("VerbForm") == ["Fin"]] for doc in corpus.docs]
    causal_frames = {frame_id for frame_id, name in enumerate(corpus.frame_names)
                     if name.startswith("Cause") or name in CAUSAL_FRAME_NAMES}

    columns = {
        "picture_unit_coverage": pd.Series(n_units, index=corpus.narratives.index) / len(CIU_LABELS),
        "beyond_picture_prop": [share_outside_picture(sentences, mentions)
                                for sentences, mentions in zip(corpus.sentence_spans, corpus.content_units)],
        "picture_anchoring": [mentions_per_10_words(mentions, n_words)
                              for mentions, n_words in zip(corpus.content_units, corpus.narratives["n_words"])],
        "unit_order_originality": unit_order_originality(unit_orders),
        "n_themes": (corpus.topic_mix >= THEME_SHARE).sum(1),
        "theme_originality": theme_originality(corpus.topic_mix),
        "n_events": [len(event_set) for event_set in event_sets],
        "causal_link_rate": per_100_framebert_words(corpus, lambda word: word.frame_id in causal_frames),
        "temporal_shift_rate": [share_of_tense_shifts(verbs) for verbs in finite_verbs],
        "temporal_rate": [time_expressions_per_100_words(doc, words) for doc, words in zip(corpus.docs, corpus.words)],
        "n_characters": [n_characters(doc) for doc in corpus.docs],
        "mental_state_rate": [mental_state_verbs_per_100_words(words) for words in corpus.words],
        "hedge_rate": hedge_rate(corpus),
        "question_prop": [share_of_questions(sentences) for sentences in corpus.sentences],
        "past_tense_prop": [share_of_past_tense(verbs) for verbs in finite_verbs],
        "first_person_narration": [share_of_first_person_subjects(doc) for doc in corpus.docs],
        "event_rarity": [mean_rarity(event_set, people_using_event, corpus.n_people) for event_set in event_sets],
        "rare_event_prop": [share_of_rare_items(event_set, people_using_event, corpus.n_people)
                            for event_set in event_sets],
        "arc_range": [arc.max() - arc.min() for arc in arcs],
        "arc_turns": [n_arc_turns(arc) for arc in arcs],
    }
    columns["structure_originality"] = structure_originality(corpus, columns)
    return pd.DataFrame(columns, index=corpus.narratives.index)
