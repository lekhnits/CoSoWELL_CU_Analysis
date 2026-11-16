"""
Family 2 - divergent thinking.

    fluency         how many ideas: idea units (clause heads), distinct ideas (clusters of near-identical
                    sentences), different picture content units
    flexibility     how many different kinds of meaning: FrameNet frames (FrameBERT), WordNet supersenses
                    (noun.person, verb.motion ...), Empath categories
    elaboration     how much detail per idea: modifiers and words per idea unit; sentences per concept
    perspectives    how many different people think, feel, perceive or speak in the text
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .picture_corpus import PictureCorpus, supersense
from .vector_helpers import entropy

MODIFIER_DEPS = {"amod", "advmod", "nummod", "npadvmod", "prep", "relcl", "acl", "appos", "compound"}
MENTAL_SUPERSENSES = {"verb.cognition", "verb.emotion", "verb.perception"}


def perspective_holders(doc) -> int:
    """Number of different subjects of thinking, feeling, perceiving or speaking verbs."""
    holders = set()
    for token in doc:
        is_mental_or_speech_verb = (token.pos_ == "VERB"
                                    and supersense(token) in MENTAL_SUPERSENSES | {"verb.communication"})
        if not is_mental_or_speech_verb:
            continue
        for child in token.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                holders.add(child.lower_ if child.pos_ == "PRON" else child.lemma_.lower())
    return len(holders)


def frames_evoked(framebert_tokens: list) -> Counter:
    """How often each FrameNet frame is evoked in the narrative; frame id 0 means "no frame"."""
    return Counter(token.frame_id for token in framebert_tokens if token.frame_id != 0)


def n_empath_categories(texts: list[str]) -> list[int]:
    """Number of different Empath categories each text touches."""
    from empath import Empath

    empath = Empath()
    return [sum(count > 0 for count in empath.analyze(text, normalize=False).values()) for text in texts]


def divergent_thinking_metrics(corpus: PictureCorpus) -> pd.DataFrame:
    frames = [frames_evoked(tokens) for tokens in corpus.framebert_tokens]
    frames_missing = corpus.framebert_missing
    idea_units = corpus.n_idea_units.clip(lower=1)          # avoids dividing by zero
    n_modifiers = [sum(token.dep_ in MODIFIER_DEPS for token in doc) for doc in corpus.docs]
    n_concepts = np.array([len(set(labels)) for labels in corpus.concept_labels])

    columns = {
        "fluency_idea_units": corpus.n_idea_units,
        "fluency_distinct_ideas": [len(set(labels)) for labels in corpus.idea_labels],
        "fluency_content_units": [len({mention.label for mention in mentions}) for mentions in corpus.content_units],
        "flexibility_frames": [np.nan if missing else len(frame_counts)
                               for frame_counts, missing in zip(frames, frames_missing)],
        "flexibility_frame_entropy": [np.nan if missing else entropy(frame_counts.values())
                                      for frame_counts, missing in zip(frames, frames_missing)],
        "flexibility_supersenses": [len(senses) for senses in corpus.supersenses],
        "flexibility_empath": n_empath_categories(corpus.texts),
        "elaboration_modifiers_per_idea": n_modifiers / idea_units,
        "elaboration_words_per_idea": corpus.narratives["n_words"] / idea_units,
        # Sentences per concept cluster: how long the narrative stays with one concept.
        "idea_expansion": corpus.n_sentences / n_concepts,
        "perspective_diversity": [perspective_holders(doc) for doc in corpus.docs],
    }
    return pd.DataFrame(columns, index=corpus.narratives.index)
