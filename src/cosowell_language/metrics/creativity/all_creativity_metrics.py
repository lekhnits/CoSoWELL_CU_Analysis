"""
Creativity metrics of the picture descriptions (results/features/cookie_creativity.csv), in seven families:

    originality_metrics              1 semantic originality and novelty
    divergent_thinking_metrics       2 divergent thinking
    discourse_organisation_metrics   3 discourse organisation and thematic development
    linguistic_richness_metrics      4 linguistic richness and complexity
    narrative_structure_metrics      5 narrative novelty and structure
    semantic_network_metrics         6 semantic network
    discourse_dynamics_metrics       7 discourse dynamics

Metrics that the language metrics already compute under the same name (ttr, idea_density,
local_coherence ...) are not repeated here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...config import KEYS
from ..language_resources import ensure_nltk_resources
from .creativity_settings import CACHE, RESOURCES
from .discourse_dynamics_metrics import discourse_dynamics_metrics
from .discourse_organisation_metrics import discourse_organisation_metrics
from .divergent_thinking_metrics import divergent_thinking_metrics
from .linguistic_richness_metrics import linguistic_richness_metrics
from .narrative_structure_metrics import narrative_structure_metrics
from .originality_metrics import originality_metrics
from .picture_corpus import build_picture_corpus
from .semantic_network_metrics import semantic_network_metrics

FAMILIES = [originality_metrics, divergent_thinking_metrics, discourse_organisation_metrics,
            linguistic_richness_metrics, narrative_structure_metrics, semantic_network_metrics,
            discourse_dynamics_metrics]


def measure_creativity(corpus_path: Path, run_models: bool = False,
                       extra_narratives: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    One row per picture narrative (>= MIN_WORDS words) with all creativity metrics. extra_narratives: new texts
    measured together with the corpus (see build_picture_corpus).
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    RESOURCES.mkdir(parents=True, exist_ok=True)
    ensure_nltk_resources()
    corpus = build_picture_corpus(corpus_path, run_models, extra_narratives)
    tables = [family(corpus) for family in FAMILIES]
    return pd.concat([corpus.narratives[KEYS], *tables], axis=1)
