"""
Metrics of new picture descriptions (Cookie Theft task) that are not in the CoSoWELL corpus, measured with the same
functions as the corpus, so the columns match results/features/cookie.csv, cookie_gaze.csv and
cookie_creativity.csv.

The language and gaze metrics are measured one text at a time. The creativity metrics compare a text with the
picture descriptions of every other writer (rarity, originality), so the new texts are measured together with the
whole corpus; this takes several minutes, and the transformer models run on the new texts.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from ..config import CORPUS_PATH, KEYS, MIN_WORDS
from .all_language_metrics import measure_gaze, measure_language

# Session label of the new texts, so their keys never clash with the corpus's sessions (t1 ... t6).
NEW_SESSION = "new"


def read_new_texts(path: Path) -> pd.DataFrame:
    """
    New texts from a CSV with a 'narrative' column and, optionally, an 'id' column (one name per writer or text).
    Without ids, the texts are numbered new_1, new_2 ...
    """
    texts = pd.read_csv(path)
    return new_texts_table(texts["narrative"].tolist(), texts["id"].tolist() if "id" in texts else None)


def new_texts_table(narratives: list[str], ids: list[str] | None = None) -> pd.DataFrame:
    """The texts in the corpus's format: id, test_session, narrative. Empty texts are dropped."""
    ids = ids or [f"new_{i + 1}" for i in range(len(narratives))]
    texts = pd.DataFrame({"id": [str(i) for i in ids], "test_session": NEW_SESSION,
                          "narrative": [str(text).strip() for text in narratives]})
    if texts["id"].duplicated().any():
        raise ValueError("every new text needs its own id")
    return texts[texts["narrative"].str.len().gt(0)].reset_index(drop=True)


def creativity_of_new_texts(texts: pd.DataFrame) -> pd.DataFrame:
    """Creativity metrics of the new texts, measured together with the corpus; texts under MIN_WORDS words get none."""
    # One thread per library, as in the pipeline: the models run text by text and thread pools slow each other.
    # These must be set before torch is loaded, which is why the import is here.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(variable, "1")
    from .creativity.all_creativity_metrics import measure_creativity

    everyone = measure_creativity(CORPUS_PATH, run_models=True, extra_narratives=texts)
    return everyone[everyone["test_session"].eq(NEW_SESSION)]


def measure_new_picture_descriptions(texts: pd.DataFrame, with_creativity: bool = True) -> pd.DataFrame:
    """
    One row per new text: its words, then the language, gaze and (optionally) creativity metrics, with the same
    column names as results/features/.
    """
    language = measure_language(texts, "cookie")
    gaze = measure_gaze(texts)
    metrics = texts.merge(language, on=KEYS).merge(gaze, on=KEYS)
    if with_creativity:
        metrics = metrics.merge(creativity_of_new_texts(texts), on=KEYS, how="left")
    too_short = metrics["n_words"] < MIN_WORDS
    if too_short.any():
        print(f"{too_short.sum()} text(s) under {MIN_WORDS} words: the corpus analysis leaves such texts out, and "
              "they have no creativity metrics")
    return metrics
