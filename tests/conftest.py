"""Shared fixtures: the release files are read once per test session. Tests skip when the data are absent."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cosowell_language.config import CORPUS_PATH, SURVEY_PATH  # noqa: E402
from cosowell_language.data_loading import load_narratives, load_survey, read_corpus_rows  # noqa: E402

needs_data = pytest.mark.skipif(not (CORPUS_PATH.exists() and SURVEY_PATH.exists()),
                                reason="CoSoWELL release files are not in data/")


@pytest.fixture(scope="session")
def corpus_rows() -> pd.DataFrame:
    """Token-level corpus rows (only the NarrativeRecord columns)."""
    return read_corpus_rows()


@pytest.fixture(scope="session")
def narratives() -> pd.DataFrame:
    return load_narratives()


@pytest.fixture(scope="session")
def survey() -> pd.DataFrame:
    return load_survey()


def keys(df: pd.DataFrame, columns=("id", "test_session")) -> set:
    return set(df[list(columns)].itertuples(index=False, name=None))
