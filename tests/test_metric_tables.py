"""Do the saved metric tables (results/features/) line up with the narratives and with the analysis sample?"""

import pandas as pd
import pytest

from cosowell_language.config import FEATURES_DIR, KEYS, MIN_WORDS, NARRATIVE_TYPES, RESULTS_DIR

from conftest import keys, needs_data

pytestmark = needs_data


def saved(name: str, columns=None) -> pd.DataFrame:
    return pd.read_csv(FEATURES_DIR / f"{name}.csv", usecols=columns)


@pytest.mark.parametrize("ntype", NARRATIVE_TYPES)
def test_one_metric_row_per_narrative(narratives, ntype):
    table = saved(ntype, KEYS + ["n_words"])
    assert not table.duplicated(KEYS).any()
    assert keys(table) == keys(narratives[narratives["ntype"].eq(ntype)])


def test_gaze_table_covers_every_picture_description(narratives):
    assert keys(saved("cookie_gaze", KEYS)) == keys(narratives[narratives["ntype"].eq("cookie")])


def test_creativity_table_covers_the_long_enough_picture_descriptions():
    cookie = saved("cookie", KEYS + ["n_words"])
    assert keys(saved("cookie_creativity", KEYS)) == keys(cookie[cookie["n_words"] >= MIN_WORDS])


def test_analysis_sample_matches_saved_sample():
    from cosowell_language.stats.analysis_data import load_analysis_data

    data = load_analysis_data(NARRATIVE_TYPES)
    sample = pd.read_csv(RESULTS_DIR / "all" / "sample.csv", index_col=0)["value"]
    assert len(data["session_scores"]) == sample["sessions"]
    assert data["session_scores"]["id"].nunique() == sample["participants"]
    assert len(data["scored_texts"]) == sample["narratives"]
    analysed = data["raw"].merge(data["scored_texts"][KEYS + ["ntype"]], on=KEYS + ["ntype"])
    assert len(analysed) == len(data["scored_texts"]) and (analysed["n_words"] >= MIN_WORDS).all()
