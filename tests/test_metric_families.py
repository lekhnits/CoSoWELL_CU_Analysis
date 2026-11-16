"""The metric families on small hand-made texts (no release data needed)."""

import math

import pytest

from cosowell_language.metrics.all_language_metrics import FOR_EVERY_NARRATIVE
from cosowell_language.metrics.first_person_pronoun_metrics import first_person_pronoun_metrics
from cosowell_language.metrics.narrative_gaze_metrics import narrative_gaze_metrics
from cosowell_language.metrics.text_parsing import parse_texts, words_of
from cosowell_language.metrics.tools.cookie_theft_content_units import content_unit_mentions


@pytest.fixture(scope="module")
def we_text():
    pytest.importorskip("spacy")
    return parse_texts(["We went to the park. I liked it and we had lunch."])[0]


def test_every_metric_name_is_unique_across_families(we_text):
    names = [name for family in FOR_EVERY_NARRATIVE for name in family(we_text)]
    assert len(names) == len(set(names))


def test_first_person_rates(we_text):
    rates = first_person_pronoun_metrics(we_text)
    assert len(words_of(we_text)) == 12
    assert math.isclose(rates["fpp_plural_rate"], 100 * 2 / 12)
    assert math.isclose(rates["fpp_singular_rate"], 100 * 1 / 12)


def test_she_refers_back_to_the_mother():
    mentions = content_unit_mentions("The mother is drying a plate. She does not see the water.")
    assert [m.label for m in mentions][:3] == ["woman", "woman_drying_washing_plates", "plate"]
    assert any(m.method == "reference" and m.label == "woman" for m in mentions)


def test_gaze_of_a_short_description():
    gaze = narrative_gaze_metrics("The boy takes a cookie. The water overflows in the sink.")
    assert gaze["gaze_n_fixations"] >= 3
    assert gaze["gaze_n_saccades"] == gaze["gaze_n_fixations"] - 1
    assert 0 <= gaze["gaze_cross_quadrant_prop"] <= 1
