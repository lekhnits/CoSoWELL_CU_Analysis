"""
Is the CoSoWELL release 2 data read correctly?

Counts, the participant / session / narrative structure, and how narratives
match survey sessions. The expected numbers are facts of release 2; a failing
test means the files or the reading code changed.
"""

import pandas as pd
import pytest

from cosowell_language.config import CORPUS_PATH, KEYS, NARRATIVE_TYPES, SURVEY_PATH
from cosowell_language.data_loading import NarrativeRecord, SurveyRecord, columns_of, read_table
from cosowell_language.data_overview import (PUBLISHED_CORPUS_BY_SESSION, PUBLISHED_CORPUS_TOTAL, SESSIONS,
                                             data_by_session, people_in_each_pair_of_sessions,
                                             retention_between_sessions, session_dates)

from conftest import keys, needs_data

pytestmark = needs_data

SURVEY_RECORDS, SURVEY_PEOPLE = 2621, 2054
SURVEY_RECORDS_PER_SESSION = {"t1": 455, "t2": 256, "t3": 412, "t4": 165, "t5": 172, "t6": 238, "t7": 803, "t8": 120}
SESSIONS_WITH_TEXTS, PEOPLE_WITH_TEXTS = 2626, 1356
SESSIONS_SUBMITTED_MORE_THAN_ONCE = 125
SESSIONS_WITH_TEXTS_AND_SURVEY = 1437

# Per session in the matched corpus file: token rows, submitted texts (doc_ids) and people who wrote.
CORPUS_TOKENS_PER_SESSION = {"t1": 158656, "t2": 326118, "t3": 258395, "t4": 228185,
                             "t5": 257565, "t6": 197069, "t7": 216246, "t8": 207201}
CORPUS_TEXTS_PER_SESSION = {"t1": 832, "t2": 1727, "t3": 1612, "t4": 1504,
                            "t5": 1644, "t6": 1216, "t7": 1300, "t8": 1224}
WRITERS_PER_SESSION = {"t1": 208, "t2": 317, "t3": 403, "t4": 376, "t5": 389, "t6": 303, "t7": 324, "t8": 306}
ANALYSED_PER_SESSION = {"t1": 208, "t2": 165, "t3": 358, "t4": 144, "t5": 74, "t6": 103, "t7": 305, "t8": 80}
ANALYSED_PEOPLE_AT_TWO_SESSIONS = {("t3", "t7"): 99, ("t6", "t7"): 42, ("t4", "t7"): 18, ("t5", "t7"): 14}


def test_survey_file_has_every_survey_record_column():
    header = read_table(SURVEY_PATH, nrows=0).columns
    assert set(columns_of(SurveyRecord)) <= set(header)


def test_corpus_file_has_every_narrative_record_column():
    header = read_table(CORPUS_PATH, nrows=0).columns
    assert set(columns_of(NarrativeRecord)) <= set(header)


def test_loaders_return_only_the_interface_columns(survey, narratives):
    assert list(survey.columns) == columns_of(SurveyRecord)
    assert list(narratives.columns) == ["id", "test_session", "ntype", "doc_id", "narrative"]


def test_survey_counts(survey):
    assert len(survey) == SURVEY_RECORDS
    assert survey["id"].nunique() == SURVEY_PEOPLE
    assert survey["test_session"].value_counts().to_dict() == SURVEY_RECORDS_PER_SESSION


def test_survey_has_one_record_per_session(survey):
    raw = read_table(SURVEY_PATH, usecols=KEYS)
    assert not raw.duplicated(KEYS).any()


def test_only_t1_is_before_covid(survey):
    phase_of_session = survey.groupby("test_session")["phase"].unique().map(list).to_dict()
    assert phase_of_session.pop("t1") == ["pre_cov"]
    assert all(phases == ["cov"] for phases in phase_of_session.values())


@pytest.mark.parametrize("column, low, high", [
    ("loneliness", 3, 9), ("health", 1, 4), ("companion", 1, 3), ("leftout", 1, 3), ("isolated", 1, 3),
    ("social_meet_freq", 1, 7), ("talk_social_network", 1, 9), ("divulge_family", 1, 4), ("age", 55, 84),
])
def test_survey_answers_are_on_their_scale(survey, column, low, high):
    values = pd.to_numeric(survey[column], errors="coerce").dropna()
    assert values.between(low, high).all()


def test_loneliness_is_the_sum_of_its_three_items(survey):
    items = survey[["companion", "leftout", "isolated"]].apply(pd.to_numeric)
    complete = items.notna().all(axis=1) & survey["loneliness"].notna()
    assert (items[complete].sum(axis=1) == survey.loc[complete, "loneliness"]).all()


def test_every_session_has_one_narrative_of_each_type(narratives):
    types_per_session = narratives.groupby(KEYS)["ntype"].apply(sorted)
    assert len(types_per_session) == SESSIONS_WITH_TEXTS
    assert (types_per_session.map(tuple) == tuple(sorted(NARRATIVE_TYPES))).all()


def test_narrative_counts(narratives):
    assert len(narratives) == 4 * SESSIONS_WITH_TEXTS
    assert narratives["id"].nunique() == PEOPLE_WITH_TEXTS
    assert set(narratives["test_session"]) == set(SURVEY_RECORDS_PER_SESSION)
    assert narratives["narrative"].str.len().gt(0).all()


def test_sessions_submitted_more_than_once(corpus_rows):
    """Some sessions were submitted two or three times, each time with all four texts."""
    submissions = corpus_rows.drop_duplicates("doc_id").groupby(KEYS + ["type"]).size()
    repeated = submissions[submissions > 1].reset_index()
    assert repeated[KEYS].drop_duplicates().shape[0] == SESSIONS_SUBMITTED_MORE_THAN_ONCE
    assert submissions.max() == 3


def test_the_first_submission_is_used(corpus_rows, narratives):
    first_doc = corpus_rows.drop_duplicates(KEYS + ["type"]).set_index(KEYS + ["type"])["doc_id"]
    used_doc = narratives.set_index(KEYS + ["ntype"])["doc_id"]
    assert (used_doc.sort_index().to_numpy() == first_doc.sort_index().to_numpy()).all()


def test_each_text_is_the_same_on_every_token_row_of_its_submission(corpus_rows):
    assert (corpus_rows.groupby("doc_id")["narrative"].nunique(dropna=False) == 1).all()


def test_each_submission_belongs_to_one_session_and_type(corpus_rows):
    assert (corpus_rows.groupby("doc_id")[KEYS + ["type"]].nunique() == 1).all().all()


def test_every_writer_answered_the_survey(narratives, survey):
    assert set(narratives["id"]) <= set(survey["id"])


def test_sessions_with_both_texts_and_survey(narratives, survey):
    assert len(keys(narratives) & keys(survey)) == SESSIONS_WITH_TEXTS_AND_SURVEY


def test_sessions_without_survey_belong_to_people_surveyed_at_another_session(narratives, survey):
    unmatched = keys(narratives) - keys(survey)
    surveyed_people = set(survey["id"])
    assert all(person in surveyed_people for person, _ in unmatched)


def test_published_table_rows_add_up_to_its_total():
    """Guards against typing errors in the published table copied into data_overview.py."""
    sums = PUBLISHED_CORPUS_BY_SESSION.drop(columns=["first_day", "last_day"]).sum().to_dict()
    expected = {name: value for name, value in PUBLISHED_CORPUS_TOTAL.items() if name != "different_people"}
    assert sums == expected


def test_published_sessions_are_in_order_and_do_not_overlap():
    table = PUBLISHED_CORPUS_BY_SESSION
    assert list(table.index) == SESSIONS
    first_day = pd.to_datetime(table["first_day"])
    last_day = pd.to_datetime(table["last_day"])
    assert (first_day <= last_day).all()
    assert (first_day.iloc[1:].to_numpy() > last_day.iloc[:-1].to_numpy()).all()


def test_corpus_counts_per_session(corpus_rows):
    assert corpus_rows.groupby("test_session").size().to_dict() == CORPUS_TOKENS_PER_SESSION
    assert corpus_rows.drop_duplicates("doc_id").groupby("test_session").size().to_dict() == CORPUS_TEXTS_PER_SESSION
    assert corpus_rows.groupby("test_session")["id"].nunique().to_dict() == WRITERS_PER_SESSION


@pytest.mark.parametrize("our_counts, published_column", [
    (CORPUS_TOKENS_PER_SESSION, "tokens"),
    (CORPUS_TEXTS_PER_SESSION, "narratives"),
    (WRITERS_PER_SESSION, "participants"),
])
def test_corpus_holds_at_least_the_published_data(our_counts, published_column):
    """
    The published counts are after the dataset authors' own exclusions, which the release files
    do not record, so the matched corpus has somewhat more in every session: about 10% more
    at t1 and t2, under 2% more at t3-t8.
    """
    published = PUBLISHED_CORPUS_BY_SESSION[published_column]
    ours = pd.Series(our_counts)
    assert (ours >= published).all()
    assert (ours / published).max() < 1.12


def test_data_by_session(narratives, survey):
    table = data_by_session(narratives, survey)
    assert table["people_with_texts"].to_dict() == WRITERS_PER_SESSION
    assert table["texts_and_survey"].to_dict() == ANALYSED_PER_SESSION
    assert table["survey_records"].to_dict() == SURVEY_RECORDS_PER_SESSION
    assert (table["texts_and_survey"] + table["texts_only"] == table["people_with_texts"]).all()
    assert (table["texts_and_survey"] + table["survey_only"] == table["survey_records"]).all()


def test_every_survey_record_has_age_and_gender(survey):
    assert survey["age"].notna().all()
    assert survey["gender"].notna().all()


def test_analysed_people_at_two_sessions_always_include_t7(narratives, survey):
    """In the analysed sample nobody took part more than twice, and every repeat includes t7 (H3)."""
    analysed = narratives[KEYS].drop_duplicates().merge(survey[KEYS], on=KEYS)
    in_both = people_in_each_pair_of_sessions(analysed)
    off_diagonal = {(first, second): in_both.loc[first, second]
                    for i, first in enumerate(SESSIONS) for second in SESSIONS[i + 1:]
                    if in_both.loc[first, second] > 0}
    assert off_diagonal == ANALYSED_PEOPLE_AT_TWO_SESSIONS
    assert analysed.groupby("id").size().max() == 2


def test_retention_flows_add_up(narratives):
    """Every writer of a session arrives one way and leaves one way; flows between sessions match."""
    retention = retention_between_sessions(narratives[KEYS].drop_duplicates())
    arrived = retention["continued"] + retention["returned"] + retention["new"]
    leaving = retention["to_next"] + retention["paused"] + retention["left_for_good"]
    assert (arrived == retention["present"]).all()
    assert (leaving.drop("t8") == retention["present"].drop("t8")).all()
    assert (retention["to_next"].iloc[:-1].to_numpy() == retention["continued"].iloc[1:].to_numpy()).all()
    assert retention["new"].sum() == PEOPLE_WITH_TEXTS
    assert retention["present"].to_dict() == WRITERS_PER_SESSION


def test_session_dates_are_read_from_the_published_table():
    assert session_dates() == {"t1": "Mar 2019", "t2": "Apr–Jun 2020", "t3": "Jun 2020", "t4": "Oct–Nov 2020",
                               "t5": "Jan–Feb 2021", "t6": "Mar–Apr 2021", "t7": "Aug 2021", "t8": "Mar 2022"}
