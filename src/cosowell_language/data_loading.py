"""
The CoSoWELL release 2 data this analysis uses, and how it is read.

Two files, each described by a dataclass that lists every column the code uses
(no other column is read):

    NarrativeRecord   data/cosowell_corpus_matched_v2.txt    one typed narrative
    SurveyRecord      data/cosowell_survey_v2_numericVar.csv one survey session

A participant is `id`; a session is (`id`, `test_session`), t1 ... t8. A
session has four narratives, one of each type, and at most one survey record.

The corpus file is token-level: one row per token, with the full narrative
repeated on every row. 125 sessions (most at t2) were submitted two or three
times, each time with all four narratives (different `doc_id`). The first
submission in file order is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import pandas as pd

from .config import CORPUS_PATH, KEYS, NARRATIVE_TYPES, SURVEY_PATH


def described(text: str):
    return field(default=None, metadata={"description": text})


@dataclass(frozen=True)
class NarrativeRecord:
    id: str = described("participant (anonymised hash)")
    test_session: str = described("session, t1 ... t8")
    type: str = described("narrative type: cookie (picture description), past, yesterday, future")
    doc_id: str = described("one submitted text; a session submitted twice has two doc_ids per type")
    narrative: str = described("the full typed text")


@dataclass(frozen=True)
class SurveyRecord:
    # session and background
    id: str = described("participant")
    test_session: str = described("session, t1 ... t8")
    phase: str = described("pre_cov (t1) or cov (t2-t8, after 13 March 2020)")
    gender: str = described("female, male, other, prefer_not_to_say")
    age: float = described("years, 55-84")
    years_education: float = described("years of education (values > 30 are treated as missing)")
    education: str = described("highest level of education, e.g. high school graduate, bachelor's degree")
    employed: str = described("employment: full, part, no")
    retired: str = described("yes, no")
    country: str = described("usa, canada")
    residency: str = described("private_apartment_or_home, residential_care_or_home, nursing_home")
    health: float = described("self-rated health, 1 = poor ... 4 = excellent")
    # objective social isolation
    living_arrangements: str = described("alone, spouse_partner, else")
    num_friends: float = described("number of friends")
    num_household: float = described("close people in the household")
    social_meet_freq: float = described("social get-togethers, 1 = never ... 7 = several times a week")
    professional_meet_freq: float = described("organised group meetings, 1-7")
    volunteer_meet_freq: float = described("volunteering, 1-7")
    talk_social_network: float = described("talks to close network, 1 = never ... 9 = every day")
    # social support
    divulge_family: float = described("can open up to family, 1 = never ... 4 = often")
    rely_family: float = described("can rely on family, 1-4")
    divulge_friend: float = described("can open up to friends, 1-4")
    rely_friend: float = described("can rely on friends, 1-4")
    # felt loneliness (3-item scale, Hughes et al., 2004)
    loneliness: float = described("sum of the three items, 3-9")
    companion: float = described("lack companionship, 1 = hardly ever ... 3 = often")
    leftout: float = described("feel left out, 1-3")
    isolated: float = described("feel isolated from others, 1-3")
    # t7-t8 only
    well_being_score: float = described("Flourishing Scale sum, 8-56")
    cognitive_reappraisal: float = described("Emotion Regulation Questionnaire facet mean, 1-7")
    expressive_suppression: float = described("Emotion Regulation Questionnaire facet mean, 1-7")


def columns_of(record) -> list[str]:
    return [f.name for f in fields(record)]


def describe(record) -> pd.DataFrame:
    """The columns of a record and what they mean, e.g. describe(SurveyRecord)."""
    return pd.DataFrame([(f.name, f.type, f.metadata["description"]) for f in fields(record)],
                        columns=["column", "type", "description"])


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read a .txt (tab-separated) or .csv release file, dropping repeated header rows."""
    path = Path(path)
    separator = "\t" if path.suffix.lower() == ".txt" else ","
    df = pd.read_csv(path, sep=separator, **kwargs)
    # The release files repeat their header line inside the data; such rows equal the column names.
    header_row = pd.Series(True, index=df.index)
    for col in df.columns:
        header_row &= df[col].astype(str).eq(str(col))
    return df.loc[~header_row].reset_index(drop=True)


def read_corpus_rows(corpus_path: Path = CORPUS_PATH) -> pd.DataFrame:
    """Token-level corpus rows, only the NarrativeRecord columns."""
    return read_table(corpus_path, usecols=columns_of(NarrativeRecord))


def one_row_per_narrative(rows: pd.DataFrame) -> pd.DataFrame:
    """First submission of each session and type (first in file order), with the text stripped."""
    narratives = rows.drop_duplicates(KEYS + ["type"]).rename(columns={"type": "ntype"})
    return narratives.assign(narrative=narratives["narrative"].fillna("").astype(str).str.strip())


def load_narratives(corpus_path: Path = CORPUS_PATH, types=NARRATIVE_TYPES) -> pd.DataFrame:
    """One row per narrative: id, test_session, ntype, doc_id, narrative (non-empty, of the given types)."""
    narratives = one_row_per_narrative(read_corpus_rows(corpus_path))
    keep = narratives["ntype"].isin(types) & narratives["narrative"].str.len().gt(0)
    return narratives.loc[keep, KEYS + ["ntype", "doc_id", "narrative"]].reset_index(drop=True)


def load_survey(survey_path: Path = SURVEY_PATH) -> pd.DataFrame:
    """Survey responses, one row per session, only the SurveyRecord columns."""
    columns = columns_of(SurveyRecord)
    survey = read_table(survey_path, usecols=columns)[columns]
    return survey.drop_duplicates(KEYS).reset_index(drop=True)
