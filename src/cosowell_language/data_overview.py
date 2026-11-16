"""
What data each session has: texts, survey records, demographics, and who took part in which sessions.

A person can write the four narratives at a session without a survey record for that same
session (their survey answers come from another session), and can answer the survey without
writing. The tables here show, session by session, how many people have which data, and how
many people of the analysed sample (texts and survey at the same session) took part in
several sessions.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from .config import KEYS, MIN_WORDS, NARRATIVE_TYPES

SESSIONS = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]

# The corpus as described by the dataset authors (CoSoWELL release 2), per test session.
# These counts are after the authors' own exclusions, which the release files do not record,
# so they are smaller than what the matched corpus file contains (tests/test_data.py checks this).
PUBLISHED_CORPUS_BY_SESSION = pd.DataFrame(
    [
        ("t1", "2019-03-01", "2019-03-30", 146_458, 9_007, 7_419, 760, 8_435, 190),
        ("t2", "2020-04-08", "2020-06-16", 295_785, 12_681, 10_493, 1_559, 16_777, 284),
        ("t3", "2020-06-17", "2020-06-30", 254_477, 12_387, 10_240, 1_600, 14_282, 400),
        ("t4", "2020-10-14", "2020-11-05", 226_113, 11_411, 9_479, 1_492, 12_340, 373),
        ("t5", "2021-01-12", "2021-02-15", 256_402, 12_357, 10_231, 1_636, 14_055, 387),
        ("t6", "2021-03-29", "2021-04-17", 195_918, 11_924, 9_165, 1_208, 10_919, 301),
        ("t7", "2021-08-05", "2021-08-18", 215_650, 12_809, 9_656, 1_292, 11_933, 322),
        ("t8", "2022-03-15", "2022-03-30", 206_875, 12_176, 9_500, 1_220, 11_114, 305),
    ],
    columns=["test_session", "first_day", "last_day", "tokens", "types", "lemmas",
             "narratives", "sentences", "participants"],
).set_index("test_session")

# The "Total" row of the published table; 1,311 is the number of different people.
PUBLISHED_CORPUS_TOTAL = {"tokens": 1_797_678, "types": 94_752, "lemmas": 76_183, "narratives": 10_767,
                          "sentences": 99_855, "participants": 2_562, "different_people": 1_311}


def session_dates() -> dict[str, str]:
    """When each session ran, from the published table, e.g. {"t2": "Apr–Jun 2020"}."""
    dates = {}
    for session, row in PUBLISHED_CORPUS_BY_SESSION.iterrows():
        first_day = pd.Timestamp(row["first_day"])
        last_day = pd.Timestamp(row["last_day"])
        if first_day.month == last_day.month:
            dates[session] = first_day.strftime("%b %Y")
        else:
            dates[session] = f"{first_day:%b}–{last_day:%b %Y}"
    return dates


def read_word_counts(features_dir) -> pd.DataFrame:
    """Words per narrative (id, test_session, ntype, n_words), from the saved metric tables in results/features/."""
    tables = []
    for ntype in NARRATIVE_TYPES:
        words = pd.read_csv(features_dir / f"{ntype}.csv", usecols=KEYS + ["n_words"])
        tables.append(words.assign(ntype=ntype))
    return pd.concat(tables, ignore_index=True)


def data_kinds_by_session(word_counts: pd.DataFrame, survey: pd.DataFrame) -> pd.DataFrame:
    """
    People per session with each kind of data: a survey record, a usable narrative of each type
    (at least MIN_WORDS words, as in the analysis), and texts and survey at the same session.
    Rows are the kinds of data, columns the sessions.
    """
    usable = word_counts[word_counts["n_words"] >= MIN_WORDS]
    rows = {"survey": survey.groupby("test_session").size()}
    for ntype in NARRATIVE_TYPES:
        rows[ntype] = usable[usable["ntype"] == ntype].groupby("test_session").size()

    writers = usable[KEYS].drop_duplicates()
    analysed = writers.merge(survey[KEYS], on=KEYS)
    rows["texts_and_survey"] = analysed.groupby("test_session").size()
    return pd.DataFrame(rows).T.reindex(columns=SESSIONS).fillna(0).astype(int)


def missing_years_of_education(survey: pd.DataFrame) -> pd.Series:
    """Missing or implausible (> 30 years, treated as missing in the analysis) years of education."""
    years = pd.to_numeric(survey["years_education"], errors="coerce")
    return years.isna() | years.gt(30)


def data_by_session(narratives: pd.DataFrame, survey: pd.DataFrame) -> pd.DataFrame:
    """
    For each session: people with a survey record, people who wrote the narratives, people with
    both (the analysed sample), texts only, survey only, and survey records missing a demographic.
    """
    writers = narratives[KEYS].drop_duplicates().assign(texts=True)
    answered = survey[KEYS].assign(survey=True)
    people = writers.merge(answered, on=KEYS, how="outer")
    # After the outer merge, a person missing from one side has NaN there, i.e. no such data.
    people["texts"] = people["texts"].eq(True)
    people["survey"] = people["survey"].eq(True)

    counts = pd.DataFrame({
        "survey_records": people.groupby("test_session")["survey"].sum(),
        "people_with_texts": people.groupby("test_session")["texts"].sum(),
        "texts_and_survey": (people["texts"] & people["survey"]).groupby(people["test_session"]).sum(),
        "texts_only": (people["texts"] & ~people["survey"]).groupby(people["test_session"]).sum(),
        "survey_only": (~people["texts"] & people["survey"]).groupby(people["test_session"]).sum(),
    })

    missing = pd.DataFrame({
        "age": pd.to_numeric(survey["age"], errors="coerce").isna(),
        "gender": survey["gender"].isna(),
        "years_education": missing_years_of_education(survey),
        "health": pd.to_numeric(survey["health"], errors="coerce").isna(),
    })
    missing_by_session = missing.groupby(survey["test_session"]).sum().add_prefix("missing_")
    return counts.join(missing_by_session).reindex(SESSIONS).astype(int)


def sessions_of_each_person(sessions: pd.DataFrame) -> pd.Series:
    """The sessions each person took part in, e.g. ("t2", "t4"), in session order."""
    return sessions.groupby("id")["test_session"].apply(lambda s: tuple(sorted(set(s), key=SESSIONS.index)))


def people_in_each_pair_of_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Session x session table: the number of people present at both sessions (the diagonal is
    the number present at that session). A person present at t2, t4 and t5 counts in the
    t2-t4, t2-t5 and t4-t5 cells.
    """
    table = pd.DataFrame(0, index=SESSIONS, columns=SESSIONS)
    for person_sessions in sessions_of_each_person(sessions):
        for session in person_sessions:
            table.loc[session, session] += 1
        for first, second in combinations(person_sessions, 2):
            table.loc[first, second] += 1
            table.loc[second, first] += 1
    return table


def retention_between_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """
    For each session, where its people come from and where they go next (the flows of the retention plot).

    Coming in: continued from the previous session, returned after missing at least one session,
    or new (first session). Going out: continue to the next session, pause (miss the next session
    but come back later), or leave for good (no later session). The last session has no out-flows.
    """
    person_sessions = sessions_of_each_person(sessions)
    rows = []
    for position, session in enumerate(SESSIONS):
        previous = SESSIONS[position - 1] if position > 0 else None
        following = SESSIONS[position + 1] if position + 1 < len(SESSIONS) else None
        later = set(SESSIONS[position + 1:])
        counts = dict(test_session=session, present=0, continued=0, returned=0, new=0,
                      to_next=0, paused=0, left_for_good=0)

        for attended in person_sessions:
            if session not in attended:
                continue
            counts["present"] += 1
            earlier = [s for s in attended if SESSIONS.index(s) < position]
            if previous in attended:
                counts["continued"] += 1
            elif earlier:
                counts["returned"] += 1
            else:
                counts["new"] += 1

            if following is None:
                continue
            if following in attended:
                counts["to_next"] += 1
            elif later & set(attended):
                counts["paused"] += 1
            else:
                counts["left_for_good"] += 1
        rows.append(counts)
    return pd.DataFrame(rows).set_index("test_session")


DATA_AT_SESSION = ["texts_and_survey", "texts_only", "survey_only", "absent"]


def data_per_person_and_session(writers: pd.DataFrame, survey: pd.DataFrame) -> pd.DataFrame:
    """
    Person x session table of what each person gave at each session: texts and survey, texts only,
    survey only, or absent (one of DATA_AT_SESSION). Everyone with texts or a survey record at any session.

    Rows are sorted by attendance: people present at t1 first, and among them those also present at
    t2 first, and so on, so each group of people starting at the same session forms one block; within
    the same attendance, by the data given at each session.
    """
    has_texts = writers[KEYS].drop_duplicates().assign(texts=True)
    has_survey = survey[KEYS].drop_duplicates().assign(survey=True)
    people = has_texts.merge(has_survey, on=KEYS, how="outer")
    people["texts"] = people["texts"].eq(True)
    people["survey"] = people["survey"].eq(True)
    people["data"] = "survey_only"
    people.loc[people["texts"] & ~people["survey"], "data"] = "texts_only"
    people.loc[people["texts"] & people["survey"], "data"] = "texts_and_survey"

    table = people.pivot(index="id", columns="test_session", values="data").reindex(columns=SESSIONS)
    table = table.fillna("absent")
    # Present sessions sort first (absent is last in DATA_AT_SESSION), session by session from t1;
    # within the same attendance, people with the same data at each session end up next to each other.
    attended = table.ne("absent").astype(int).add_prefix("present_")
    data_codes = table.apply(lambda column: column.map(DATA_AT_SESSION.index)).add_prefix("data_")
    sort_keys = list(attended.columns) + list(data_codes.columns)
    order = attended.join(data_codes).sort_values(sort_keys, ascending=[False] * len(SESSIONS) + [True] * len(SESSIONS))
    return table.loc[order.index]


def people_by_combination_of_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """How many people took part in exactly each combination of sessions (e.g. 't2 + t4'), most common first."""
    combination = sessions_of_each_person(sessions)
    counts = combination.map(" + ".join).value_counts().rename_axis("sessions").rename("people").reset_index()
    counts.insert(1, "number_of_sessions", counts["sessions"].str.count(r"\+") + 1)
    return counts


# Demographics table (report Table 4): survey column -> row label and its categories in display order.
DEMOGRAPHIC_CATEGORIES = {
    "gender": ("Sex", {"female": "Women", "male": "Men", "other": "Other", "prefer_not_to_say": "Not stated"}),
    "education": ("Highest education", {
        "some high school": "Some high school", "high school graduate": "High school graduate",
        "some college, no degree": "Some college, no degree", "associate's degree": "Associate's degree",
        "bachelor's degree": "Bachelor's degree", "master's degree": "Master's degree", "doctorate": "Doctorate"}),
    "employed": ("Employment", {"full": "Full time", "part": "Part time", "no": "Not employed"}),
    "retired": ("Retired", {"yes": "Yes", "no": "No"}),
    "country": ("Country", {"usa": "USA", "canada": "Canada"}),
    "living_arrangements": ("Lives with", {"spouse_partner": "Spouse or partner", "alone": "Alone",
                                           "else": "Others"}),
    "residency": ("Residence", {"private_apartment_or_home": "Private home or apartment",
                                "residential_care_or_home": "Residential care", "nursing_home": "Nursing home"}),
    "health": ("Self-rated health", {1: "Poor", 2: "Fair", 3: "Good", 4: "Excellent"}),
}
AGE_GROUPS = {"55–64": (55, 64), "65–74": (65, 74), "75 and older": (75, 200)}


def mean_sd(values: pd.Series, digits: int = 1) -> str:
    return f"{values.mean():.{digits}f} ({values.std():.{digits}f})"


def count_percent(mask: pd.Series) -> str:
    return f"{int(mask.sum()):,} ({mask.mean():.1%})"


def demographic_rows(people: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(characteristic, category, value) for one group of people, one row per person."""
    age = pd.to_numeric(people["age"], errors="coerce")
    education_years = pd.to_numeric(people["years_education"], errors="coerce")
    education_years = education_years.where(education_years <= 30)      # as in survey_scores
    loneliness = pd.to_numeric(people["loneliness"], errors="coerce")
    rows = [("People", "", f"{len(people):,}"),
            ("Age, years", "Mean (SD)", mean_sd(age)),
            ("Age, years", "Range", f"{age.min():.0f}–{age.max():.0f}")]
    for label, (low, high) in AGE_GROUPS.items():
        rows.append(("Age, years", label, count_percent(age.between(low, high))))
    rows.append(("Years of education", "Mean (SD)", mean_sd(education_years)))
    for column, (characteristic, categories) in DEMOGRAPHIC_CATEGORIES.items():
        answers = people[column]
        if column == "health":
            answers = pd.to_numeric(answers, errors="coerce")
        for code, name in categories.items():
            rows.append((characteristic, name, count_percent(answers.eq(code))))
        if answers.isna().any():
            rows.append((characteristic, "Missing", count_percent(answers.isna())))
    rows.append(("Loneliness score (3–9)", "Mean (SD)", mean_sd(loneliness)))
    return rows


def participant_demographics(survey: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Demographics of the analysed people at their first analysed session (sessions: KEYS of the analysed
    sessions), for everyone and for the people with two sessions.
    """
    analysed = survey.merge(sessions[KEYS].drop_duplicates(), on=KEYS)
    order = analysed["test_session"].str[1:].astype(int)
    first = analysed[order.eq(order.groupby(analysed["id"]).transform("min"))]
    two_sessions = first["id"].isin(analysed.loc[analysed["id"].duplicated(), "id"])
    columns = ["characteristic", "category"]
    everyone = pd.DataFrame(demographic_rows(first), columns=[*columns, "all_people"])
    repeaters = pd.DataFrame(demographic_rows(first[two_sessions]), columns=[*columns, "two_sessions"])
    # The people with two sessions are a subset, so a 'Missing' row of theirs always exists for everyone too;
    # where only everyone has one, theirs is 0.
    table = everyone.merge(repeaters, on=columns, how="left")
    return table.fillna({"two_sessions": "0 (0.0%)"})
