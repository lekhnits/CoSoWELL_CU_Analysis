"""Order of a person's sessions (t1 ... t8)."""

from __future__ import annotations

import pandas as pd


def session_number(test_session: pd.Series) -> pd.Series:
    """'t3' -> 3, for ordering a person's sessions."""
    return test_session.str[1:].astype(int)


def first_session_of_each_person(d: pd.DataFrame) -> pd.Series:
    """True for the rows of each person's earliest session."""
    order = session_number(d["test_session"])
    earliest = order.groupby(d["id"]).transform("min")
    return order == earliest
