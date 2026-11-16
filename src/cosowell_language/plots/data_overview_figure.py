"""
Data overview (double column), figure 0: one row per participant, one column per session,
coloured by what the person gave at that session (texts and survey, texts only, survey only).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from ..data_overview import DATA_AT_SESSION
from .figure_style import AQUA, BLUE, DOUBLE_COLUMN, INK, MID, YELLOW, mm, plt, save, sns

DATA_COLOURS = {"texts_and_survey": BLUE, "texts_only": AQUA, "survey_only": YELLOW, "absent": MID}
DATA_LABELS = {"texts_and_survey": "Texts and survey (analysed)", "texts_only": "Texts only",
               "survey_only": "Survey only"}


def draw_data_overview_figure(data_per_person: pd.DataFrame, dates: dict[str, str], path: Path) -> None:
    """data_per_person: person x session table from data_per_person_and_session, already sorted."""
    codes = data_per_person.apply(lambda column: column.map(DATA_AT_SESSION.index))
    colours = ListedColormap([DATA_COLOURS[kind] for kind in DATA_AT_SESSION])

    fig, ax = plt.subplots(figsize=mm(DOUBLE_COLUMN, 120))
    sns.heatmap(codes, ax=ax, cmap=colours, vmin=-.5, vmax=len(DATA_AT_SESSION) - .5, cbar=False,
                yticklabels=False, rasterized=True)

    sessions = list(data_per_person.columns)
    ax.set_xticks([position + .5 for position in range(len(sessions))],
                  [f"{session}\n{dates[session]}" for session in sessions], rotation=0)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("")
    ax.set_ylabel(f"Participants (n = {len(data_per_person):,}), sorted by the sessions they attended")
    label_first_session_groups(ax, data_per_person)
    label_people_per_session(ax, data_per_person)
    # Thin white lines between sessions keep neighbouring columns apart.
    for position in range(1, len(sessions)):
        ax.axvline(position, color="white", lw=.6)

    ax.legend(handles=[Patch(color=DATA_COLOURS[kind], lw=0, label=label) for kind, label in DATA_LABELS.items()],
              loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, handlelength=1, columnspacing=1.2)
    save(fig, path)


def label_first_session_groups(ax, data_per_person: pd.DataFrame) -> None:
    """
    A bracket right of the heatmap for each group of people who started at the same session, with the
    number of people in it. The rows are sorted by attendance, so each group is one run of rows.
    """
    first_session = data_per_person.ne("absent").idxmax(axis=1).to_numpy()
    # The bracket sits just outside the axes: x in axes units, y in data units (rows).
    transform = ax.get_yaxis_transform()
    gap = len(first_session) * .004          # space between neighbouring brackets, in rows
    start = 0
    for row in range(1, len(first_session) + 1):
        if row < len(first_session) and first_session[row] == first_session[start]:
            continue
        top, bottom = start + gap, row - gap
        ax.plot([1.012, 1.02, 1.02, 1.012], [top, top, bottom, bottom], color=INK, lw=.5,
                transform=transform, clip_on=False)
        ax.text(1.026, (start + row) / 2, f"{row - start:,}", ha="left", va="center", fontsize=6,
                color=INK, transform=transform)
        start = row


def label_people_per_session(ax, data_per_person: pd.DataFrame) -> None:
    """Number of people present at each session (any data), above its column."""
    present = data_per_person.ne("absent").sum()
    for position, people in enumerate(present):
        ax.text(position + .5, 1.01, f"{people:,}", ha="center", va="bottom", fontsize=6, color=INK,
                transform=ax.get_xaxis_transform())
