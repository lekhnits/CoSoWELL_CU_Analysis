"""
"Narrative gaze" (picture task): scanpath metrics from the order of the Cookie Theft content units.

These are not eye-tracking data. The ordered content-unit (CU) mentions of a
picture description are treated as a scanpath over the picture:

    AOI          one of the 23 CUs, anchored at its (x, y) position on the
                 1838 x 1300 px reference image (approximate, set by hand)
    fixation     a CU mention plus the following words until a *different* CU is
                 mentioned; consecutive mentions of the same CU are merged.
                 Duration = number of words in the segment.
    saccade      a transition between two different CUs; amplitude = distance
                 between the CU anchors, as a share of the image diagonal
    regression   a saccade to a CU that has already been fixated
    latency      words written before the first CU
    dwell        words spent on the CUs (all fixation durations together)
    entropy      stationary: how evenly the dwell spreads over the CUs;
                 transition: how unpredictable the next CU is from the current one
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .tools.cookie_theft_content_units import CIU_LABELS, content_unit_mentions

WORD_RE = re.compile(r"\b[\w']+\b")  # same tokenizer as the content-unit matching, so word positions line up
IMAGE_WIDTH, IMAGE_HEIGHT = 1838, 1300
IMAGE_DIAGONAL = math.hypot(IMAGE_WIDTH, IMAGE_HEIGHT)

# Approximate anchor of each CU on the reference picture, as a share of width and height.
CIU_IMAGE_ANCHORS: dict[str, tuple[float, float]] = {
    "boy": (0.32, 0.39), "girl": (0.14, 0.62), "woman": (0.61, 0.54), "kitchen": (0.51, 0.54),
    "outside": (0.84, 0.34), "cookie": (0.22, 0.13), "jar": (0.22, 0.16), "stool": (0.26, 0.72),
    "sink": (0.76, 0.70), "plate": (0.69, 0.45), "dishcloth": (0.69, 0.55), "water": (0.72, 0.78),
    "window": (0.81, 0.27), "cupboard": (0.28, 0.23), "dishes": (0.91, 0.66), "curtains": (0.80, 0.22),
    "boy_taking_stealing": (0.30, 0.19), "boy_or_stool_falling": (0.27, 0.63),
    "woman_drying_washing_plates": (0.66, 0.48), "water_overflowing": (0.72, 0.77),
    "action_performed_by_girl": (0.18, 0.47), "woman_unconcerned_by_overflowing": (0.61, 0.35),
    "woman_indifferent_to_children": (0.56, 0.35),
}
# The quarter of the picture each CU is assigned to: 1 top left, 2 bottom left, 3 top right, 4 bottom right.
QUADRANT = {
    **dict.fromkeys(["boy", "action_performed_by_girl", "cupboard", "boy_taking_stealing", "cookie", "jar"], 1),
    **dict.fromkeys(["girl", "stool", "boy_or_stool_falling"], 2),
    **dict.fromkeys(["woman_unconcerned_by_overflowing", "woman_indifferent_to_children", "woman", "dishcloth",
                     "plate", "woman_drying_washing_plates", "kitchen", "outside", "window", "curtains"], 3),
    **dict.fromkeys(["sink", "dishes", "water", "water_overflowing"], 4),
}
# A CU is "social" (a person) when its label contains a person word.
PERSON_UNITS = {label for label in CIU_LABELS if set(label.split("_")) & {"boy", "girl", "woman"}}


def fixation_at(label: str, duration: int) -> dict:
    x, y = CIU_IMAGE_ANCHORS[label]
    return {
        "cu": label,
        "x": x * IMAGE_WIDTH,
        "y": y * IMAGE_HEIGHT,
        "quadrant": QUADRANT[label],
        "social": label in PERSON_UNITS,
        "duration": duration,
    }


def fixations_of(mentions: list, n_words: int) -> list[dict]:
    """Merge consecutive mentions of the same unit; a fixation lasts until the next different unit."""
    fixations: list[dict] = []
    following_mentions = [*mentions[1:], None]
    for mention, following in zip(mentions, following_mentions):
        next_start = following.token_index if following else n_words
        n_words_on_unit = max(next_start - mention.token_index, 1)
        if fixations and fixations[-1]["cu"] == mention.label:
            fixations[-1]["duration"] += n_words_on_unit
        else:
            fixations.append(fixation_at(mention.label, n_words_on_unit))
    return fixations


def mark_revisits(fixations: list[dict]) -> list[dict]:
    """A fixation is a revisit when its CU was already fixated earlier."""
    seen: set[str] = set()
    for fixation in fixations:
        fixation["revisit"] = fixation["cu"] in seen
        seen.add(fixation["cu"])
    return fixations


def entropy(counts) -> float:
    """Shannon entropy in bits of a set of counts."""
    counts = np.asarray(list(counts), dtype=float)
    if counts.sum():
        counts = counts[counts > 0]
    if len(counts) == 0:
        return np.nan
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def fixation_metrics(fixations: list[dict], latency: int, n_words: int) -> dict:
    durations = np.array([f["duration"] for f in fixations], dtype=float)
    total_dwell = durations.sum() if len(durations) else 0.0
    has_any = len(durations) > 0
    return {
        "gaze_n_fixations": len(fixations),
        "gaze_fix_dur_mean": durations.mean() if has_any else np.nan,
        "gaze_fix_dur_median": float(np.median(durations)) if has_any else np.nan,
        "gaze_fix_dur_sd": durations.std() if len(durations) > 1 else np.nan,
        "gaze_fix_dur_max": durations.max() if has_any else np.nan,
        "gaze_total_dwell_words": total_dwell,
        "gaze_dwell_prop": total_dwell / n_words if n_words else np.nan,
        "gaze_latency_words": latency,
        "gaze_latency_prop": latency / n_words if n_words else np.nan,
    }


def saccade_metrics(fixations: list[dict], n_words: int) -> dict:
    """Saccades are the moves between consecutive fixations; amplitudes are shares of the image diagonal."""
    saccades = list(zip(fixations, fixations[1:]))
    n = len(saccades)
    dx = np.array([to["x"] - start["x"] for start, to in saccades], dtype=float)
    dy = np.array([to["y"] - start["y"] for start, to in saccades], dtype=float)
    amplitudes = np.array([math.hypot(x, y) for x, y in zip(dx, dy)], dtype=float)
    crosses_quadrant = [start["quadrant"] != to["quadrant"] for start, to in saccades]
    revisits = [f["revisit"] for f in fixations[1:]]
    return {
        "gaze_n_saccades": n,
        "gaze_saccade_rate": 100 * n / n_words if n_words else np.nan,
        "gaze_saccade_amp_mean": amplitudes.mean() / IMAGE_DIAGONAL if n else np.nan,
        "gaze_saccade_amp_sd": amplitudes.std() / IMAGE_DIAGONAL if n > 1 else np.nan,
        "gaze_scanpath_length": amplitudes.sum() / IMAGE_DIAGONAL,
        # A saccade is horizontal when it moves at least as far sideways as up or down.
        "gaze_horizontal_prop": float(np.mean(np.abs(dx) >= np.abs(dy))) if n else np.nan,
        "gaze_rightward_prop": float(np.mean(dx > 0)) if n else np.nan,
        "gaze_cross_quadrant_prop": float(np.mean(crosses_quadrant)) if n else np.nan,
        "gaze_regression_prop": float(np.mean(revisits)) if n else np.nan,
    }


def transition_entropy(fixations: list[dict]) -> float:
    """Entropy of the next unit given the current one (Krejtz et al., 2015), weighted by how often each unit is left."""
    saccades = list(zip(fixations, fixations[1:]))
    if not saccades:
        return np.nan
    next_units: dict[str, Counter] = {}
    for start, to in saccades:
        next_units.setdefault(start["cu"], Counter())[to["cu"]] += 1
    weighted = [sum(counts.values()) / len(saccades) * entropy(counts.values()) for counts in next_units.values()]
    return sum(weighted)


def spread_metrics(fixations: list[dict]) -> dict:
    """How the description spreads over the picture: coverage, entropy, attention to the people."""
    dwell = Counter()
    for fixation in fixations:
        dwell[fixation["cu"]] += fixation["duration"]
    durations = np.array([f["duration"] for f in fixations], dtype=float)
    total_dwell = durations.sum() if len(durations) else 0.0
    social_dwell = sum(f["duration"] for f in fixations if f["social"])
    return {
        "gaze_coverage": len(dwell) / len(CIU_LABELS),
        "gaze_stationary_entropy": entropy(dwell.values()),
        "gaze_transition_entropy": transition_entropy(fixations),
        "gaze_social_dwell_prop": social_dwell / total_dwell if total_dwell else np.nan,
        "gaze_first_fix_social": float(fixations[0]["social"]) if fixations else np.nan,
    }


def narrative_gaze_metrics(text: str) -> dict:
    """Every gaze metric of one picture description (raw text: no spaCy parse needed)."""
    n_words = len(WORD_RE.findall(text))
    mentions = [m for m in content_unit_mentions(text) if m.token_index is not None]
    latency = mentions[0].token_index if mentions else n_words
    fixations = mark_revisits(fixations_of(mentions, n_words))
    return {
        **fixation_metrics(fixations, latency, n_words),
        **saccade_metrics(fixations, n_words),
        **spread_metrics(fixations),
    }
