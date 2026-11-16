"""
Cookie Theft content units (CUs): the dictionary and the matching.

The 23 standard content units of the Cookie Theft picture (Goodglass & Kaplan)
are matched with a keyword dictionary. Pronouns and nearby referring words
("she", "it", ...) are linked to the most recent compatible unit, so
"the mother ... she" counts as two mentions of the unit "woman".

    content_unit_mentions(text)   ordered mentions, repeats kept
    count_content_units(text)     (different units mentioned, all mentions)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

CIU_LABELS: list[str] = [
    "boy",
    "girl",
    "woman",
    "kitchen",
    "outside",
    "cookie",
    "jar",
    "stool",
    "sink",
    "plate",
    "dishcloth",
    "water",
    "window",
    "cupboard",
    "dishes",
    "curtains",
    "boy_taking_stealing",
    "boy_or_stool_falling",
    "woman_drying_washing_plates",
    "water_overflowing",
    "action_performed_by_girl",
    "woman_unconcerned_by_overflowing",
    "woman_indifferent_to_children",
]

COMMON_CIU_KEYWORDS: dict[str, set[str]] = {
    "boy": {
        "adolescent", "boy", "brother", "child", "guy", "he", "himself",
        "his", "kid", "lad", "little boy", "male", "man", "shirt", "shoe",
        "sock", "son", "young boy", "young man", "youth",
    },
    "girl": {
        "child", "daughter", "dress", "girl", "kid", "lass", "little girl",
        "sister", "skirt", "young girl",
    },
    "woman": {
        "adult", "apron", "domestic", "dress", "fifties", "heels",
        "high-heels", "homemaker", "housewife", "lady", "ladys", "mama",
        "mom", "mommy", "mother", "mum", "mummy", "wife", "woman",
    },
    "kitchen": {"counter", "culinary", "home", "interior", "kitchen"},
    "outside": {
        "background", "backyard", "bush", "day", "exterior", "flower",
        "garage", "garden", "grass", "grow", "lawn", "leaves", "out",
        "outdoor", "outdoors", "outside", "path", "pathway", "scenery",
        "shrub", "shrubbery", "sidewalk", "snow", "spring", "summer",
        "sunny", "tree", "walkway", "yard",
    },
    "cookie": {"biscuit", "chocolatechip", "cookie", "cookies", "pastry", "snack", "treat"},
    "jar": {"canister", "container", "cookie jar", "cookie tin", "holder", "jar", "lid", "pot", "tin", "vessel"},
    "stool": {
        "bench", "chair", "footstool", "furniture", "ladder", "perch",
        "step", "step ladder", "step stool", "stepladder", "stepstool",
        "stool", "three", "threelegged",
    },
    "sink": {"basin", "drain", "faucet", "kitchen sink", "sink"},
    "plate": {"plate", "plates", "saucer"},
    "dishcloth": {
        "cloth", "dish cloth", "dish towel", "dishcloth", "dishrag",
        "dishtowel", "drying cloth", "handtowel", "napkin", "rag", "sponge",
        "tea towel", "textile", "towel",
    },
    "water": {
        "deluge", "faucet water", "flood", "floor", "flow", "inundation", "liquid", "moisture", "puddle",
        "tap water", "torrent", "water",
    },
    "window": {"casement", "glass", "pane", "window", "windows"},
    "cupboard": {"cabinet", "cupboard", "door", "handle", "pantry", "shelf", "shelves", "storage"},
    "dishes": {"bowl", "crockery", "cup", "dish", "dishes", "dishware"},
    "curtains": {"curtain", "curtains", "drape", "dressing", "fabric", "hang", "tie", "tieback", "wave", "wind"},
    "boy_taking_stealing": {
        "acquire", "climb", "climbing", "extract", "filch", "grab",
        "grabbing", "pilfer", "raid", "reach", "reaching", "rob", "secure",
        "snatch", "snatching", "sneak", "snitch", "steal", "stealing",
        "step up", "take", "taking", "theft", "try",
    },
    "boy_or_stool_falling": {
        "backwards", "balance", "balancing", "collapse", "crash", "fall",
        "falling", "hurt", "overbalance", "overturn", "tilt", "tip",
        "tipping", "topple", "toppling", "tumble", "tumbling", "unstable",
    },
    "woman_drying_washing_plates": {
        "clean", "dry", "drying", "rinse", "rinsing", "scrub", "wash",
        "washing", "wipe", "wiping",
    },
    "water_overflowing": {
        "deluge", "drip", "flood", "flooding", "flow", "inundation",
        "overfill", "overflow", "overflowing", "overrun", "pour", "run",
        "run over", "running", "running over", "spill", "spilling",
        "splash", "torrent",
    },
    "action_performed_by_girl": {
        "ask", "asking", "extend", "extending", "finger", "gesture",
        "giggle", "hand", "laugh", "lip", "motion", "mouth", "outstretched",
        "point", "pointing", "quiet", "reach out", "reaching out", "request",
        "say", "shout", "shh", "signal", "speak", "stand", "standing",
        "tell", "wait", "waiting", "warn",
    },
    "woman_unconcerned_by_overflowing": {
        "attention", "aware", "care", "concern", "daydream", "daze",
        "disregard", "distract", "distracted", "heedless", "ignore",
        "ignoring", "ignorant", "neglectful", "nonchalant", "not notice",
        "not noticing", "notice", "oblivious", "realize", "stand", "unaware",
        "unconcerned", "unnoticed",
    },
    "woman_indifferent_to_children": {
        "apathetic", "attention", "aware", "back", "behind", "clue",
        "disregard", "distract", "distracted", "focus", "heedless", "ignore",
        "ignoring", "ignorant", "indifferent", "nonchalant", "notice",
        "oblivious", "turn", "unaware", "unconcern", "unconcerned",
    },
}


@dataclass(frozen=True)
class SpanMatch:
    label: str
    matched_form: str
    start: int
    end: int
    token_index: int | None = None
    method: str | None = None
    sentence_index: int | None = None


REFERENTIAL_FORMS: dict[str, set[str]] = {
    "male": {"he", "him", "his", "himself"},
    "female": {"she", "her", "hers", "herself"},
    "neutral": {"it", "its", "itself", "this", "that", "these", "those"},
    "plural": {"they", "them", "their", "theirs", "themselves"},
}
REFERENTIAL_WORDS = set().union(*REFERENTIAL_FORMS.values())
PERSON_LABELS = {"boy", "girl", "woman"}

# Referring word -> (units it can refer to, max words back, max sentences back)
REFERENCE_RULES = {
    "male": ({"boy"}, 20, 1),
    "female": ({"girl", "woman"}, 20, 1),
    "neutral": (set(CIU_LABELS) - PERSON_LABELS, 12, 0),
    "plural": (set(CIU_LABELS), 12, 0),
}
WORD_RE = re.compile(r"\b[\w']+\b")


def content_unit_mentions(text: str) -> list[SpanMatch]:
    """Ordered content-unit mentions (repeats kept, referring words resolved)."""
    tokens = tokenize(text)
    direct, matched_tokens = dictionary_matches(text, tokens)
    return add_resolved_references(tokens, direct, matched_tokens)


def count_content_units(text: str) -> tuple[int, int]:
    """(number of different units mentioned, total number of mentions)."""
    counts = Counter(span.label for span in content_unit_mentions(text))
    return len(counts), sum(counts.values())


def sentence_number(text: str, offset: int) -> int:
    """Rough sentence index of a character offset (sentence ends counted before it)."""
    return len(re.findall(r"[.!?]+(?:\s+|$)", text[:offset]))


def tokenize(text: str) -> list[dict]:
    """Words of the raw text with their character positions, word number and sentence number."""
    tokens = []
    for i, match in enumerate(WORD_RE.finditer(text)):
        tokens.append({"text": match.group(0), "lower": match.group(0).lower(), "start": match.start(),
                       "end": match.end(), "token_index": i, "sentence_index": sentence_number(text, match.start())})
    return tokens


def dictionary_forms() -> dict[tuple[str, ...], set[str]]:
    """Keyword (as a tuple of words) -> the units it names. Referring words are left to reference resolution."""
    forms: dict[tuple[str, ...], set[str]] = {}
    for label, keywords in COMMON_CIU_KEYWORDS.items():
        for form in keywords:
            clean = form.lower().strip()
            key = tuple(WORD_RE.findall(clean))
            if key and clean not in REFERENTIAL_WORDS:
                forms.setdefault(key, set()).add(label)
    return forms


def pick_unit(candidates: set[str], earlier: list[SpanMatch]) -> str | None:
    """An ambiguous keyword ("child") goes to the candidate mentioned most recently."""
    if len(candidates) == 1:
        return next(iter(candidates))
    return next((mention.label for mention in reversed(earlier) if mention.label in candidates), None)


def dictionary_matches(text: str, tokens: list[dict]) -> tuple[dict[int, SpanMatch], set[int]]:
    """
    Longest keyword match at each position, left to right, without overlaps.
    Returns the matches keyed by their first token, and every token they cover.
    """
    forms = dictionary_forms()
    lengths = sorted({len(key) for key in forms}, reverse=True)
    covered: set[int] = set()
    matches: dict[int, SpanMatch] = {}
    for start in range(len(tokens)):
        if start in covered:
            continue
        for length in lengths:
            end = start + length
            span = set(range(start, end))
            if end > len(tokens) or span & covered:
                continue
            candidates = forms.get(tuple(token["lower"] for token in tokens[start:end]))
            label = pick_unit(candidates, list(matches.values())) if candidates else None
            if label is None:
                continue
            first, last = tokens[start], tokens[end - 1]
            matches[start] = SpanMatch(label, text[first["start"]:last["end"]], first["start"], last["end"],
                                       token_index=start, method="dictionary", sentence_index=first["sentence_index"])
            covered |= span
            break
    return matches, covered


def referred_unit(word: str, token_index: int, sentence_index: int, earlier: list[SpanMatch]) -> str | None:
    """The most recent compatible unit close enough before a referring word ("she" -> woman)."""
    kind = next((kind for kind, forms in REFERENTIAL_FORMS.items() if word in forms), None)
    if kind is None:
        return None
    candidates, max_words_back, max_sentences_back = REFERENCE_RULES[kind]
    for previous in reversed(earlier):
        if previous.label not in candidates:
            continue
        if previous.token_index is None or previous.sentence_index is None:
            return previous.label
        words_back = token_index - previous.token_index
        sentences_back = sentence_index - previous.sentence_index
        if 0 < words_back <= max_words_back and 0 <= sentences_back <= max_sentences_back:
            return previous.label
    return None


def add_resolved_references(tokens: list[dict], direct: dict[int, SpanMatch], covered: set[int]) -> list[SpanMatch]:
    """All mentions in text order: dictionary matches plus referring words linked to a unit."""
    mentions: list[SpanMatch] = []
    for i, token in enumerate(tokens):
        if i in direct:
            mentions.append(direct[i])
            continue
        if i in covered:
            continue
        label = referred_unit(token["lower"], token["token_index"], token["sentence_index"], mentions)
        if label is not None:
            mentions.append(SpanMatch(label, token["text"], token["start"], token["end"], token_index=i,
                                      method="reference", sentence_index=token["sentence_index"]))
    return mentions
