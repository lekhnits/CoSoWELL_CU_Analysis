"""
Script 2: SemR – Semantic Relevance Measure
============================================
Implements the Semantic Relevance (SemR) algorithm described in:

    Stegmann, G., et al. (2022).
    "Automated semantic relevance as an indicator of cognitive decline:
    Out-of-sample validation on a large-scale longitudinal dataset."
    Alzheimer's & Dementia: Diagnosis, Assessment & Disease Monitoring,
    14(1), e12294.
    https://doi.org/10.1002/dad2.12294
    Full text (open access): https://pmc.ncbi.nlm.nih.gov/articles/PMC8865737/

Algorithm (from the paper):
  1. Transcribe speech.
  2. Categorise each word: does it match a target element or not?
     The 23 target elements are those from Ahmed et al. (2010).
     Synonyms are accepted (e.g. "young man" for "boy").
  3. Count unique elements mentioned.
  4. SemR = unique elements mentioned / total words produced.
"""

import re
from collections import defaultdict

import spacy

# ---------------------------------------------------------------------------
# 1.  The 23 target elements for the Cookie Theft picture
#     (Ahmed et al. 2010, Table 1; also used by Stegmann et al. 2022)
#
#     Each entry is (canonical_label, {accepted surface forms / synonyms}).
#     Surface forms are lower-cased; multi-word expressions are checked via
#     phrase matching.
# ---------------------------------------------------------------------------

COOKIE_THEFT_ELEMENTS: list[tuple[str, set[str]]] = [
    ("boy",         {"boy", "kid", "child", "son", "lad", "young man", "little boy",
                     "young boy", "youth"}),
    ("girl",        {"girl", "daughter", "lass", "young girl", "little girl",
                     "sister", "child"}),
    ("woman",       {"woman", "mother", "mom", "lady", "housewife", "wife",
                     "mum", "mommy", "mummy"}),
    ("cookie",      {"cookie", "biscuit", "treat", "snack", "cookies"}),
    ("jar",         {"jar", "cookie jar", "container", "tin", "pot",
                     "cookie tin"}),
    ("stool",       {"stool", "step stool", "stepstool", "footstool",
                     "ladder", "step ladder"}),
    ("sink",        {"sink", "basin", "kitchen sink"}),
    ("water",       {"water", "tap water", "faucet water"}),
    ("plate",       {"plate", "dish", "dishes", "dishware", "crockery",
                     "saucer", "plates"}),
    ("dishcloth",   {"dishcloth", "towel", "cloth", "rag", "drying cloth",
                     "tea towel", "dish towel", "dishrag"}),
    ("window",      {"window", "windows"}),
    ("curtains",    {"curtain", "curtains"}),
    ("cupboard",    {"cupboard", "cabinet", "shelf", "shelves", "pantry"}),
    ("kitchen",     {"kitchen"}),
    ("outside",     {"outside", "outdoors", "yard", "garden", "exterior",
                     "out", "outdoor"}),
    ("stealing",    {"steal", "stealing", "take", "taking", "grab", "grabbing",
                     "reach", "reaching", "snatch", "snatching", "filch",
                     "pilfer", "theft"}),
    ("falling",     {"fall", "falling", "tip", "tipping", "topple",
                     "toppling", "overbalance", "tumble", "tumbling"}),
    ("drying",      {"dry", "drying", "wipe", "wiping", "wash", "washing",
                     "rinse", "rinsing"}),
    ("overflowing", {"overflow", "overflowing", "spill", "spilling",
                     "flood", "flooding", "run over", "running over",
                     "run", "running"}),
    ("girl standing", {"stand", "standing", "wait", "waiting"}),
    ("hand out",    {"hand", "outstretched", "reach out", "reaching out",
                     "point", "pointing", "extend", "extending"}),
    ("woman oblivious", {"oblivious", "unaware", "ignore", "ignoring",
                         "unconcerned", "indifferent", "heedless",
                         "distracted", "unnoticed", "notice",
                         "not notice", "not noticing"}),
    ("climbing",    {"climb", "climbing", "step up", "clamber",
                     "clamber up", "perch", "perching", "balance",
                     "balancing"}),
]

# ---------------------------------------------------------------------------
# 2.  Build fast lookup structures
# ---------------------------------------------------------------------------

# Single-token surface form → canonical label
_TOKEN_TO_LABEL: dict[str, str] = {}
# Multi-word phrase → canonical label  (phrase length → list of (phrase, label))
_PHRASE_TO_LABEL: dict[int, list[tuple[str, str]]] = defaultdict(list)

for label, forms in COOKIE_THEFT_ELEMENTS:
    for form in forms:
        toks = form.split()
        if len(toks) == 1:
            _TOKEN_TO_LABEL[form] = label
        else:
            _PHRASE_TO_LABEL[len(toks)].append((form, label))

_MAX_PHRASE_LEN = max(_PHRASE_TO_LABEL.keys(), default=1)


# ---------------------------------------------------------------------------
# 3.  spaCy setup
# ---------------------------------------------------------------------------

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise SystemExit(
        "spaCy model not found.\n"
        "Run: python -m spacy download en_core_web_sm"
    )


# ---------------------------------------------------------------------------
# 4.  Core SemR function
# ---------------------------------------------------------------------------

def compute_semr(
    transcript: str,
    elements: list[tuple[str, set[str]]] | None = None,
    return_details: bool = True,
) -> dict:
    """
    Compute the Semantic Relevance (SemR) score for a picture-description
    transcript.

    Parameters
    ----------
    transcript : str
        Raw or automatically transcribed participant speech.
    elements : list | None
        Custom list of (label, {surface forms}).  Defaults to the 23 Cookie
        Theft elements (Ahmed et al. 2010).
    return_details : bool
        If True, include per-element match details in the output.

    Returns
    -------
    dict with keys:
        semr            – float, SemR score (0–1)
        unique_elements – int, number of unique picture elements identified
        total_words     – int, total word count
        matched         – list of dicts (element, matched_form, position)
                          [only when return_details=True]
        not_matched     – list of canonical labels not found
    """
    # ── Rebuild lookup if custom elements provided ──────────────────────────
    if elements is not None:
        global _TOKEN_TO_LABEL, _PHRASE_TO_LABEL, _MAX_PHRASE_LEN
        _TOKEN_TO_LABEL = {}
        _PHRASE_TO_LABEL = defaultdict(list)
        for lbl, forms in elements:
            for form in forms:
                toks = form.split()
                if len(toks) == 1:
                    _TOKEN_TO_LABEL[form] = lbl
                else:
                    _PHRASE_TO_LABEL[len(toks)].append((form, lbl))
        _MAX_PHRASE_LEN = max(_PHRASE_TO_LABEL.keys(), default=1)

    # ── Tokenise ─────────────────────────────────────────────────────────────
    doc       = nlp(transcript.lower())
    word_toks = [t for t in doc if not t.is_space and not t.is_punct]
    total_words = len(word_toks)

    if total_words == 0:
        return {"semr": 0.0, "unique_elements": 0, "total_words": 0,
                "matched": [], "not_matched": [l for l, _ in COOKIE_THEFT_ELEMENTS]}

    # Build a flat list of (surface, lemma, original_index) for phrase search
    surface_list = [t.text for t in word_toks]
    lemma_list   = [t.lemma_ for t in word_toks]

    matched_labels: dict[str, dict] = {}   # canonical → first match detail

    # ── Multi-word phrase scan (longest match first) ─────────────────────────
    covered_indices: set[int] = set()
    for phrase_len in sorted(_PHRASE_TO_LABEL.keys(), reverse=True):
        for phrase, label in _PHRASE_TO_LABEL[phrase_len]:
            phrase_toks = phrase.split()
            for start in range(len(surface_list) - phrase_len + 1):
                window = surface_list[start:start + phrase_len]
                if window == phrase_toks:
                    indices = set(range(start, start + phrase_len))
                    if not indices & covered_indices:
                        covered_indices |= indices
                        if label not in matched_labels:
                            matched_labels[label] = {
                                "element":      label,
                                "matched_form": phrase,
                                "position":     start,
                            }

    # ── Single-token scan ────────────────────────────────────────────────────
    for idx, (surf, lemma) in enumerate(zip(surface_list, lemma_list)):
        if idx in covered_indices:
            continue
        for candidate in {surf, lemma}:
            if candidate in _TOKEN_TO_LABEL:
                label = _TOKEN_TO_LABEL[candidate]
                if label not in matched_labels:
                    matched_labels[label] = {
                        "element":      label,
                        "matched_form": candidate,
                        "position":     idx,
                    }
                break

    # ── Compute SemR ─────────────────────────────────────────────────────────
    unique_elements = len(matched_labels)
    semr = unique_elements / total_words

    # ── Which elements were NOT mentioned? ───────────────────────────────────
    all_labels   = {lbl for lbl, _ in COOKIE_THEFT_ELEMENTS}
    not_matched  = sorted(all_labels - set(matched_labels.keys()))

    result: dict = {
        "semr":            round(semr, 6),
        "unique_elements": unique_elements,
        "total_words":     total_words,
        "not_matched":     not_matched,
    }
    if return_details:
        result["matched"] = sorted(matched_labels.values(),
                                   key=lambda x: x["position"])
    return result


# ---------------------------------------------------------------------------
# 5.  Batch helper (list of transcripts → list of SemR scores)
# ---------------------------------------------------------------------------

def batch_semr(transcripts: list[str], **kwargs) -> list[dict]:
    """Run compute_semr over a list of transcripts."""
    return [compute_semr(t, **kwargs) for t in transcripts]


# ---------------------------------------------------------------------------
# 6.  CLI / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json, sys

    # Two example transcripts: a healthy control and a simulated impaired one
    examples = {
        "healthy_control": (
            "I see a woman standing at the kitchen sink washing and drying dishes. "
            "The water is overflowing from the sink onto the floor but she seems "
            "completely unaware of it. Behind her, a boy is standing on a stool "
            "trying to steal cookies from a cookie jar that's on a shelf in the "
            "cupboard. The stool looks like it's tipping and he might fall. "
            "A girl is standing next to him with her hand outstretched, asking for "
            "a cookie. There's a window with curtains above the sink and you can "
            "see a garden outside."
        ),
        "impaired_speaker": (
            "There is a woman. She is doing something. "
            "And there is a boy. He is getting something. "
            "The floor is wet. There is a girl too."
        ),
    }

    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as fh:
            examples = {"input_file": fh.read()}

    for name, transcript in examples.items():
        print("=" * 60)
        print(f"Speaker: {name}")
        print("=" * 60)
        result = compute_semr(transcript)
        print(f"SemR score       : {result['semr']:.4f}")
        print(f"Unique elements  : {result['unique_elements']} / 23")
        print(f"Total words      : {result['total_words']}")
        print("\nMatched elements:")
        for m in result["matched"]:
            print(f"  [{m['element']:30s}] matched via '{m['matched_form']}'")
        print("\nNot matched:", result["not_matched"])
        print()