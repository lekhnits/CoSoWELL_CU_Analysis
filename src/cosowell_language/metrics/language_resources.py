"""
Language resources, each loaded once: spaCy, Empath, VADER, the spell checker,
WordNet look-ups and word frequencies.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import SPACY_MODEL

# A noun counts when the hypernym chain of its first WordNet sense contains one of these.
RELATIONSHIP_ROOTS = ("relative.n.01", "spouse.n.01", "friend.n.01", "neighbor.n.01", "acquaintance.n.01",
                      "lover.n.01", "co-worker.n.01", "colleague.n.01", "companion.n.01", "family.n.01")
SICKNESS_ROOTS = ("ill_health.n.01", "pain.n.01", "injury.n.01", "symptom.n.01")
EMPATH_SOCIAL = ["family", "friends", "communication", "meeting"]
EMPATH_SICK_PAIN = ["pain", "health", "injury", "medical_emergency", "suffering"]
# A vague noun has at least VAGUE_MIN_SENSES WordNet senses and a Zipf frequency of at least VAGUE_MIN_ZIPF
# (thing, place, way ...), or is a WordNet top-level noun.
VAGUE_MIN_SENSES = 10
VAGUE_MIN_ZIPF = 5.3


def ensure_nltk_resources() -> None:
    import nltk

    for resource, path in [("wordnet", "corpora/wordnet.zip"), ("omw-1.4", "corpora/omw-1.4.zip"),
                           ("vader_lexicon", "sentiment/vader_lexicon.zip")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


@lru_cache(maxsize=None)
def nlp():
    import spacy

    return spacy.load(SPACY_MODEL)


@lru_cache(maxsize=None)
def empath_lexicon():
    from empath import Empath

    return Empath()


@lru_cache(maxsize=None)
def empath_word_sets() -> dict[str, set[str]]:
    """The Empath categories the metrics use, plus two combined sets: social and illness/pain words."""
    cats = empath_lexicon().cats
    sets = {name: set(cats[name]) for name in
            ["positive_emotion", "negative_emotion", "sadness", *EMPATH_SOCIAL, *EMPATH_SICK_PAIN]}
    sets["social"] = set().union(*(sets[category] for category in EMPATH_SOCIAL))
    sets["sick_pain"] = set().union(*(sets[category] for category in EMPATH_SICK_PAIN))
    return sets


@lru_cache(maxsize=None)
def vader():
    from nltk.sentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=None)
def spell_checker():
    from spellchecker import SpellChecker

    return SpellChecker(language="en")


@lru_cache(maxsize=None)
def wordnet_root_synsets(names: tuple[str, ...]) -> frozenset:
    from nltk.corpus import wordnet as wn

    return frozenset(wn.synset(name) for name in names)


@lru_cache(maxsize=None)
def first_noun_sense_and_its_hypernyms(lemma: str) -> frozenset:
    """The first WordNet sense of the noun and every more general sense above it (mother -> parent -> relative ...).
    Only the first sense is used because it is the most common one; later senses are often unrelated."""
    from nltk.corpus import wordnet as wn

    synsets = wn.synsets(lemma, pos=wn.NOUN)
    if not synsets:
        return frozenset()
    first_sense = synsets[0]
    more_general_senses = first_sense.closure(lambda sense: sense.hypernyms() + sense.instance_hypernyms())
    return frozenset(more_general_senses) | {first_sense}


def is_relationship_noun(lemma: str) -> bool:
    return bool(first_noun_sense_and_its_hypernyms(lemma) & wordnet_root_synsets(RELATIONSHIP_ROOTS))


def is_sickness_noun(lemma: str) -> bool:
    return bool(first_noun_sense_and_its_hypernyms(lemma) & wordnet_root_synsets(SICKNESS_ROOTS))


@lru_cache(maxsize=None)
def is_cognition_verb(lemma: str) -> bool:
    from nltk.corpus import wordnet as wn

    synsets = wn.synsets(lemma, pos=wn.VERB)
    return bool(synsets) and synsets[0].lexname() == "verb.cognition"


@lru_cache(maxsize=None)
def is_vague_noun(lemma: str) -> bool:
    """Highly polysemous, highly frequent nouns (thing, place, way ...) or WordNet top-level nouns."""
    from nltk.corpus import wordnet as wn

    synsets = wn.synsets(lemma, pos=wn.NOUN)
    if not synsets:
        return False
    if synsets[0].lexname() == "noun.Tops":
        return True
    return len(synsets) >= VAGUE_MIN_SENSES and zipf(lemma) >= VAGUE_MIN_ZIPF


@lru_cache(maxsize=None)
def zipf(word: str) -> float:
    from wordfreq import zipf_frequency

    return zipf_frequency(word, "en")


def torch_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
