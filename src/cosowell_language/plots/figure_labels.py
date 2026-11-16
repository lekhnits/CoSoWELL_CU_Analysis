"""Readable names for metrics, survey terms and narrative types in figures."""

from __future__ import annotations

METRICS = {
    "fpp_plural_rate": "'We' words", "fpp_singular_rate": "'I' words",
    "social_plus_we_rate": "Social words + 'we'", "social_word_rate": "Social words",
    "n_relationships_named": "Relationships named", "relationship_word_rate": "Relationship nouns",
    "differentiation_rate": "Negation and contrast", "pronoun_noun_ratio": "Pronoun–noun ratio",
    "sickness_pain_rate": "Illness and pain words", "neg_emo_sick_pain_rate": "Negative or illness words",
    "vader_negative": "Negative tone", "vader_compound": "Overall tone", "vader_positive": "Positive tone",
    "mean_word_zipf": "Word frequency", "fragment_prop": "Sentence fragments",
    "transition_surprisal": "Topic transition surprisal", "mean_dependency_distance": "Dependency distance",
    "n_events": "Events", "ttr": "Type–token ratio",
}
TERMS = {
    "loneliness": "Loneliness", "isolation_index": "Objective isolation",
    "health": "Self-rated health",
}
SCOPES = {"pooled": "All texts", "cookie": "Picture", "past": "Past", "yesterday": "Yesterday", "future": "Future"}


def metric(code: str) -> str:
    if code in METRICS:
        return METRICS[code]
    if code.startswith("empath_"):
        return f"Empath: {code[7:].replace('_', ' ')}"
    return code.replace("_", " ").capitalize()


def term(code: str) -> str:
    return TERMS.get(code, code)


def scope(code: str) -> str:
    return SCOPES.get(code, code)


# Research question of each finding (config.FINDINGS).
HYPOTHESIS = {"RQ1 isolation": "RQ1", "RQ2 loneliness": "RQ2", "RQ1-2 isolation vs loneliness": "RQ1–2"}


def finding_term(finding: str, code: str) -> str:
    """e.g. 'RQ1: Objective isolation'."""
    return f"{HYPOTHESIS.get(finding, finding.split()[0])}: {term(code)}"


# Isolation markers shown in figure 2, with their figure labels.
ISOLATION_MARKERS = {"fpp_plural_rate": "'We' words", "fpp_singular_rate": "'I' words",
                     "social_plus_we_rate": "Social words + 'we'", "n_relationships_named": "Relationships named",
                     "relationship_word_rate": "Relationship nouns", "social_word_rate": "Social words",
                     "empath_wedding": "Wedding words", "empath_celebration": "Celebration words"}
