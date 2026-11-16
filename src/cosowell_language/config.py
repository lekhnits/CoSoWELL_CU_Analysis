"""Paths and analysis constants shared by all modules."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FEATURES_DIR = RESULTS_DIR / "features"
FIGURES_DIR = RESULTS_DIR / "figures"                # every figure of the report

CORPUS_PATH = DATA_DIR / "cosowell_corpus_matched_v2.txt"
SURVEY_PATH = DATA_DIR / "cosowell_survey_v2_numericVar.csv"

KEYS = ["id", "test_session"]                       # one writing/survey session
NARRATIVE_TYPES = ["cookie", "past", "yesterday", "future"]
TRANSFORMER_TYPES = {"cookie"}                      # transformer features are computed for these only
# Extra feature sets per narrative type: results/features/<type>_<set>.csv (steps 1b and 1c).
EXTRA_FEATURES = {"cookie": ["gaze", "creativity"]}
# Metrics of steps 1b/1c that duplicate another metric (rank correlation >= 0.99); dropped when loading.
DUPLICATE_METRICS = {
    "gaze_n_saccades": "gaze_n_fixations - 1", "gaze_latency_prop": "1 - gaze_dwell_prop",
    "gaze_coverage": "cu_unique / 23", "fluency_content_units": "cu_unique", "picture_unit_coverage": "cu_unique / 23",
    "picture_anchoring": "10 x cu_per_word", "hedge_rate": "tentative_rate (same hedge tagger)",
    "past_tense_prop": "past_tense_share", "fluency_distinct_ideas": "n_sentences (rho = 1.00)",
    "flexibility_frame_entropy": "flexibility_frames (rho = 0.99)",
}
MIN_WORDS = 20                                      # shorter narratives are excluded
SEED = 2026

# Pretrained models
SPACY_MODEL = "en_core_web_sm"
HEDGE_MODEL = "jeniakim/hedgehog"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Covariates in every model (z-scored). health: 1 = poor ... 4 = excellent.
COVARIATES = ["age", "years_education", "health", "female", "covid_phase"]
LENGTH_COVARIATE = "log_n_words"
LENGTH_FEATURES = {"n_words", "n_sentences"}        # not adjusted for length

# Survey scores analysed (column names after `survey.build_session_table`).
MEASURES = ["loneliness", "isolation_index"]

# One language model per research question: survey terms entered jointly, plus extra covariates.
# RQ1-2 is the follow-up of RQ1 and RQ2: both scores in one model, so each is net of the other.
FINDINGS = {
    "RQ1 isolation": dict(terms=["isolation_index"]),
    "RQ2 loneliness": dict(terms=["loneliness"]),
    "RQ1-2 isolation vs loneliness": dict(terms=["loneliness", "isolation_index"]),
}
