"""Models, resources and thresholds of the creativity metrics."""

from ...config import DATA_DIR

CACHE = DATA_DIR / "cache" / "creativity"      # per-text transformer outputs
RESOURCES = DATA_DIR / "resources"             # GloVe, FrameBERT frame list, concreteness norms

SEED = 0
DSI_MODEL, DSI_LAYERS = "google-bert/bert-large-uncased", (6, 7)
GLOVE = "glove-wiki-gigaword-300"
METAPHOR_MODEL = "CreativeLang/metaphor_detection_roberta_seq"
NOVEL_METAPHOR_MODEL = "CreativeLang/novel_metaphors"
FRAME_MODEL = "liyucheng/frame_finder"
ROBERTA_TOKENIZER = "FacebookAI/roberta-base"
LM_MODEL = "distilbert/distilgpt2"
FRAMEBERT_INFERENCE = "https://raw.githubusercontent.com/liyucheng09/MetaphorFrame/main/inference.py"
CONCRETENESS_URL = ("https://raw.githubusercontent.com/ArtsEngine/concreteness/master/"
                    "Concreteness_ratings_Brysbaert_et_al_BRM.txt")

RARE_SHARE_OF_PEOPLE = 0.01  # a word or event is "rare" when at most 1% of participants use it
UNIQUE_IDEA_PCTL = 80        # sentence counts as a unique idea above this corpus percentile of originality
LARGE_JUMP_PCTL = 90         # consecutive-sentence distance above this corpus percentile = large jump
CONCEPT_CLUSTER_DIST = 0.6   # cosine distance cut for concept clusters within a narrative
N_CORPUS_TOPICS = 20         # corpus-wide sentence clusters for the transition Markov chain
N_NEAREST_NARRATIVES = 10    # originality_knn: compared with the 10 most similar narratives of others

# Reference description of the picture (task relevance).
COOKIE_PICTURE = (
    "A boy is standing on a stool that is tipping over, taking cookies from a cookie jar in the cupboard. "
    "A girl is reaching up for a cookie and laughing. A woman is drying a dish at the sink while the sink "
    "overflows and water spills onto the floor. Through the open window there is a garden and a path, and "
    "dishes sit on the counter.")
