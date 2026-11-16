"""
The picture descriptions as the creativity metrics see them, and the results several metric
families share (sentence embeddings, originality of each sentence, FrameBERT tags, LDA topics ...).

Several metrics compare a narrative with the narratives of other people (rarity, originality),
so the whole corpus is prepared at once, by build_picture_corpus, before any metric is computed.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import EMBEDDING_MODEL, KEYS, MIN_WORDS, SPACY_MODEL
from ...data_loading import load_narratives
from ..coherence_metrics import REPEATED_IDEA_COS
from ..language_resources import torch_device
from ..syntax_metrics import CLAUSE_DEPS
from ..text_parsing import is_word
from ..tools.cookie_theft_content_units import content_unit_mentions
from .creativity_settings import CONCEPT_CLUSTER_DIST, GLOVE, RARE_SHARE_OF_PEOPLE, RESOURCES, SEED
from .framebert_tagging import TaggedWord, frame_names, framebert_tag
from .model_output_cache import cached, cached_per_text
from .vector_helpers import unit_norm

CONTENT_WORD_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
N_LDA_TOPICS = 12


@dataclass
class PictureCorpus:
    """Picture descriptions with >= MIN_WORDS words, in participant / session order."""

    narratives: pd.DataFrame            # id, test_session, narrative, n_words
    docs: list                          # spaCy parse of each narrative
    person: np.ndarray                  # 0 ... n_people - 1, the participant of each narrative
    n_people: int
    run_models: bool                    # may transformer outputs missing from the caches be computed?
    has_new_texts: bool = False         # texts added to the corpus (metrics/new_picture_descriptions.py)
    device: object = None

    # Per narrative: sentences, words and content words (nouns, verbs, adjectives, adverbs; no stop words).
    sentence_spans: list = field(default_factory=list)
    sentences: list = field(default_factory=list)
    words: list = field(default_factory=list)
    content_words: list = field(default_factory=list)

    # Every sentence of every narrative in one list, and where each narrative's sentences are in it.
    all_sentences: list = field(default_factory=list)
    n_sentences: np.ndarray = None
    sentence_offsets: np.ndarray = None
    sentence_owner: np.ndarray = None   # narrative of each sentence
    sentence_person: np.ndarray = None  # participant of each sentence

    # Shared results, filled in by build_picture_corpus.
    all_sentence_embeddings: np.ndarray = None
    sentence_embeddings: list = None
    narrative_embeddings: np.ndarray = None
    fit_to_own_narrative: list = None
    all_sentence_originality: np.ndarray = None
    sentence_originality: list = None
    idea_labels: list = None
    concept_labels: list = None
    glove: object = None
    lemma_sets: list = None
    people_using_lemma: Counter = None
    supersenses: list = None
    content_units: list = None
    n_idea_units: pd.Series = None
    frame_names: list = None
    framebert_tokens: list = None
    framebert_missing: np.ndarray = None
    framebert_words: list = None
    topic_mix: np.ndarray = None

    @property
    def texts(self) -> list[str]:
        return self.narratives["narrative"].tolist()

    @property
    def n_narratives(self) -> int:
        return len(self.narratives)


def build_picture_corpus(corpus_path: Path, run_models: bool,
                         extra_narratives: pd.DataFrame | None = None) -> PictureCorpus:
    """
    Every picture description of the corpus, plus extra_narratives (id, test_session, narrative) if given: new
    texts are measured against the whole corpus, as rarity and originality compare a text with everyone else's.
    """
    narratives, docs = parse_long_enough_picture_descriptions(corpus_path, extra_narratives)
    corpus = PictureCorpus(narratives=narratives, docs=docs, person=pd.factorize(narratives["id"])[0],
                           n_people=narratives["id"].nunique(), run_models=run_models, device=torch_device(),
                           has_new_texts=extra_narratives is not None)
    split_into_sentences_and_words(corpus)
    print(f"{corpus.n_narratives} narratives >= {MIN_WORDS} words from {corpus.n_people} participants")

    add_sentence_embeddings(corpus)
    add_sentence_originality(corpus)
    add_sentence_clusters(corpus)
    corpus.glove = load_glove()
    corpus.lemma_sets = [{lemma_of(word) for word in content} for content in corpus.content_words]
    corpus.people_using_lemma = people_using_each_item(corpus, corpus.lemma_sets)
    corpus.supersenses = [supersense_counts(content) for content in corpus.content_words]
    corpus.content_units = [content_unit_mentions(text) for text in corpus.texts]
    corpus.n_idea_units = pd.Series([len(idea_units(doc)) for doc in docs], index=narratives.index)
    add_framebert_tags(corpus)
    corpus.topic_mix = lda_topic_mix(corpus)
    return corpus


def parse_long_enough_picture_descriptions(corpus_path: Path,
                                           extra_narratives: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list]:
    import spacy
    from tqdm.auto import tqdm

    narratives = load_narratives(corpus_path, ["cookie"]).drop(columns="ntype")
    if extra_narratives is not None:
        narratives = pd.concat([narratives, extra_narratives[KEYS + ["narrative"]]], ignore_index=True)
    narratives = narratives.sort_values(KEYS).reset_index(drop=True)
    nlp = spacy.load(SPACY_MODEL)
    texts = narratives["narrative"].tolist()
    docs = list(tqdm(nlp.pipe(texts, batch_size=64), total=len(narratives), desc="spaCy"))
    narratives["n_words"] = [sum(is_word(token) for token in doc) for doc in docs]
    keep = (narratives["n_words"] >= MIN_WORDS).to_numpy()
    kept_docs = [doc for doc, is_kept in zip(docs, keep) if is_kept]
    return narratives[keep].reset_index(drop=True), kept_docs


def split_into_sentences_and_words(corpus: PictureCorpus) -> None:
    for doc in corpus.docs:
        spans = [sentence for sentence in doc.sents if any(is_word(token) for token in sentence)]
        words = [token for token in doc if is_word(token)]
        content = [word for word in words if word.pos_ in CONTENT_WORD_POS and not word.is_stop and word.is_alpha]
        corpus.sentence_spans.append(spans)
        corpus.sentences.append([span.text.strip() for span in spans])
        corpus.words.append(words)
        corpus.content_words.append(content)
    corpus.n_sentences = np.array([len(sentences) for sentences in corpus.sentences])
    corpus.sentence_offsets = np.r_[0, np.cumsum(corpus.n_sentences)]
    corpus.all_sentences = [sentence for sentences in corpus.sentences for sentence in sentences]
    corpus.sentence_owner = np.repeat(np.arange(corpus.n_narratives), corpus.n_sentences)
    corpus.sentence_person = corpus.person[corpus.sentence_owner]


def sentences_of_each_narrative(corpus: PictureCorpus, values_per_sentence: np.ndarray) -> list[np.ndarray]:
    """Cut a value-per-sentence array (in all_sentences order) into one piece per narrative."""
    offsets = corpus.sentence_offsets
    return [values_per_sentence[offsets[i]:offsets[i + 1]] for i in range(corpus.n_narratives)]


def encode_sentences(sentences: list[str], device) -> list[np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device=str(device))
    return list(model.encode(sentences, batch_size=64, normalize_embeddings=True, show_progress_bar=True))


def add_sentence_embeddings(corpus: PictureCorpus) -> None:
    """Unit-length embedding of each sentence; a narrative's embedding is the normalised mean of its sentences."""
    embeddings = cached_per_text("sentence_embedding", corpus.all_sentences,
                                 lambda sentences: encode_sentences(sentences, corpus.device), may_compute=True)
    corpus.all_sentence_embeddings = np.vstack(embeddings)
    corpus.sentence_embeddings = sentences_of_each_narrative(corpus, corpus.all_sentence_embeddings)
    centroids = np.vstack([embeddings.mean(0) for embeddings in corpus.sentence_embeddings])
    corpus.narrative_embeddings = unit_norm(centroids)
    # How well each sentence fits its own narrative: similarity to the narrative's centroid.
    corpus.fit_to_own_narrative = [embeddings @ unit_norm(embeddings.mean(0))
                                   for embeddings in corpus.sentence_embeddings]


def add_sentence_originality(corpus: PictureCorpus) -> None:
    """Originality of a sentence: 1 - similarity to the most similar sentence written by any other participant."""
    embeddings, person = corpus.all_sentence_embeddings, corpus.sentence_person
    block = 1024                     # the full similarity matrix does not fit in memory, so it is built in blocks
    closest_other_similarity = np.empty(len(embeddings), dtype=np.float32)
    for start in range(0, len(embeddings), block):
        rows = slice(start, start + block)
        similarity = embeddings[rows] @ embeddings.T
        same_person = person[rows, None] == person[None, :]
        similarity[same_person] = -np.inf
        closest_other_similarity[rows] = similarity.max(1)
    corpus.all_sentence_originality = 1 - closest_other_similarity
    corpus.sentence_originality = sentences_of_each_narrative(corpus, corpus.all_sentence_originality)


def sentence_clusters(embeddings: np.ndarray, distance: float, linkage: str) -> np.ndarray:
    """Group a narrative's sentences by meaning (agglomerative clustering on cosine distance)."""
    from sklearn.cluster import AgglomerativeClustering

    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int)
    clustering = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage=linkage,
                                         distance_threshold=distance)
    return clustering.fit_predict(embeddings)


def add_sentence_clusters(corpus: PictureCorpus) -> None:
    # Ideas: nearly identical sentences (cosine >= 0.85) count as one idea.
    corpus.idea_labels = [sentence_clusters(embeddings, 1 - REPEATED_IDEA_COS, "complete")
                          for embeddings in corpus.sentence_embeddings]
    # Concepts: looser groups of sentences about the same thing.
    corpus.concept_labels = [sentence_clusters(embeddings, CONCEPT_CLUSTER_DIST, "average")
                             for embeddings in corpus.sentence_embeddings]


def load_glove():
    from gensim.models import KeyedVectors

    glove_kv = RESOURCES / f"{GLOVE}.kv"
    if not glove_kv.exists():            # one-off conversion to gensim's format, then memory-mapped
        import gensim.downloader as gensim_api
        gensim_api.load(GLOVE).save(str(glove_kv))
    return KeyedVectors.load(str(glove_kv), mmap="r")


def glove_vector(corpus: PictureCorpus, token):
    """GloVe vector of the word, or None if GloVe does not know it."""
    if token.lower_ not in corpus.glove.key_to_index:
        return None
    return corpus.glove[token.lower_]


def lemma_of(token) -> str:
    return token.lemma_.lower()


def people_using_each_item(corpus: PictureCorpus, item_sets: list[set]) -> Counter:
    """How many participants use each item (lemma, event ...); a person's sessions count once."""
    items_of_person = [set() for _ in range(corpus.n_people)]
    for person, items in zip(corpus.person, item_sets):
        items_of_person[person] |= items
    return Counter(item for items in items_of_person for item in items)


def mean_rarity(items: set, people_using: Counter, n_people: int) -> float:
    """
    Mean rarity of the narrative's items. Rarity of an item = log(number of people / number of people who use it),
    as in tf-idf: 0 when everyone uses it, larger the fewer people do.
    """
    if not items:
        return np.nan
    return np.mean([math.log(n_people / people_using[item]) for item in items])


def share_of_rare_items(items: set, people_using: Counter, n_people: int) -> float:
    """Share of the narrative's items used by at most RARE_SHARE_OF_PEOPLE of all participants."""
    if not items:
        return np.nan
    return np.mean([people_using[item] <= RARE_SHARE_OF_PEOPLE * n_people for item in items])


def supersense(token):
    """WordNet lexicographer file of the first sense (e.g. noun.person, verb.cognition)."""
    from nltk.corpus import wordnet as wn

    wordnet_pos = {"NOUN": wn.NOUN, "PROPN": wn.NOUN, "VERB": wn.VERB}.get(token.pos_)
    if wordnet_pos is None:
        return None
    synsets = wn.synsets(token.lemma_.lower(), pos=wordnet_pos)
    return synsets[0].lexname() if synsets else None


def supersense_counts(content: list) -> Counter:
    """How many content words of each WordNet supersense; words WordNet does not know are left out."""
    senses = [supersense(word) for word in content]
    return Counter(sense for sense in senses if sense)


def idea_units(doc) -> list:
    """Clause heads: verbs and auxiliaries that head a clause (an auxiliary of a main verb is not counted)."""
    units = []
    for token in doc:
        heads_clause = token.pos_ in {"VERB", "AUX"} and (token.dep_ in CLAUSE_DEPS or token.dep_ == "conj")
        is_helping_verb = token.pos_ == "AUX" and token.head.pos_ == "VERB" and token.dep_ != "ROOT"
        if heads_clause and not is_helping_verb:
            units.append(token)
    return units


def add_framebert_tags(corpus: PictureCorpus) -> None:
    """Per token: is it a metaphor, a novel metaphor, which FrameNet frame does it evoke; missing when not cached."""
    tags = cached_per_text("framebert", corpus.texts, lambda texts: framebert_tag(texts, corpus.device),
                           may_compute=corpus.run_models)
    corpus.frame_names = frame_names()
    corpus.framebert_missing = np.array([text_tags is None for text_tags in tags])
    corpus.framebert_tokens = [[TaggedWord(*tag) for tag in text_tags or []] for text_tags in tags]
    # Only real words (FrameBERT also tags punctuation).
    corpus.framebert_words = [[token for token in tokens if any(character.isalpha() for character in token.text)]
                              for tokens in corpus.framebert_tokens]


def per_100_framebert_words(corpus: PictureCorpus, is_counted) -> list[float]:
    """Per narrative: % of its FrameBERT words for which is_counted(word) is true; NaN when FrameBERT is missing."""
    values = []
    for words, missing in zip(corpus.framebert_words, corpus.framebert_missing):
        if missing:
            values.append(np.nan)
        else:
            values.append(100 * sum(is_counted(word) for word in words) / max(len(words), 1))
    return values


def lda_topic_mix(corpus: PictureCorpus) -> np.ndarray:
    """Share of each of 12 LDA topics in each narrative (content-word lemmas)."""
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel

    lda_documents = [[lemma_of(word) for word in content] for content in corpus.content_words]
    dictionary = Dictionary(lda_documents)
    # Words in fewer than 5 narratives or in more than half of them carry no topic information.
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    bags_of_words = [dictionary.doc2bow(document) for document in lda_documents]

    def fit_lda():
        return LdaModel(bags_of_words, id2word=dictionary, num_topics=N_LDA_TOPICS, passes=15, random_state=SEED)

    # The cache is named by the number of texts, which only identifies the corpus itself; with new texts added,
    # the model is fitted afresh, so a different set of new texts never reuses it.
    lda = fit_lda() if corpus.has_new_texts else cached(f"lda_{corpus.n_narratives}", fit_lda)
    topic_shares = [[share for _, share in lda.get_document_topics(bag, minimum_probability=0)]
                    for bag in bags_of_words]
    return np.array(topic_shares)
