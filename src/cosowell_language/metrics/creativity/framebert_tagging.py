"""
FrameBERT tagging (Li et al.): for every word, whether it is a metaphor, a novel metaphor, and which FrameNet
frame it evokes. Three RoBERTa word taggers run over each sentence:

    metaphor model         is the word used metaphorically?
    novel-metaphor model   is the metaphor new rather than conventional? It is given the metaphor labels as an
                           extra input, so it runs after the metaphor model
    frame model            the FrameNet frame the word evokes (frame id 0 = no frame)
"""

from __future__ import annotations

import ast
import json
from typing import NamedTuple

from .creativity_settings import (FRAME_MODEL, FRAMEBERT_INFERENCE, METAPHOR_MODEL, NOVEL_METAPHOR_MODEL, RESOURCES,
                                  ROBERTA_TOKENIZER)


class TaggedWord(NamedTuple):
    text: str
    is_metaphor: int
    is_novel_metaphor: int
    frame_id: int


def frame_names() -> list[str]:
    """FrameNet frame names in FrameBERT's label order (read once from the FrameBERT repository)."""
    path = RESOURCES / "framebert_frames.json"
    if not path.exists():
        import urllib.request
        source_code = urllib.request.urlopen(FRAMEBERT_INFERENCE).read().decode()
        frame_list_assignment = next(
            statement for statement in ast.parse(source_code).body
            if isinstance(statement, ast.Assign) and getattr(statement.targets[0], "id", "") == "frame_list")
        path.write_text(json.dumps(ast.literal_eval(frame_list_assignment.value)))
    return json.loads(path.read_text())


def sentences_as_words(texts: list[str]) -> list[tuple[int, list[str]]]:
    """(text number, words) of every non-empty sentence of every text."""
    from nltk.tokenize import sent_tokenize, word_tokenize

    sentences = []
    for text_number, text in enumerate(texts):
        for sentence in sent_tokenize(text):
            words = word_tokenize(sentence)
            if words:
                sentences.append((text_number, words))
    return sentences


def first_subtoken_positions(encoding, sentence_number: int) -> list[int]:
    """RoBERTa splits a word into sub-tokens; the label of a word is the label of its first sub-token."""
    positions = []
    previous_word = None
    for position, word in enumerate(encoding.word_ids(sentence_number)):
        if word is not None and word != previous_word:
            positions.append(position)
        previous_word = word
    return positions


def framebert_tag(texts: list[str], device, batch_size: int = 32) -> list[list[tuple]]:
    """Per text: one (word, is_metaphor, is_novel_metaphor, frame_id) tuple per word."""
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_TOKENIZER, add_prefix_space=True)
    metaphor_model = AutoModelForTokenClassification.from_pretrained(METAPHOR_MODEL, num_labels=2)
    novel_metaphor_model = AutoModelForTokenClassification.from_pretrained(NOVEL_METAPHOR_MODEL, num_labels=2,
                                                                           type_vocab_size=2)
    frame_model = AutoModelForTokenClassification.from_pretrained(FRAME_MODEL)
    metaphor_model, novel_metaphor_model, frame_model = (
        model.to(device).eval() for model in (metaphor_model, novel_metaphor_model, frame_model))

    sentences = sentences_as_words(texts)
    tags_of_text = [[] for _ in texts]
    for batch_start in tqdm(range(0, len(sentences), batch_size), desc="FrameBERT"):
        batch = sentences[batch_start:batch_start + batch_size]
        encoding = tokenizer([words for _, words in batch], is_split_into_words=True, truncation=True,
                             max_length=256, padding=True, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in encoding.items()}
        word_positions = [first_subtoken_positions(encoding, i) for i in range(len(batch))]

        with torch.no_grad():
            metaphor_labels = metaphor_model(**inputs).logits.argmax(-1)
            # The novel-metaphor model is told which words are metaphors, on each word's first sub-token.
            metaphor_input = torch.zeros_like(metaphor_labels)
            for i, positions in enumerate(word_positions):
                metaphor_input[i, positions] = metaphor_labels[i, positions]
            novel_labels = novel_metaphor_model(**inputs, token_type_ids=metaphor_input).logits.argmax(-1)
            frame_labels = frame_model(**inputs).logits.argmax(-1)

        metaphor_labels = metaphor_labels.cpu().numpy()
        novel_labels = novel_labels.cpu().numpy()
        frame_labels = frame_labels.cpu().numpy()
        for i, (text_number, words) in enumerate(batch):
            # zip stops at the last word that survived truncation to 256 sub-tokens.
            for word, position in zip(words, word_positions[i]):
                is_metaphor = int(metaphor_labels[i, position])
                # Only a metaphor can be a novel metaphor.
                is_novel_metaphor = int(is_metaphor and novel_labels[i, position])
                frame_id = int(frame_labels[i, position])
                tags_of_text[text_number].append((word, is_metaphor, is_novel_metaphor, frame_id))
    return tags_of_text
