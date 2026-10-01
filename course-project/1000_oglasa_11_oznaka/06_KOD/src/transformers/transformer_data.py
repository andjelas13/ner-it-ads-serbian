"""Chunk-safe Hugging Face alignment: one canonical word tokenizer, no silent truncation."""
from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.classical.models import SequenceExample, make_sequences, sequences_to_entities
from src.common.constants import BIO_LABELS, ID_TO_LABEL, LABEL_TO_ID


@dataclass
class Chunk:
    input_ids: list[int]; attention_mask: list[int]; labels: list[int]
    sequence_index: int; word_ids: list[int | None]


def model_name(alias: str) -> str:
    return {"bertic": "classla/bcms-bertic", "mbert": "google-bert/bert-base-multilingual-cased"}[alias]


def subword_length_stats(sequences: list[SequenceExample], tokenizer: Any) -> dict[str, float | int]:
    # ``verbose=False`` matters here: this is descriptive statistics, not a
    # forward pass, so a >512 sequence is expected and must not look like a
    # model indexing error in notebook output.
    lengths = [len(tokenizer([t.text for t in seq.tokens], is_split_into_words=True,
                             add_special_tokens=True, verbose=False)["input_ids"]) for seq in sequences]
    if not lengths: return {"n": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "recommended_max_length": 128}
    q = np.percentile(lengths, [50, 90, 95, 99])
    # Keep a multiple of 8; p95 gets headroom, but never hard-code 512 without data.
    recommended = min(512, max(128, int(np.ceil((q[2] + 16) / 8) * 8)))
    return {"n": len(lengths), "p50": float(q[0]), "p90": float(q[1]), "p95": float(q[2]), "p99": float(q[3]),
            "max": int(max(lengths)), "recommended_max_length": recommended}


def _is_omittable_format_token(text: str, label: str) -> bool:
    """Only an O-labelled token made solely of Unicode format/space chars may vanish."""
    return (
        label == "O"
        and bool(text)
        and any(unicodedata.category(character) == "Cf" for character in text)
        and all(
            character.isspace() or unicodedata.category(character) == "Cf"
            for character in text
        )
    )


def build_chunks(sequences: list[SequenceExample], tokenizer: Any, max_length: int, stride: int = 32) -> list[Chunk]:
    chunks: list[Chunk] = []
    covered: dict[int, set[int]] = defaultdict(set)
    for sequence_index, seq in enumerate(sequences):
        words = [token.text for token in seq.tokens]
        encoded = tokenizer(words, is_split_into_words=True, truncation=True, max_length=max_length, stride=stride,
                            return_overflowing_tokens=True, return_attention_mask=True)
        for batch_index in range(len(encoded["input_ids"])):
            word_ids = encoded.word_ids(batch_index=batch_index)
            labels, seen_word = [], set()
            for word_id in word_ids:
                if word_id is None: labels.append(-100)
                elif word_id in seen_word: labels.append(-100)
                else:
                    labels.append(LABEL_TO_ID[seq.labels[word_id]]); seen_word.add(word_id); covered[sequence_index].add(word_id)
            chunks.append(Chunk(input_ids=encoded["input_ids"][batch_index], attention_mask=encoded["attention_mask"][batch_index],
                                labels=labels, sequence_index=sequence_index, word_ids=word_ids))
    missing = {i: set(range(len(seq.tokens)))-covered[i] for i, seq in enumerate(sequences)}
    missing = {i: ids for i, ids in missing.items() if ids}
    # A few fast tokenizers can omit word_ids at an overflow boundary.  Never
    # silently drop those words: encode each omitted word as a standalone
    # overflow-safe fallback chunk, preserving its original global word id.
    # This is rare and is logged by the caller through the chunk count.
    for sequence_index, word_ids_missing in missing.items():
        seq = sequences[sequence_index]
        for original_word_id in sorted(word_ids_missing):
            encoded = tokenizer([seq.tokens[original_word_id].text], is_split_into_words=True,
                                truncation=True, max_length=max_length, stride=stride,
                                return_overflowing_tokens=True, return_attention_mask=True)
            for batch_index in range(len(encoded["input_ids"])):
                local_word_ids = encoded.word_ids(batch_index=batch_index)
                seen_word, labels, remapped_ids = set(), [], []
                for local_word_id in local_word_ids:
                    global_word_id = original_word_id if local_word_id is not None else None
                    remapped_ids.append(global_word_id)
                    if global_word_id is None or global_word_id in seen_word:
                        labels.append(-100)
                    else:
                        labels.append(LABEL_TO_ID[seq.labels[global_word_id]])
                        seen_word.add(global_word_id)
                        covered[sequence_index].add(global_word_id)
                chunks.append(Chunk(input_ids=encoded["input_ids"][batch_index], attention_mask=encoded["attention_mask"][batch_index],
                                    labels=labels, sequence_index=sequence_index, word_ids=remapped_ids))
    still_missing = {i: set(range(len(seq.tokens)))-covered[i] for i, seq in enumerate(sequences)}
    still_missing = {i: ids for i, ids in still_missing.items() if ids}
    if still_missing:
        # Known BCMS WordPiece edge case: zero-width Unicode direction/spacing
        # marks produce no subword and therefore no word_id.  They have no
        # textual model representation.  O-only marks are safely omitted and
        # reconstructed as O; a labelled word may never be omitted silently.
        fatal = {
            sequence_index: {
                word_id
                for word_id in ids
                if not _is_omittable_format_token(
                    sequences[sequence_index].tokens[word_id].text,
                    sequences[sequence_index].labels[word_id],
                )
            }
            for sequence_index, ids in still_missing.items()
        }
        fatal = {sequence_index: ids for sequence_index, ids in fatal.items() if ids}
        if fatal:
            raise ValueError(f"Tokenizer dropped non-omittable words after fallback: {fatal}")
    return chunks


class TorchChunkDataset:
    def __init__(self, chunks: list[Chunk]): self.chunks = chunks
    def __len__(self): return len(self.chunks)
    def __getitem__(self, index: int):
        import torch
        value = self.chunks[index]
        return {"input_ids": torch.tensor(value.input_ids), "attention_mask": torch.tensor(value.attention_mask),
                "labels": torch.tensor(value.labels), "chunk_index": torch.tensor(index)}


def collator(features: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    # Dynamic padding to the longest sequence in this batch; labels use -100.
    keys = ["input_ids", "attention_mask", "labels"]
    maximum = max(len(item["input_ids"]) for item in features)
    output = {"chunk_index": torch.stack([item["chunk_index"] for item in features])}
    for key in keys:
        fill = -100 if key == "labels" else 0
        output[key] = torch.stack([torch.nn.functional.pad(item[key], (0, maximum-len(item[key])), value=fill) for item in features])
    return output


def chunks_to_word_labels(
    sequences: list[SequenceExample], chunks: list[Chunk], predicted_ids: list[np.ndarray]
) -> list[list[str]]:
    word_predictions: dict[int, dict[int, str]] = defaultdict(dict)
    for chunk, row in zip(chunks, predicted_ids):
        first = set()
        for pos, word_id in enumerate(chunk.word_ids):
            if word_id is not None and word_id not in first and word_id not in word_predictions[chunk.sequence_index]:
                word_predictions[chunk.sequence_index][word_id] = ID_TO_LABEL[int(row[pos])]
                first.add(word_id)
    # Tokenizers can intentionally omit zero-width Unicode marks.  They are
    # unlabelled O tokens; defaulting them to O preserves every source offset
    # and cannot create or remove an entity.
    return [[word_predictions[i].get(j, "O") for j in range(len(sequence.tokens))]
            for i, sequence in enumerate(sequences)]


def chunks_to_entities(sequences: list[SequenceExample], chunks: list[Chunk], predicted_ids: list[np.ndarray]) -> set[tuple[str, str, int, int, str]]:
    return sequences_to_entities(sequences, chunks_to_word_labels(sequences, chunks, predicted_ids))
