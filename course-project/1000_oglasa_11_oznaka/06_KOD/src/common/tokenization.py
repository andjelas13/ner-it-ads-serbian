"""Gold-independent domain tokenizer and lossless char-span/BIO conversion."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from src.data.prepare_ner_dataset import annotation_field, annotation_label


TOKENIZER_VERSION = "it-domain-text-only-v2"


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    """Tokenize solely from raw text, without inspecting gold annotations.

    Punctuation/symbols are standalone tokens.  Alphanumeric runs are split at
    letter/digit transitions and Unicode camel/acronym boundaries.  These are
    deterministic domain rules for flattened IT specifications such as
    ``i5-45708gb``, ``GPUGeForce`` and ``ekranaIPS``.
    """
    output: list[Token] = []
    forced_boundaries: set[int] = set()
    # Common concatenated storage token: ``skladistenjeeMCP``.  The boundary
    # before eMCP is visible in the raw casing and therefore label-free.
    for match in re.finditer(r"eMCP", text):
        forced_boundaries.update((match.start(), match.end()))
    # Serbian inflection is sometimes glued to the technical unit (bitna,
    # bitnom...).  Keep the objective unit separate from its grammatical tail.
    for match in re.finditer(r"(?i)bit(?=(?:na|ni|no|ne|nu|nim|nom|nog|nih|noj|nu)\b)", text):
        forced_boundaries.add(match.end())
    # Flattened catalogue titles may omit whitespace between a four-digit CPU
    # model and the next capacity, or between DDR generation and capacity.
    for pattern in (r"(?i)i[3579]-\d{4}(?=\d)", r"(?i)DDR\d(?=\d)"):
        for match in re.finditer(pattern, text):
            forced_boundaries.add(match.end())
    start = 0

    def kind(character: str) -> str:
        category = unicodedata.category(character)
        if category.startswith("L"):
            if character.islower():
                return "lower"
            if character.isupper():
                return "upper"
            return "letter"
        if category.startswith("N"):
            return "number"
        return "other"

    def flush(end: int) -> None:
        nonlocal start
        if start < end:
            output.append(Token(text[start:end], start, end))

    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            start = index
            continue
        current_kind = kind(text[index])
        if current_kind == "other":
            output.append(Token(text[index], index, index + 1))
            index += 1
            start = index
            continue
        start = index
        index += 1
        while index < len(text) and not text[index].isspace() and kind(text[index]) != "other":
            previous_kind = kind(text[index - 1])
            next_kind = kind(text[index])
            after_kind = kind(text[index + 1]) if index + 1 < len(text) else "other"
            letter_number = {previous_kind, next_kind} <= {"lower", "upper", "letter", "number"} and (
                previous_kind == "number" or next_kind == "number"
            ) and previous_kind != next_kind
            camel = previous_kind == "lower" and next_kind == "upper"
            acronym = previous_kind == "upper" and next_kind == "upper" and after_kind == "lower"
            if letter_number or camel or acronym or index in forced_boundaries:
                flush(index)
                start = index
            index += 1
        flush(index)
        start = index
    return output


def bio_valid(labels: Iterable[str]) -> bool:
    previous = "O"
    for label in labels:
        if label == "O":
            previous = label
            continue
        if not (label.startswith("B-") or label.startswith("I-")):
            return False
        kind = label[2:]
        if label.startswith("I-") and (previous == "O" or previous[2:] != kind):
            return False
        previous = label
    return True


def repair_bio(labels: Iterable[str]) -> list[str]:
    repaired: list[str] = []
    previous = "O"
    for label in labels:
        if label.startswith("I-") and (previous == "O" or previous[2:] != label[2:]):
            label = "B-" + label[2:]
        if label != "O" and not (label.startswith("B-") or label.startswith("I-")):
            label = "O"
        repaired.append(label)
        previous = label
    return repaired


def invalid_bio_diagnostics(label_sequences: Iterable[Iterable[str]]) -> dict[str, float | int]:
    """Count invalid BIO transitions before the deterministic repair step."""
    invalid = 0
    total = 0
    for labels in label_sequences:
        previous = "O"
        for label in labels:
            total += 1
            malformed = label != "O" and not (label.startswith("B-") or label.startswith("I-"))
            invalid_i = label.startswith("I-") and (previous == "O" or previous[2:] != label[2:])
            if malformed or invalid_i:
                invalid += 1
            previous = label if not malformed else "O"
    return {
        "invalid_bio_transition_count": invalid,
        "token_count": total,
        "invalid_bio_transition_rate": invalid / total if total else 0.0,
    }


def spans_to_bio(text: str, annotations: list[dict], field: str) -> tuple[list[Token], list[str]]:
    relevant = [a for a in annotations if annotation_field(a) == field]
    tokens = tokenize(text)
    labels = ["O"] * len(tokens)
    starts = {token.start: i for i, token in enumerate(tokens)}
    ends = {token.end: i for i, token in enumerate(tokens)}
    for ann in sorted(relevant, key=lambda a: int(a["start"])):
        start, end, tag = int(ann["start"]), int(ann["end"]), annotation_label(ann)
        if start not in starts or end not in ends:
            raise ValueError(f"{field}: span [{start},{end}) does not align with tokenizer: {text[start:end]!r}")
        left, right = starts[start], ends[end]
        if any(labels[i] != "O" for i in range(left, right + 1)):
            raise ValueError(f"{field}: overlapping BIO assignment [{start},{end})")
        labels[left] = "B-" + tag
        for i in range(left + 1, right + 1):
            labels[i] = "I-" + tag
    return tokens, labels


def bio_to_spans(tokens: list[Token], labels: Iterable[str]) -> list[tuple[int, int, str]]:
    labels = repair_bio(labels)
    output: list[tuple[int, int, str]] = []
    current_start: int | None = None
    current_tag: str | None = None
    previous_end: int | None = None
    for token, label in zip(tokens, labels):
        prefix, tag = (label.split("-", 1) if label != "O" else ("O", None))
        continues = prefix == "I" and tag == current_tag and previous_end is not None
        if not continues and current_start is not None:
            output.append((current_start, previous_end, current_tag))  # type: ignore[arg-type]
            current_start, current_tag = None, None
        if prefix != "O" and not continues:
            current_start, current_tag = token.start, tag
        previous_end = token.end
    if current_start is not None:
        output.append((current_start, previous_end, current_tag))  # type: ignore[arg-type]
    return output
