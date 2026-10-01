"""Shared label-free utilities for report analysis scripts."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common.constants import LABELS
from src.common.tokenization import tokenize
from src.data.prepare_ner_dataset import annotation_field, annotation_label, examples, load_json

Entity = tuple[str, str, int, int, str]


def entities(path: Path) -> tuple[list[dict[str, Any]], set[Entity]]:
    """Ucitaj spanove i zadrzi samo oznake iz aktivne sheme (``LABELS``).

    Kalibracioni fajlovi su anotirani sirom shemom od jedanaest oznaka, iz prve
    verzije projekta, a ovaj eksperiment koristi pet.  Bez ovog uslova strict i
    relaxed slaganje i Cohenova kappa racunali bi se i preko sest oznaka koje
    vise ne ulaze u zadatak.  ``semantic()`` je vec filtriran, jer iterira kroz
    ``LABELS`` -- zbog toga se ranije samo semanticka mera poklapala sa
    izvestajem, koji brojeve navodi "nakon zadrzavanja samo pet ciljnih oznaka".
    """
    ads = examples(load_json(path))
    return ads, {
        (str(ad["id"]), annotation_field(ann), int(ann["start"]), int(ann["end"]), annotation_label(ann))
        for ad in ads for ann in ad.get("annotations", [])
        if annotation_label(ann) in LABELS
    }


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def strict(gold: set[Entity], predicted: set[Entity], label: str | None = None) -> dict[str, float | int]:
    if label:
        gold = {item for item in gold if item[-1] == label}; predicted = {item for item in predicted if item[-1] == label}
    return prf(len(gold & predicted), len(predicted - gold), len(gold - predicted))


def _overlap(left: Entity, right: Entity) -> bool:
    return left[0] == right[0] and left[1] == right[1] and left[-1] == right[-1] and max(left[2], right[2]) < min(left[3], right[3])


def relaxed(gold: set[Entity], predicted: set[Entity], label: str | None = None) -> dict[str, float | int]:
    if label:
        gold = {item for item in gold if item[-1] == label}; predicted = {item for item in predicted if item[-1] == label}
    left, right = sorted(gold), sorted(predicted)
    adjacent = [[j for j, candidate in enumerate(right) if _overlap(item, candidate)] for item in left]
    matched: dict[int, int] = {}
    def augment(index: int, seen: set[int]) -> bool:
        for candidate in adjacent[index]:
            if candidate in seen: continue
            seen.add(candidate)
            if candidate not in matched or augment(matched[candidate], seen):
                matched[candidate] = index; return True
        return False
    tp = sum(augment(index, set()) for index in range(len(left)))
    return prf(tp, len(right) - tp, len(left) - tp)


def semantic_grid(ads: list[dict[str, Any]], spans: set[Entity]) -> list[str]:
    by_sequence: dict[tuple[str, str], list[Entity]] = defaultdict(list)
    for span in spans: by_sequence[span[:2]].append(span)
    output: list[str] = []
    for ad in ads:
        for field in ("title", "body"):
            text = str(ad.get(field, "")); relevant = by_sequence[(str(ad["id"]), field)]
            for token in tokenize(text):
                matches = [span for span in relevant if span[2] < token.end and token.start < span[3]]
                output.append(max(matches, key=lambda span: min(span[3], token.end) - max(span[2], token.start))[-1] if matches else "O")
    return output


def semantic(gold_ads: list[dict[str, Any]], gold: set[Entity], predicted: set[Entity]) -> dict[str, Any]:
    first, second = semantic_grid(gold_ads, gold), semantic_grid(gold_ads, predicted)
    total_tp = total_fp = total_fn = 0; per_label: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        tp = sum(a == label and b == label for a, b in zip(first, second))
        fp = sum(a != label and b == label for a, b in zip(first, second))
        fn = sum(a == label and b != label for a, b in zip(first, second))
        per_label[label] = prf(tp, fp, fn); total_tp += tp; total_fp += fp; total_fn += fn
    output: dict[str, Any] = prf(total_tp, total_fp, total_fn)
    output["per_label"] = per_label; output["token_count"] = len(first)
    output["accuracy_with_O"] = sum(a == b for a, b in zip(first, second)) / len(first) if first else 0.0
    return output


def cohen_kappa(gold_ads: list[dict[str, Any]], first_spans: set[Entity], second_spans: set[Entity]) -> float:
    first, second = semantic_grid(gold_ads, first_spans), semantic_grid(gold_ads, second_spans)
    if not first: return 0.0
    classes = set(first) | set(second)
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    expected = sum(first.count(label) * second.count(label) for label in classes) / (len(first) ** 2)
    return (observed - expected) / (1 - expected) if expected != 1 else 1.0
