"""Single entity-level scorer used by both CPU and Kaggle packages."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from src.common.constants import LABELS

Entity = tuple[str, str, int, int, str]


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def strict_scores(gold: Iterable[Entity], predicted: Iterable[Entity]) -> dict[str, Any]:
    gold_set, pred_set = set(gold), set(predicted)
    result: dict[str, Any] = _prf(len(gold_set & pred_set), len(pred_set - gold_set), len(gold_set - pred_set))
    per_tag: dict[str, dict[str, float | int]] = {}
    f1s, supports = [], []
    for tag in LABELS:
        g = {e for e in gold_set if e[-1] == tag}
        p = {e for e in pred_set if e[-1] == tag}
        row = _prf(len(g & p), len(p - g), len(g - p))
        row["support"] = len(g)
        per_tag[tag] = row
        if g:
            f1s.append(float(row["f1"]))
            supports.append(len(g))
    result["per_tag"] = per_tag
    result["macro_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    result["weighted_f1"] = sum(v * s for v, s in zip(f1s, supports)) / sum(supports) if supports else 0.0
    return result


def boundary_scores(gold: Iterable[Entity], predicted: Iterable[Entity]) -> dict[str, float | int]:
    g = {(a, f, s, e) for a, f, s, e, _ in gold}
    p = {(a, f, s, e) for a, f, s, e, _ in predicted}
    return _prf(len(g & p), len(p - g), len(g - p))


def label_confusion(gold: Iterable[Entity], predicted: Iterable[Entity]) -> dict[str, dict[str, int]]:
    gold_at = {(a, f, s, e): tag for a, f, s, e, tag in gold}
    pred_at = {(a, f, s, e): tag for a, f, s, e, tag in predicted}
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for key in gold_at.keys() & pred_at.keys():
        if gold_at[key] != pred_at[key]:
            matrix[gold_at[key]][pred_at[key]] += 1
    return {key: dict(value) for key, value in matrix.items()}


def _overlap(a: Entity, b: Entity) -> bool:
    return a[0] == b[0] and a[1] == b[1] and a[-1] == b[-1] and max(a[2], b[2]) < min(a[3], b[3])


def _maximum_cardinality_matches(gold: list[Entity], predicted: list[Entity]) -> int:
    """Exact maximum bipartite matching for valid same-label overlaps."""
    adjacency = [[j for j, right in enumerate(predicted) if _overlap(left, right)] for left in gold]
    matched_gold_for_prediction: dict[int, int] = {}

    def augment(gold_index: int, seen: set[int]) -> bool:
        for predicted_index in adjacency[gold_index]:
            if predicted_index in seen:
                continue
            seen.add(predicted_index)
            previous = matched_gold_for_prediction.get(predicted_index)
            if previous is None or augment(previous, seen):
                matched_gold_for_prediction[predicted_index] = gold_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(gold)))


def relaxed_overlap_scores(gold: Iterable[Entity], predicted: Iterable[Entity]) -> dict[str, float | int]:
    """Maximum-cardinality one-to-one matching of same-label overlapping spans."""
    g, p = set(gold), set(predicted)
    # Entities in different ad, field or label can never overlap. Partitioning
    # therefore preserves the exact maximum-cardinality result while avoiding
    # an O(all_entities^2) adjacency matrix for pooled 10-fold OOF scoring.
    by_gold: dict[tuple[str, str, str], list[Entity]] = defaultdict(list)
    by_predicted: dict[tuple[str, str, str], list[Entity]] = defaultdict(list)
    for item in g: by_gold[(item[0], item[1], item[4])].append(item)
    for item in p: by_predicted[(item[0], item[1], item[4])].append(item)
    matches = sum(_maximum_cardinality_matches(by_gold[key], by_predicted[key]) for key in by_gold.keys() | by_predicted.keys())
    return _prf(matches, len(p) - matches, len(g) - matches)


def semantic_token_scores(
    gold_sequences: Iterable[Iterable[str]], predicted_sequences: Iterable[Iterable[str]]
) -> dict[str, Any]:
    """Token classification scores after collapsing B-X/I-X to semantic X."""
    gold_flat: list[str] = []
    predicted_flat: list[str] = []
    for gold_labels, predicted_labels in zip(gold_sequences, predicted_sequences):
        gold_row, predicted_row = list(gold_labels), list(predicted_labels)
        if len(gold_row) != len(predicted_row):
            raise ValueError("Gold and predicted BIO rows must have identical token counts")
        gold_flat.extend(label[2:] if label != "O" else "O" for label in gold_row)
        predicted_flat.extend(label[2:] if label != "O" else "O" for label in predicted_row)

    per_tag: dict[str, dict[str, float | int]] = {}
    total_tp = total_fp = total_fn = 0
    for tag in LABELS:
        tp = sum(gold == tag and predicted == tag for gold, predicted in zip(gold_flat, predicted_flat))
        fp = sum(gold != tag and predicted == tag for gold, predicted in zip(gold_flat, predicted_flat))
        fn = sum(gold == tag and predicted != tag for gold, predicted in zip(gold_flat, predicted_flat))
        row = _prf(tp, fp, fn)
        row["support"] = sum(gold == tag for gold in gold_flat)
        per_tag[tag] = row
        total_tp += tp; total_fp += fp; total_fn += fn
    result: dict[str, Any] = _prf(total_tp, total_fp, total_fn)
    result["macro_f1"] = sum(float(per_tag[tag]["f1"]) for tag in LABELS) / len(LABELS)
    result["per_tag"] = per_tag
    result["token_count"] = len(gold_flat)
    return result


def error_categories(gold: Iterable[Entity], predicted: Iterable[Entity]) -> dict[str, int]:
    """Diagnostic categories; exact TP are excluded from errors."""
    g, p = set(gold), set(predicted)
    matched = g & p
    remaining_g, remaining_p = list(g - matched), list(p - matched)
    counts: Counter[str] = Counter()
    used_g, used_p = set(), set()
    # Exact boundary, wrong tag.
    for i, left in enumerate(remaining_g):
        for j, right in enumerate(remaining_p):
            if i not in used_g and j not in used_p and left[:4] == right[:4]:
                counts["label_confusion"] += 1; used_g.add(i); used_p.add(j)
    # Same tag, overlap: boundary error.
    for i, left in enumerate(remaining_g):
        for j, right in enumerate(remaining_p):
            if i not in used_g and j not in used_p and _overlap(left, right):
                counts["boundary_error"] += 1; used_g.add(i); used_p.add(j)
    counts["missed_gold"] = len(remaining_g) - len(used_g)
    counts["spurious_prediction"] = len(remaining_p) - len(used_p)
    return dict(counts)


def full_report(
    gold: Iterable[Entity],
    predicted: Iterable[Entity],
    gold_bio: Iterable[Iterable[str]] | None = None,
    predicted_bio: Iterable[Iterable[str]] | None = None,
) -> dict[str, Any]:
    gold, predicted = list(gold), list(predicted)
    report = strict_scores(gold, predicted)
    report["boundary_only"] = boundary_scores(gold, predicted)
    report["relaxed_overlap"] = relaxed_overlap_scores(gold, predicted)
    report["label_confusion"] = label_confusion(gold, predicted)
    report["error_categories"] = error_categories(gold, predicted)
    if gold_bio is not None and predicted_bio is not None:
        report["semantic_token"] = semantic_token_scores(gold_bio, predicted_bio)
    return report
