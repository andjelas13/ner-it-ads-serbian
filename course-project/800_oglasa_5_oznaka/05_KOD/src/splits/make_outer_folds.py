"""Strict group-aware multilabel split implementation used by every rerun package."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.common.constants import LABELS
from src.data.prepare_ner_dataset import annotation_label


def presence_matrix(ads: list[dict[str, Any]]) -> np.ndarray:
    matrix = np.zeros((len(ads), len(LABELS)), dtype=np.int8)
    label_index = {label: index for index, label in enumerate(LABELS)}
    for row, ad in enumerate(ads):
        for annotation in ad.get("annotations", []):
            label = annotation_label(annotation)
            if label in label_index:
                matrix[row, label_index[label]] = 1
    return matrix


def _id_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _exact_body_groups(ads: Iterable[dict[str, Any]]) -> list[list[str]]:
    by_body: dict[str, list[str]] = defaultdict(list)
    singletons: list[list[str]] = []
    for ad in ads:
        ad_id = str(ad["id"])
        body = ad.get("body", "")
        if isinstance(body, str) and body:
            by_body[body].append(ad_id)
        else:
            singletons.append([ad_id])
    groups = singletons + list(by_body.values())
    groups = [sorted(group, key=_id_key) for group in groups]
    return sorted(groups, key=lambda group: _id_key(group[0]))


def _fold_cost(
    counts: np.ndarray,
    target_size: int,
    total_counts: np.ndarray,
    total_ads: int,
) -> float:
    expected = total_counts * (target_size / total_ads)
    return float((np.square(counts - expected) / np.maximum(expected, 1.0)).sum())


def _group_folds(
    groups: list[list[str]],
    presence: dict[str, np.ndarray],
    n_splits: int,
    seed: int,
) -> list[list[str]]:
    try:
        from iterstrat import __version__ as iterstrat_version
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    except ImportError as exc:
        raise RuntimeError(
            "iterative-stratification is required to generate a new final split. "
            "Install requirements.txt first."
        ) from exc
    group_or = np.stack([
        np.maximum.reduce([presence[ad_id] for ad_id in group])
        for group in groups
    ])
    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    assignment = np.full(len(groups), -1, dtype=np.int16)
    for fold, (_, indices) in enumerate(
        splitter.split(np.zeros(len(groups)), group_or)
    ):
        assignment[indices] = fold
    if np.any(assignment < 0):
        raise RuntimeError("MultilabelStratifiedKFold left an unassigned group")

    group_sizes = np.asarray([len(group) for group in groups], dtype=np.int16)
    contributions = np.stack([
        np.sum([presence[ad_id] for ad_id in group], axis=0)
        for group in groups
    ]).astype(np.float64)
    total_ads = int(group_sizes.sum())
    targets = np.asarray([
        total_ads // n_splits + (fold < total_ads % n_splits)
        for fold in range(n_splits)
    ], dtype=np.int32)
    sizes = np.asarray([
        int(group_sizes[assignment == fold].sum()) for fold in range(n_splits)
    ], dtype=np.int32)
    counts = np.stack([
        contributions[assignment == fold].sum(axis=0)
        for fold in range(n_splits)
    ])
    total_counts = contributions.sum(axis=0)

    while not np.array_equal(sizes, targets):
        best: tuple[float, int, int, int] | None = None
        for source in np.where(sizes > targets)[0]:
            group_indices = np.where(assignment == source)[0]
            for destination in np.where(sizes < targets)[0]:
                capacity = int(targets[destination] - sizes[destination])
                for group_index in group_indices:
                    weight = int(group_sizes[group_index])
                    if weight > capacity:
                        continue
                    value = contributions[group_index]
                    before = _fold_cost(counts[source], int(targets[source]), total_counts, total_ads)
                    before += _fold_cost(counts[destination], int(targets[destination]), total_counts, total_ads)
                    after = _fold_cost(counts[source] - value, int(targets[source]), total_counts, total_ads)
                    after += _fold_cost(counts[destination] + value, int(targets[destination]), total_counts, total_ads)
                    candidate = (after - before, int(group_index), int(source), int(destination))
                    if best is None or candidate < best:
                        best = candidate
        if best is None:
            raise RuntimeError(
                f"Cannot repair fold sizes {sizes.tolist()} to {targets.tolist()}"
            )
        _, group_index, source, destination = best
        value = contributions[group_index]
        weight = int(group_sizes[group_index])
        assignment[group_index] = destination
        sizes[source] -= weight
        sizes[destination] += weight
        counts[source] -= value
        counts[destination] += value

    rng = np.random.RandomState(seed + 50000)
    weight_buckets = {
        int(weight): np.where(group_sizes == weight)[0]
        for weight in np.unique(group_sizes)
    }
    for _ in range(20_000):
        left = int(rng.randint(len(groups)))
        bucket = weight_buckets[int(group_sizes[left])]
        right = int(bucket[rng.randint(len(bucket))])
        left_fold, right_fold = int(assignment[left]), int(assignment[right])
        if left_fold == right_fold:
            continue
        left_value, right_value = contributions[left], contributions[right]
        before = _fold_cost(counts[left_fold], int(targets[left_fold]), total_counts, total_ads)
        before += _fold_cost(counts[right_fold], int(targets[right_fold]), total_counts, total_ads)
        after_left = counts[left_fold] - left_value + right_value
        after_right = counts[right_fold] - right_value + left_value
        after = _fold_cost(after_left, int(targets[left_fold]), total_counts, total_ads)
        after += _fold_cost(after_right, int(targets[right_fold]), total_counts, total_ads)
        if after + 1e-12 < before:
            assignment[left], assignment[right] = right_fold, left_fold
            counts[left_fold], counts[right_fold] = after_left, after_right

    folds = []
    for fold in range(n_splits):
        test_ids = [
            ad_id
            for group_index in np.where(assignment == fold)[0]
            for ad_id in groups[int(group_index)]
        ]
        folds.append(sorted(test_ids, key=_id_key))
    return folds


def make_folds(
    ads: list[dict[str, Any]], n_splits: int = 10, seed: int = 42
) -> list[list[str]]:
    ids = [str(ad["id"]) for ad in ads]
    rows = presence_matrix(ads)
    presence = {ad_id: rows[index] for index, ad_id in enumerate(ids)}
    return _group_folds(_exact_body_groups(ads), presence, n_splits, seed)


def create_split_payload(
    ads: list[dict[str, Any]], n_splits: int = 10, seed: int = 42
) -> dict[str, Any]:
    try:
        from iterstrat import __version__ as iterstrat_version
    except ImportError as exc:
        raise RuntimeError("iterative-stratification is required to generate a new final split.") from exc
    folds = make_folds(ads, n_splits=n_splits, seed=seed)
    all_ids = {str(ad["id"]) for ad in ads}
    result = {
        "seed": seed,
        "n_splits": n_splits,
        "unit": "whole_ad",
        "labels": LABELS,
        "splitter": "MultilabelStratifiedKFold + deterministic exact-size group repair",
        "iterative_stratification_version": iterstrat_version,
        "fallback_allowed": False,
        "grouping": "non-empty byte-exact body",
        "folds": [
            {
                "fold": index,
                "test_ad_ids": fold,
                "train_ad_ids": sorted(all_ids - set(fold), key=_id_key),
            }
            for index, fold in enumerate(folds)
        ],
    }
    validate_split(result, all_ids)
    _validate_duplicate_groups(result, _exact_body_groups(ads))
    return result


def _validate_duplicate_groups(value: dict[str, Any], groups: list[list[str]]) -> None:
    for fold in value["folds"]:
        test = set(fold["test_ad_ids"])
        for group in groups:
            if len(group) > 1 and test & set(group) and not set(group) <= test:
                raise ValueError(f"Exact-body group was split: {group}")


def validate_split(split: dict[str, Any], all_ids: set[str]) -> None:
    held_out: list[str] = []
    for fold in split["folds"]:
        train = set(fold["train_ad_ids"])
        test = set(fold["test_ad_ids"])
        if train & test or train | test != all_ids:
            raise ValueError(f"Invalid train/test partition in fold {fold['fold']}")
        held_out.extend(test)
    if Counter(held_out) != Counter(all_ids):
        raise ValueError("Each ad must be outer-test exactly once")


def save_split(payload: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_fold(path: str | Path, fold_id: int) -> tuple[set[str], set[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    row = next(item for item in payload["folds"] if int(item["fold"]) == fold_id)
    return set(row["train_ad_ids"]), set(row["test_ad_ids"])


def main() -> int:
    import argparse
    import csv
    from src.data.prepare_ner_dataset import examples, load_json, validate_dataset

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/processed/ner_dataset.json")
    parser.add_argument("--output", type=Path, default=root / "splits/outer_folds.json")
    parser.add_argument("--csv", type=Path, default=root / "splits/outer_folds.csv")
    parser.add_argument("--summary", type=Path, default=root / "splits/split_summary.json")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-ads", type=int, default=800)
    args = parser.parse_args()

    data = load_json(args.data)
    validate_dataset(data, args.expected_ads)
    payload = create_split_payload(examples(data), args.folds, args.seed)
    save_split(payload, args.output)
    rows = [{"ad_id": ad_id, "outer_fold": fold["fold"], "role": "test"} for fold in payload["folds"] for ad_id in fold["test_ad_ids"]]
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ad_id", "outer_fold", "role"])
        writer.writeheader(); writer.writerows(rows)
    summary = {"n_folds": args.folds, "fold_sizes": [len(fold["test_ad_ids"]) for fold in payload["folds"]], "splitter": payload["splitter"], "grouping": payload["grouping"], "fallback_allowed": False}
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
