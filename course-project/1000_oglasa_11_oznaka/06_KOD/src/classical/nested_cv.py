"""Nested-CV entry point for NB, SVM, XGBoost and CRF."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from src.classical.models import GRIDS, IndependentTokenModel, make_model, make_sequences, sequences_to_entities
from src.data.prepare_ner_dataset import entity_set, examples, load_json
from src.features.token_features import sequence_features
from src.common.constants import BIO_LABELS, LABEL_TO_ID
from src.features.sparse_backend import to_csr, vocabulary
from src.evaluation.metrics import full_report, strict_scores
from src.common.protocol import run_identity
from src.splits.make_outer_folds import load_fold
from src.common.tokenization import invalid_bio_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data/processed/ner_dataset.json"
DEFAULT_SPLITS = PROJECT_ROOT / "splits/outer_folds.json"


def _ads_by_ids(ads: list[dict[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
    return [ad for ad in ads if str(ad["id"]) in ids]


def _load_bundled_inner_split(
    outer_train: list[dict[str, Any]], outer_fold: int, inner_folds: int,
    split_directory: Path,
) -> dict[str, Any]:
    path = split_directory / f"inner_outer_{outer_fold:02d}_{inner_folds}fold.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Required canonical inner split is missing: {path}. "
            "Dynamic/fallback split generation is intentionally disabled."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {str(ad["id"]) for ad in outer_train}
    if int(payload.get("n_splits", -1)) != inner_folds:
        raise ValueError(f"Wrong n_splits in {path}")
    if payload.get("fallback_allowed") is not False:
        raise ValueError(f"Non-canonical split metadata in {path}")
    held_out: list[str] = []
    for fold in payload["folds"]:
        train = set(fold["train_ad_ids"])
        valid = set(fold["test_ad_ids"])
        if train & valid or train | valid != expected:
            raise ValueError(f"Invalid bundled inner partition in {path}")
        held_out.extend(valid)
    if Counter(held_out) != Counter(expected):
        raise ValueError(f"Bundled inner split does not cover outer-train once: {path}")
    return payload


def _score_cached_sequence_model(model_name: str, params: dict[str, Any], train_seq: list[Any],
                                 valid_seq: list[Any], gold: set[Any],
                                 runtime_options: dict[str, Any]) -> dict[str, Any]:
    fit_start = time.perf_counter()
    model = make_model(model_name, params, runtime_options).fit(train_seq)
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    prediction = sequences_to_entities(valid_seq, model.predict(valid_seq))
    predict_seconds = time.perf_counter() - predict_start
    return {"f1": float(strict_scores(gold, prediction)["f1"]),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "n_iter": None, "hit_iteration_limit": False}


def _build_sequence_cache(ads: list[dict[str, Any]]) -> tuple[dict[str, list[Any]], dict[str, set[Any]]]:
    """Compute deterministic, label-free feature dictionaries once per ad.

    Tokenization/BIO conversion is source-only and features are deterministic;
    caching them across folds cannot leak learned vocabulary or labels.  The
    train-only ``DictVectorizer.fit`` remains below in every inner split.
    """
    sequences: dict[str, list[Any]] = {}
    gold: dict[str, set[Any]] = {}
    for ad in ads:
        ad_id = str(ad["id"])
        rows = make_sequences([ad])
        for row in rows:
            row.cached_features = sequence_features(row.tokens, row.field)
        sequences[ad_id] = rows
        gold[ad_id] = entity_set({"examples": [ad]})
    return sequences, gold


def _cached_rows(cache: dict[str, list[Any]], ids: set[str]) -> list[Any]:
    def sort_key(ad_id: str) -> tuple[int, int | str]:
        return (0, int(ad_id)) if ad_id.isdigit() else (1, ad_id)
    return [row for ad_id in sorted(ids, key=sort_key) for row in cache[ad_id]]


def _prepare_independent_split(train_seq: list[Any], valid_seq: list[Any], gold: set[Any]) -> dict[str, Any]:
    """Fit the vocabulary once per inner split, not once per HP setting.

    NB's eight alpha/prior variants and SVM's six variants share exactly the
    same feature matrices.  This changes no data, no fold and no score; it
    removes the former 8x repeated tokenisation/vectorisation bottleneck.
    """
    train_features = [f for seq in train_seq for f in seq.cached_features]
    valid_features = [f for seq in valid_seq for f in seq.cached_features]
    # The vocabulary is learned only from inner-train. Direct CSR construction
    # avoids DictVectorizer's large temporary Python-object representation.
    vectorizer = vocabulary(train_features, min_df=1)
    x_train = to_csr(train_features, vectorizer)
    x_valid = to_csr(valid_features, vectorizer)
    return {"train_seq": train_seq, "valid_seq": valid_seq, "vectorizer": vectorizer,
            "x_train": x_train, "x_valid": x_valid,
            "y_train": np.fromiter((LABEL_TO_ID[label] for seq in train_seq for label in seq.labels),
                                   dtype=np.int16), "gold": gold}


def _score_prepared_independent(model_name: str, params: dict[str, Any], prepared: dict[str, Any],
                                runtime_options: dict[str, Any]) -> dict[str, Any]:
    adapter = IndependentTokenModel(model_name, params, runtime_options)
    model = adapter._build()
    fit_start = time.perf_counter()
    model.fit(prepared["x_train"], prepared["y_train"])
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    flat = [BIO_LABELS[int(value)] for value in model.predict(prepared["x_valid"]).tolist()]
    predict_seconds = time.perf_counter() - predict_start
    cursor, predicted_labels = 0, []
    for seq in prepared["valid_seq"]:
        predicted_labels.append(flat[cursor:cursor + len(seq.tokens)])
        cursor += len(seq.tokens)
    predicted = sequences_to_entities(prepared["valid_seq"], predicted_labels)
    # Inner HP selection requires only strict entity micro-F1.  Boundary,
    # relaxed-overlap and pairwise error diagnostics are intentionally kept
    # for the one final outer-test report, not repeated 40+ times here.
    n_iter_value = getattr(model, "n_iter_", None)
    if n_iter_value is not None:
        n_iter = int(np.max(np.asarray(n_iter_value)))
    else:
        n_iter = None
    max_iter = int(runtime_options.get("svm_max_iter", 5000))
    return {"f1": float(strict_scores(prepared["gold"], predicted)["f1"]),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "n_iter": n_iter,
            "hit_iteration_limit": bool(model_name == "svm" and n_iter is not None and n_iter >= max_iter)}


def tune(model_name: str, outer_train: list[dict[str, Any]], outer_fold: int, inner_folds: int,
         sequence_cache: dict[str, list[Any]], gold_cache: dict[str, set[Any]],
         runtime_options: dict[str, Any], hp_workers: int,
         split_directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_bundled_inner_split(outer_train, outer_fold, inner_folds, split_directory)
    rows = []
    best_params, best_score = None, -1.0
    keys = [json.dumps(params, sort_keys=True) for params in GRIDS[model_name]]
    per_params: dict[str, list[dict[str, Any] | None]] = {key: [None] * inner_folds for key in keys}
    if model_name != "crf":
        # Keep exactly one sparse inner split in memory.  A five-way cache is
        # faster on a workstation but can page heavily on Colab's CPU RAM.
        for inner_index, fold in enumerate(payload["folds"], 1):
            print(f"  preparing {model_name} outer={outer_fold} inner={inner_index}/{inner_folds}", flush=True)
            train_ids, valid_ids = set(fold["train_ad_ids"]), set(fold["test_ad_ids"])
            prepared = _prepare_independent_split(_cached_rows(sequence_cache, train_ids), _cached_rows(sequence_cache, valid_ids),
                                                  set().union(*(gold_cache[ad_id] for ad_id in valid_ids)))
            if model_name == "svm" and hp_workers > 1:
                with ThreadPoolExecutor(max_workers=min(hp_workers, len(GRIDS[model_name]))) as pool:
                    futures = {pool.submit(_score_prepared_independent, model_name, params, prepared, runtime_options): params
                               for params in GRIDS[model_name]}
                    for future in as_completed(futures):
                        params = futures[future]
                        per_params[json.dumps(params, sort_keys=True)][inner_index - 1] = future.result()
            else:
                for params in GRIDS[model_name]:
                    per_params[json.dumps(params, sort_keys=True)][inner_index - 1] = \
                        _score_prepared_independent(model_name, params, prepared, runtime_options)
    else:
        for inner_index, fold in enumerate(payload["folds"], 1):
            train_ids, valid_ids = set(fold["train_ad_ids"]), set(fold["test_ad_ids"])
            train_seq, valid_seq = _cached_rows(sequence_cache, train_ids), _cached_rows(sequence_cache, valid_ids)
            gold = set().union(*(gold_cache[ad_id] for ad_id in valid_ids))
            for params in GRIDS[model_name]:
                per_params[json.dumps(params, sort_keys=True)][inner_index - 1] = \
                    _score_cached_sequence_model(model_name, params, train_seq, valid_seq, gold, runtime_options)
    for params in GRIDS[model_name]:
        diagnostics = per_params[json.dumps(params, sort_keys=True)]
        assert all(value is not None for value in diagnostics)
        completed = [value for value in diagnostics if value is not None]
        values = [float(value["f1"]) for value in completed]
        mean = float(np.mean(values))
        rows.append({"params": params, "inner_entity_f1": values, "mean_inner_entity_f1": mean,
                     "inner_fit_seconds": [value["fit_seconds"] for value in completed],
                     "inner_predict_seconds": [value["predict_seconds"] for value in completed],
                     "inner_n_iter": [value["n_iter"] for value in completed],
                     "inner_hit_iteration_limit": [value["hit_iteration_limit"] for value in completed]})
        print(f"  {model_name} outer={outer_fold} HP={params} inner_F1={mean:.4f}", flush=True)
        if mean > best_score:
            best_params, best_score = params, mean
    assert best_params is not None
    return best_params, rows


def run_fold(model_name: str, dataset_path: Path, split_path: Path, output_dir: Path, fold: int, inner_folds: int,
             cached: tuple[list[dict[str, Any]], dict[str, list[Any]], dict[str, set[Any]]] | None = None,
             runtime_options: dict[str, Any] | None = None, hp_workers: int = 1) -> dict[str, Any]:
    start = time.perf_counter()
    runtime_options = runtime_options or {}
    if cached is None:
        data = load_json(dataset_path); ads = examples(data); sequence_cache, gold_cache = _build_sequence_cache(ads)
    else:
        ads, sequence_cache, gold_cache = cached
    train_ids, test_ids = load_fold(split_path, fold)
    outer_train, outer_test = _ads_by_ids(ads, train_ids), _ads_by_ids(ads, test_ids)
    tuning_start = time.perf_counter()
    params, tuning = tune(model_name, outer_train, fold, inner_folds, sequence_cache, gold_cache,
                          runtime_options, hp_workers, split_path.parent)
    tuning_seconds = time.perf_counter() - tuning_start
    train_seq, test_seq = _cached_rows(sequence_cache, train_ids), _cached_rows(sequence_cache, test_ids)
    gold = set().union(*(gold_cache[ad_id] for ad_id in test_ids))
    if model_name == "crf":
        final_fit_start = time.perf_counter()
        model = make_model(model_name, params, runtime_options).fit(train_seq)
        final_fit_seconds = time.perf_counter() - final_fit_start
        final_predict_start = time.perf_counter()
        labels = model.predict(test_seq)
        predicted = sequences_to_entities(test_seq, labels)
        final_predict_seconds = time.perf_counter() - final_predict_start
    else:
        # Same leakage-safe direct CSR path as inner tuning.  In particular,
        # do not return to DictVectorizer only for the final outer fit.
        prepared = _prepare_independent_split(train_seq, test_seq, gold)
        adapter = IndependentTokenModel(model_name, params, runtime_options)
        model = adapter._build()
        final_fit_start = time.perf_counter()
        model.fit(prepared["x_train"], prepared["y_train"])
        final_fit_seconds = time.perf_counter() - final_fit_start
        final_predict_start = time.perf_counter()
        flat = [BIO_LABELS[int(value)] for value in model.predict(prepared["x_valid"]).tolist()]
        final_predict_seconds = time.perf_counter() - final_predict_start
        cursor, labels = 0, []
        for sequence in test_seq:
            labels.append(flat[cursor:cursor + len(sequence.tokens)]); cursor += len(sequence.tokens)
        predicted = sequences_to_entities(test_seq, labels)
    gold_labels = [sequence.labels for sequence in test_seq]
    report = full_report(gold, predicted, gold_labels, labels)
    report["invalid_bio_before_repair"] = invalid_bio_diagnostics(labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof = [{"ad_id": a, "field": f, "start": s, "end": e, "label": label} for a, f, s, e, label in sorted(predicted)]
    (output_dir / f"fold_{fold:02d}_oof_predictions.json").write_text(json.dumps(oof, ensure_ascii=False, indent=2), encoding="utf-8")
    token_oof = [{"ad_id": sequence.ad_id, "field": sequence.field,
                  "tokens": [{"start": token.start, "end": token.end} for token in sequence.tokens],
                  "predicted_bio": predicted_labels}
                 for sequence, predicted_labels in zip(test_seq, labels)]
    (output_dir / f"fold_{fold:02d}_oof_token_predictions.json").write_text(
        json.dumps(token_oof, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"model": model_name, "outer_fold": fold, **run_identity(dataset_path, split_path),
              "best_params": params, "inner_tuning": tuning,
              "metrics": report, "n_train_ads": len(outer_train), "n_test_ads": len(outer_test),
              "runtime_seconds": time.perf_counter()-start,
              "timing_seconds": {"inner_tuning": tuning_seconds, "final_fit": final_fit_seconds,
                                   "final_predict": final_predict_seconds},
              "runtime_options": runtime_options, "hp_workers": hp_workers,
              "environment": {"python": sys.version, "platform": platform.platform()}}
    (output_dir / f"fold_{fold:02d}_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["nb", "svm", "xgb", "crf", "all"], default="nb")
    parser.add_argument("--fold", type=int, default=None, help="One outer fold; omit for all ten.")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--hp-workers", type=int, default=2,
                        help="Parallel HP fits sharing one CSR matrix; applied only to SVM.")
    parser.add_argument("--svm-tol", type=float, default=1e-3,
                        help="Larger values stop LinearSVC earlier; benchmark before changing the final protocol.")
    parser.add_argument("--xgb-n-jobs", type=int, default=-1)
    parser.add_argument("--xgb-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    models = [args.model] if args.model != "all" else ["nb", "svm", "xgb", "crf"]
    folds = [args.fold] if args.fold is not None else list(range(10))
    data = load_json(args.data); ads = examples(data); sequence_cache, gold_cache = _build_sequence_cache(ads)
    cached = (ads, sequence_cache, gold_cache)
    runtime_options = {"svm_tol": args.svm_tol, "svm_max_iter": 5000,
                       "xgb_n_jobs": args.xgb_n_jobs, "xgb_device": args.xgb_device,
                       "crf_max_iterations": 100}
    for name in models:
        for fold in folds:
            print(f"START model={name} outer_fold={fold}", flush=True)
            result_name = {"nb": "naive_bayes", "svm": "svm", "xgb": "xgboost", "crf": "crf"}[name]
            print(json.dumps(run_fold(name, args.data, args.splits, args.results / result_name, fold, args.inner_folds,
                                      cached, runtime_options, max(1, args.hp_workers)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
