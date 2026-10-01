"""Small model-specific CLI layer over the shared nested-CV implementation.

NB, SVM and CRF are intentionally CPU-only launchers.  XGBoost is the one
classical model configured for a CUDA-enabled Google Colab T4 runtime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.classical.nested_cv import _build_sequence_cache, run_fold
from src.data.prepare_ner_dataset import examples, load_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_RESULT_NAMES = {"nb": "naive_bayes", "svm": "svm", "xgb": "xgboost", "crf": "crf"}


def run_model(model: str, inner_folds: int, *, gpu: bool) -> int:
    parser = argparse.ArgumentParser(description=f"Run final nested-CV protocol for {MODEL_RESULT_NAMES[model]}.")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(10)), metavar="FOLD")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/processed/ner_dataset.json")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "splits/outer_folds.json")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--hp-workers", type=int, default=1,
                        help="Parallel hyperparameter fits; applies only to SVM.")
    parser.add_argument("--svm-tol", type=float, default=1e-3)
    parser.add_argument("--xgb-n-jobs", type=int, default=2 if gpu else -1)
    args = parser.parse_args()
    if any(fold < 0 or fold > 9 for fold in args.folds):
        parser.error("outer folds must be in [0, 9]")
    if gpu and model != "xgb":
        raise RuntimeError("Only the XGBoost launcher is GPU-enabled.")

    ads = examples(load_json(args.data))
    cache, gold = _build_sequence_cache(ads)
    runtime = {
        "svm_tol": args.svm_tol,
        "svm_max_iter": 5000,
        "xgb_n_jobs": args.xgb_n_jobs,
        "xgb_device": "cuda" if gpu else "cpu",
        "crf_max_iterations": 100,
    }
    output = args.results / MODEL_RESULT_NAMES[model]
    for fold in args.folds:
        print(f"START model={model} outer_fold={fold}", flush=True)
        result = run_fold(model, args.data, args.splits, output, fold, inner_folds,
                          (ads, cache, gold), runtime, max(1, args.hp_workers))
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0
