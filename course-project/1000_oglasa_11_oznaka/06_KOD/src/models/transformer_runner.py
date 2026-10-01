"""Single-T4 Colab launcher shared by BERTiÄ‡ and mBERT.

One model is trained per outer fold.  Each run stores metrics and OOF
predictions after every epoch of the same seven-epoch training trajectory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.transformers.run_transformer_cv import DEFAULT_EPOCHS, run_fold


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_model(alias: str) -> int:
    parser = argparse.ArgumentParser(description=f"Run {alias} on one Colab T4 GPU.")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(10)), metavar="FOLD")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/processed/ner_dataset.json")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "splits/outer_folds.json")
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--checkpoint-policy", choices=["none", "last", "all"], default="none")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()
    if any(fold < 0 or fold > 9 for fold in args.folds):
        parser.error("outer folds must be in [0, 9]")
    for fold in args.folds:
        run_fold(alias, fold, args.data, args.splits, args.results, args.max_length,
                 args.stride, args.epochs, args.batch_size,
                 args.gradient_accumulation_steps, args.checkpoint_policy, args.log_every)
    return 0
