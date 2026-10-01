"""Create canonical pooled final artifacts from all completed OOF fold files.

The runner is deliberately separate from training: it may be re-run safely after
the ten folds of a model finish, without changing predictions or model weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.evaluate_predictions import discover_oof, evaluate_paths, write_final_artifacts


CLASSICAL = ("naive_bayes", "svm", "xgboost", "crf")
TRANSFORMERS = ("bertic", "mbert")
REQUIRED_TRANSFORMER_EPOCHS = frozenset(range(1, 8))


def _epochs_with_complete_oof(model_dir: Path) -> list[int]:
    epochs: set[int] = set()
    for path in model_dir.glob("fold_*/epoch_*_oof_predictions.json"):
        try:
            epochs.add(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    complete: list[int] = []
    for epoch in sorted(epochs):
        try:
            discover_oof(model_dir, epoch)
            complete.append(epoch)
        except ValueError:
            pass
    return complete


def _finalize(model_dir: Path, data: Path, epoch: int | None) -> Path:
    spans, tokens, metrics = discover_oof(model_dir, epoch)
    report, _ = evaluate_paths(data, spans, tokens)
    output = model_dir / "final" / (f"epoch_{epoch}" if epoch is not None else "")
    write_final_artifacts(output, report, metrics, model_dir)
    return output


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--data", type=Path, default=root / "data/processed/ner_dataset.json")
    parser.add_argument("--models", nargs="*", default=[*CLASSICAL, *TRANSFORMERS])
    parser.add_argument(
        "--require-all", action="store_true",
        help="Require all 10 OOF folds; for BERTic/mBERT require every epoch 1..7.",
    )
    args = parser.parse_args()
    written: list[str] = []
    missing: list[str] = []
    for model in args.models:
        directory = args.results / model
        if model in TRANSFORMERS:
            epochs = _epochs_with_complete_oof(directory)
            missing_epochs = sorted(REQUIRED_TRANSFORMER_EPOCHS - set(epochs))
            if args.require_all and missing_epochs:
                missing.append(f"{model} missing complete epochs {missing_epochs}")
                continue
            if not epochs:
                missing.append(model)
                continue
            for epoch in epochs:
                written.append(str(_finalize(directory, args.data, epoch)))
        else:
            try:
                written.append(str(_finalize(directory, args.data, None)))
            except ValueError:
                missing.append(model)
    print({"finalized": written, "incomplete": missing})
    if missing and args.require_all:
        raise SystemExit("Incomplete OOF outputs: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
