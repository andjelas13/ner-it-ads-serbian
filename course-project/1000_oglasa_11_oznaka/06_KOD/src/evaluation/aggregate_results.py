"""Diagnostic fold-mean summary only; never use this output as final pooled OOF metrics.

The canonical final metrics are produced by ``src.evaluation.finalize_results``.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.metrics import full_report


def _load_jsons(directory: Path, suffix: str) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in directory.rglob(suffix)] if directory.exists() else []


def aggregate(input_dirs: list[Path], output: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for directory in input_dirs:
        for item in _load_jsons(directory, "*_metrics.json"):
            if "metrics" in item and "model" in item:
                key = item["model"] + (f"_epoch_{item['epoch']}" if "epoch" in item else "")
                grouped[key].append(item)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, files in sorted(grouped.items()):
        f1 = [item["metrics"]["f1"] for item in files]
        p = [item["metrics"]["precision"] for item in files]
        r = [item["metrics"]["recall"] for item in files]
        semantic = [item["metrics"].get("semantic_token", {}).get("f1") for item in files]
        semantic = [value for value in semantic if value is not None]
        invalid_count = sum(item["metrics"].get("invalid_bio_before_repair", {}).get(
            "invalid_bio_transition_count", 0) for item in files)
        invalid_tokens = sum(item["metrics"].get("invalid_bio_before_repair", {}).get(
            "token_count", 0) for item in files)
        rows.append({"model": model, "folds": len(files), "strict_micro_f1_mean": float(np.mean(f1)),
                     "strict_micro_f1_std": float(np.std(f1)), "precision_mean": float(np.mean(p)),
                     "recall_mean": float(np.mean(r)),
                     "semantic_token_micro_f1_mean": float(np.mean(semantic)) if semantic else None,
                     "invalid_bio_transition_count": invalid_count,
                     "invalid_bio_transition_rate": invalid_count / invalid_tokens if invalid_tokens else 0.0})
    with open(output / "aggregate_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["model"]); writer.writeheader(); writer.writerows(rows)
    (output / "aggregate_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/combined"))
    args = parser.parse_args()
    print("DIAGNOSTIC ONLY: fold means are not final pooled OOF metrics.")
    aggregate(args.inputs, args.output); return 0


if __name__ == "__main__":
    raise SystemExit(main())
