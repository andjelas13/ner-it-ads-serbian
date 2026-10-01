"""Assert that stored canonical metrics equal a fresh score from saved OOF files."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from src.evaluation.evaluate_predictions import discover_oof, evaluate_paths


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {prefix: float(value)}
    if isinstance(value, dict):
        merged: dict[str, float] = {}
        for key, child in value.items():
            merged.update(_numeric_leaves(child, f"{prefix}.{key}" if prefix else str(key)))
        return merged
    return {}


def _per_label_rows(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Canonical numeric representation of the generated per_label.csv."""
    return {
        label: {key: float(value) for key, value in values.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
        for label, values in report["per_tag"].items()
    }


def _read_per_label(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            str(row["label"]): {
                key: float(value) for key, value in row.items()
                if key != "label" and value not in (None, "")
            }
            for row in rows
        }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, help="Required for a transformer epoch final artifact.")
    parser.add_argument("--data", type=Path, default=root / "data/processed/ner_dataset.json")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    prediction_paths, token_paths, _ = discover_oof(args.model_dir, args.epoch)
    recomputed, _ = evaluate_paths(args.data, prediction_paths, token_paths)
    final_dir = args.model_dir / "final" / (f"epoch_{args.epoch}" if args.epoch is not None else "")
    stored = json.loads((final_dir / "metrics.json").read_text(encoding="utf-8"))
    expected, observed = _numeric_leaves(stored), _numeric_leaves(recomputed)
    missing = sorted(set(expected) - set(observed))
    mismatches = {
        key: {"stored": expected[key], "recomputed": observed[key]}
        for key in sorted(set(expected) & set(observed))
        if not math.isclose(expected[key], observed[key], abs_tol=args.tolerance, rel_tol=0.0)
    }
    stored_per_label = _read_per_label(final_dir / "per_label.csv")
    recomputed_per_label = _per_label_rows(recomputed)
    per_label_missing = sorted(set(stored_per_label) ^ set(recomputed_per_label))
    per_label_mismatches: dict[str, dict[str, dict[str, float]]] = {}
    for label in sorted(set(stored_per_label) & set(recomputed_per_label)):
        stored_values, recomputed_values = stored_per_label[label], recomputed_per_label[label]
        for key in sorted(set(stored_values) | set(recomputed_values)):
            if key not in stored_values or key not in recomputed_values or not math.isclose(
                stored_values[key], recomputed_values[key], abs_tol=args.tolerance, rel_tol=0.0,
            ):
                per_label_mismatches.setdefault(label, {})[key] = {
                    "stored": stored_values.get(key), "recomputed": recomputed_values.get(key),
                }
    payload = {
        "model_dir": str(args.model_dir), "epoch": args.epoch, "tolerance": args.tolerance,
        "checked_numeric_values": len(expected), "missing_recomputed_values": missing,
        "mismatches": mismatches,
        "per_label_rows_checked": len(stored_per_label),
        "per_label_missing_or_extra_labels": per_label_missing,
        "per_label_mismatches": per_label_mismatches,
        "matches": not missing and not mismatches and not per_label_missing and not per_label_mismatches,
    }
    (final_dir / "recompute_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if not payload["matches"]:
        raise SystemExit("Stored final metrics do not match an OOF recomputation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
