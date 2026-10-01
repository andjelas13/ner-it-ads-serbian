"""Pairwise human agreement on the shared 50-ad calibration set."""
from __future__ import annotations

import argparse, csv, json
from itertools import combinations
from pathlib import Path
from typing import Any

from src.analysis.common import cohen_kappa, entities, relaxed, semantic, strict
from src.common.constants import LABELS


def _default_inputs(root: Path) -> dict[str, Path]:
    directory = root / "data/analysis/calibration_50"
    return {"A": directory / "annotator_A.json", "B": directory / "annotator_B.json", "C": directory / "annotator_C.json", "D": directory / "annotator_D.json"}


def main() -> int:
    root = Path(__file__).resolve().parents[2]; parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="*", help="Optional four files in A B C D order.")
    parser.add_argument("--output", type=Path, default=root / "report_artifacts/analysis/human_agreement")
    args = parser.parse_args(); paths = _default_inputs(root) if not args.inputs else dict(zip("ABCD", args.inputs, strict=True))
    loaded = {name: entities(path) for name, path in paths.items()}; rows: list[dict[str, Any]] = []; per_label: list[dict[str, Any]] = []
    for left, right in combinations("ABCD", 2):
        ads, gold = loaded[left]; _, predicted = loaded[right]
        semantic_scores = semantic(ads, gold, predicted)
        row = {"left": left, "right": right, "strict_f1": strict(gold, predicted)["f1"], "relaxed_f1": relaxed(gold, predicted)["f1"], "semantic_token_f1": semantic_scores["f1"], "cohen_kappa_semantic_with_O": cohen_kappa(ads, gold, predicted)}
        rows.append(row)
        for label in LABELS:
            per_label.append({"left": left, "right": right, "label": label, "strict_f1": strict(gold, predicted, label)["f1"], "relaxed_f1": relaxed(gold, predicted, label)["f1"], "semantic_token_f1": semantic_scores["per_label"][label]["f1"]})
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, values in (("pairwise.csv", rows), ("per_label.csv", per_label)):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    per_label_semantic = {
        label: sum(float(row["semantic_token_f1"]) for row in per_label if row["label"] == label) / len(rows)
        for label in LABELS
    }
    payload = {"pairs": rows, "mean_strict_f1": sum(float(row["strict_f1"]) for row in rows)/len(rows), "mean_relaxed_f1": sum(float(row["relaxed_f1"]) for row in rows)/len(rows), "mean_semantic_token_f1": sum(float(row["semantic_token_f1"]) for row in rows)/len(rows), "mean_cohen_kappa_semantic_with_O": sum(float(row["cohen_kappa_semantic_with_O"]) for row in rows)/len(rows), "per_label_mean_semantic_token_f1": per_label_semantic}
    (args.output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(payload, ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
