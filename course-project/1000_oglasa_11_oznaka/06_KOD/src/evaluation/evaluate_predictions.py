"""Evaluate saved OOF span and token predictions with the central NER scorer.

The command is deliberately model-agnostic: a model writes standard OOF JSON,
then this module reconstructs the matching gold subset and writes one report.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from src.data.prepare_ner_dataset import entity_set, examples, load_json
from src.evaluation.metrics import full_report
from src.classical.models import make_sequences
from src.common.tokenization import invalid_bio_diagnostics


def _entities(rows: list[dict[str, Any]]) -> set[tuple[str, str, int, int, str]]:
    return {
        (str(row["ad_id"]), str(row["field"]), int(row["start"]), int(row["end"]), str(row["label"]))
        for row in rows
    }


def _read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"Expected JSON list in {path}")
        rows.extend(value)
    return rows


def _gold_bio(data: dict[str, Any], requested: set[tuple[str, str]]) -> list[list[str]]:
    lookup = {(sequence.ad_id, sequence.field): sequence.labels for sequence in make_sequences(examples(data))}
    missing = requested - set(lookup)
    if missing:
        raise ValueError(f"Missing gold token rows for {len(missing)} sequences")
    return [lookup[key] for key in sorted(requested)]


def _predicted_bio(paths: list[Path], requested: set[tuple[str, str]]) -> list[list[str]]:
    rows = _read_rows(paths)
    lookup = {(str(row["ad_id"]), str(row["field"])): list(row["predicted_bio"]) for row in rows}
    missing = requested - set(lookup)
    if missing:
        raise ValueError(f"Missing token prediction rows for {len(missing)} sequences")
    return [lookup[key] for key in sorted(requested)]


def discover_oof(model_dir: Path, epoch: int | None = None) -> tuple[list[Path], list[Path], list[Path]]:
    """Find exactly one OOF span/token file and metric file per outer fold."""
    if epoch is None:
        span_files = sorted(model_dir.glob("fold_*_oof_predictions.json"))
        token_files = sorted(model_dir.glob("fold_*_oof_token_predictions.json"))
        metric_files = sorted(model_dir.glob("fold_*_metrics.json"))
    else:
        span_files = sorted(model_dir.glob(f"fold_*/epoch_{epoch}_oof_predictions.json"))
        token_files = sorted(model_dir.glob(f"fold_*/epoch_{epoch}_oof_token_predictions.json"))
        metric_files = sorted(model_dir.glob(f"fold_*/epoch_{epoch}_metrics.json"))
    def keys(paths: list[Path]) -> set[int]:
        found = set()
        for path in paths:
            match = re.search(r"fold_(\d+)", str(path))
            if match:
                found.add(int(match.group(1)))
        return found
    expected = set(range(10))
    if keys(span_files) != expected or keys(token_files) != expected or keys(metric_files) != expected:
        raise ValueError("Model-level finalization requires exactly one span/token/metric OOF artifact for each outer fold 0..9")
    return span_files, token_files, metric_files


def evaluate_paths(data_path: Path, prediction_paths: list[Path], token_paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pooled OOF scorer. Each ad is evaluated once, in its outer-test fold."""
    predicted_rows = _read_rows(prediction_paths)
    predicted = _entities(predicted_rows)
    data = load_json(data_path)
    token_rows = _read_rows(token_paths)
    requested_ads = {str(row["ad_id"]) for row in token_rows}
    requested_sequences = {(str(row["ad_id"]), str(row["field"])) for row in token_rows}
    gold = entity_set(data, requested_ads)
    predicted_bio = _predicted_bio(token_paths, requested_sequences)
    report = full_report(gold, predicted, _gold_bio(data, requested_sequences), predicted_bio)
    report["invalid_bio_before_repair"] = invalid_bio_diagnostics(predicted_bio)
    report["pooled_outer_test_ads"] = len(requested_ads)
    report["pooled_outer_test_sequences"] = len(requested_sequences)
    return report, token_rows


def write_final_artifacts(
    output: Path, report: dict[str, Any], metric_paths: list[Path] | None = None,
    model_dir: Path | None = None,
) -> None:
    """Write the one canonical pooled result consumed by report generation."""
    fold_rows: list[dict[str, Any]] = []
    if metric_paths:
        invalid_count = invalid_tokens = 0
        all_fold_invalid_diagnostics = True
        for path in metric_paths:
            item = json.loads(path.read_text(encoding="utf-8"))
            invalid = item["metrics"].get("invalid_bio_before_repair")
            if invalid is None:
                all_fold_invalid_diagnostics = False
                invalid = {}
            invalid_count += int(invalid.get("invalid_bio_transition_count", 0))
            invalid_tokens += int(invalid.get("token_count", 0))
            fold_rows.append({
                "outer_fold": item["outer_fold"],
                "strict_f1": item["metrics"]["f1"],
                "relaxed_f1": item["metrics"]["relaxed_overlap"]["f1"],
                "semantic_token_f1": item["metrics"].get("semantic_token", {}).get("f1"),
                "invalid_bio_transition_count": invalid.get("invalid_bio_transition_count", 0),
                "token_count": invalid.get("token_count", 0),
                "invalid_bio_transition_rate": invalid.get("invalid_bio_transition_rate", 0.0),
                "source": str(path.relative_to(model_dir)) if model_dir else str(path),
            })
        aggregated_invalid = {
            "invalid_bio_transition_count": invalid_count,
            "token_count": invalid_tokens,
            "invalid_bio_transition_rate": invalid_count / invalid_tokens if invalid_tokens else 0.0,
        }
        # The scorer derives the same diagnostic directly from OOF labels. A
        # mismatch here means a current fold artifact is internally inconsistent.
        # Older artifacts may lack this optional diagnostic altogether.
        if all_fold_invalid_diagnostics and report.get("invalid_bio_before_repair") != aggregated_invalid:
            raise ValueError("Fold invalid-BIO diagnostics disagree with pooled OOF token predictions")
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "per_label.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [{"label": label, **metrics} for label, metrics in report["per_tag"].items()]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["label"])
        writer.writeheader(); writer.writerows(rows)
    if fold_rows:
        fold_rows.sort(key=lambda row: int(row["outer_fold"]))
        with (output / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0])); writer.writeheader(); writer.writerows(fold_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data", type=Path, default=root / "data/processed/ner_dataset.json")
    parser.add_argument("--predictions", type=Path, nargs="+",
                        help="One or more *_oof_predictions.json files.")
    parser.add_argument("--token-predictions", type=Path, nargs="*", default=[])
    parser.add_argument("--model-dir", type=Path,
                        help="Discover all 10 fold OOF files for one model.")
    parser.add_argument("--epoch", type=int,
                        help="Transformer epoch to pool; omit for classical models.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if bool(args.predictions) == bool(args.model_dir):
        parser.error("provide exactly one of --predictions or --model-dir")
    if args.model_dir:
        prediction_paths, token_paths, metric_paths = discover_oof(args.model_dir, args.epoch)
        output = args.output or args.model_dir / "final" / (f"epoch_{args.epoch}" if args.epoch is not None else "")
    else:
        prediction_paths, token_paths, metric_paths = args.predictions, args.token_predictions, []
        if not token_paths:
            parser.error("--token-predictions is required for pooled semantic-token metrics")
        output = args.output
    assert output is not None
    report, _ = evaluate_paths(args.data, prediction_paths, token_paths)
    write_final_artifacts(output, report, metric_paths, args.model_dir)
    print(json.dumps({key: report[key] for key in ("precision", "recall", "f1")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
