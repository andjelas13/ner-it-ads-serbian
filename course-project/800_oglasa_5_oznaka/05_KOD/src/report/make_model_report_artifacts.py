"""Build report tables exclusively from canonical pooled OOF final artifacts.

It intentionally does not average per-fold F1 scores: the primary row for every
model is its pooled OOF micro score in ``results/<model>/final/metrics.json``.
Fold-level mean/std remain descriptive diagnostics in ``fold_metrics.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


# učitava JSON fajl sa rezultatima
def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# prosek po foldovima ostaje samo dijagnostika; glavni broj je pooled vrednost
def _fold_summary(path: Path) -> tuple[float | str, float | str]:
    if not path.exists():
        return "", ""
    with path.open(encoding="utf-8", newline="") as handle:
        values = [float(row["strict_f1"]) for row in csv.DictReader(handle)]
    return (float(np.mean(values)), float(np.std(values))) if values else ("", "")


# brojevi se samo čitaju iz već finalizovanih rezultata, ništa se ovde ne računa
def _pooled_row(model: str, stage: str, final_dir: Path) -> dict[str, Any]:
    metrics = _read(final_dir / "metrics.json")
    fold_mean, fold_std = _fold_summary(final_dir / "fold_metrics.csv")
    return {
        "model": model,
        "stage": stage,
        "pooled_outer_test_ads": metrics.get("pooled_outer_test_ads", ""),
        "strict_f1_pooled": metrics["f1"],
        "strict_precision_pooled": metrics["precision"],
        "strict_recall_pooled": metrics["recall"],
        "relaxed_f1_pooled": metrics["relaxed_overlap"]["f1"],
        "semantic_token_f1_pooled": metrics.get("semantic_token", {}).get("f1", ""),
        "invalid_bio_transition_count": metrics.get("invalid_bio_before_repair", {}).get("invalid_bio_transition_count", ""),
        "invalid_bio_transition_rate": metrics.get("invalid_bio_before_repair", {}).get("invalid_bio_transition_rate", ""),
        "fold_strict_f1_mean": fold_mean,
        "fold_strict_f1_std": fold_std,
        "source": str(final_dir / "metrics.json"),
    }


# čita tabelu po oznakama koju je napravio evaluator
def _per_label_rows(model: str, stage: str, final_dir: Path) -> list[dict[str, Any]]:
    """Read the canonical per-label table emitted by the pooled OOF evaluator."""
    path = final_dir / "per_label.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [{"model": model, "stage": stage, **row, "source": str(path)} for row in csv.DictReader(handle)]


# upisuje redove u CSV sa zadatim zaglavljem
def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


# crta krivu strict F1 po epohama za BERTić i mBERT
def _plot_epoch_curves(epoch_rows: list[dict[str, Any]], figure_dir: Path) -> None:
    if not epoch_rows:
        return
    import matplotlib.pyplot as plt
    for model in sorted({str(row["model"]) for row in epoch_rows}):
        points = sorted((int(str(row["stage"]).split("_")[1]), float(row["strict_f1_pooled"]))
                        for row in epoch_rows if row["model"] == model)
        plt.plot([point[0] for point in points], [point[1] for point in points], marker="o", label=model)
    plt.xlabel("Epoch"); plt.ylabel("Pooled strict span micro-F1"); plt.legend(); plt.tight_layout()
    plt.savefig(figure_dir / "transformer_epoch_curves.png", dpi=160); plt.close()


# crta strict F1 po oznakama, jedan stubić po modelu
def _plot_per_label(per_label_rows: list[dict[str, Any]], figure_dir: Path) -> None:
    """One comparable strict-F1 plot for final classical and epoch-7 Transformer outputs."""
    selected = [row for row in per_label_rows if row["stage"] == "final" or row["stage"] == "epoch_7"]
    if not selected:
        return
    import matplotlib.pyplot as plt
    labels = sorted({str(row["label"]) for row in selected})
    models = sorted({str(row["model"]) for row in selected})
    x = np.arange(len(labels)); width = 0.8 / max(len(models), 1)
    for index, model in enumerate(models):
        lookup = {str(row["label"]): float(row["f1"]) for row in selected if row["model"] == model}
        plt.bar(x - 0.4 + width / 2 + index * width, [lookup.get(label, 0.0) for label in labels], width, label=model)
    plt.xticks(x, labels, rotation=45, ha="right"); plt.ylabel("Strict span F1"); plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(figure_dir / "per_label_strict_f1.png", dpi=160); plt.close()


# crta stopu nevalidnih BIO prelaza po modelu
def _plot_invalid_bio(invalid_rows: list[dict[str, Any]], figure_dir: Path) -> None:
    selected = [row for row in invalid_rows if row["stage"] == "final" or row["stage"] == "epoch_7"]
    if not selected:
        return
    import matplotlib.pyplot as plt
    names = [str(row["model"]) for row in selected]
    rates = [float(row["invalid_bio_transition_rate"]) for row in selected]
    plt.bar(names, rates); plt.xticks(rotation=35, ha="right"); plt.ylabel("Invalid BIO transition rate before repair"); plt.tight_layout()
    plt.savefig(figure_dir / "invalid_bio_comparison.png", dpi=160); plt.close()


# prolazi kroz results/, skuplja pooled rezultate i pravi tabele i grafikone
def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--output", type=Path, default=root / "report_artifacts")
    args = parser.parse_args()
    table_dir = args.output / "tables"; figure_dir = args.output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in args.results.iterdir() if path.is_dir()) if args.results.exists() else []:
        final = model_dir / "final"
        if (final / "metrics.json").exists():
            rows.append(_pooled_row(model_dir.name, "final", final))
            per_label_rows.extend(_per_label_rows(model_dir.name, "final", final))
        for epoch_dir in sorted(final.glob("epoch_*"), key=lambda path: int(path.name.split("_")[1])) if final.exists() else []:
            if (epoch_dir / "metrics.json").exists():
                row = _pooled_row(model_dir.name, epoch_dir.name, epoch_dir)
                rows.append(row); epoch_rows.append(row)
                per_label_rows.extend(_per_label_rows(model_dir.name, epoch_dir.name, epoch_dir))
    columns = ["model", "stage", "pooled_outer_test_ads", "strict_f1_pooled", "strict_precision_pooled",
               "strict_recall_pooled", "relaxed_f1_pooled", "semantic_token_f1_pooled",
               "invalid_bio_transition_count", "invalid_bio_transition_rate",
               "fold_strict_f1_mean", "fold_strict_f1_std", "source"]
    _write_csv(table_dir / "model_metrics.csv", rows, columns)
    (table_dir / "model_metrics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    per_label_columns = ["model", "stage", "label", "tp", "fp", "fn", "precision", "recall", "f1", "support", "source"]
    _write_csv(table_dir / "per_label_metrics.csv", per_label_rows, per_label_columns)
    invalid_rows = [
        {key: row[key] for key in ("model", "stage", "invalid_bio_transition_count", "invalid_bio_transition_rate", "source")}
        for row in rows if row["invalid_bio_transition_count"] != ""
    ]
    _write_csv(table_dir / "invalid_bio_comparison.csv", invalid_rows,
               ["model", "stage", "invalid_bio_transition_count", "invalid_bio_transition_rate", "source"])
    try:
        _plot_epoch_curves(epoch_rows, figure_dir)
        _plot_per_label(per_label_rows, figure_dir)
        _plot_invalid_bio(invalid_rows, figure_dir)
    except ImportError:
        pass
    print(json.dumps({"pooled_rows": len(rows), "per_label_rows": len(per_label_rows), "tables": str(table_dir), "figures": str(figure_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
