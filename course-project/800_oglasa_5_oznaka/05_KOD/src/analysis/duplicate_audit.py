"""Report byte-exact title/body duplicate groups and their CV-fold placement."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from src.data.prepare_ner_dataset import examples, load_json

def main() -> int:
    root = Path(__file__).resolve().parents[2]; parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/processed/ner_dataset.json"); parser.add_argument("--splits", type=Path, default=root / "splits/outer_folds.json"); parser.add_argument("--output", type=Path, default=root / "report_artifacts/analysis/duplicate_audit.json"); args = parser.parse_args()
    groups = {field: defaultdict(list) for field in ("title", "body")}
    for ad in examples(load_json(args.data)):
        for field in groups:
            text = str(ad.get(field, ""))
            if text: groups[field][text].append(str(ad["id"]))
    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    fold_for = {str(ad_id): int(fold["fold"]) for fold in split_payload["folds"] for ad_id in fold["test_ad_ids"]}
    result = {field: [{"ad_ids": ids, "chars": len(text), "folds": sorted({fold_for[item] for item in ids}), "cross_fold": len({fold_for[item] for item in ids}) > 1} for text, ids in values.items() if len(ids) > 1] for field, values in groups.items()}
    payload = {
        "duplicate_groups": result,
        "cross_fold_exact_body_groups": sum(row["cross_fold"] for row in result["body"]),
        "cross_fold_exact_title_groups": sum(row["cross_fold"] for row in result["title"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(payload, ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
