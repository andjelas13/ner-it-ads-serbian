"""Validate the frozen dataset/split protocol before starting an experiment."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.common.protocol import run_identity
from src.common.tokenization import spans_to_bio, tokenize
from src.data.prepare_ner_dataset import examples, fields, load_json, validate_dataset
from src.splits.make_outer_folds import _exact_body_groups, validate_split, _validate_duplicate_groups


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/processed/ner_dataset.json")
    parser.add_argument("--splits", type=Path, default=root / "splits/outer_folds.json")
    parser.add_argument("--output", type=Path, default=root / "splits/protocol_validation.json")
    parser.add_argument("--expected-ads", type=int, default=800)
    args = parser.parse_args()
    data = load_json(args.data)
    ads = examples(data)
    dataset_report = validate_dataset(data, args.expected_ads)
    split = json.loads(args.splits.read_text(encoding="utf-8"))
    all_ids = {str(ad["id"]) for ad in ads}
    validate_split(split, all_ids)
    groups = _exact_body_groups(ads)
    _validate_duplicate_groups(split, groups)
    changed = []
    for ad in ads:
        for field, text in fields(ad):
            raw_tokens = tokenize(text)
            bio_tokens, _ = spans_to_bio(text, ad.get("annotations", []), field)
            if raw_tokens != bio_tokens:
                changed.append({"ad_id": str(ad["id"]), "field": field})
    if changed:
        raise ValueError(f"Annotations changed tokenizer output: {changed[:10]}")
    fold_sizes = [len(fold["test_ad_ids"]) for fold in split["folds"]]
    payload = {**run_identity(args.data, args.splits), "dataset": dataset_report,
               "n_folds": len(split["folds"]), "fold_sizes": fold_sizes,
               "exact_body_duplicate_groups": sum(len(group) > 1 for group in groups),
               "tokenizer_annotation_independent": True,
               "test_membership": dict(Counter(ad_id for fold in split["folds"] for ad_id in fold["test_ad_ids"]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ads": dataset_report["ads"], "annotations": dataset_report["annotations"], "fold_sizes": fold_sizes}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
