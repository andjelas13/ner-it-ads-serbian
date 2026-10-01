"""Create frozen inner splits from the one final outer split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.prepare_ner_dataset import examples, load_json
from src.splits.make_outer_folds import create_split_payload, load_fold


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/processed/ner_dataset.json")
    parser.add_argument("--outer", type=Path, default=root / "splits/outer_folds.json")
    parser.add_argument("--output-dir", type=Path, default=root / "splits")
    parser.add_argument("--inner-folds", type=int, choices=[5, 10], required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ads = examples(load_json(args.data))
    for outer_fold in range(10):
        train_ids, _ = load_fold(args.outer, outer_fold)
        outer_train = [ad for ad in ads if str(ad["id"]) in train_ids]
        payload = create_split_payload(outer_train, n_splits=args.inner_folds, seed=1000 + outer_fold)
        output = args.output_dir / f"inner_outer_{outer_fold:02d}_{args.inner_folds}fold.json"
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved {output}");
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
