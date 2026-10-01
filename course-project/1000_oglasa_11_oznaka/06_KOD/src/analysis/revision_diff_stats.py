"""Quantify INIT-to-FINAL annotation changes, including overlapping boundary edits."""
from __future__ import annotations

import argparse, json
from pathlib import Path
from src.analysis.common import Entity, entities

def _overlap(left: Entity, right: Entity) -> bool:
    return left[0] == right[0] and left[1] == right[1] and left[-1] == right[-1] and max(left[2], right[2]) < min(left[3], right[3])

def main() -> int:
    root = Path(__file__).resolve().parents[2]; parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True); parser.add_argument("--final", type=Path, default=root / "data/annotations/final_span_annotations.json"); parser.add_argument("--output", type=Path, default=root / "report_artifacts/analysis/revision_diff.json"); args = parser.parse_args()
    _, initial = entities(args.initial); _, final = entities(args.final); unchanged = initial & final; removed, added = initial - final, final - initial
    available = set(added); boundary = 0
    for old in removed:
        candidates = [new for new in available if _overlap(old, new)]
        if candidates:
            available.remove(max(candidates, key=lambda new: min(old[3], new[3])-max(old[2], new[2]))); boundary += 1
    payload = {"initial_annotations": len(initial), "final_annotations": len(final), "unchanged": len(unchanged), "removed": len(removed), "added": len(added), "same_label_overlap_boundary_corrections_at_least": boundary}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(payload, ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
