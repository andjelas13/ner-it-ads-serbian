"""Compare GPT annotation against the declared calibration reference annotation."""
from __future__ import annotations

import argparse, json
from pathlib import Path
from src.analysis.common import cohen_kappa, entities, relaxed, semantic, strict
from src.evaluation.metrics import error_categories

def main() -> int:
    root = Path(__file__).resolve().parents[2]; directory = root / "data/analysis/calibration_50"; parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=directory / "annotator_A.json"); parser.add_argument("--gpt", type=Path, default=directory / "gpt_annotation.json"); parser.add_argument("--output", type=Path, default=root / "report_artifacts/analysis/gpt_vs_human.json"); args = parser.parse_args()
    ads, reference = entities(args.reference); _, gpt = entities(args.gpt)
    payload = {"reference": str(args.reference), "gpt": str(args.gpt), "strict": strict(reference, gpt), "relaxed": relaxed(reference, gpt), "semantic": semantic(ads, reference, gpt), "cohen_kappa_semantic_with_O": cohen_kappa(ads, reference, gpt), "error_categories": error_categories(reference, gpt)}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(payload, ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
