"""Compute corpus size, source, label and length statistics from final data."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from src.analysis.common import entities

# računa statistiku korpusa i upisuje je u JSON, uz ispis na ekran
def main() -> int:
    root = Path(__file__).resolve().parents[2]; parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/processed/ner_dataset.json"); parser.add_argument("--output", type=Path, default=root / "report_artifacts/analysis/corpus_stats.json"); args = parser.parse_args()
    ads, spans = entities(args.data); labels = Counter(item[-1] for item in spans)
    # lanac završava na source_url, jer prva dva polja u kanonskom skupu ne postoje
    def source(ad: dict) -> str:
        value = str(ad.get("source", ad.get("source_site", ad.get("source_url", "")))).lower()
        if "halooglasi" in value: return "HaloOglasi"
        if "kupujemprodajem" in value: return "KupujemProdajem"
        return "unknown"
    sources = Counter(source(ad) for ad in ads)
    lengths = {field: [len(str(ad.get(field, ""))) for ad in ads] for field in ("title", "body")}
    payload = {"ads": len(ads), "annotations": len(spans), "sources": dict(sorted(sources.items())), "label_counts": dict(sorted(labels.items())), "lengths": {field: {"mean": sum(values)/len(values), "min": min(values), "max": max(values)} for field, values in lengths.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(payload, ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
