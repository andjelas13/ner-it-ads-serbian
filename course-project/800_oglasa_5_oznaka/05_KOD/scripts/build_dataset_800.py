"""Gradi kanonski JSON skup od 800 oglasa i pet oznaka."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.common.constants import LABELS

ROOT = Path(__file__).resolve().parents[1]


# učitava 800 oglasa iz JSONL fajla i vraća ih po ad_id
def read_ads(path: Path) -> dict[str, dict[str, Any]]:
    ads: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ad_id = str(row["ad_id"])
            if ad_id in ads:
                raise ValueError(f"{path}:{line_number}: duplikat ad_id {ad_id}")
            ads[ad_id] = row
    return ads


# preuzima kategoriju oglasa iz sirovog skupa od 1000, ako postoji
def read_categories(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(ad["id"]): str(ad.get("category", "")) for ad in payload.get("ads", [])}


# učitava TSV sa finalnim spanovima; ofseti su 0-based i end-exclusive
def read_spans(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


# spaja tekstove i spanove, uz proveru svakog ofseta i teksta spana
def build(ads: dict[str, dict[str, Any]], spans: list[dict[str, Any]],
          categories: dict[str, str]) -> tuple[list[dict[str, Any]], Counter[str], int]:
    grouped: dict[str, list[dict[str, Any]]] = {ad_id: [] for ad_id in ads}
    label_count: Counter[str] = Counter()
    dropped = 0
    for row in spans:
        ad_id = str(row["ad_id"]).strip()
        label = str(row["label"]).strip()
        field = str(row["field"]).strip()
        if label not in LABELS:
            dropped += 1
            continue
        if ad_id not in ads:
            raise ValueError(f"span pokazuje na nepostojeci oglas {ad_id}")
        if field not in {"title", "body"}:
            raise ValueError(f"oglas {ad_id}: nepoznato polje {field!r}")
        start, end = int(row["start"]), int(row["end"])
        text = str(ads[ad_id].get(field, ""))
        if not 0 <= start < end <= len(text):
            raise ValueError(f"oglas {ad_id}/{field}: ofset [{start},{end}) izlazi iz teksta")
        # TSV ne sme da sadrzi pravi prelom reda, pa su \n i \t u koloni ``text``
        # zapisani kao dva znaka.  Vracamo ih pre poredjenja sa izvornim tekstom.
        expected = str(row.get("text", "")).replace("\\n", "\n").replace("\\t", "\t")
        if expected and text[start:end] != expected:
            raise ValueError(
                f"oglas {ad_id}/{field}: tekst {text[start:end]!r} != ocekivano {expected!r}"
            )
        grouped[ad_id].append({"field": field, "start": start, "end": end,
                               "text": text[start:end], "label": label})
        label_count[label] += 1

    examples: list[dict[str, Any]] = []
    for ad_id in sorted(ads, key=int):
        ad = ads[ad_id]
        annotations = sorted(grouped[ad_id], key=lambda item: (item["field"], item["start"]))
        # Kljucevi su namerno isti kao u ranijem kanonskom fajlu za 1.000 oglasa:
        # id, category, title, body, source_url, annotations.  Naziv izvora
        # (Halo Oglasi / KupujemProdajem) se ne upisuje kao zaseban kljuc, jer ga
        # ``src/analysis/corpus_stats.py`` cita iz ``source_url``.
        examples.append({
            "id": int(ad_id),
            "category": categories.get(ad_id, ""),
            "title": str(ad.get("title", "")),
            "body": str(ad.get("body", "")),
            "source_url": str(ad.get("url", "")),
            "annotations": annotations,
        })
    return examples, label_count, dropped


# ceo posao: učita, spoji, proveri i upiše kanonski JSON
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ads", type=Path, default=ROOT / "data/raw/oglasi_800.jsonl")
    parser.add_argument("--spans", type=Path,
                        default=ROOT / "data/annotations/finalne_anotacije_800_span.tsv")
    parser.add_argument("--categories", type=Path, default=ROOT / "data/raw/ads_1000.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/annotations/final_span_annotations_800_5labels.json")
    parser.add_argument("--expected-ads", type=int, default=800)
    args = parser.parse_args()

    ads = read_ads(args.ads)
    if len(ads) != args.expected_ads:
        raise ValueError(f"Ocekivano {args.expected_ads} oglasa, ucitano {len(ads)}")
    examples, label_count, dropped = build(ads, read_spans(args.spans),
                                           read_categories(args.categories))
    payload = {
        "schema": "it_ner_annotation_v1",
        "phase3_role": "gold_800_5labels",
        "provenance": {
            "sources": [str(args.ads.name), str(args.spans.name)],
            "labels": LABELS,
            "note": "Rucna (gold) anotacija: 800 oglasa, pet oznaka.",
            "dropped_out_of_scheme_spans": dropped,
        },
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "ads": len(examples),
                      "annotations": sum(label_count.values()),
                      "labels": dict(sorted(label_count.items())),
                      "dropped_out_of_scheme_spans": dropped}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
