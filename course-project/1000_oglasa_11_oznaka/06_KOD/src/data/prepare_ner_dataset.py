"""Canonical JSON loader and strict annotation validation."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.common.constants import LABELS


# učitava JSON i svodi ga na oblik sa listom oglasa pod ključem examples
def load_json(path: str | Path) -> dict[str, Any]:
    # utf-8-sig zbog BOM oznake koju Windows alati upišu na početak fajla
    with open(path, encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return {"examples": value}
    if "examples" not in value:
        raise ValueError(f"{path}: expected top-level examples")
    return value


# vraća samu listu oglasa, bilo da je prosleđena putanja ili već učitan rečnik
def examples(data_or_path: dict[str, Any] | str | Path) -> list[dict[str, Any]]:
    data = load_json(data_or_path) if isinstance(data_or_path, (str, Path)) else data_or_path
    return data["examples"]


# prolazi kroz naslov i telo oglasa; prazna polja preskače
def fields(ad: dict[str, Any]) -> Iterable[tuple[str, str]]:
    """Only title/body are model sequences; missing fields are skipped."""
    for field in ("title", "body"):
        value = ad.get(field, "")
        if isinstance(value, str) and value:
            yield field, value


# kaže da li span pripada naslovu ili telu; podržava i starije nazive
def annotation_field(annotation: dict[str, Any]) -> str:
    value = annotation.get("field", annotation.get("source", "body"))
    return value if value in {"title", "body"} else "body"


# vraća oznaku spana; naš alat izvozi label, a tag se prihvata radi prenosivosti
def annotation_label(annotation: dict[str, Any]) -> str:
    """The historical tool exported ``label``; accept ``tag`` for portability."""
    return str(annotation.get("label", annotation.get("tag", "")))


# svodi jednu anotaciju na petorku ad_id, polje, početak, kraj, oznaka
def canonical_span(ad: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    field = annotation_field(annotation)
    text = ad.get(field, "")
    start, end = int(annotation["start"]), int(annotation["end"])
    return {"ad_id": str(ad["id"]), "field": field, "start": start, "end": end,
            "label": annotation_label(annotation), "text": text[start:end]}


# stroga provera ulaza: broj oglasa, duplikati, dozvoljene oznake, ofseti i preklapanja
def validate_dataset(data_or_path: dict[str, Any] | str | Path, expected_n: int | None = None) -> dict[str, Any]:
    ads = examples(data_or_path)
    if expected_n is not None and len(ads) != expected_n:
        raise ValueError(f"Expected {expected_n} ads, got {len(ads)}")
    seen: set[str] = set()
    label_count: Counter[str] = Counter()
    annotations = 0
    for ad in ads:
        ad_id = str(ad.get("id"))
        if not ad_id or ad_id == "None" or ad_id in seen:
            raise ValueError(f"Duplicate/missing ad id: {ad_id}")
        seen.add(ad_id)
        by_field: dict[str, list[tuple[int, int]]] = {"title": [], "body": []}
        for ann in ad.get("annotations", []):
            tag = annotation_label(ann)
            if tag not in LABELS:
                raise ValueError(f"ad {ad_id}: unsupported tag {tag!r}")
            field = annotation_field(ann)
            text = ad.get(field, "")
            try:
                start, end = int(ann["start"]), int(ann["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"ad {ad_id}: invalid offsets {ann}") from exc
            if not (0 <= start < end <= len(text)):
                raise ValueError(f"ad {ad_id}/{field}: invalid [{start}, {end})")
            by_field[field].append((start, end))
            label_count[tag] += 1
            annotations += 1
        # preklapanje se hvata ovde jer BIO shema ne trpi token u dva entiteta
        for field, spans in by_field.items():
            spans.sort()
            if any(right[0] < left[1] for left, right in zip(spans, spans[1:])):
                raise ValueError(f"ad {ad_id}/{field}: overlapping annotations")
    # otisak liste ad_id-jeva služi kao dokaz nad kojim skupom je model treniran
    return {"ads": len(ads), "annotations": annotations, "labels": dict(sorted(label_count.items())),
            "ad_ids_sha256": hashlib.sha256("\n".join(sorted(seen)).encode()).hexdigest()}


# pravi skup svih spanova, u obliku u kom se porede sa predikcijama modela
def entity_set(data_or_path: dict[str, Any] | str | Path, ad_ids: set[str] | None = None) -> set[tuple[str, str, int, int, str]]:
    output = set()
    for ad in examples(data_or_path):
        if ad_ids is not None and str(ad["id"]) not in ad_ids:
            continue
        for ann in ad.get("annotations", []):
            span = canonical_span(ad, ann)
            output.add((span["ad_id"], span["field"], span["start"], span["end"], span["label"]))
    return output


# upisuje kanonski JSON, zajedno sa izveštajem o proveri
def save_dataset(ads: list[dict[str, Any]], path: str | Path, provenance: dict[str, Any]) -> None:
    output = {"schema": "it_ner_annotation_v1", "phase3_role": "silver_1000",
              "provenance": provenance, "examples": ads}
    Path(path).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


# ceo posao: učita, proveri i upiše .json i .jsonl sa tokenima i BIO oznakama
def main() -> int:
    """Validate final spans and materialize one canonical JSON + JSONL dataset."""
    import argparse
    from src.common.constants import BIO_LABELS
    from src.common.tokenization import spans_to_bio

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/annotations/final_span_annotations.json")
    parser.add_argument("--output", type=Path, default=root / "data/processed/ner_dataset.json")
    parser.add_argument("--jsonl", type=Path, default=root / "data/processed/ner_dataset.jsonl")
    parser.add_argument("--expected-ads", type=int, default=1000)
    args = parser.parse_args()

    source = load_json(args.input)
    report = validate_dataset(source, args.expected_ads)
    ads = examples(source)
    try:
        source_name = str(args.input.resolve().relative_to(root.resolve()))
    except ValueError:
        source_name = args.input.name
    provenance = {"source": source_name, "validation": report, "tokenizer": "it-domain-text-only-v2", "bio_labels": BIO_LABELS}
    save_dataset(ads, args.output, provenance)
    with args.jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for ad in ads:
            sequences = []
            for field, text in fields(ad):
                tokens, bio = spans_to_bio(text, ad.get("annotations", []), field)
                sequences.append({"field": field, "tokens": [{"text": token.text, "start": token.start, "end": token.end, "bio": label} for token, label in zip(tokens, bio)]})
            handle.write(json.dumps({"ad_id": str(ad["id"]), "title": ad.get("title", ""), "body": ad.get("body", ""), "sequences": sequences}, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "jsonl": str(args.jsonl), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
