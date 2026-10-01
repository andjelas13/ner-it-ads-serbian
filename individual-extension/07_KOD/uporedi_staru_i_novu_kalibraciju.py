#!/usr/bin/env python3
"""Poredimo stare i ponovo urađene anotacije Anđele i Nikole na istih 50 oglasa.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD_ROOT = ROOT / "01_STARA_KALIBRACIJA" / "izvedeno_5_oznaka"
TEXTS_PATH = OLD_ROOT / "tekstovi_50_oglasa_utf8.jsonl.txt"
DEFAULT_RESULTS = ROOT / "04_REZULTATI"


sys.path.insert(0, str(HERE))
from analiziraj_llm_i_rucne import (  
    classify_component,
    connected_components,
)
from src.analysis.common import cohen_kappa, relaxed, semantic, strict 


LABELS = ("CPU", "GPU", "RAM", "SKLADISTE", "CENA")
LABEL_SET = set(LABELS)

Entity = tuple[str, str, int, int, str]


def normalized_label(value: object) -> str:
  
    label = str(value)
    return "SKLADISTE" if label == "SKLADIŠTE" else label


def load_texts() -> list[dict]:

    ads = []
    for line_number, line in enumerate(TEXTS_PATH.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        ads.append(
            {
                "id": str(row["ad_id"]),
                "title": str(row.get("title", "")),
                "body": str(row.get("body", "")),
                "url": str(row.get("url", "")),
            }
        )
    ids = [ad["id"] for ad in ads]
    if len(ads) != 50 or len(set(ids)) != 50:
        raise ValueError(f"Tekstualni skup mora imati 50 različitih oglasa, pronađeno {len(set(ids))}")
    return ads


def validate_entity(ad_by_id: dict[str, dict], item: Entity, source: str) -> None:
  
    ad_id, field, start, end, label = item
    if ad_id not in ad_by_id:
        raise ValueError(f"{source}: nepoznat oglas {ad_id}")
    if field not in {"title", "body"}:
        raise ValueError(f"{source}: nepoznato polje {field!r}")
    if label not in LABEL_SET:
        raise ValueError(f"{source}: nedozvoljena oznaka {label!r}")
    text = str(ad_by_id[ad_id][field])
    if not 0 <= start < end <= len(text):
        raise ValueError(f"{source}: neispravan span {ad_id}/{field} [{start}, {end})")


def load_old_tsv(path: Path, ads: list[dict]) -> set[Entity]:
   
    ad_by_id = {str(ad["id"]): ad for ad in ads}
    output: set[Entity] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"ad_id", "field", "start", "end", "label", "text"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"{path}: TSV nema očekivane kolone")
      
        for row_number, row in enumerate(reader, start=2):
            label = normalized_label(row["label"])
            if label not in LABEL_SET:
                continue
            item: Entity = (
                str(row["ad_id"]),
                str(row["field"]),
                int(row["start"]),
                int(row["end"]),
                label,
            )
            validate_entity(ad_by_id, item, f"{path.name}, red {row_number}")
            actual = str(ad_by_id[item[0]][item[1]])[item[2] : item[3]]
            if actual != row["text"]:
                raise ValueError(f"{path.name}, red {row_number}: tekst spana se ne poklapa sa offsetima")
            if item in output:
                raise ValueError(f"{path.name}, red {row_number}: duplirana anotacija")
            output.add(item)
    return output


def load_new_json(path: Path, ads: list[dict]) -> set[Entity]:

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    examples = payload.get("examples") if isinstance(payload, dict) else None
    if not isinstance(examples, list):
        raise ValueError(f"{path}: JSON nema listu examples")
    if len(examples) != 50:
        raise ValueError(f"{path}: očekivano je 50 oglasa, pronađeno {len(examples)}")

    ad_by_id = {str(ad["id"]): ad for ad in ads}
    seen_ids: set[str] = set()
    output: set[Entity] = set()
    for example in examples:
        ad_id = str(example.get("id"))
        if ad_id in seen_ids:
            raise ValueError(f"{path}: oglas {ad_id} se pojavljuje više puta")
        seen_ids.add(ad_id)
        if ad_id not in ad_by_id:
            raise ValueError(f"{path}: nepoznat oglas {ad_id}")

        for field in ("title", "body"):
            if str(example.get(field, "")) != str(ad_by_id[ad_id][field]):
                raise ValueError(f"{path}: tekst oglasa {ad_id}/{field} se razlikuje od kalibracionog skupa")
        annotations = example.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError(f"{path}: oglas {ad_id} nema ispravnu listu annotations")
        for annotation in annotations:
            label = normalized_label(annotation.get("label", annotation.get("tag", "")))
            item: Entity = (
                ad_id,
                str(annotation.get("field", "body")),
                int(annotation["start"]),
                int(annotation["end"]),
                label,
            )
            validate_entity(ad_by_id, item, path.name)
            actual = str(ad_by_id[ad_id][item[1]])[item[2] : item[3]]
         
      
            if "text" in annotation and str(annotation["text"]) != actual:
                raise ValueError(f"{path}: tekst anotacije {ad_id}/{item[1]} se ne poklapa sa offsetima")
            if item in output:
                raise ValueError(f"{path}: duplirana anotacija {item}")
            output.add(item)
  
    if seen_ids != set(ad_by_id):
        missing = sorted(set(ad_by_id) - seen_ids, key=int)
        raise ValueError(f"{path}: nedostaju oglasi {missing}")
    return output


def metric_block(ads: list[dict], left: set[Entity], right: set[Entity]) -> dict:
    """

      * ``exact_matches`` — broj spanova koji su u oba skupa potpuno isti;
      * ``strict`` — slaganje uz uslov da su i granice i oznaka identične;
      * ``relaxed`` — slaganje uz preklapanje spanova sa istom oznakom, gde se
        svaki span sme upariti samo jednom;
      * ``semantic_token`` — slaganje po tokenima: svakom tokenu teksta dodeli se
        oznaka spana koji ga pokriva, pa se dva niza oznaka porede;
      * ``cohen_kappa`` — Cohenova kappa nad istim nizovima tokena, tj. slaganje
        umanjeno za onoliko koliko bi se dva anotatora složila slučajno.


    """
   
    semantic_result = semantic(ads, left, right)
    return {
        "left_spans": len(left),
        "right_spans": len(right),
        "exact_matches": len(left & right),
        "strict": strict(left, right),
        "relaxed": relaxed(left, right),
        "semantic_token": semantic_result,
        "cohen_kappa": cohen_kappa(ads, left, right),
        "per_label": [
            {
                "label": label,
                "left_spans": sum(item[-1] == label for item in left),
                "right_spans": sum(item[-1] == label for item in right),
                "strict": strict(left, right, label),
                "relaxed": relaxed(left, right, label),
                "semantic_token": semantic_result["per_label"][label],
            }
            for label in LABELS
        ],
    }


def difference_rows(
    round_name: str,
    ads: list[dict],
    left: set[Entity],
    right: set[Entity],
) -> list[dict]:
  
    ad_by_id = {str(ad["id"]): ad for ad in ads}
    exact = left & right
    components = connected_components(left - exact, right - exact)
    rows = []
    for index, (left_items, right_items) in enumerate(components, start=1):
        items = left_items + right_items
     
        ad_id, field = items[0][0], items[0][1]
        text = str(ad_by_id[ad_id][field])
        start = min(item[2] for item in items)
        end = max(item[3] for item in items)
        context_start = max(0, start - 70)
        context_end = min(len(text), end + 70)

        def spans(values: list[Entity]) -> list[dict]:
           
            return [
                {
                    "start": item[2],
                    "end": item[3],
                    "label": item[4],
                    "text": text[item[2] : item[3]],
                }
                for item in values
            ]

        rows.append(
            {
                "round": round_name,
               
                "case_id": f"{round_name[:1].upper()}{index:04d}",
                "ad_id": ad_id,
                "field": field,
                "category": classify_component(left_items, right_items),
                "context": text[context_start:context_end],
                "andjela_spans": spans(left_items),
                "nikola_spans": spans(right_items),
            }
        )
    return rows


def comparison(
    name: str,
    ads: list[dict],
    left: set[Entity],
    right: set[Entity],
) -> tuple[dict, list[dict]]:
  
    rows = difference_rows(name, ads, left, right)
    block = metric_block(ads, left, right)
    block["difference_cases"] = len(rows)
    block["ads_with_differences"] = len({row["ad_id"] for row in rows})
    block["difference_categories"] = dict(Counter(row["category"] for row in rows).most_common())
    return block, rows


def fmt(value: float) -> str:
    
    return f"{value:.4f}".replace(".", ",")


def relative_path(path: Path) -> str:
  
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
   
    parser = argparse.ArgumentParser(
        description="Poredi stare i nove anotacije istih 50 kalibracionih oglasa."
    )
    parser.add_argument("--andjela-letter", required=True, choices=list("ABCD"))
    parser.add_argument("--nikola-letter", required=True, choices=list("ABCD"))
    parser.add_argument("--new-andjela", required=True, type=Path)
    parser.add_argument("--new-nikola", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> int:
    """Učitava četiri skupa anotacija, računa četiri poređenja i upisuje rezultate."""
    args = parse_args()

    if args.andjela_letter == args.nikola_letter:
        raise ValueError("Anđela i Nikola ne mogu imati isto slovo u staroj kalibraciji")

    ads = load_texts()
    old_andjela_path = OLD_ROOT / f"anotator_{args.andjela_letter}_5_oznaka.tsv"
    old_nikola_path = OLD_ROOT / f"anotator_{args.nikola_letter}_5_oznaka.tsv"
    old_andjela = load_old_tsv(old_andjela_path, ads)
    old_nikola = load_old_tsv(old_nikola_path, ads)
    new_andjela = load_new_json(args.new_andjela, ads)
    new_nikola = load_new_json(args.new_nikola, ads)

    comparisons: dict[str, dict] = {}
    comparisons["old_pair"], old_rows = comparison("stara", ads, old_andjela, old_nikola)
    comparisons["new_pair"], new_rows = comparison("nova", ads, new_andjela, new_nikola)
    comparisons["andjela_stability"], _ = comparison("andjela_stabilnost", ads, old_andjela, new_andjela)
    comparisons["nikola_stability"], _ = comparison("nikola_stabilnost", ads, old_nikola, new_nikola)

    result = {
        "metadata": {
            "ads": 50,
            "labels": list(LABELS),
            "andjela_old_letter": args.andjela_letter,
            "nikola_old_letter": args.nikola_letter,
            "old_andjela_file": relative_path(old_andjela_path),
            "old_nikola_file": relative_path(old_nikola_path),
            "new_andjela_file": relative_path(args.new_andjela),
            "new_nikola_file": relative_path(args.new_nikola),
        },
        "comparisons": comparisons,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rezultati_stara_nova_kalibracija.json"
    csv_path = output_dir / "razlike_stara_nova_kalibracija.csv"
    
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        columns = [
            "round",
            "case_id",
            "ad_id",
            "field",
            "category",
            "context",
            "andjela_spans",
            "nikola_spans",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in old_rows + new_rows:
       
            copy = dict(row)
            copy["andjela_spans"] = json.dumps(copy["andjela_spans"], ensure_ascii=False)
            copy["nikola_spans"] = json.dumps(copy["nikola_spans"], ensure_ascii=False)
            writer.writerow(copy)

   
    print(json.dumps({key: value["strict"]["f1"] for key, value in comparisons.items()}, indent=2))
    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
