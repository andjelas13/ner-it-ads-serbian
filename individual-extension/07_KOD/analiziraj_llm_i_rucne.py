"""Ponovljivo poređenje ručnih anotacija i prvog LLM prolaza na 800 oglasa.


  Svaka anotacija svodi se na petorku (oglas, polje, početak, kraj, oznaka), pa se
  dva skupa porede kao skupovi.

  Slaganje se meri na tri načina: strict (span se poklapa u granicama i oznaci),
  relaxed (spanovi se dodeljuju jedan drugom kada se preklapaju i imaju istu
  oznaku) i semantic-token (poređenje po tokenima, gde se broji i koliko se
  poklapa deo teksta). Uz njih ide i Cohenova κ nad tokenima, koja uzima u obzir
  slaganje koje bi nastalo i slučajno.

  Razlike se ne gledaju span po span, već po mestu u tekstu: spanovi koji se
  preklapaju spajaju se u povezanu komponentu i takva komponenta je jedan
  „slučaj razlike“. Svaki slučaj dobija strukturnu kategoriju, koja opisuje oblik razlike, 
  a ne ko je u pravu.


"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "02_PODACI"
RESULTS = ROOT / "04_REZULTATI"
REVIEW = ROOT / "06_RUCNA_PROVERA"

sys.path.insert(0, str(HERE))
from src.analysis.common import cohen_kappa, relaxed, semantic, strict  # noqa: E402
from src.common.tokenization import tokenize  # noqa: E402


HUMAN_PATH = DATA / "rucne_anotacije_800_oglasa_5_oznaka.json"
LLM_PATH = DATA / "llm_prvi_prolaz_1000_oglasa_11_oznaka.json"
PARTS_PATH = DATA / "podela_800_oglasa_na_4_dela.json"
LABELS = ("CPU", "GPU", "RAM", "SKLADISTE", "CENA")
LABEL_SET = set(LABELS)
DROPPED_LABELS = ("EKRAN", "BATERIJA", "BRAND", "MODEL", "GARANCIJA", "MESTO")

BOOTSTRAP_REPETITIONS = 2_000
BOOTSTRAP_SEED = 20260914

Entity = tuple[str, str, int, int, str]


@dataclass(frozen=True)
class DifferenceCase:
    """Jedan slučaj razlike"""
    case_id: str
    ad_id: str
    field: str
    part: int
    category: str
    primary_label: str
    human: tuple[Entity, ...]
    llm: tuple[Entity, ...]
    context_start: int
    context_end: int
    context: str
    dropped_nearby: tuple[Entity, ...]


def load_examples(path: Path) -> list[dict]:
   
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload["examples"] if isinstance(payload, dict) else payload


def annotation_label(annotation: dict) -> str:
  
    return str(annotation.get("label", annotation.get("tag", "")))


def annotation_field(annotation: dict) -> str:
   
    value = str(annotation.get("field", annotation.get("source", "body")))
    return value if value in {"title", "body"} else "body"


def entity(ad_id: str | int, annotation: dict) -> Entity:
   
    return (
        str(ad_id),
        annotation_field(annotation),
        int(annotation["start"]),
        int(annotation["end"]),
        annotation_label(annotation),
    )


def entities(ads: Iterable[dict], labels: set[str], ad_ids: set[str] | None = None) -> set[Entity]:
    """
         LLM izlaz od 1.000 oglasa svodi na istih 800 oglasa i pet oznaka
        koje ima ručni skup.
        """
    return {
        entity(ad["id"], annotation)
        for ad in ads
        if ad_ids is None or str(ad["id"]) in ad_ids
        for annotation in ad.get("annotations", [])
        if annotation_label(annotation) in labels
    }


def validate_inputs(human_ads: list[dict], llm_ads: list[dict]) -> dict:
 
    human_by_id = {str(ad["id"]): ad for ad in human_ads}
    llm_by_id = {str(ad["id"]): ad for ad in llm_ads}
    if len(human_by_id) != 800:
        raise ValueError(f"Očekivano je 800 različitih ručno anotiranih oglasa, nađeno {len(human_by_id)}")
    if len(llm_by_id) != 1000:
        raise ValueError(f"Očekivano je 1000 različitih LLM oglasa, nađeno {len(llm_by_id)}")
    if not set(human_by_id) <= set(llm_by_id):
        raise ValueError("LLM izlaz ne sadrži sve oglase iz ručnog skupa")

    mismatched_texts: list[dict] = []
    invalid_offsets: list[dict] = []
    for ad_id, human_ad in human_by_id.items():
        llm_ad = llm_by_id[ad_id]
        for field in ("title", "body"):
            if human_ad.get(field, "") != llm_ad.get(field, ""):
                mismatched_texts.append({"ad_id": ad_id, "field": field})
        for source, ad in (("rucno", human_ad), ("llm", llm_ad)):
            for ann in ad.get("annotations", []):
                field = annotation_field(ann)
                text = str(ad.get(field, ""))
                start, end = int(ann["start"]), int(ann["end"])
                if not (0 <= start < end <= len(text)) or text[start:end] != str(ann.get("text", text[start:end])):
                    invalid_offsets.append({"source": source, "ad_id": ad_id, "annotation": ann})
    if mismatched_texts:
        raise ValueError(f"Tekstovi se razlikuju u {len(mismatched_texts)} polja")
    if invalid_offsets:
        raise ValueError(f"Pronađeno je {len(invalid_offsets)} neispravnih karakter-ofseta")
    return {
        "human_ads": len(human_by_id),
        "llm_ads": len(llm_by_id),
        "shared_ads": len(human_by_id),
        "mismatched_text_fields": 0,
        "invalid_offsets": 0,
    }


def overlaps(left: Entity, right: Entity, require_label: bool = False) -> bool:

    return (
        left[0] == right[0]
        and left[1] == right[1]
        and (not require_label or left[-1] == right[-1])
        and max(left[2], right[2]) < min(left[3], right[3])
    )


def connected_components(human: set[Entity], llm: set[Entity]) -> list[tuple[list[Entity], list[Entity]]]:
    """Grupiše spanove koji se preklapaju u povezane komponente

         sve ručne i LLM spanove koji se
        lančano preklapaju posmatramo zajedno. Tako se jedno mesto broji kao jedan
        slučaj i kada na jednoj strani stoji jedan, a na drugoj više spanova.
        Preklapanje se gleda bez obzira na oznaku, da bi se videla i zamena oznake.
        """
    by_human: dict[tuple[str, str], list[Entity]] = defaultdict(list)
    by_llm: dict[tuple[str, str], list[Entity]] = defaultdict(list)
    for item in human:
        by_human[item[:2]].append(item)
    for item in llm:
        by_llm[item[:2]].append(item)

    output: list[tuple[list[Entity], list[Entity]]] = []
    for key in sorted(by_human.keys() | by_llm.keys(), key=lambda value: (int(value[0]), value[1])):
        hs = sorted(by_human[key], key=lambda value: (value[2], value[3], value[4]))
        ls = sorted(by_llm[key], key=lambda value: (value[2], value[3], value[4]))
        nodes = [("h", index) for index in range(len(hs))] + [("l", index) for index in range(len(ls))]
        adjacency: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
        for i, left in enumerate(hs):
            for j, right in enumerate(ls):
                if overlaps(left, right):
                    adjacency[("h", i)].add(("l", j))
                    adjacency[("l", j)].add(("h", i))
        seen: set[tuple[str, int]] = set()
        for node in nodes:
            if node in seen:
                continue
            stack = [node]
            seen.add(node)
            component: list[tuple[str, int]] = []
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbour in adjacency[current]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
            output.append(
                (
                    sorted((hs[index] for side, index in component if side == "h"), key=lambda x: (x[2], x[3], x[4])),
                    sorted((ls[index] for side, index in component if side == "l"), key=lambda x: (x[2], x[3], x[4])),
                )
            )
    return output


def classify_component(human: Sequence[Entity], llm: Sequence[Entity]) -> str:
    """

        SUVISNO — span postoji samo u LLM izlazu; PROPUSTENO — samo u ručnoj
        anotaciji; SPAJANJE — LLM ima jedan span tamo gde ručna ima više;
        FRAGMENTACIJA — obrnuto; SLOZENO — obe strane imaju više spanova.
        Kada obe strane imaju tačno jedan span: POGRESNA_OZNAKA (iste granice,
        različita oznaka), GRANICA_I_OZNAKA (i granica i oznaka se razlikuju),
        GRANICA_SIRA (LLM span obuhvata ručni), GRANICA_UZA (obrnuto) i
        GRANICA_POMERENA (granice se samo delimično poklapaju).

        """
    if not human:
        return "SUVISNO"
    if not llm:
        return "PROPUSTENO"
    if len(human) > 1 and len(llm) == 1:
        return "SPAJANJE"
    if len(human) == 1 and len(llm) > 1:
        return "FRAGMENTACIJA"
    if len(human) > 1 and len(llm) > 1:
        return "SLOZENO"

    h, l = human[0], llm[0]
    same_label = h[-1] == l[-1]
    same_bounds = h[2:4] == l[2:4]
    if same_bounds and not same_label:
        return "POGRESNA_OZNAKA"
    if not same_label:
        return "GRANICA_I_OZNAKA"
    if l[2] <= h[2] and l[3] >= h[3]:
        return "GRANICA_SIRA"
    if l[2] >= h[2] and l[3] <= h[3]:
        return "GRANICA_UZA"
    return "GRANICA_POMERENA"


def primary_label(human: Sequence[Entity], llm: Sequence[Entity]) -> str:
  
    return (human or llm)[0][-1]


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
   
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def per_ad_counts(human_ads: list[dict], human: set[Entity], llm: set[Entity]) -> dict[str, dict[str, tuple[int, int, int]]]:
    """Za svaki oglas broji pogotke, viškove i propuste po sve tri mere.
        """
    output: dict[str, dict[str, tuple[int, int, int]]] = {}
    for ad in human_ads:
        ad_id = str(ad["id"])
        h = {item for item in human if item[0] == ad_id}
        l = {item for item in llm if item[0] == ad_id}
        strict_row = strict(h, l)
        relaxed_row = relaxed(h, l)

        semantic_row = semantic([ad], h, l)
        output[ad_id] = {
            "strict": (int(strict_row["tp"]), int(strict_row["fp"]), int(strict_row["fn"])),
            "relaxed": (int(relaxed_row["tp"]), int(relaxed_row["fp"]), int(relaxed_row["fn"])),
            "semantic": (int(semantic_row["tp"]), int(semantic_row["fp"]), int(semantic_row["fn"])),
        }
    return output


def bootstrap_intervals(counts: dict[str, dict[str, tuple[int, int, int]]]) -> dict[str, dict[str, list[float]]]:
  
    rng = random.Random(BOOTSTRAP_SEED)
    ids = sorted(counts, key=int)
    values: dict[str, dict[str, list[float]]] = {
        name: {measure: [] for measure in ("precision", "recall", "f1")}
        for name in ("strict", "relaxed", "semantic")
    }
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = [rng.choice(ids) for _ in ids]
        for name in values:
            tp = sum(counts[ad_id][name][0] for ad_id in sampled)
            fp = sum(counts[ad_id][name][1] for ad_id in sampled)
            fn = sum(counts[ad_id][name][2] for ad_id in sampled)
            row = prf(tp, fp, fn)
            for measure in values[name]:
                values[name][measure].append(float(row[measure]))
    return {
        name: {
            measure: [float(np.percentile(series, 2.5)), float(np.percentile(series, 97.5))]
            for measure, series in measures.items()
        }
        for name, measures in values.items()
    }


def semantic_for_field(human_ads: list[dict], human: set[Entity], llm: set[Entity], field: str) -> dict:
   
    h = {item for item in human if item[1] == field}
    l = {item for item in llm if item[1] == field}
    other = "body" if field == "title" else "title"
    ads: list[dict] = []
    for original in human_ads:
        ad = dict(original)
        ad[other] = ""
        ads.append(ad)
    return semantic(ads, h, l)


def case_sort_key(component: tuple[list[Entity], list[Entity]]) -> tuple:
  
    items = component[0] + component[1]
    first = min(items, key=lambda value: (int(value[0]), value[1], value[2], value[3], value[4]))
    return int(first[0]), first[1], min(item[2] for item in items), min(item[3] for item in items)


def build_cases(
    human_ads: list[dict],
    components: list[tuple[list[Entity], list[Entity]]],
    dropped: set[Entity],
    part_by_ad: dict[str, int],
) -> list[DifferenceCase]:
    
    ads = {str(ad["id"]): ad for ad in human_ads}
    dropped_by_sequence: dict[tuple[str, str], list[Entity]] = defaultdict(list)
    for item in dropped:
        dropped_by_sequence[item[:2]].append(item)
    output: list[DifferenceCase] = []
    for index, (human, llm) in enumerate(sorted(components, key=case_sort_key), start=1):
        items = human + llm
        ad_id, field = items[0][0], items[0][1]
        text = str(ads[ad_id].get(field, ""))
        left, right = min(item[2] for item in items), max(item[3] for item in items)
        context_start, context_end = max(0, left - 90), min(len(text), right + 90)
        nearby = tuple(
            sorted(
                (
                    item
                    for item in dropped_by_sequence[(ad_id, field)]
                    if item[2] < right + 3 and item[3] > left - 3
                ),
                key=lambda value: (value[2], value[3], value[4]),
            )
        )
        output.append(
            DifferenceCase(
                case_id=f"K{index:04d}",
                ad_id=ad_id,
                field=field,
                part=part_by_ad[ad_id],
                category=classify_component(human, llm),
                primary_label=primary_label(human, llm),
                human=tuple(human),
                llm=tuple(llm),
                context_start=context_start,
                context_end=context_end,
                context=text[context_start:context_end],
                dropped_nearby=nearby,
            )
        )
    return output


def serialise_spans(spans: Sequence[Entity], full_text: str) -> list[dict]:

    return [
        {"start": item[2], "end": item[3], "label": item[4], "text": full_text[item[2]:item[3]]}
        for item in spans
    ]


def write_cases(cases: list[DifferenceCase], human_ads: list[dict]) -> None:

    REVIEW.mkdir(parents=True, exist_ok=True)
    ads = {str(ad["id"]): ad for ad in human_ads}
    json_rows: list[dict] = []
    for case in cases:
        full_text = str(ads[case.ad_id].get(case.field, ""))
        json_rows.append(
            {
                "case_id": case.case_id,
                "ad_id": case.ad_id,
                "field": case.field,
                "part": case.part,
                "structural_category": case.category,
                "primary_label": case.primary_label,
                "context_start": case.context_start,
                "context_end": case.context_end,
                "context": case.context,
                "human_spans": serialise_spans(case.human, full_text),
                "llm_spans": serialise_spans(case.llm, full_text),
                "dropped_llm_spans_nearby": serialise_spans(case.dropped_nearby, full_text),
            }
        )
    (REVIEW / "slucajevi_1114.json").write_text(
        json.dumps({"cases": json_rows}, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    with (REVIEW / "slucajevi_1114.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        columns = [
            "case_id", "ad_id", "field", "part", "structural_category", "primary_label", "context",
            "human_spans", "llm_spans", "dropped_llm_spans_nearby",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in json_rows:
            csv_row = dict(row)
            for key in ("human_spans", "llm_spans", "dropped_llm_spans_nearby"):
                csv_row[key] = json.dumps(csv_row[key], ensure_ascii=False)
            writer.writerow({key: csv_row.get(key, "") for key in columns})


def distribution_table(cases: list[DifferenceCase], attribute: str) -> list[dict]:
    counts = Counter(getattr(case, attribute) for case in cases)
    total = len(cases)
    return [
        {attribute: key, "count": count, "percent": 100 * count / total}
        for key, count in counts.most_common()
    ]


def metric_row(human_ads: list[dict], human: set[Entity], llm: set[Entity]) -> dict:
  
    return {
        "strict": strict(human, llm),
        "relaxed": relaxed(human, llm),
        "semantic": semantic(human_ads, human, llm),
        "cohen_kappa": cohen_kappa(human_ads, human, llm),
    }


def normalized_entity(item: Entity, texts: dict[tuple[str, str], str]) -> Entity:

    trim = " \t\r\n.,;:!?()[]{}<>\"'„“”’`*•-–—"
    ad_id, field, start, end, label = item
    text = texts[(ad_id, field)]
    while start < end and text[start] in trim:
        start += 1
    while end > start and text[end - 1] in trim:
        end -= 1
    return ad_id, field, start, end, label


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    human_ads = load_examples(HUMAN_PATH)
    llm_ads = load_examples(LLM_PATH)
    validation = validate_inputs(human_ads, llm_ads)
    human_ids = {str(ad["id"]) for ad in human_ads}
 
    human = entities(human_ads, LABEL_SET)
    llm = entities(llm_ads, LABEL_SET, human_ids)
    dropped = entities(llm_ads, set(DROPPED_LABELS), human_ids)

    parts_payload = json.loads(PARTS_PATH.read_text(encoding="utf-8"))
    part_by_ad = {
        str(ad_id): part_index
        for part_index, part in enumerate(parts_payload["parts"], start=1)
        for ad_id in part
    }
    if set(part_by_ad) != human_ids:
        raise ValueError("Podela na četiri dela ne odgovara skupu od 800 oglasa")

    exact = human & llm
    components = connected_components(human - exact, llm - exact)
    cases = build_cases(human_ads, components, dropped, part_by_ad)
    write_cases(cases, human_ads)

    overall = metric_row(human_ads, human, llm)
    intervals = bootstrap_intervals(per_ad_counts(human_ads, human, llm))

    per_label: list[dict] = []
    for label in LABELS:
        h = {item for item in human if item[-1] == label}
        l = {item for item in llm if item[-1] == label}
        row = metric_row(human_ads, h, l)
        per_label.append({"label": label, "human_spans": len(h), "llm_spans": len(l), **row})

    per_field: list[dict] = []
    for field in ("title", "body"):
        h = {item for item in human if item[1] == field}
        l = {item for item in llm if item[1] == field}
        row = {"strict": strict(h, l), "relaxed": relaxed(h, l), "semantic": semantic_for_field(human_ads, h, l, field)}
        other = "body" if field == "title" else "title"
        field_ads = []
        for original in human_ads:
            ad = dict(original)
            ad[other] = ""
            field_ads.append(ad)
        row["cohen_kappa"] = cohen_kappa(field_ads, h, l)
        per_field.append({"field": field, "human_spans": len(h), "llm_spans": len(l), **row})

    per_part: list[dict] = []
    for part in range(1, 5):
        ids = {ad_id for ad_id, value in part_by_ad.items() if value == part}
        ads = [ad for ad in human_ads if str(ad["id"]) in ids]
        h = {item for item in human if item[0] in ids}
        l = {item for item in llm if item[0] in ids}
        difference_count = sum(case.part == part for case in cases)
        per_part.append(
            {
                "part": part,
                "ads": len(ids),
                "human_spans": len(h),
                "llm_spans": len(l),
                "difference_cases": difference_count,
                "differences_per_100_human_spans": 100 * difference_count / len(h),
                **metric_row(ads, h, l),
            }
        )

    cases_per_ad = Counter(case.ad_id for case in cases)
    top_ads = [
        {"ad_id": ad_id, "difference_cases": count, "percent_all_cases": 100 * count / len(cases)}
        for ad_id, count in cases_per_ad.most_common(20)
    ]
    top_8_count = sum(item["difference_cases"] for item in top_ads[:8])

    llm_all_selected = entities(llm_ads, LABEL_SET | set(DROPPED_LABELS), human_ids)
    removed_counts = Counter(item[-1] for item in llm_all_selected if item[-1] in DROPPED_LABELS)
    direct_dropped_overlap = sum(
        any(overlaps(item, other) for item in case.dropped_nearby for other in case.human + case.llm)
        for case in cases
    )
    near_dropped = sum(bool(case.dropped_nearby) for case in cases)

    texts = {
        (str(ad["id"]), field): str(ad.get(field, ""))
        for ad in human_ads
        for field in ("title", "body")
    }
    normalized_human = {normalized_entity(item, texts) for item in human}
    normalized_llm = {normalized_entity(item, texts) for item in llm}

    without_outliers = {"709", "840"}
    h_wo = {item for item in human if item[0] not in without_outliers}
    l_wo = {item for item in llm if item[0] not in without_outliers}
    ads_wo = [ad for ad in human_ads if str(ad["id"]) not in without_outliers]
    components_wo = connected_components(h_wo - (h_wo & l_wo), l_wo - (h_wo & l_wo))

    result = {
        "metadata": {
            "reference": "ručne anotacije tima",
            "comparison": "prvi LLM prolaz",
            "labels": list(LABELS),
            "bootstrap_unit": "oglas",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "validation": validation,
        "counts": {
            "human_spans": len(human),
            "llm_spans_after_filter": len(llm),
            "exact_matches": len(exact),
            "difference_cases": len(cases),
            "ads_with_differences": len(cases_per_ad),
        },
        "overall": overall,
        "bootstrap_95_percent_intervals": intervals,
        "per_label": per_label,
        "per_field": per_field,
        "per_part": per_part,
        "difference_categories": distribution_table(cases, "category"),
        "differences_by_primary_label": distribution_table(cases, "primary_label"),
        "differences_by_field": distribution_table(cases, "field"),
        "top_ads": top_ads,
        "concentration": {
            "top_8_cases": top_8_count,
            "top_8_percent": 100 * top_8_count / len(cases),
        },
        "old_scheme": {
            "removed_spans_total": sum(removed_counts.values()),
            "removed_spans_by_label": dict(removed_counts.most_common()),
            "cases_with_direct_overlap_of_dropped_label": direct_dropped_overlap,
            "cases_with_dropped_label_within_3_characters": near_dropped,
            "note": "Mera sa pragom od 3 karaktera je kontekstualna blizina, ne direktno preklapanje.",
        },
        "sensitivity": {
            "strict_after_trimming_outer_space_and_punctuation": strict(normalized_human, normalized_llm),
            "without_ads_709_and_840": {
                "counts": {
                    "human_spans": len(h_wo),
                    "llm_spans": len(l_wo),
                    "difference_cases": len(components_wo),
                },
                "metrics": metric_row(ads_wo, h_wo, l_wo),
                "difference_categories": dict(Counter(classify_component(h, l) for h, l in components_wo).most_common()),
            },
        },
    }
    (RESULTS / "rezultati_llm_naspram_rucnih.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))
    print(json.dumps({"difference_categories": result["difference_categories"]}, ensure_ascii=False, indent=2))
    print(f"Rezultati: {RESULTS / 'rezultati_llm_naspram_rucnih.json'}")
    print(f"Slučajevi: {REVIEW / 'slucajevi_1114.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
