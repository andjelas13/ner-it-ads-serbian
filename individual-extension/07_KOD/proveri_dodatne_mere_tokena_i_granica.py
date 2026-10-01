#!/usr/bin/env python3
"""Detaljne token-level i boundary mere nad LLM i ručnim anotacijama.

"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "02_PODACI"
RESULTS = ROOT / "04_REZULTATI"

sys.path.insert(0, str(HERE))
from analiziraj_llm_i_rucne import (  # noqa: E402
    LABELS,
    LABEL_SET,
    connected_components,
    entities,
    load_examples,
    strict,
    validate_inputs,
)
from src.analysis.common import semantic_grid  # noqa: E402


Entity = tuple[str, str, int, int, str]


def safe_div(numerator: int | float, denominator: int | float) -> float:

    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(
    gold: Sequence[str], predicted: Sequence[str], classes: Sequence[str]
) -> dict:
    """Jednoklasne klasifikacione mere i matrica konfuzije.

    Ulaz su dva poravnata niza oznaka iste dužine: ``gold`` je ručna strana,
    ``predicted`` LLM strana, a i-ti element jednog i drugog niza je oznaka
    istog tokena. ``classes`` je spisak dozvoljenih oznaka i ujedno redosled
    vrsta i kolona u matrici konfuzije.

    Za svaku oznaku se računaju tp, fp, fn, preciznost, odziv i F1, a za ceo
    niz tačnost (udeo tokena oko kojih se strane slažu), macro-F1 i balanced
    accuracy.

    Balanced accuracy
    je prosek odziva po oznakama, to jest prosečan udeo pogođenih tokena unutar
    svake oznake posebno. Obe mere se navode zato što je ``O`` daleko najčešća
    oznaka, pa bi gola tačnost izgledala visoko čak i kada su entiteti loše
    pogođeni.
    """
    if len(gold) != len(predicted):
        raise ValueError("Nizovi oznaka moraju imati istu dužinu")
 
    confusion: dict[str, dict[str, int]] = {
        first: {second: 0 for second in classes} for first in classes
    }
    for first, second in zip(gold, predicted):
        confusion[first][second] += 1

    per_class: dict[str, dict] = {}
    for label in classes:
      
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in classes if other != label)
        fn = sum(confusion[label][other] for other in classes if other != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "gold_tokens": sum(confusion[label].values()),
            "predicted_tokens": sum(confusion[other][label] for other in classes),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    accuracy = safe_div(sum(confusion[label][label] for label in classes), len(gold))
    return {
        "tokens": len(gold),
        "accuracy_micro_f1": accuracy,
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(classes),
        "balanced_accuracy": sum(row["recall"] for row in per_class.values()) / len(classes),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def ner_micro(per_class: dict[str, dict], labels: Iterable[str]) -> dict[str, float | int]:
    """Mikro-prosek preko izabranih oznaka, dakle bez oznake ``O``.

  
    """
    chosen = [per_class[label] for label in labels]
    tp = sum(int(row["tp"]) for row in chosen)
    fp = sum(int(row["fp"]) for row in chosen)
    fn = sum(int(row["fn"]) for row in chosen)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
    }


def merge_intervals(spans: Sequence[Entity]) -> list[tuple[int, int]]:
  
    merged: list[list[int]] = []
    for start, end in sorted((item[2], item[3]) for item in spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_length(intervals: Sequence[tuple[int, int]]) -> int:

    return sum(end - start for start, end in intervals)


def intersection_length(
    first: Sequence[tuple[int, int]], second: Sequence[tuple[int, int]]
) -> int:
  
    left = right = total = 0
    while left < len(first) and right < len(second):
        start = max(first[left][0], second[right][0])
        end = min(first[left][1], second[right][1])
        total += max(0, end - start)
        if first[left][1] <= second[right][1]:
            left += 1
        else:
            right += 1
    return total


def trim_whitespace(spans: set[Entity], ads_by_id: dict[str, dict]) -> set[Entity]:
    """Odseci razmake sa oba kraja svakog spana i vrati očišćen skup.

    """
    output: set[Entity] = set()
    for ad_id, field, start, end, label in spans:
        text = str(ads_by_id[ad_id].get(field, ""))
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            output.add((ad_id, field, start, end, label))
    return output


def boundary_metrics_for_label(
    human: set[Entity], llm: set[Entity], label: str
) -> tuple[dict, list[dict]]:
    """Mere preklapanja granica za jednu oznaku.

    jedna oblast može da sadrži više spanova sa jedne i više sa druge
    strane (kada je jedna strana pojam iscepkala ili spojila), ili spanove
    samo sa jedne strane — to je onda pojam koji druga strana nije uočila.

    Za oblast u kojoj obe strane imaju bar jedan span računaju se:
      - hull-IoU: presek kroz uniju dva omotača, gde je omotač raspon od
        najmanjeg početka do najvećeg kraja svih spanova te strane u oblasti;
        gleda samo gde pojam počinje i gde se završava,
      - hull-Dice: dvostruki presek omotača kroz zbir njihovih dužina; blaži
        je od IoU jer preklapanje broji u odnosu na obe dužine,
      - union-IoU: isto kao IoU, ali nad stvarno pokrivenim karakterima
        (spojenim intervalima) umesto nad omotačem, pa rupe između iscepkanih
        spanova ne ulaze u pokrivenost,
      - pomeraji početka i kraja u karakterima i podatak da li su omotači
        potpuno isti.

    """
    human_label = {item for item in human if item[-1] == label}
    llm_label = {item for item in llm if item[-1] == label}
    components = connected_components(human_label, llm_label)
    records: list[dict] = []
    pooled_dice_numerator = pooled_dice_denominator = 0

    for left, right in components:
        record = {
            "label": label,
            "human_spans": len(left),
            "llm_spans": len(right),
            "paired": bool(left and right),
            "hull_iou": 0.0,
            "hull_dice": 0.0,
            "union_iou": 0.0,
            "start_shift": None,
            "end_shift": None,
            "same_hull": False,
        }
        if left and right:
            left_hull = (min(item[2] for item in left), max(item[3] for item in left))
            right_hull = (min(item[2] for item in right), max(item[3] for item in right))
            hull_intersection = max(
                0, min(left_hull[1], right_hull[1]) - max(left_hull[0], right_hull[0])
            )
            hull_union = max(left_hull[1], right_hull[1]) - min(left_hull[0], right_hull[0])
            hull_denominator = (left_hull[1] - left_hull[0]) + (
                right_hull[1] - right_hull[0]
            )
            # Za union-IoU se ne gleda omotač nego stvarno pokriveni karakteri,
            # pa se spanovi svake strane prvo spoje u razdvojene intervale.
            left_union = merge_intervals(left)
            right_union = merge_intervals(right)
            union_intersection = intersection_length(left_union, right_union)
            union_total = (
                interval_length(left_union)
                + interval_length(right_union)
                - union_intersection
            )
            record.update(
                {
                    "hull_iou": safe_div(hull_intersection, hull_union),
                    "hull_dice": safe_div(2 * hull_intersection, hull_denominator),
                    "union_iou": safe_div(union_intersection, union_total),
                    "start_shift": abs(left_hull[0] - right_hull[0]),
                    "end_shift": abs(left_hull[1] - right_hull[1]),
                    "same_hull": left_hull == right_hull,
                }
            )
            # Pooled hull-Dice se skuplja kao jedan razlomak preko svih
            # uparenih oblasti (zbir preseka kroz zbir dužina), a ne kao
            # prosek pojedinačnih Dice vrednosti; tako duži pojmovi imaju
            # srazmerno veći uticaj nego kratki.
            pooled_dice_numerator += 2 * hull_intersection
            pooled_dice_denominator += hull_denominator
        records.append(record)

    paired = [row for row in records if row["paired"]]
    summary = {
        "human_spans": len(human_label),
        "llm_spans": len(llm_label),
        "strict": strict(human_label, llm_label),
        "overlap_groups": len(records),
        "paired_groups": len(paired),
        "macro_hull_iou_all_groups": safe_div(
            math.fsum(float(row["hull_iou"]) for row in records), len(records)
        ),
        "pooled_hull_dice_paired_groups": safe_div(
            pooled_dice_numerator, pooled_dice_denominator
        ),
        "macro_union_iou_all_groups": safe_div(
            math.fsum(float(row["union_iou"]) for row in records), len(records)
        ),
    }
    return summary, records


def main() -> int:
    
    human_ads = load_examples(DATA / "rucne_anotacije_800_oglasa_5_oznaka.json")
    llm_ads = load_examples(DATA / "llm_prvi_prolaz_1000_oglasa_11_oznaka.json")
    validation = validate_inputs(human_ads, llm_ads)
    human_ids = {str(ad["id"]) for ad in human_ads}
    human = entities(human_ads, LABEL_SET)
    llm = entities(llm_ads, LABEL_SET, human_ids)
    ads_by_id = {str(ad["id"]): ad for ad in human_ads}

    # Od karakterskih spanova do tokena: ``semantic_grid`` deli naslov i telo
    # svakog oglasa tokenizatorom koji gleda samo tekst, nikad anotacije, pa
    # svakom tokenu dodeljuje oznaku onog spana sa kojim se najviše preklapa,
    # odnosno ``O`` ako ga ne dodiruje nijedan span. Oba niza se prave nad
    # istim tekstovima, pa su iste dužine i poravnata token po token.
    human_grid = semantic_grid(human_ads, human)
    llm_grid = semantic_grid(human_ads, llm)
    class_order = ["O", *LABELS]
    full_token = classification_metrics(human_grid, llm_grid, class_order)
    full_token["ner_micro_without_O"] = ner_micro(full_token["per_class"], LABELS)

    binary_human = ["O" if label == "O" else "ENTITET" for label in human_grid]
    binary_llm = ["O" if label == "O" else "ENTITET" for label in llm_grid]
    binary = classification_metrics(binary_human, binary_llm, ["O", "ENTITET"])

    both_positive = [
        (first, second)
        for first, second in zip(human_grid, llm_grid)
        if first != "O" and second != "O"
    ]
    conditional = classification_metrics(
        [first for first, _ in both_positive],
        [second for _, second in both_positive],
        list(LABELS),
    )

    # Raspodela neslaganja po vrsti: model označio tamo gde ručna strana nije,
    # ručna strana označila tamo gde model nije, i slučaj kada obe označe isti
    # token ali različitom oznakom. Zbir te tri vrste je ukupno neslaganje.
    llm_tag_human_o = sum(
        first == "O" and second != "O" for first, second in zip(human_grid, llm_grid)
    )
    human_tag_llm_o = sum(
        first != "O" and second == "O" for first, second in zip(human_grid, llm_grid)
    )
    different_non_o = sum(
        first != "O" and second != "O" and first != second
        for first, second in zip(human_grid, llm_grid)
    )
    disagreements = llm_tag_human_o + human_tag_llm_o + different_non_o

    # Od ovog mesta se prelazi sa tokena na granice spanova: za svaku oznaku
    # posebno, pa se svi zapisi po oblasti skupljaju u jednu listu.
    per_label_boundary: dict[str, dict] = {}
    all_records: list[dict] = []
    for label in LABELS:
        summary, records = boundary_metrics_for_label(human, llm, label)
        per_label_boundary[label] = summary
        all_records.extend(records)

    paired = [row for row in all_records if row["paired"]]
    
    # ``1:1`` je ista podela, ``1:vise`` znači da je model isti pojam
    # iscepkao, ``vise:1`` da ga je spojio, a ``0:1`` i ``1:0`` su oblasti
    # koje postoje samo kod jedne strane.
    split_merge = Counter()
    for row in all_records:
        left, right = int(row["human_spans"]), int(row["llm_spans"])
        if left == 0:
            split_merge["0:1"] += 1
        elif right == 0:
            split_merge["1:0"] += 1
        elif left == 1 and right == 1:
            split_merge["1:1"] += 1
        elif left == 1 and right > 1:
            split_merge["1:vise"] += 1
        elif left > 1 and right == 1:
            split_merge["vise:1"] += 1
        else:
            split_merge["vise:vise"] += 1

    starts = [int(row["start_shift"]) for row in paired]
    ends = [int(row["end_shift"]) for row in paired]
    hull_ious = [float(row["hull_iou"]) for row in paired]
    # Pooled hull-Dice za sve oznake zajedno. Ne dobija se prosekom vrednosti
    # po oznakama, jer je to jedan razlomak, pa se brojilac i imenilac ponovo
    # skupljaju u prolazu kroz sve oznake.
    pooled_num = pooled_den = 0
    for label in LABELS:
        components = connected_components(
            {item for item in human if item[-1] == label},
            {item for item in llm if item[-1] == label},
        )
        for left, right in components:
            if not left or not right:
                continue
            left_start, left_end = min(item[2] for item in left), max(item[3] for item in left)
            right_start, right_end = min(item[2] for item in right), max(item[3] for item in right)
            intersection = max(0, min(left_end, right_end) - max(left_start, right_start))
            pooled_num += 2 * intersection
            pooled_den += (left_end - left_start) + (right_end - right_start)

    strict_total = strict(human, llm)
    whitespace_strict = strict(
        trim_whitespace(human, ads_by_id), trim_whitespace(llm, ads_by_id)
    )
    split_merge_count = (
        split_merge["1:vise"] + split_merge["vise:1"] + split_merge["vise:vise"]
    )

    result = {
        "metadata": {
            "ads": len(human_ads),
            "tokens": len(human_grid),
            "labels": list(LABELS),
            "human_spans": len(human),
            "llm_spans": len(llm),
            "source_note": "Nezavisan proračun ovom projektnom skriptom nad izvornim podacima.",
            "validation": validation,
        },
        "token_semantics_O_plus_5_labels": full_token,
        "entity_presence_O_vs_ENTITY": binary,
        "conditional_label_when_both_entity": conditional,
        "token_disagreements": {
            "llm_tag_human_O": llm_tag_human_o,
            "llm_tag_human_O_share": safe_div(llm_tag_human_o, disagreements),
            "human_tag_llm_O": human_tag_llm_o,
            "human_tag_llm_O_share": safe_div(human_tag_llm_o, disagreements),
            "different_non_O_label": different_non_o,
            "different_non_O_label_share": safe_div(different_non_o, disagreements),
            "total": disagreements,
            "tag_vs_O_total": llm_tag_human_o + human_tag_llm_o,
            "tag_vs_O_share": safe_div(llm_tag_human_o + human_tag_llm_o, disagreements),
        },
        "boundary_analysis": {
            "strict": strict_total,
            "whitespace_strict": whitespace_strict,
            "whitespace_changed_result": strict_total != whitespace_strict,
            "per_label": per_label_boundary,
            "all_groups": len(all_records),
            "paired_groups": len(paired),
            "paired_groups_share": safe_div(len(paired), len(all_records)),
            "macro_hull_iou_all_groups": safe_div(
                math.fsum(float(row["hull_iou"]) for row in all_records), len(all_records)
            ),
            "pooled_hull_dice_paired_groups": safe_div(pooled_num, pooled_den),
            "macro_union_iou_all_groups": safe_div(
                math.fsum(float(row["union_iou"]) for row in all_records), len(all_records)
            ),
            # Koliko su granice blizu kada obe strane vide isti pojam: pragovi
            # hull-IoU 0,50 i 0,80, isti početak, isti kraj i pomeraji do
            # jednog odnosno dva karaktera.
            "paired_boundary_closeness": {
                "median_hull_iou": statistics.median(hull_ious),
                "hull_iou_ge_0_50_count": sum(value >= 0.50 for value in hull_ious),
                "hull_iou_ge_0_50_share": safe_div(
                    sum(value >= 0.50 for value in hull_ious), len(paired)
                ),
                "hull_iou_ge_0_80_count": sum(value >= 0.80 for value in hull_ious),
                "hull_iou_ge_0_80_share": safe_div(
                    sum(value >= 0.80 for value in hull_ious), len(paired)
                ),
                "same_start_count": sum(value == 0 for value in starts),
                "same_start_share": safe_div(sum(value == 0 for value in starts), len(paired)),
                "same_end_count": sum(value == 0 for value in ends),
                "same_end_share": safe_div(sum(value == 0 for value in ends), len(paired)),
                "same_hull_count": sum(bool(row["same_hull"]) for row in paired),
                "same_hull_share": safe_div(
                    sum(bool(row["same_hull"]) for row in paired), len(paired)
                ),
                "both_ends_within_1_count": sum(
                    first <= 1 and second <= 1 for first, second in zip(starts, ends)
                ),
                "both_ends_within_1_share": safe_div(
                    sum(first <= 1 and second <= 1 for first, second in zip(starts, ends)),
                    len(paired),
                ),
                "both_ends_within_2_count": sum(
                    first <= 2 and second <= 2 for first, second in zip(starts, ends)
                ),
                "both_ends_within_2_share": safe_div(
                    sum(first <= 2 and second <= 2 for first, second in zip(starts, ends)),
                    len(paired),
                ),
                "median_start_shift_chars": statistics.median(starts),
                "median_end_shift_chars": statistics.median(ends),
            },
            "segmentation_components": {
                "counts": dict(split_merge),
                "split_merge_paired_count": split_merge_count,
                "split_merge_paired_share": safe_div(split_merge_count, len(paired)),
                "disjoint_count": split_merge["0:1"] + split_merge["1:0"],
            },
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "dodatne_mere_tokena_i_granica.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
