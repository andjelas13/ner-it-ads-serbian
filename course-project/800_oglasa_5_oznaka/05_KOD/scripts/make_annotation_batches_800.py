"""Bira 800 oglasa iz korpusa od 1000 i deli ih na cetiri dela po 200."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


# Seed je fiksiran da bi podela uvek bila ista. Ne menjati: po ovoj podeli su
# oglasi vec anotirani i predati.
SEED_SELECT = 20260904

TOTAL_SELECTED = 800          # koliko oglasa se bira iz korpusa od 1000
BATCHES = 4                   # broj anotatora
BATCH_SIZE = TOTAL_SELECTED // BATCHES   # 200 oglasa po anotatoru

LABELS = ["CPU", "GPU", "RAM", "SKLADISTE", "CENA"]


# učitava oglase iz JSON-a; prihvata i objekat sa examples i golu listu
def load_corpus(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["examples"] if isinstance(payload, dict) else payload


# razvrstava oglase po kategoriji i promeša svaku grupu
def group_by_category(ads: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for ad in ads:
        buckets[ad.get("category", "?")].append(ad)
    for key in buckets:
        rng.shuffle(buckets[key])
    return buckets


# bira zadati broj oglasa tako da udeli kategorija prate korpus
def select_proportional(buckets: dict[str, list[dict]], total_source: int,
                        target: int, rng: random.Random) -> list[dict]:
    selected: list[dict] = []
    leftover: list[dict] = []
    for category in sorted(buckets):
        pool = buckets[category]
        take = round(target * len(pool) / total_source)
        selected.extend(pool[:take])
        leftover.extend(pool[take:])
    # Zaokruzivanje po kategorijama retko da tacno 800, pa se razlika dopuni
    # ili skrati iz ostatka.
    rng.shuffle(leftover)
    while len(selected) < target:
        selected.append(leftover.pop())
    while len(selected) > target:
        leftover.append(selected.pop())
    return selected


# deli oglase naizmenično na četiri dela, unutar svake kategorije
def split_round_robin(selected: list[dict], rng: random.Random) -> list[list[dict]]:
    parts: list[list[dict]] = [[] for _ in range(BATCHES)]
    by_category: dict[str, list[dict]] = defaultdict(list)
    for ad in selected:
        by_category[ad.get("category", "?")].append(ad)
    # Naizmenicna podela unutar kategorije znaci da svaki deo dobija priblizno
    # isti udeo svake vrste oglasa.
    for category in sorted(by_category):
        pool = by_category[category]
        rng.shuffle(pool)
        for index, ad in enumerate(pool):
            parts[index % BATCHES].append(ad)
    for part in parts:
        rng.shuffle(part)
    return parts


# premešta oglase dok svaki deo ne dobije tačno 200
def balance_to_equal_size(parts: list[list[dict]]) -> None:
    sizes = lambda: [len(p) for p in parts]
    while max(sizes()) > BATCH_SIZE:
        source = sizes().index(max(sizes()))
        target = sizes().index(min(sizes()))
        counts_source: dict[str, int] = defaultdict(int)
        counts_target: dict[str, int] = defaultdict(int)
        for ad in parts[source]:
            counts_source[ad.get("category", "?")] += 1
        for ad in parts[target]:
            counts_target[ad.get("category", "?")] += 1
        moved = max(counts_source, key=lambda c: counts_source[c] - counts_target[c])
        for position, ad in enumerate(parts[source]):
            if ad.get("category", "?") == moved:
                parts[target].append(parts[source].pop(position))
                break


# ispisuje tabelu: udeo kategorije u korpusu i koliko je pripalo kom delu
def report(source: list[dict], parts: list[list[dict]]) -> None:
    total = len(source)
    categories = sorted({ad.get("category", "?") for ad in source})
    header = f"{'kategorija':20}{'korpus %':>10}   " + "".join(
        f"{'deo ' + str(i + 1):>8}" for i in range(BATCHES)
    )
    print(header)
    for category in categories:
        in_source = sum(1 for ad in source if ad.get("category") == category)
        row = "".join(
            f"{sum(1 for ad in part if ad.get('category') == category):8}"
            for part in parts
        )
        print(f"{category:20}{100 * in_source / total:9.1f}%   {row}")
    print(f"{'UKUPNO':20}{'':10}   " + "".join(f"{len(p):8}" for p in parts))


# ceo posao: učita korpus, izabere 800, podeli na četiri dela i upiše podelu
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ulaz", type=Path,
                        default=Path("data/annotations/final_span_annotations.json"),
                        help="JSON sa 1000 oglasa")
    parser.add_argument("--izlaz", type=Path, default=Path("podela_800_4x200.json"),
                        help="gde se upisuje podela (podrazumevano podela_800_4x200.json)")
    args = parser.parse_args()

    print(f"[INFO] Ulazni korpus: {args.ulaz}")
    ads = load_corpus(args.ulaz)
    print(f"[INFO] Ucitano oglasa: {len(ads):,}")

    # Redosled poziva generatora slucajnih brojeva odredjuje podelu i ne sme se
    # menjati -- po ovoj podeli su oglasi vec anotirani.
    rng_select = random.Random(SEED_SELECT)
    buckets = group_by_category(ads, rng_select)
    selected = select_proportional(buckets, len(ads), TOTAL_SELECTED, rng_select)
    parts = split_round_robin(selected, rng_select)
    balance_to_equal_size(parts)

    # Provere: tacno 800 oglasa, tacno 200 po delu, bez preklapanja.
    assert sum(len(p) for p in parts) == TOTAL_SELECTED
    assert all(len(p) == BATCH_SIZE for p in parts)
    ids = [ad["id"] for part in parts for ad in part]
    assert len(set(ids)) == TOTAL_SELECTED, "preklapanje id-jeva izmedju delova"

    print()
    report(ads, parts)
    print()

    args.izlaz.parent.mkdir(parents=True, exist_ok=True)
    args.izlaz.write_text(
        json.dumps(
            {
                "seed_select": SEED_SELECT,
                "labels": LABELS,
                "batch_size": BATCH_SIZE,
                "parts": [[ad["id"] for ad in part] for part in parts],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Upisano {TOTAL_SELECTED:,} oglasa u {BATCHES} dela -> {args.izlaz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
