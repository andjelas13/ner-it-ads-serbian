#!/usr/bin/env python3

"""
    Prikuplja 1.000 javnih IT oglasa sa Halo Oglasa i KupujemProdajem.

    Skripta je namerno restartabilana: svaki uspešno preuzet oglas upisuje se u
    checkpoint. Konačni JSON/JSONL/CSV nastaju tek kada su sve kvote popunjene.
"""

import argparse
import csv
import html as html_module
import json
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


# Omogucava konzoli da ispise karaktere poput č, ć, š, đ itd.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except OSError:
        pass

# Direktorijumi/ulazni/izlazni fajlovi
ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
CHECKPOINT = ROOT / "checkpoint_oglasi.jsonl"
OUTPUT_JSON = ROOT / "IT_oglasi_1000.json"
OUTPUT_JSONL = ROOT / "IT_oglasi_1000.jsonl"
OUTPUT_CSV = ROOT / "IT_oglasi_1000_manifest.csv"
REPORT_MD = ROOT / "IZVESTAJ_O_SKUPU.md"
CANDIDATES_JSON = ROOT / "candidate_pools.json"
WEB_RECOVERY_JSONL = ROOT / "kp_web_recovery.jsonl"

HALO_LIST_BASE = "https://smsprint.halooglasi.com"
HALO_CANONICAL_BASE = "https://www.halooglasi.com"
KP_BASE = "https://www.kupujemprodajem.com"
KP_HEALTHCHECK_URL = (
    "https://www.kupujemprodajem.com/kompjuteri-laptop-i-tablet/laptopovi/"
    "lenovo-13th-i3-1315u-40gb-ddr4-ram-1tb-nvme-ssd/oglas/192208041"
)

SEED = 20260824
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)

# Kategorije i ciljana raspodela: 700 Halo oglasa i 300 KupujemProdajem oglasa.
# QUOTAS čuva relativne težine kategorija, a SOURCE_QUOTAS njihove tačne kvote.
QUOTAS: dict[str, int] = {
    "Laptop": 65,
    "Telefon": 65,
    "Tablet": 35,
    "Desktop računar": 55,
    "Monitor": 35,
    "GPU": 50,
    "RAM": 45,
    "Skladište": 60,
    "CPU": 35,
    "Baterija": 25,
    "Matična ploča": 15,
    "Napajanje": 15,
}

SOURCE_QUOTAS: dict[str, dict[str, int]] = {
    "Halo Oglasi": {
        "Laptop": 91,
        "Telefon": 91,
        "Tablet": 49,
        "Desktop računar": 77,
        "Monitor": 49,
        "GPU": 70,
        "RAM": 63,
        "Skladište": 84,
        "CPU": 49,
        "Baterija": 35,
        "Matična ploča": 21,
        "Napajanje": 21,
    },
    "KupujemProdajem": {
        "Laptop": 45,
        "Telefon": 45,
        "Tablet": 25,
        "Desktop računar": 35,
        "Monitor": 25,
        "GPU": 30,
        "RAM": 25,
        "Skladište": 35,
        "CPU": 15,
        "Baterija": 8,
        "Matična ploča": 7,
        "Napajanje": 5,
    },
}


# Usluzna klasa koja opisuje kategoriju HaloOglasa
# dataclass anotacija generise dunder (double underscore (__)) metode automatski.
# argument frozen cini polja klase immutable (ne mogu se menjati ili izbrisati)
@dataclass(frozen=True)
class HaloCategory:
    label: str
    paths: tuple[str, ...]
    require_pattern: str | None = None


# Torka koja sadrzi sve dozvoljene kategorije za scrape-ovane HaloOglase
HALO_CATEGORIES: tuple[HaloCategory, ...] = (
    HaloCategory("Laptop", ("/racunari/laptop-racunari",)),
    HaloCategory("Telefon", ("/telefoni",), require_pattern=r"mobilni telefon|smartfon|iphone|telefon"),
    HaloCategory("Tablet", ("/racunari/tablet-racunari",)),
    HaloCategory("Desktop računar", ("/racunari/desktop-racunari",)),
    HaloCategory("Monitor", ("/racunari/lcd-monitori", "/racunari/crt-monitori")),
    HaloCategory("GPU", ("/racunari/graficke-karte",)),
    HaloCategory("RAM", ("/racunari/memorije",)),
    HaloCategory(
        "Skladište",
        (
            "/racunari/hard-diskovi",
            "/racunari/ssd-diskovi",
            "/racunari/eksterni-hard-diskovi",
            "/racunari/eksterni-ssd",
            "/racunari/usb-flash-memorija",
        ),
    ),
    HaloCategory("CPU", ("/racunari/procesori",)),
    HaloCategory(
        "Baterija",
        ("/racunari/punjaci-za-racunare", "/telefoni/baterije", "/telefoni/iphone-baterije", "/telefoni/samsung-baterije"),
        require_pattern=r"baterij|battery|akku|akumulator|power\s*bank",
    ),
    HaloCategory("Matična ploča", ("/racunari/maticne-ploce",)),
    HaloCategory("Napajanje", ("/racunari/napajanja",)),
)


# Pretrage služe samo za pronalaženje kandidata. Konačna kategorija se proverava
# iz putanje detaljnog oglasa, pa rezultat za npr. "monitor" ne može slučajno
# uključiti bebi-monitor ili auto-kameru.
KP_SEARCHES: tuple[str, ...] = (
    "laptop",
    "polovan laptop",
    "telefon",
    "mobilni telefon",
    "tablet",
    "tablet racunar",
    "desktop racunar",
    "gaming pc racunar",
    "mini pc",
    "monitor racunar",
    "lcd monitor",
    "graficka karta",
    "gpu graficka",
    "ram memorija",
    "ddr4 ram",
    "ssd disk",
    "hdd disk",
    "nvme",
    "procesor cpu",
    "intel amd procesor",
    "baterija laptop",
    "baterija za laptop",
    "maticna ploca",
    "motherboard maticna",
    "napajanje racunar",
    "atx napajanje psu",
)


KP_PATH_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Baterija", ("/kompjuteri-laptop-i-tablet/laptopovi-baterije/",)),
    ("RAM", ("/kompjuteri-desktop/ram-memorije/", "/kompjuteri-laptop-i-tablet/laptopovi-memorije/")),
    (
        "Skladište",
        (
            "/kompjuteri-desktop/hard-diskovi/",
            "/kompjuteri-desktop/hard-diskovi-ssd/",
            "/kompjuteri-desktop/hard-diskovi-eksterni/",
            "/kompjuteri-laptop-i-tablet/laptopovi-hard-diskovi/",
            "/kompjuteri-desktop/usb-flash/",
        ),
    ),
    ("GPU", ("/kompjuteri-desktop/graficke-kartice/",)),
    ("CPU", ("/kompjuteri-desktop/procesori/", "/kompjuteri-laptop-i-tablet/laptopovi-procesori/")),
    ("Monitor", ("/kompjuteri-desktop/monitori/",)),
    ("Matična ploča", ("/kompjuteri-desktop/maticne-ploce/", "/kompjuteri-laptop-i-tablet/laptopovi-maticne-ploce/")),
    ("Napajanje", ("/kompjuteri-desktop/napajanja/",)),
    ("Tablet", ("/kompjuteri-laptop-i-tablet/tableti/",)),
    ("Laptop", ("/kompjuteri-laptop-i-tablet/laptopovi/",)),
    (
        "Desktop računar",
        (
            "/kompjuteri-desktop/kompjuteri/",
            "/kompjuteri-desktop/polovni-kompjuteri/",
            "/kompjuteri-desktop/apple-desktop/",
            "/kompjuteri-desktop/mini-pc/",
        ),
    ),
    ("Telefon", ("/mobilni-telefoni/",)),
)


thread_local = threading.local()
request_spacing_lock = threading.Lock()
last_request_at: dict[str, float] = defaultdict(float)
kp_status_lock = threading.Lock()
kp_consecutive_block_responses = 0


# Klasa za obradu izuzetaka
class SiteTemporarilyBlocked(RuntimeError):
    """Prekini izvršavanje pre nego privremena blokada sajta blokira zahteve."""


checkpoint_lock = threading.Lock()


# Funkcija ispisuje konacne rezultate
def write_outputs(items: list[dict[str, Any]]) -> None:
    # Sortiramo oglase po izvoru, kategoriji i source_ad_id-ju
    ordered = sorted(items, key=lambda x: (str(x["source"]), str(x["category"]), str(x["source_ad_id"])))
    # Generisemo ID za svaki oglas
    for index, item in enumerate(ordered, start=1):
        item["id"] = index

    payload = {
        "schema": "it-oglasi-mass-unannotated-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "selection_seed": SEED,
        "count": len(ordered),
        "sources": ["Halo Oglasi", "KupujemProdajem"],
        "ads": ordered,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT_JSONL.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in ordered),
        encoding="utf-8",
    )
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "source", "category", "source_category", "source_ad_id", "title", "url", "body_chars", "retrieved_at"],
        )
        writer.writeheader()
        for item in ordered:
            writer.writerow(
                {
                    "id": item["id"],
                    "source": item["source"],
                    "category": item["category"],
                    "source_category": item["source_category"],
                    "source_ad_id": item["source_ad_id"],
                    "title": item["title"],
                    "url": item["url"],
                    "body_chars": len(item["body"]),
                    "retrieved_at": item["retrieved_at"],
                }
            )

    counts = count_by_source_category(ordered)
    source_counts = Counter(str(item["source"]) for item in ordered)
    total_chars = sum(len(str(item["title"])) + len(str(item["body"])) for item in ordered)
    rows = "\n".join(
        f"| {label} | {counts[('Halo Oglasi', label)]} | {counts[('KupujemProdajem', label)]} | "
        f"{counts[('Halo Oglasi', label)] + counts[('KupujemProdajem', label)]} |"
        for label in QUOTAS
    )
    # REPORT.MD generisanje
    REPORT_MD.write_text(
        "# Izveštaj o skupu od 1.000 IT oglasa\n\n"
        f"- Ukupno oglasa: **{len(ordered)}**\n"
        f"- Halo Oglasi: **{source_counts['Halo Oglasi']}**\n"
        f"- KupujemProdajem: **{source_counts['KupujemProdajem']}**\n"
        f"- Ukupan broj karaktera u naslovima i telima: **{total_chars:,}**\n"
        f"- Seed za mešanje kandidata: `{SEED}`\n"
        "- Ranijih 50 oglasa iz anotacionog alata isključeno je prema ID-u izvornog oglasa.\n"
        "- Svaki zapis ima neprazan naslov i telo, jedinstven ID izvornog oglasa i URL iz odgovarajuće IT kategorije.\n"
        "- Dataset ne sadrži NER anotacije.\n\n"
        "## Raspodela\n\n"
        "| Kategorija | Halo Oglasi | KupujemProdajem | Ukupno |\n"
        "|---|---:|---:|---:|\n"
        f"{rows}\n\n"
        "## Fajlovi\n\n"
        "- `IT_oglasi_1000.json` — glavni, čitljiv JSON sa metapodacima.\n"
        "- `IT_oglasi_1000.jsonl` — jedan oglas po redu, pogodno za masovnu obradu.\n"
        "- `IT_oglasi_1000_manifest.csv` — pregled naslova, kategorija i URL-ova.\n"
        "- `scraper.py` — restartabilni scraper.\n",
        encoding="utf-8",
    )


# Funkcija vrsi genricke provere odabranih 1000 oglasa
def validate(items: list[dict[str, Any]]) -> None:
    if len(items) != 1000:
        raise RuntimeError(f"Očekivano 1000 oglasa, dobijeno {len(items)}")
    keys = [ad_key(str(item["url"])) for item in items]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Dataset sadrži duplikate oglasa")
    for index, item in enumerate(items, start=1):
        for field in ("source", "category", "title", "body", "url", "source_ad_id"):
            if not str(item.get(field, "")).strip():
                raise RuntimeError(f"Oglas {index} nema obavezno polje {field}")
        if item["source"] not in {"Halo Oglasi", "KupujemProdajem"}:
            raise RuntimeError(f"Nepoznat izvor u oglasu {index}")
        if item["category"] not in QUOTAS:
            raise RuntimeError(f"Nepoznata kategorija u oglasu {index}")
    counts = count_by_source_category(items)
    for source in ("Halo Oglasi", "KupujemProdajem"):
        for label, quota in SOURCE_QUOTAS[source].items():
            if counts[(source, label)] != quota:
                raise RuntimeError(
                    f"Pogrešna kvota {source}/{label}: {counts[(source, label)]}, očekivano {quota}"
                )


# Funkcija od svih unique sakupljenih oglasa bira tacan broj koji ispunjava kvotu
def select_output_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # RNG daje deterministicke rezultate
    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    for source in ("Halo Oglasi", "KupujemProdajem"):
        for label, quota in SOURCE_QUOTAS[source].items():
            pool = [item for item in items if item["source"] == source and item["category"] == label]
            if len(pool) < quota:
                raise RuntimeError(
                    f"Nema dovoljno zapisa za {source}/{label}: {len(pool)}, potrebno {quota}"
                )
            rng.shuffle(pool)
            selected.extend(pool[:quota])
    return selected


# Izvlaci kategoriju za KP URL
def kp_source_category(url: str) -> str:
    parts = [part.replace("-", " ") for part in urlparse(url).path.strip("/").split("/")]
    if "oglas" in parts:
        parts = parts[: parts.index("oglas")]
    return " > ".join(parts[:2])


# Dodaje oglase u JSON cache fajl (Thread safe upis)
def append_checkpoint(item: dict[str, Any]) -> None:
    line = json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
    with checkpoint_lock:
        with CHECKPOINT.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()


# Funkcija parsira tekst i formatira tekst (naslov ili telo oglasa) u zeljenu formu
def clean_text(value: str) -> str:
    # unescape ce dohvatati ce pretvoriti specijalne karaktere \nbsp, \t itd. u stringove (# non breaking space, "  " itd.)
    value = html_module.unescape(value)
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# Funkcija vraca formatirani KP oglas uz validiranje njegove ispravnosti
def scrape_kp(url: str, expected_category: str) -> dict[str, Any] | None:
    # Dohvati oglas
    response = fetch(url)
    # Kreiraj njegovu apsolutnu putanju
    final_url = canonical_url(response.url)
    # Izvuci kategoriju
    actual_category = kp_category_from_url(final_url)
    # Provera da li se kategorije poklapaju
    if actual_category != expected_category:
        return None
    # Parsiramo dobijeni HTML
    soup = BeautifulSoup(response.text, "html.parser")
    page_text = clean_text(soup.get_text("\n"))
    lowered = page_text.lower()
    if "oglas više nije aktivan" in lowered or "oglas je obrisan" in lowered:
        return None
    heading = soup.select_one("h1")
    description = soup.select_one('[class*="AdViewDescription"][class*="sectionContent"]')
    # Parsiraj naslov
    title = clean_text(heading.get_text(" ", strip=True) if heading else "")
    # Parsiraj telo
    body = clean_text(description.get_text("\n", strip=True) if description else "")
    # Ne razmatramo oglase koji kupuju/otkpuljuju
    if re.match(r"^\s*(kupujem|otkup\b)", title, flags=re.I):
        return None
    # Ne razmatramo oglas koji nema naslov, telo ili je prekratak
    if not title or not body or len(body) < 8:
        return None
    # Dohvati ID KP oglasa
    match = re.search(r"/oglas/(\d+)", urlparse(final_url).path)
    # Vrati formatiran KP oglas
    return {
        "source": "KupujemProdajem",
        "category": actual_category,
        "source_category": kp_source_category(final_url),
        "title": title,
        "body": body,
        "url": final_url,
        "source_ad_id": match.group(1) if match else "",
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# Funkcija parsira HTML i pretvara ga u clean_text()
def html_to_text(fragment: str) -> str:
    soup = BeautifulSoup(fragment, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for element in soup.find_all(["p", "div", "li", "tr"]):
        element.append("\n")
    return clean_text(soup.get_text())


# Funkcija izvlaci JSON objekat iz stranice uz provere gresaka
def extract_json_object_after(source: str, marker: str) -> dict[str, Any]:
    position = source.find(marker)
    if position < 0:
        raise ValueError(f"Nije pronađen marker: {marker}")
    start = position + len(marker)
    value, _ = json.JSONDecoder().raw_decode(source[start:])
    if not isinstance(value, dict):
        raise ValueError("Očekivan JSON objekat")
    return value


# Funkcija dohvata HaloOglas za datu kategoriju i URL (kategorija se prosledjuje radi provere)
def scrape_halo(url: str, category: HaloCategory) -> dict[str, Any] | None:
    # Provera
    if not halo_url_matches_category(url, category):
        return None
    # Dohvati oglas
    response = fetch(url)
    # Izvuci ugradjeni JSON objekat iz stranice
    classified = extract_json_object_after(response.text, "QuidditaEnvironment.CurrentClassified=")
    # Razmatramo samo aktivne oglase
    if not classified.get("IsCurrentAdActive", True):
        return None
    # Izvlacimo potrebne podatke da bi sproveli provere
    title = clean_text(str(classified.get("Title", "")))
    body = html_to_text(str(classified.get("TextHtml", "")))
    text_for_check = f"{title}\n{body}".lower()
    # Provera za oglase iz kategorije telefona i baterije koje ne prolaze proveru definisanu u HaloCategory.require_pattern polju
    if category.require_pattern and not re.search(category.require_pattern, text_for_check, flags=re.I):
        return None
    # Ne razmatramo oglase koji poicnju sa kupujem i otkup
    if re.match(r"^\s*(kupujem|otkup\b)", title, flags=re.I):
        return None
    # Ne razmatramo oglase koji nemaju naslov, telo ili su prekratki
    if not title or not body or len(body) < 8:
        return None
    names = [clean_text(str(x)) for x in classified.get("CategoryNames", [])]
    canonical = canonical_url(response.url)
    # Vrati formatiran oglas
    return {
        "source": "Halo Oglasi",
        "category": category.label,
        "source_category": " > ".join(names),
        "title": title,
        "body": body,
        "url": canonical,
        "source_ad_id": str(classified.get("Id", "")),
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# Funkcija otvara grupu oglasa i proverava njihov sadrzaj
def collect_group(source: str, label: str, candidates: list[str], already: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    # Dohvati broj oglasa koji je potrebno scrape-ovati
    target = SOURCE_QUOTAS[source][label]
    # Broj oglasa koji su vec scrape-ovani
    have = sum(1 for item in already if item["source"] == source and item["category"] == label)
    # Broj oglasa koji fali
    need = max(0, target - have)
    # Funkcija nece nastaviti ako je taj broj 0
    if need == 0:
        return []
    # Kreiramo listu postojecih oglasa (njihovih kljuceva) kako ne bi dobili duplikate
    existing_keys = { ad_key(str(item["url"])) for item in already }
    # Kreiramo listu kandidata koje cemo da ispitujemo
    candidates = [url for url in candidates if ad_key(url) not in existing_keys]
    # Za KP oglase prioritizujemo novije (noviji imaju veci ID po pravilu, ako ne mozemo da izvucemo ID stavicemo ih na dno liste (ID = 0))
    if source == "KupujemProdajem":
        candidates.sort(
            key=lambda url: int(re.search(r"/oglas/(\d+)", url).group(1))
            if re.search(r"/oglas/(\d+)", url)
            else 0,
            reverse=True,
        )
    # Warning da nije sakupljeno dovoljno oglasa kandidata
    if len(candidates) < need:
        print(f"Upozorenje: {source}/{label} ima {len(candidates)} kandidata za još {need} oglasa", file=sys.stderr)

    # Ovaj objekat se koristi za scraping HaloOglasa (scrape_halo() mora da proveri URL i kategoriju)
    # Za KP oglase se ovap rovera ne radi, kategorija se nasledjuje iz konacnog URL-a
    category_object = next((x for x in HALO_CATEGORIES if x.label == label), None)
    results: list[dict[str, Any]] = []
    attempted = 0  # Broji pokusaje pristupa jednom oglasu
    cursor = 0  # Sluzi kao pokazivac na trenutni oglas
    # Radimo u manjim talasima: cim se kvota popuni, ne saljemo nepotrebne zahteve.
    while len(results) < need and cursor < len(candidates):
        # Ukoliko ne podizemo vise niti
        if workers == 1:
            url = candidates[cursor]
            cursor += 1
            attempted += 1
            # Proveri da li se radi o KP ili HaloOglasi oglasu i respektivno pozovi funkcije za dohvatanje
            try:
                if source == "Halo Oglasi":
                    assert category_object is not None
                    item = scrape_halo(url, category_object)
                else:
                    item = scrape_kp(url, label)
            except SiteTemporarilyBlocked:
                raise
            except Exception as exc:
                print(f"  Preskačem {url}: {exc}", file=sys.stderr)
                item = None
            # Dodaj ispravne oglase u listu
            if item is not None:
                key = ad_key(str(item["url"]))
                if key not in existing_keys:
                    existing_keys.add(key)
                    results.append(item)
                    append_checkpoint(item)
            # Ako ne uspemo posle 10 pokusaja odustajemo od tog linka
            if attempted % 10 == 0 or len(results) >= need:
                print(
                    f"  {source}/{label}: {have + len(results)}/{target} "
                    f"(pokušano {attempted}/{len(candidates)})",
                    flush=True,
                )
            continue

        # Ukoliko radimo sa vise niti, dohvatamo oglase u batch-ovima
        batch_size = min(max(workers * 3, need - len(results)), len(candidates) - cursor)
        batch = candidates[cursor: cursor + batch_size]
        cursor += batch_size
        # Podizemo niti u paraleli
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_url = {}
            # Za svaki URL u batch-u podizemo jednog worker-a
            for url in batch:
                if source == "Halo Oglasi":
                    assert category_object is not None
                    future = executor.submit(scrape_halo, url, category_object)
                else:
                    future = executor.submit(scrape_kp, url, label)
                # Ovde se cuvaju Future objekti
                future_to_url[future] = url
            # Yieldujemo na as_completed kada se zavrsi nit
            for future in as_completed(future_to_url):
                attempted += 1
                try:
                    item = future.result()
                except SiteTemporarilyBlocked:
                    raise
                except Exception as exc:
                    print(f"  Preskačem {future_to_url[future]}: {exc}", file=sys.stderr)
                    continue
                if item is None:
                    continue
                # Izvuci kljuc iz URL
                key = ad_key(str(item["url"]))
                # Ne dodajemo duplikate
                if key in existing_keys:
                    continue
                # Dodaj novi oglas u listu
                existing_keys.add(key)
                results.append(item)
                append_checkpoint(item)
                if len(results) >= need:
                    break
        print(
            f"  {source}/{label}: {have + len(results)}/{target} "
            f"(pokušano {attempted}/{len(candidates)})",
            flush=True,
        )
        time.sleep(0.15)
    return results[:need]


# Funkcija cuva trenutno stanje u cache
# Pretvaramo pools recnik u JSON format
def save_candidate_cache(pools: dict[tuple[str, str], list[str]]) -> None:
    raw = {
        f"{source}\t{label}": list(dict.fromkeys(urls))
        for (source, label), urls in pools.items()
        if urls
    }
    CANDIDATES_JSON.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# Funkcija dohvata kategoriju za datu putanju koristeci internu listu KP_PATH_RULES
# Primer: /kompjuteri-desktop/graficke-kartice/ -> ("GPU", ("/kompjuteri-desktop/graficke-kartice/",))
def kp_category_from_url(url: str) -> str | None:
    path = urlparse(url).path.lower()
    for label, fragments in KP_PATH_RULES:
        if any(fragment in path for fragment in fragments):
            return label
    return None


# Funkcija kreira formatiran KP URL za dati query i stranicu
def kp_search_url(query: str, page: int) -> str:
    # urlencode se koristi da lepo slozi stvari u link, kako ne bi doslo do gresaka kao u slucaju manuelnog kreiranja linka
    return f"{KP_BASE}/pretraga?{urlencode({'keywords': query, 'so': '1', 'page': str(page)})}"


# Funkcija vraca listu kandidata URL za jednu kategoriju (bez duplikata)
def kp_search_links(query: str, pages: int) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    # Iteriramo po stranicama kategorije
    for page in range(1, pages + 1):
        # Dohvatamo stranicu
        soup = BeautifulSoup(fetch(kp_search_url(query, page)).text, "html.parser")
        page_count = 0
        # Dohvatamo svaki <a> tag sa href atributom
        for anchor in soup.select('a[href*="/oglas/"]'):
            # Dohvati href
            href = anchor.get("href", "")
            # Izvuci ID iz href
            if not re.search(r"/oglas/\d+", href):
                continue
            # Od relativne putanje href pravimo apsolutnu
            url = canonical_url(urljoin(KP_BASE, href))
            # Dohvati ID tog url-a
            key = ad_key(url)
            # Ako se oglas nije pojavljivao dodaj ga u listu kandidata
            if key not in seen:
                seen.add(key)
                links.append(url)
                page_count += 1
        # Ne nastavljamo pretragu ako na celoj stranici nije pronadjen nijedan kandidat
        if page_count == 0:
            break
        time.sleep(0.12)
    return links


# Funkcija uspostavlja sesiju za svaku nit koja je poziva
# Sesija se uspostavlja jednom za svaku nit, a onda se obradjuju zahtevi niti
def session() -> requests.Session:
    # Threadlocal omogucava da se dohvati aktivna nit, da ne bi doslo do slucaja u kom dve niti dele jednu sesiju
    value = getattr(thread_local, "session", None)
    if value is None:
        value = requests.Session()
        # Parametri sesije
        value.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept-Language": "sr-RS,sr;q=0.9,en;q=0.7",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )
        thread_local.session = value
    return value


# Funkcija koja usporava zahteve kako ne bi opteretili server
def throttle(url: str) -> None:
    host = urlparse(url).netloc.lower()
    minimum_gap = 6.0 if "kupujemprodajem.com" in host else 0.10
    # Brava ne dozvoljava nitima da preskoce delay
    with request_spacing_lock:
        # time.monotonic() fja nije osetljiva na promene sys-clocka
        # ona trenutno (fragment sekunde) vraca vreme i koristi se kao merac vremena (razlika izmedju dva poziva)
        remaining = last_request_at[host] + minimum_gap - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        last_request_at[host] = time.monotonic()


# Funkcija dohvata stranicu sa linka
def fetch(url: str, attempts: int = 4) -> requests.Response:
    global kp_consecutive_block_responses
    last_error: Exception | None = None
    # Pokusavamo 4 puta
    for attempt in range(attempts):
        try:
            # Delay
            throttle(url)
            # 15s da uspostavimo konekciju, 45s da procitamo odgovor
            response = session().get(url, timeout=(15, 45), allow_redirects=True)
            # Uspesno dohvatanje
            if response.status_code == 200:
                if "kupujemprodajem.com" in urlparse(url).netloc.lower():
                    with kp_status_lock:
                        kp_consecutive_block_responses = 0
                return response
            # Ukoliko dobijemo error pokusavamo 4 puta
            # Ukoliko i dalje dobijamo error, proverimo healthcheck sajta (hardcode oglas koji sigurno radi)
            # Ako i ta provera ne prodje ispisujemo gresku
            if response.status_code in {404, 410} and "kupujemprodajem.com" in urlparse(url).netloc.lower():
                with kp_status_lock:
                    kp_consecutive_block_responses += 1
                    blocked = kp_consecutive_block_responses >= 3
                if blocked:
                    throttle(KP_HEALTHCHECK_URL)
                    health = session().get(
                        KP_HEALTHCHECK_URL,
                        timeout=(15, 45),
                        allow_redirects=True,
                    )
                    if health.status_code == 200 and health.content:
                        with kp_status_lock:
                            kp_consecutive_block_responses = 0
                    else:
                        raise SiteTemporarilyBlocked(
                            "KupujemProdajem returned empty HTTP 404/410 responses and the "
                            "known-active health check failed; the checkpoint is safe."
                        )
                return response
            if response.status_code not in {403, 408, 425, 429, 500, 502, 503, 504}:
                response.raise_for_status()
            last_error = RuntimeError(f"HTTP {response.status_code}: {url}")
        except SiteTemporarilyBlocked:
            raise
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
        time.sleep(1.0 * (2**attempt) + random.random() * 0.4)
    raise RuntimeError(str(last_error) if last_error else f"Neuspešno preuzimanje: {url}")


# Pomocna funkcija za kreiranje url za datu putanju i stranus
def listing_page_url(path: str, page: int) -> str:
    # Provera dozvoljava funkciji da radi cak iako path sadrzi query string
    separator = "&" if "?" in path else "?"
    return urljoin(HALO_LIST_BASE, f"{path}{separator}page={page}")


# Funkcija vraca listu kandidata URL za jednu kategoriju (bez duplikata) za HaloOglase
def halo_listing_links(path: str, pages: int) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for page in range(1, pages + 1):
        # Dohvatamo celu stranu
        soup = BeautifulSoup(fetch(listing_page_url(path, page)).text, "html.parser")
        page_links: list[str] = []
        # Iteriramo po svim <a> elementima koji imaju href atribut
        for anchor in soup.select('a[href]'):
            # Dohvatamo ocekivanu putanju
            href = anchor.get("href", "")
            # Regex ce zadrzati samo ID HaloOglasa (jedinstven broj od 10 cifara)
            if re.search(r"/\d{10,}(?:\?|$)", href):
                # Kreiraj apsolutnu putanju
                url = urljoin(HALO_LIST_BASE, href)
                # Provera link/kategorija
                if not halo_url_matches_path(url, path):
                    continue
                # Dohvatamo ID iz URL
                key = ad_key(url)
                # Ukoliko kandidat nije u seen listi dodajemo ga
                if key not in seen:
                    seen.add(key)
                    links.append(url)
                    page_links.append(url)
        # Ako stranica nije dala nijedan validan link, ne nastavljamo na novu stranicu vec prekidamo pretragu
        if not page_links:
            break
        # Tajmer je postavljen kako ne bi opteretili server s kog scrape-ujemo
        time.sleep(0.12)
    return links


# Counter klasa broji ponovljene vrednosti
# Ako npr items sadrzi:
# [
#     {"source": "Halo Oglasi", "category": "Laptop"},
#     {"source": "Halo Oglasi", "category": "Laptop"},
#     {"source": "KupujemProdajem", "category": "GPU"},
# ]
# Counter objekat ce izgledati:
# Counter({
#     ("Halo Oglasi", "Laptop"): 2,
#     ("KupujemProdajem", "GPU"): 1,
# })
def count_by_source_category(items: Iterable[dict[str, Any]]) -> Counter[tuple[str, str]]:
    return Counter((str(item["source"]), str(item["category"])) for item in items)


# Ucitava kadnidate koji su odabrani u prethodnim iteracijama
def load_candidate_cache() -> dict[tuple[str, str], list[str]]:
    pools: dict[tuple[str, str], list[str]] = defaultdict(list)
    if not CANDIDATES_JSON.is_file():
        return pools
    raw = json.loads(CANDIDATES_JSON.read_text(encoding="utf-8"))
    for compound_key, urls in raw.items():
        source, label = compound_key.split("\t", 1)
        pools[(source, label)].extend(str(url) for url in urls)

    # Povratna lista pool izgleda ovako:
    # {
    #     ("Halo Oglasi", "Laptop"): [url1, url2, ...],
    #     ("Halo Oglasi", "GPU"): [url3, url4, ...],
    #     ("KupujemProdajem", "Laptop"): [url5, url6, ...],
    #     ...
    # }
    return pools


# Funkcija scrape-uje kandidate oglase koji se dalje analiziraju
# Kreira pools recnik koji mapira (source, category) parove u listi mogucih URL-ova.
# collect_group() funckija kasnije otvara te URL-ove i  bira koji oglasi su upotrebljivi
def collect_candidates(rng: random.Random, listing_pages: int, kp_pages: int) -> dict[tuple[str, str], list[str]]:
    # Ucitavamo linkove kandidate iz prethodnih pokretanja skripte
    pools = load_candidate_cache()
    # globally_seen je skup koji sadrzi sve ID kandidata koji su ucitani iz cache-a
    globally_seen: set[str] = {ad_key(url) for urls in pools.values() for url in urls}
    # Brojac koji broji koliko je oglasa do sada (checkpoint) vec obradjeno
    checkpoint_counts = count_by_source_category(load_checkpoint())

    # Provera da li su kvote vec ispunjene za HaloOglase i KupujemProdajem
    if all(
        checkpoint_counts[("Halo Oglasi", label)] >= quota
        for label, quota in SOURCE_QUOTAS["Halo Oglasi"].items()
    ):
        listing_pages = 0
        print("Halo Oglasi is already complete; skipping its listing requests.", flush=True)
    if all(
        checkpoint_counts[("KupujemProdajem", label)] >= quota
        for label, quota in SOURCE_QUOTAS["KupujemProdajem"].items()
    ):
        kp_pages = 0
        print("KupujemProdajem quota is already complete; skipping its requests.", flush=True)
    elif any(source == "KupujemProdajem" and urls for (source, _), urls in pools.items()):
        kp_pages = 0
        print("Using the saved KupujemProdajem candidate pool.", flush=True)

    # Pocetak prikupjanja Halo Oglasa
    print("Prikupljam Halo listing linkove...", flush=True)
    halo_by_label: dict[str, list[str]] = defaultdict(list)
    # Podizemo 6 niti koje ce raditi otkrivanje linkova kandidata
    with ThreadPoolExecutor(max_workers=6) as executor:
        # Za svaku kategoriju se podize jedna nit
        jobs = {
            executor.submit(halo_listing_links, path, listing_pages): (category.label, path)
            for category in HALO_CATEGORIES
            for path in category.paths
        }
        # Kada nit zavrsi svoj posao, as_completed ce yieldovati Future objekte
        for future in as_completed(jobs):
            label, path = jobs[future]
            try:
                halo_by_label[label].extend(future.result())
            except Exception as exc:
                print(f"  Upozorenje za Halo {path}: {exc}", file=sys.stderr)
    # Deduplikacija URL-ova
    # RNG koji nam omogucava determinizam u uklanjanju duplikata
    for category in HALO_CATEGORIES:
        local = halo_by_label[category.label]
        rng.shuffle(local)
        for url in local:
            key = ad_key(url)
            if key not in globally_seen:
                globally_seen.add(key)
                pools[("Halo Oglasi", category.label)].append(url)
        print(f"  {category.label}: {len(pools[('Halo Oglasi', category.label)])} kandidata", flush=True)

    # Pocetak prikupjanja KP oglasa
    print("Prikupljam KP linkove iz više pretraga...", flush=True)
    all_kp: list[str] = []
    # KP oglasi se prikupljaju jedan po jedan, ne u paraleli
    with ThreadPoolExecutor(max_workers=1) as executor:
        jobs = { executor.submit( kp_search_links, query, kp_pages ): query for query in KP_SEARCHES }
        # Kao i kod HaloOglasa, as_completed vraca Future objekat kada se nit zavrsi
        for future in as_completed(jobs):
            query = jobs[future]
            try:
                # Dodajemo oglas u listu kandidata
                all_kp.extend(future.result())
            except SiteTemporarilyBlocked:
                raise
            except Exception as exc:
                print(f"  Upozorenje za KP pretragu {query!r}: {exc}", file=sys.stderr)
    # RNG koji nam omogucava determinizam u uklanjanju duplikata
    rng.shuffle(all_kp)
    for url in all_kp:
        # Dohvati ID url-a
        key = ad_key(url)
        # Ako je vec u listi vidjenih preskoci ga
        if key in globally_seen:
            continue
        # Dohvatamo labelu
        label = kp_category_from_url(url)
        if label is None:
            continue
        # Dodaj oglas u listu kandidata
        globally_seen.add(key)
        pools[("KupujemProdajem", label)].append(url)
    for label in QUOTAS:
        print(f"  {label}: {len(pools[('KupujemProdajem', label)])} kandidata", flush=True)
    # Sacuvaj trenutno izabrane oglase u cache (deo koji cini skriptu restartabilnom)
    save_candidate_cache(pools)
    return pools


# Funkcija rewrite-uje fajl ukoliko je neki oglas izbacen
def rewrite_checkpoint(items: Iterable[dict[str, Any]]) -> None:
    CHECKPOINT.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in items),
        encoding="utf-8",
    )


# Funkcija proverava da li konkretan HaloOglas i URL pripadaju dozvoljenoj kategoriji
# Primer:
#   url = "https://www.halooglasi.com/racunari/laptop-racunari/lenovo/thinkpad/1234567890"
#   listing_path = "/racunari/laptop-racunari"
def halo_url_matches_path(url: str, listing_path: str) -> bool:
    # F-ja ulr_parse deli link na kategorije
    # Iz dokumentacije:     Parse a URL into 6 components:
    #                       <scheme>://<netloc>/<path>;<params>?<query>#<fragment>
    # detail_path ce npr. izgledati: /racunari/laptop-racunari/lenovo/thinkpad/1234567890
    detail_path = urlparse(url).path.lower()
    # Uklanja / na kraju linka
    # "/racunari/laptop-racunari/"  ->  "/racunari/laptop-racunari"
    normalized_listing = listing_path.rstrip("/").lower()
    # Za telefone je potrebna detaljnija provera
    if normalized_listing == "/telefoni":
        return bool(
            re.match(
                r"^/telefoni/[^/]*mobilni-telefoni/[^/]+/\d{10,}/?$",
                detail_path,
            )
        )
    # Uporedi kategorije i vrati True/False
    return detail_path.startswith(normalized_listing + "/")


# Funckija iterira po svim mogucim kategorijama i pokusava da match-uje dodeljenu kategoriju sa kategorijom iz linka
def halo_url_matches_category(url: str, category: HaloCategory) -> bool:
    # Any vraca True ako je bilo koji element kolekcije bool(x) = True
    return any(halo_url_matches_path(url, path) for path in category.paths)


# Funckija proverava da li HaloOglas zaista istu kategoriju u url koja mu je dodeljena
def item_matches_assigned_category(item: dict[str, Any]) -> bool:
    if item.get("source") != "Halo Oglasi":
        return True
    label = str(item.get("category", ""))
    # Izvuci kategoriju koja je dodeljena oglasa
    category = next((value for value in HALO_CATEGORIES if value.label == label), None)
    # Uporedi tu kategoriju sa kategorijom koja se nalazi u URL
    return bool(category and halo_url_matches_category(str(item.get("url", "")), category))


# Funkcija za standardizaciju URL-a
def canonical_url(url: str) -> str:
    # Parsiraj URL u 6 komponeneti i izvuci netloc
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    # Tretiraj smsprint.halooglasi.com host kao regularan Halo Oglasi link
    if host == "smsprint.halooglasi.com":
        host = "www.halooglasi.com"
    # kid i tracking parametri nisu identitet oglasa i uklanjaju se
    kept = [(k, v) for k, v in parse_qsl(parsed.query) if k.lower() not in {"kid", "utm_source", "utm_medium", "utm_campaign"}]
    return urlunparse(("https", host, parsed.path.rstrip("/"), "", urlencode(kept), ""))


# Kreiraj jedinstveni ID od url-a
def ad_key(url: str) -> str:
    parsed = urlparse(url)
    if "kupujemprodajem.com" in parsed.netloc:
        match = re.search(r"/oglas/(\d+)", parsed.path)
        return f"kp:{match.group(1)}" if match else canonical_url(url)
    match = re.search(r"/(\d{10,})(?:/)?$", parsed.path)
    return f"halo:{match.group(1)}" if match else canonical_url(url)


# Vraca listu recnika koji sadrze key:value za trenutno scrape-ovane oglase
def load_checkpoint() -> list[dict[str, Any]]:
    # Provera da li fajl postoji
    if not CHECKPOINT.is_file():
        return []
    # Parsiraj file na putanji CHECKPOINT = ROOT / "checkpoint_oglasi.jsonl" ukoliko on postoji
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(CHECKPOINT.read_text(encoding="utf-8").splitlines(), start=1):
        # Check if the loaded line is empty
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Neispravan checkpoint, red {line_number}: {exc}") from exc
        if isinstance(value, dict):
            rows.append(value)
    return rows


# Pomocna funkcija koja definise ulazne argumente scraper-a
# Argumenti se prosledjuju kao parmetar nakon imena skripte u Run konfiguraciji
# Ukoliko ih nema postavljaju se na default vrednosti
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--halo-pages", type=int, default=10, help="Broj listing strana po Halo putanji")
    parser.add_argument("--kp-pages", type=int, default=8, help="Broj strana po KP pretrazi")
    parser.add_argument("--workers", type=int, default=4, help="Paralelni zahtevi po grupi")
    parser.add_argument("--only-finalize", action="store_true", help="Samo validira checkpoint i pravi izlaze")
    return parser.parse_args()


def main() -> int:
    # Parsiraj argumente
    args = parse_args()
    # Kreiraj direktorijum ukoliko on ne postoji. Ovo omogucava skripti da bude restartabilna
    ROOT.mkdir(parents=True, exist_ok=True)
    # Ucitaj trenutno stanje. ukoliko je skripta bila prekinuta pre kraja,
    # svi do tada ucitani oglasi ce biti sacuvani u CHECKPOINT = ROOT / "checkpoint_oglasi.jsonl"
    items = load_checkpoint()

    # Provera koju prolaze oglasi cija je dodeljena kategorija i kategorija koja se nalazi u URL ista
    # KupujemProdajem oglasi ce automatski proci, dok ce nad HaloOglasima biti izvrsena provera
    category_valid_items = [item for item in items if item_matches_assigned_category(item)]

    # Ispis koliko oglasa je uklonjeno
    if len(category_valid_items) != len(items):
        removed = len(items) - len(category_valid_items)
        print(f"Uklanjam {removed} zapisa koji nisu u dodeljenoj IT kategoriji.", flush=True)
        rewrite_checkpoint(category_valid_items)
        items = category_valid_items

    # Kod koji pokrece scarping
    # only_finalized argument definise da li ce se raditi scrape-ing ili ne
    if not args.only_finalize:
        # Sakupljamo oglase kadnidate koji mogu biti ukljuceni u konacnu selekciju
        pools = collect_candidates(random.Random(SEED), args.halo_pages, args.kp_pages)
        for source in ("Halo Oglasi", "KupujemProdajem"):
            # Iteriramo po kategorijama i pokusavamo da ispunimo kvote za svaku
            for label in QUOTAS:
                # collect group otvara konkretne oglase
                additions = collect_group(source, label, pools[(source, label)], items, args.workers)
                items.extend(additions)

    # Deduplikacija checkpointa po izvornom ID-u, uz očuvanje prvog uspešnog zapisa
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    # Iteriramo po listi i izbacujemo duplikate korsiteci skupove (ne dozvoljavaju duplikate)
    for item in items:
        key = ad_key(str(item["url"]))
        if key not in seen:
            seen.add(key)
            unique.append(item)

    output_items = select_output_items(unique)
    validate(output_items)
    write_outputs(output_items)
    print(f"Gotovo: {OUTPUT_JSON}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
