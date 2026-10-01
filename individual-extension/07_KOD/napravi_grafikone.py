#!/usr/bin/env python3
"""Generiše grafikone za izveštaj dopune iz proverljivog JSON izlaza.

"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "04_REZULTATI" / "rezultati_llm_naspram_rucnih.json"
OUT = ROOT / "05_IZVESTAJ" / "slike"

COLORS = {
    "blue": "#315A7D",
    "teal": "#2D7F79",
    "orange": "#D78535",
    "red": "#B34E4E",
    "gray": "#6D7782",
    "light": "#DDE6ED",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 220,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
   
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def categories_chart(data: dict) -> None:
    
    rows = data["difference_categories"]
    
    display = {
        "SUVISNO": "SUVIŠNO",
        "GRANICA_SIRA": "GRANICA ŠIRA",
        "PROPUSTENO": "PROPUŠTENO",
        "FRAGMENTACIJA": "FRAGMENTACIJA",
        "SPAJANJE": "SPAJANJE",
        "GRANICA_UZA": "GRANICA UŽA",
        "POGRESNA_OZNAKA": "POGREŠNA OZNAKA",
        "GRANICA_I_OZNAKA": "GRANICA I OZNAKA",
        "GRANICA_POMERENA": "GRANICA POMERENA",
        "SLOZENO": "SLOŽENO",
    }
   
    labels = [display[r["category"]] for r in rows][::-1]
    counts = [r["count"] for r in rows][::-1]
    percents = [r["percent"] for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(7.35, 4.25))
    bars = ax.barh(labels, counts, color=COLORS["blue"], alpha=0.92)
    ax.set_xlabel("Broj slučajeva razlike")
    ax.set_title("Raspodela 1.114 strukturnih slučajeva razlike")
    ax.grid(axis="x", color="#D7DEE5", linewidth=0.7, alpha=0.8)
   
    ax.set_axisbelow(True)
   
    ax.set_xlim(0, max(counts) * 1.17)
    for bar, count, pct in zip(bars, counts, percents):
       
        ax.text(
            bar.get_width() + 5,
            bar.get_y() + bar.get_height() / 2,
            f"{count}  ({pct:.1f}%)".replace(".", ","),
            va="center",
            fontsize=8.5,
        )
    save(fig, "slika_4_2_strukturne_kategorije.png")


def label_metrics_chart(data: dict) -> None:
   
    rows = data["per_label"]
   
    labels = [r["label"].replace("SKLADISTE", "SKLADIŠTE") for r in rows]
    strict = [r["strict"]["f1"] for r in rows]
    relaxed = [r["relaxed"]["f1"] for r in rows]
    semantic = [r["semantic"]["f1"] for r in rows]

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.35, 3.85))
    ax.bar(x - width, strict, width, label="Strict F1", color=COLORS["blue"])
    ax.bar(x, relaxed, width, label="Relaxed F1", color=COLORS["teal"])
    ax.bar(x + width, semantic, width, label="Semantic-token F1", color=COLORS["orange"])
    ax.set_xticks(x, labels)
   
    ax.set_ylim(0.70, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Slaganje ručnih i LLM oznaka po semantičkoj klasi")
    ax.grid(axis="y", color="#D7DEE5", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
   
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    for group in (strict, relaxed, semantic):
      
        offset = (-width if group is strict else width if group is semantic else 0)
        for xpos, value in zip(x + offset, group):
           
            ax.text(xpos, value + 0.006, f"{value:.3f}".replace(".", ","), ha="center", fontsize=7.2)
    save(fig, "slika_4_1_mere_po_oznaci.png")


def main() -> None:
    
   
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    set_style()
    categories_chart(data)
    label_metrics_chart(data)


if __name__ == "__main__":
    main()
