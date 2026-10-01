"""Immutable identity for restart-safe final experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .tokenization import TOKENIZER_VERSION


PROTOCOL_VERSION = "opj-phase3-text-only-tokenizer-v2-2026-08-27"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_identity(dataset_path: Path, split_path: Path) -> dict[str, str]:
    return {
        "dataset_sha256": file_sha256(dataset_path),
        "outer_split_sha256": file_sha256(split_path),
        "tokenizer_version": TOKENIZER_VERSION,
        "protocol_version": PROTOCOL_VERSION,
    }


def result_matches(
    path: Path,
    dataset_path: Path,
    split_path: Path,
    model: str,
    fold: int,
    epoch: int | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    expected: dict[str, Any] = run_identity(dataset_path, split_path)
    expected.update({"model": model, "outer_fold": fold})
    if epoch is not None:
        expected["epoch"] = epoch
    return all(payload.get(key) == value for key, value in expected.items())
