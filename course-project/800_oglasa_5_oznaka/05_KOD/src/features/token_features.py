"""Required BASE token features. Regexes are evidence features, never label rules."""
from __future__ import annotations

import re
from typing import Sequence

from src.common.tokenization import Token

PRICE = re.compile(r"^\d+(?:[.,]\d+)?\s*(?:€|eur|din|rsd|€)$", re.I)
CAPACITY = re.compile(r"^\d+(?:[.,]\d+)?(?:gb|tb|mb|mhz|ghz|wh|mah|w|hz)$", re.I)
CPU = re.compile(r"^(?:i[3579]|ryzen|core|xeon|celeron|pentium|snapdragon|m[1-4])", re.I)
GPU = re.compile(r"^(?:rtx|gtx|rx|radeon|geforce|arc|quadro)", re.I)
RESOLUTION = re.compile(r"^\d{3,4}[x×]\d{3,4}$", re.I)
WARRANTY = re.compile(r"^(?:garancija|meseci|meseca|godina|godine|god)$", re.I)
MODEL = re.compile(r"^(?=.*[a-zA-Z])(?=.*\d)[a-zA-Z0-9][a-zA-Z0-9._/+\-]{1,}$")
LOCATION = re.compile(r"^(?:beograd|novi|sad|nis|kragujevac|subotica|zemun|srbija)$", re.I)


def _shape(value: str) -> str:
    return "".join("d" if c.isdigit() else "X" if c.isupper() else "x" if c.islower() else c for c in value)


def _basic(prefix: str, token: Token | None) -> dict[str, object]:
    if token is None:
        return {prefix + "=<NONE>": True}
    value = token.text
    lower = value.lower()
    return {prefix + "lower=" + lower: True, prefix + "shape=" + _shape(value): True,
            prefix + "isupper": value.isupper(), prefix + "istitle": value.istitle(),
            prefix + "isdigit": value.isdigit(), prefix + "has_digit": any(c.isdigit() for c in value),
            prefix + "len=" + str(min(len(value), 20)): True}


def token_features(tokens: Sequence[Token], index: int, field: str) -> dict[str, object]:
    token = tokens[index]
    value = token.text
    feat: dict[str, object] = _basic("w=", token)
    feat.update(_basic("prev=", tokens[index-1] if index else None))
    feat.update(_basic("next=", tokens[index+1] if index + 1 < len(tokens) else None))
    feat.update({"field=" + field: True, "BOS": index == 0, "EOS": index == len(tokens)-1,
                 "prefix1=" + value[:1].lower(): True, "prefix2=" + value[:2].lower(): True,
                 "suffix1=" + value[-1:].lower(): True, "suffix2=" + value[-2:].lower(): True,
                 "regex_price": bool(PRICE.match(value)), "regex_capacity": bool(CAPACITY.match(value)),
                 "regex_cpu": bool(CPU.match(value)), "regex_gpu": bool(GPU.match(value)),
                 "regex_resolution": bool(RESOLUTION.match(value)), "regex_warranty": bool(WARRANTY.match(value)),
                 "regex_model": bool(MODEL.match(value)), "regex_location": bool(LOCATION.match(value))})
    return feat


def sequence_features(tokens: Sequence[Token], field: str) -> list[dict[str, object]]:
    return [token_features(tokens, i, field) for i in range(len(tokens))]
