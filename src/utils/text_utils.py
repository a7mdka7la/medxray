"""Tiny text helpers for cleaning MIMIC reports and parsing model output.

MIMIC reports are noisy: section headers in mixed case, lots of de-identified
markers like `___`, and the FINDINGS/IMPRESSION sections are not always
present. These helpers do the bare minimum to make evaluation fair.
"""

from __future__ import annotations

import re
from typing import Dict

_DEID = re.compile(r"_+")
_WS = re.compile(r"\s+")

# A pragmatic list — MIMIC has a long tail of header variants.
_SECTION_PATTERNS: Dict[str, re.Pattern[str]] = {
    "findings": re.compile(r"(?is)\bfindings?\s*:\s*(.*?)(?=(?:\bimpression|\brecommendations?|\Z))"),
    "impression": re.compile(r"(?is)\bimpression\s*:\s*(.*?)(?=(?:\brecommendations?|\bfindings?|\Z))"),
    "recommendations": re.compile(r"(?is)\brecommendations?\s*:\s*(.*?)(?=(?:\bfindings?|\bimpression|\Z))"),
}


def normalize_report(text: str) -> str:
    if not text:
        return ""
    text = _DEID.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip()


def extract_section(text: str, name: str) -> str:
    pat = _SECTION_PATTERNS.get(name.lower())
    if pat is None:
        return ""
    m = pat.search(text or "")
    if not m:
        return ""
    return normalize_report(m.group(1))


def parse_structured_report(text: str) -> Dict[str, str]:
    """Best-effort parse of a free-text report into the three sections."""
    return {
        "findings": extract_section(text, "findings"),
        "impression": extract_section(text, "impression"),
        "recommendations": extract_section(text, "recommendations"),
    }


def section_or_empty(parsed: Dict[str, str], name: str) -> str:
    return parsed.get(name, "") or ""
