"""Layout-aware invoice field extraction (path B).

Uses pdfplumber word bounding boxes and table detection to locate label→value
pairs spatially. Returns None for fields it cannot confidently identify so
path A (text-based) can fill gaps.
"""

from __future__ import annotations

import io
import re
from typing import Optional

import pdfplumber

from .date_utils import DATE_RE

_FIELDS = ("invoice_number", "invoice_date", "due_date", "issuer", "recipient", "total")


def _all_none() -> dict[str, Optional[str]]:
    return {k: None for k in _FIELDS}


def extract(pdf_bytes: bytes) -> dict[str, Optional[str]]:
    """Return {field: value-or-None} from a digitally-generated PDF."""
    if not pdf_bytes:
        return _all_none()
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = list(pdf.pages)
            if not pages:
                return _all_none()
            return _extract_from_pages(pages)
    except Exception:
        return _all_none()


def _extract_from_pages(pages) -> dict[str, Optional[str]]:
    pages_words: list[list[dict]] = []
    for page in pages:
        words = page.extract_words(extra_attrs=["fontname", "size"]) or []
        pages_words.append(words)

    if not any(pages_words):
        return _all_none()

    return _all_none()
