"""Shared date regex and helper. Extracted from information_extraction.py
so layout_extractor.py can reuse it without a circular import."""

from __future__ import annotations

import re
from typing import List

_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"

_DATE_PATTERNS = [
    # DD/MM/YYYY, MM-DD-YYYY, etc. (day-first — matches SROIE layout)
    r"\b((?:0?[1-9]|[12]\d|3[01])[/\-\.](?:0?[1-9]|1[0-2])[/\-\.](?:\d{4}|\d{2}))\b",
    # MM/DD/YYYY — month-first alternative (US-style). Day-first above still
    # wins on ambiguous inputs because it's checked first.
    r"\b((?:0?[1-9]|1[0-2])[/\-\.](?:0?[1-9]|[12]\d|3[01])[/\-\.](?:\d{4}|\d{2}))\b",
    # YYYY/MM/DD or YYYY-MM-DD
    r"\b(\d{4}[/\-\.](?:0?[1-9]|1[0-2])[/\-\.](?:0?[1-9]|[12]\d|3[01]))\b",
    # DD-MON-YYYY or DD-MON-YY
    r"\b((?:0?[1-9]|[12]\d|3[01])[/\-\.]" + _MONTHS + r"[/\-\.](?:\d{4}|\d{2}))\b",
    # Compact DDMMYYYY or YYYYMMDD — year must start with 19 or 20 so random
    # 8-digit invoice numbers don't falsely parse as dates.
    r"\b((?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])(?:19|20)\d{2})\b",
    r"\b((?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))\b",
    # "15 March 2024" or "March 15, 2024"
    r"\b((?:0?[1-9]|[12]\d|3[01])\s+" + _MONTHS + r"\.?\s+\d{2,4})\b",
    r"\b(" + _MONTHS + r"\.?\s+(?:0?[1-9]|[12]\d|3[01]),?\s+\d{2,4})\b",
    # "MARCH.06.2024" / "March-06-2024"
    r"\b(" + _MONTHS + r"[\./\-](?:0?[1-9]|[12]\d|3[01])[\./\-]\d{2,4})\b",
]

DATE_RE = re.compile("|".join(_DATE_PATTERNS), re.IGNORECASE)


def find_dates(text: str) -> List[str]:
    return [next(g for g in m.groups() if g) for m in DATE_RE.finditer(text)]
