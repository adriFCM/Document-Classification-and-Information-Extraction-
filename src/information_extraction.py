"""Rule-based invoice field extraction.

Lifted from notebooks/03_information_extraction.ipynb. Public entry point is
`extract_invoice_fields(text)` which returns all six target fields, with
missing values as None.

Scores on the SROIE 2019 set (973 receipts):
    invoice_date : 95.9 % exact
    issuer       : 61.5 % exact / 90.1 % relaxed
    total        : 64.4 % exact / 71.0 % relaxed
invoice_number, due_date and recipient are unscored — SROIE has no labels for
them, so they are validated qualitatively on real invoice PDFs.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional


# --- dates ----------------------------------------------------------------
_MONTHS = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'

_DATE_PATTERNS = [
    # DD/MM/YYYY, MM-DD-YYYY, etc. — separator-delimited, validate day ≤ 31 and month ≤ 12
    r'\b((?:0?[1-9]|[12]\d|3[01])[/\-\.](?:0?[1-9]|1[0-2])[/\-\.](?:\d{4}|\d{2}))\b',
    # YYYY/MM/DD or YYYY-MM-DD
    r'\b(\d{4}[/\-\.](?:0?[1-9]|1[0-2])[/\-\.](?:0?[1-9]|[12]\d|3[01]))\b',
    # DD-MON-YYYY or DD-MON-YY  (e.g. 24-MAR-2018, 25-DEC-18)
    r'\b((?:0?[1-9]|[12]\d|3[01])[/\-\.]' + _MONTHS + r'[/\-\.](?:\d{4}|\d{2}))\b',
    # Compact DDMMYYYY or YYYYMMDD (8 digits only, basic range check)
    r'\b((?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])\d{4})\b',
    r'\b(\d{4}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))\b',
    # "15 March 2024" or "March 15, 2024"
    r'\b((?:0?[1-9]|[12]\d|3[01])\s+' + _MONTHS + r'\.?\s+\d{2,4})\b',
    r'\b(' + _MONTHS + r'\.?\s+(?:0?[1-9]|[12]\d|3[01]),?\s+\d{2,4})\b',
]
_DATE_RE = re.compile('|'.join(_DATE_PATTERNS), re.IGNORECASE)


def _find_dates(text: str):
    return [next(g for g in m.groups() if g) for m in _DATE_RE.finditer(text)]


def extract_invoice_date(text: str) -> Optional[str]:
    for line in text.splitlines():
        if re.search(r'\b(invoice\s*date|date)\b', line, re.IGNORECASE):
            hits = _find_dates(line)
            if hits:
                return hits[0]
    hits = _find_dates(text)
    return hits[0] if hits else None


def extract_due_date(text: str) -> Optional[str]:
    for line in text.splitlines():
        if re.search(r'\b(due\s*date|payment\s*due|pay\s*by)\b', line, re.IGNORECASE):
            hits = _find_dates(line)
            if hits:
                return hits[0]
    m = re.search(r'\bnet\s*(\d{1,3})\b', text, re.IGNORECASE)
    if m:
        return f'NET{m.group(1)}'
    return None


# --- invoice number -------------------------------------------------------
_INVOICE_NO_RE = re.compile(
    r'(?:invoice\s*(?:no|number|num|#)\.?\s*[:#]?\s*'
    r'|inv\s*[#:]\s*'
    r'|bill\s*no\.?\s*[:#]?\s*)'
    r'([A-Z0-9][A-Z0-9\-/]{2,})',
    re.IGNORECASE,
)


def extract_invoice_number(text: str) -> Optional[str]:
    m = _INVOICE_NO_RE.search(text)
    return m.group(1) if m else None


# --- total ----------------------------------------------------------------
_AMOUNT_RE = re.compile(
    r'(?:RM|MYR|USD|EUR|GBP|NOK|SEK|DKK|\$|€|£)?\s*'
    # Either a properly grouped number (1,234 / 1,234.56) OR a plain run of
    # digits with optional decimals. Grouping alternative requires at least
    # one [,\s]\d{3} block, otherwise \d{1,3} could consume part of a larger
    # number like '4000' and leave '0' behind.
    r'(\d{1,3}(?:[,\s]\d{3})+(?:\.\d{2})?|\d+(?:\.\d{2})?)'
    r'\s*(?:RM|MYR|USD|EUR|GBP|NOK|SEK|DKK)?'
)
_TOTAL_KEYWORDS = re.compile(
    r'\b(grand\s*total|total\s*amount|total\s*due|total|amount\s*due|balance\s*due)\b',
    re.IGNORECASE,
)


def _as_float(s: str) -> float:
    try:
        return float(s.replace(',', '').replace(' ', ''))
    except ValueError:
        return 0.0


def extract_total(text: str) -> Optional[str]:
    """Return the grand-total amount as a string, or None.

    SROIE quirks handled:
    1. Label and amount often live on separate lines (`TOTAL:\\n9.00`) — if the
       keyword line has no amount, look ahead up to 2 non-empty lines.
    2. Receipts print several totals (`TOTAL`, `ROUND D TOTAL`, ...). A line
       whose label contains `round` is the final post-rounding figure, so it
       wins. Otherwise fall back to max across total-keyword lines (the grand
       total is typically the largest amount on a total line).
    """
    lines = text.splitlines()
    candidates: list[tuple[str, bool]] = []
    for i, line in enumerate(lines):
        if not _TOTAL_KEYWORDS.search(line):
            continue
        if re.search(r'sub\s*total', line, re.IGNORECASE):
            continue
        is_round = bool(re.search(r'round', line, re.IGNORECASE))
        amounts = _AMOUNT_RE.findall(line)
        if amounts:
            candidates.append((amounts[-1], is_round))
            continue
        seen = 0
        for j in range(i + 1, min(i + 5, len(lines))):
            nxt = lines[j].strip()
            if not nxt:
                continue
            seen += 1
            nxt_amounts = _AMOUNT_RE.findall(nxt)
            if nxt_amounts:
                candidates.append((nxt_amounts[-1], is_round))
                break
            if seen >= 2:
                break
    if not candidates:
        return None
    round_hits = [c for c, r in candidates if r]
    if round_hits:
        return round_hits[-1]
    return max((c for c, _ in candidates), key=_as_float)


# --- issuer ---------------------------------------------------------------
_COMPANY_SUFFIX = re.compile(
    r'\b(SDN\.?\s*B[HN]D|BHD|BND|LLC|LTD|INC|CORP|CO\.?|ENTERPRISE|TRADING|PTE|GMBH)\b',
    re.IGNORECASE,
)


_RECIPIENT_MARKER = re.compile(
    r'\b(bill\s*to|sold\s*to|ship\s*to|customer|recipient|invoice\s*to)\b',
    re.IGNORECASE,
)

# Skip standalone document-type words when they're styled as a header
# (large "INVOICE" / "RECEIPT" titles above the letterhead).
_DOC_TYPE_RE = re.compile(
    r'^(invoice|receipt|bill|statement|quotation|quote|tax\s*invoice)$',
    re.IGNORECASE,
)


def extract_issuer(text: str) -> Optional[str]:
    # Scan a bit deeper than 6 lines so we survive a logo/letterhead block, but
    # stop at the first recipient marker — everything past that belongs to the
    # customer, not the issuer.
    head_lines: list[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if _RECIPIENT_MARKER.search(ln):
            break
        if _DOC_TYPE_RE.match(ln):
            continue
        head_lines.append(ln)
        if len(head_lines) >= 8:
            break
    with_suffix = [ln for ln in head_lines if _COMPANY_SUFFIX.search(ln)]
    if with_suffix:
        return max(with_suffix, key=len)
    return head_lines[0] if head_lines else None


# --- recipient ------------------------------------------------------------
_RECIPIENT_LABEL_RE = re.compile(
    r'(?:bill\s*to|sold\s*to|ship\s*to|invoice\s*to|customer|recipient)\s*[:\-]?\s*(.*)',
    re.IGNORECASE,
)


def extract_recipient(text: str) -> Optional[str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _RECIPIENT_LABEL_RE.search(line)
        if not m:
            continue
        tail = m.group(1).strip()
        if tail:
            return tail
        # Label alone on its own line (e.g. "Bill To" / "Bill To:") —
        # return the next non-empty line.
        for j in range(i + 1, min(i + 4, len(lines))):
            nxt = lines[j].strip()
            if nxt:
                return nxt
    return None


# --- public API -----------------------------------------------------------
@dataclass
class InvoiceFields:
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    issuer: Optional[str] = None
    recipient: Optional[str] = None
    total: Optional[str] = None


def extract_invoice_fields(text: str) -> dict:
    return asdict(InvoiceFields(
        invoice_number=extract_invoice_number(text),
        invoice_date=extract_invoice_date(text),
        due_date=extract_due_date(text),
        issuer=extract_issuer(text),
        recipient=extract_recipient(text),
        total=extract_total(text),
    ))
