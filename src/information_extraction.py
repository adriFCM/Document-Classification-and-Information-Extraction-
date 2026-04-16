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
from pathlib import Path
from typing import Optional

# Optional CRF model for issuer / recipient. Loaded lazily so the module still
# imports if sklearn-crfsuite or the model file are missing.
_CRF_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "crf_invoice.pkl"
_crf = None
_crf_tried = False


def _get_crf():
    global _crf, _crf_tried
    if _crf_tried:
        return _crf
    _crf_tried = True
    try:
        from .crf_extractor import CRFInvoiceExtractor  # type: ignore
    except ImportError:
        try:
            from crf_extractor import CRFInvoiceExtractor  # type: ignore
        except ImportError:
            return None
    if not _CRF_MODEL_PATH.exists():
        return None
    try:
        _crf = CRFInvoiceExtractor.load(_CRF_MODEL_PATH)
    except Exception:
        _crf = None
    return _crf


# --- dates ----------------------------------------------------------------
from .date_utils import DATE_RE as _DATE_RE, find_dates as _find_dates  # noqa: F401


def extract_invoice_date(text: str) -> Optional[str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.search(r'\b(invoice\s*date|date\s*of\s*issue|date)\b', line, re.IGNORECASE):
            hits = _find_dates(line)
            if hits:
                return hits[0]
            # Label-only line — look ahead up to 2 non-empty lines.
            seen = 0
            for j in range(i + 1, min(i + 5, len(lines))):
                nxt = lines[j].strip()
                if not nxt:
                    continue
                seen += 1
                nxt_hits = _find_dates(nxt)
                if nxt_hits:
                    return nxt_hits[0]
                if seen >= 2:
                    break
    hits = _find_dates(text)
    return hits[0] if hits else None


_DUE_DATE_LABEL = re.compile(
    r'\b(due[\s_]*date|payment[\s_]*due|pay[\s_]*by)\b', re.IGNORECASE,
)


def extract_due_date(text: str) -> Optional[str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _DUE_DATE_LABEL.search(line):
            hits = _find_dates(line)
            if hits:
                return hits[0]
            seen = 0
            for j in range(i + 1, min(i + 5, len(lines))):
                nxt = lines[j].strip()
                if not nxt:
                    continue
                seen += 1
                nxt_hits = _find_dates(nxt)
                if nxt_hits:
                    return nxt_hits[0]
                if seen >= 2:
                    break
    m = re.search(r'\bnet\s*(\d{1,3})\b', text, re.IGNORECASE)
    if m:
        return f'NET{m.group(1)}'
    return None


# --- invoice number -------------------------------------------------------
_INVOICE_NO_LABEL = re.compile(
    r'(?:invoice\s*(?:no|number|num|#)\.?|inv\s*[#:]|bill\s*no\.?)'
    r'\s*[:#]?\s*',
    re.IGNORECASE,
)
_INVOICE_NO_VALUE = re.compile(r'(#?[A-Z0-9][A-Z0-9\-/]{2,})')
# Standalone "#1234" style near top of doc, used as fallback.
_HASH_NUM_RE = re.compile(r'^\s*(#\d{3,})\s*$', re.MULTILINE)


def extract_invoice_number(text: str) -> Optional[str]:
    # Label may be on one line with the value on a following (possibly blank-
    # separated) line — scan line by line so we can look ahead a few lines.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        lm = _INVOICE_NO_LABEL.search(line)
        if not lm:
            continue
        tail = line[lm.end():]
        vm = _INVOICE_NO_VALUE.search(tail)
        if vm:
            return vm.group(1)
        for j in range(i + 1, min(i + 4, len(lines))):
            nxt = lines[j].strip()
            if not nxt:
                continue
            vm = _INVOICE_NO_VALUE.match(nxt)
            if vm:
                return vm.group(1)
            break
    m = _HASH_NUM_RE.search(text)
    return m.group(1) if m else None


# --- total ----------------------------------------------------------------
_AMOUNT_RE = re.compile(
    r'(?:RM|MYR|USD|EUR|GBP|NOK|SEK|DKK|\$|€|£)?\s*'
    # 1) Thousands-grouped with optional dot-decimal: 1,234 or 1,234.56
    # 2) European decimal comma: 149,97 (exactly 2 digits after, nothing
    #    right after — avoids eating into thousands-grouped numbers)
    # 3) Plain: 164 or 164.97
    r'(\d{1,3}(?:[,\s]\d{3})+(?:\.\d{2})?|\d+,\d{2}(?!\d)|\d+(?:\.\d{2})?)'
    r'\s*(?:RM|MYR|USD|EUR|GBP|NOK|SEK|DKK)?'
)
_TOTAL_KEYWORDS = re.compile(
    r'\b(grand\s*total|total\s*amount|total\s*due|totals?|amount\s*due|balance\s*due)\b',
    re.IGNORECASE,
)


def _as_float(s: str) -> float:
    s = s.strip().replace(' ', '')
    # European decimal comma: exactly 2 digits after a single comma, no dot.
    if re.fullmatch(r'\d+,\d{2}', s):
        s = s.replace(',', '.')
    else:
        s = s.replace(',', '')
    try:
        return float(s)
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
    r'\b(bill(?:ed)?\s*to|sold\s*to|ship\s*to|customer|recipient|invoice\s*to'
    r'|issued\s*to|client)\b',
    re.IGNORECASE,
)

_SELLER_LABEL_RE = re.compile(
    r'^\s*(seller|from|vendor|bill\s*from)\s*[:\-]?\s*$',
    re.IGNORECASE,
)

# Skip standalone document-type words when they're styled as a header
# (large "INVOICE" / "RECEIPT" titles above the letterhead), and also skip
# template/cover lines like "White Minimalist Business Invoice" whose last
# word is a document type.
_DOC_TYPE_RE = re.compile(
    r'^(invoice|receipt|bill|statement|quotation|quote|tax\s*invoice'
    r'|invoice\s*/\s*tax\s*receipt)$',
    re.IGNORECASE,
)
_TEMPLATE_TITLE_RE = re.compile(
    r'\b(invoice|receipt|bill|statement|quotation)\s*$',
    re.IGNORECASE,
)


def extract_issuer(text: str) -> Optional[str]:
    crf = _get_crf()
    if crf is not None:
        pred = crf.predict(text).get("ISSUER")
        if pred:
            return pred
    return _rule_based_issuer(text)


def _rule_based_issuer(text: str) -> Optional[str]:
    lines = text.splitlines()

    # If a Seller:-style label exists, take the first non-empty line below it.
    for i, ln in enumerate(lines):
        if _SELLER_LABEL_RE.match(ln):
            for j in range(i + 1, min(i + 5, len(lines))):
                nxt = lines[j].strip()
                if nxt:
                    return nxt
            break

    # Otherwise, original heuristic: scan head lines, prefer company-suffix.
    head_lines: list[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if _RECIPIENT_MARKER.search(ln):
            break
        if _DOC_TYPE_RE.match(ln):
            continue
        # Template cover titles end with a document word, e.g.
        # "White Minimalist Business Invoice" — skip them too.
        if _TEMPLATE_TITLE_RE.search(ln) and len(ln.split()) >= 3:
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
    r'(?:bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|issued\s*to'
    r'|customer|recipient|client)\s*[:\-]?\s*(.*)',
    re.IGNORECASE,
)
# "To:" alone on its own line — too generic to search inside text, so only
# match when the line is effectively just the label.
_TO_LABEL_LINE_RE = re.compile(r'^\s*to\s*[:\-]\s*(.*)$', re.IGNORECASE)

# Skip lines that are clearly not a recipient name (amounts, dates, labels).
_NON_NAME_RE = re.compile(
    r'(\$|€|£|\bRM\b|\bUSD\b|\bEUR\b|\bGBP\b|\d{2,}|:)',
    re.IGNORECASE,
)
# Common table headers that should never be returned as a recipient.
_TABLE_HEADER_RE = re.compile(
    r'^(description|item|items|qty|quantity|rate|price|amount|total|tax|subtotal'
    r'|hours|um|net\s+price|net\s+worth|gross\s+worth|vat|no\.?)\b',
    re.IGNORECASE,
)
# Secondary field labels that may sit on the same line as "Bill To:" — used to
# trim the tail so we don't return "X Invoice number: Y".
_SECONDARY_LABEL_RE = re.compile(
    r'\s+(invoice\s*(?:no|number|num|#)|date|term|due|po\s*#|tel|phone|email)\b',
    re.IGNORECASE,
)


def _clean_recipient_tail(tail: str) -> str:
    m = _SECONDARY_LABEL_RE.search(tail)
    if m:
        tail = tail[: m.start()]
    return tail.strip(" :;-,")


def _next_name_line(lines, start):
    for j in range(start, min(start + 5, len(lines))):
        nxt = lines[j].strip()
        if not nxt:
            continue
        if _TABLE_HEADER_RE.search(nxt):
            continue
        if _NON_NAME_RE.search(nxt):
            continue
        return nxt
    return None


def _resolve_recipient(tail: str, lines, i):
    cleaned = _clean_recipient_tail(tail)
    if cleaned and not _TABLE_HEADER_RE.search(cleaned):
        return cleaned
    return _next_name_line(lines, i + 1)


def extract_recipient(text: str) -> Optional[str]:
    crf = _get_crf()
    if crf is not None:
        pred = crf.predict(text).get("RECIPIENT")
        if pred:
            return pred
    return _rule_based_recipient(text)


def _rule_based_recipient(text: str) -> Optional[str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _RECIPIENT_LABEL_RE.search(line)
        if m:
            res = _resolve_recipient(m.group(1).strip(), lines, i)
            if res:
                return res
            continue
        tm = _TO_LABEL_LINE_RE.match(line)
        if tm:
            res = _resolve_recipient(tm.group(1).strip(), lines, i)
            if res:
                return res
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


def extract_invoice_fields(text: str, pdf_bytes: Optional[bytes] = None) -> dict:
    # pdf_bytes is accepted for path-B (layout-aware) wiring in a later task;
    # currently unused so behaviour is identical to the text-only call.
    return asdict(InvoiceFields(
        invoice_number=extract_invoice_number(text),
        invoice_date=extract_invoice_date(text),
        due_date=extract_due_date(text),
        issuer=extract_issuer(text),
        recipient=extract_recipient(text),
        total=extract_total(text),
    ))
