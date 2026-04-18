"""Invoice field extraction — column-aware spatial anchoring.

Public API: ``extract_invoice_fields(text, pdf_bytes=None, image_bytes=None) -> dict``
Returns the six target fields (invoice_number, invoice_date, due_date,
issuer, recipient, total) with missing values as ``None``.

Primary path: given word bounding boxes (pdfplumber for digital PDFs,
tesseract OCR for scanned images), locate each field by anchoring on its
label and pulling the nearest value(s) using 2D spatial rules. Invoices
commonly use a 2-column layout (``Seller:`` vs ``Client:``) that flattens
to interleaved text; bbox extraction recovers the column structure.

Fallback path: 1D regex on plain text when no word boxes are available.
"""

from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import pdfplumber


# =====================================================================
# Regex primitives
# =====================================================================

_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"

_DATE_PATTERNS = [
    r"\b((?:0?[1-9]|[12]\d|3[01])[/\-\.](?:0?[1-9]|1[0-2])[/\-\.](?:\d{4}|\d{2}))\b",
    r"\b((?:0?[1-9]|1[0-2])[/\-\.](?:0?[1-9]|[12]\d|3[01])[/\-\.](?:\d{4}|\d{2}))\b",
    r"\b(\d{4}[/\-\.](?:0?[1-9]|1[0-2])[/\-\.](?:0?[1-9]|[12]\d|3[01]))\b",
    r"\b((?:0?[1-9]|[12]\d|3[01])\s+" + _MONTHS + r"\.?\s+\d{2,4})\b",
    r"\b(" + _MONTHS + r"\.?\s+(?:0?[1-9]|[12]\d|3[01]),?\s+\d{2,4})\b",
    r"\b(" + _MONTHS + r"[\./\-](?:0?[1-9]|[12]\d|3[01])[\./\-]\d{2,4})\b",
]
DATE_RE = re.compile("|".join(_DATE_PATTERNS), re.IGNORECASE)

_YEAR_ONLY_RE = re.compile(r"^(?:19|20)\d{2}$")

_AMOUNT_RE = re.compile(
    r"(?:[\$€£]|RM|USD|EUR|GBP)?\s*"
    r"(?:\d{1,3}(?:\.\d{3})+,\d{2}"                # EU: 3.750,00
    r"|\d{1,3}(?:,\d{3})+(?:\.\d{2})?"             # US: 3,750.00
    r"|\d{1,3}(?:[\s\u00A0]\d{3})+(?:[.,]\d{2})?"  # Space thousands: 3 750,00
    r"|\d+[.,]\d{2}"                               # Simple: 3.75 / 3,75
    r"|\d+)"
)

_INV_NUMBER_VALUE_RE = re.compile(r"^#?[A-Z0-9][A-Z0-9\-/]{2,19}$", re.IGNORECASE)


# =====================================================================
# Field labels — ordered by specificity (specific → generic)
# =====================================================================

_LABELS: dict[str, list[str]] = {
    "invoice_number": [
        r"invoice\s*(?:no|num|number|#)\.?",
        r"inv\s*(?:no|num|number|#)?\.?",
        r"bill\s*(?:no|number)\.?",
        r"reference\s*(?:no|number|#)\.?",
        r"ref\s*(?:no|#)?\.?",
        r"order\s*(?:no|number|#)\.?",
        r"po\s*(?:no|#)?\.?",
    ],
    "invoice_date": [
        r"date\s*of\s*issue",
        r"invoice\s*date",
        r"issue\s*date",
        r"bill\s*date",
        r"issued\s*on",
        r"dated",
        r"^date$",
    ],
    "due_date": [
        r"due\s*date",
        r"payment\s*due(?:\s*date)?",
        r"pay\s*by",
        r"^due$",
    ],
}

_ISSUER_LABELS = [
    r"seller", r"from", r"vendor", r"bill\s*from",
    r"issued\s*by", r"supplier", r"remit\s*to",
]
_RECIPIENT_LABELS = [
    r"bill(?:ed)?\s*to", r"sold\s*to", r"ship(?:ped)?\s*to",
    r"invoice\s*to", r"issued\s*to", r"client", r"customer",
    r"recipient", r"attention", r"attn", r"^to$",
]
_TOTAL_LABELS = [
    r"grand\s*total", r"total\s*due", r"amount\s*due",
    r"balance\s*due", r"total\s*amount", r"total\s*payable",
    r"^total$", r"total",
]

# Stop words that signal the end of an issuer/recipient block.
_STOP_BLOCK_RE = re.compile(
    r"^(tax\s*id|vat(?:\s*no)?|iban|swift|phone|tel|fax|email|e-?mail"
    r"|website|www\.|http|payment|bank|account|contact|reg(?:istration)?"
    r"|invoice|items?|description|qty|quantity|date)\b",
    re.IGNORECASE,
)


# =====================================================================
# Word-box utilities
# =====================================================================

def _y_overlap(a: dict, b: dict) -> float:
    top = max(a["top"], b["top"])
    bot = min(a["bottom"], b["bottom"])
    if bot <= top:
        return 0.0
    span = min(a["bottom"] - a["top"], b["bottom"] - b["top"])
    return (bot - top) / span if span > 0 else 0.0


def _line_height(words: list[dict]) -> float:
    if not words:
        return 12.0
    return sum(w["bottom"] - w["top"] for w in words) / len(words)


def _group_rows(words: list[dict], tol: float) -> list[list[dict]]:
    """Cluster words into visual rows by y-coordinate."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w["top"], w["x0"]))
    rows: list[list[dict]] = [[ordered[0]]]
    for w in ordered[1:]:
        if abs(w["top"] - rows[-1][0]["top"]) < tol:
            rows[-1].append(w)
        else:
            rows.append([w])
    for row in rows:
        row.sort(key=lambda w: w["x0"])
    return rows


def _find_label_span(
    label_re: re.Pattern, words: list[dict]
) -> Optional[tuple[int, int]]:
    """Locate a label (possibly spanning multiple words) in the word list.

    Returns the (start, end_exclusive) index range; preference given to
    labels ending with ':' and at row start (column-header position).
    """
    n = len(words)
    candidates: list[tuple[int, int, int]] = []
    for i in range(n):
        pieces = [words[i]["text"]]
        for j in range(i, n):
            if j > i and _y_overlap(words[i], words[j]) < 0.5:
                break
            if j > i:
                pieces.append(words[j]["text"])
            raw = " ".join(pieces).strip()
            joined = raw.lower().rstrip(":").strip()
            if label_re.fullmatch(joined):
                has_colon = raw.endswith(":")
                left = words[i]
                is_row_start = not any(
                    k != i
                    and w["x1"] <= left["x0"] - 5
                    and _y_overlap(left, w) >= 0.5
                    for k, w in enumerate(words)
                )
                score = (2 if has_colon else 0) + (1 if is_row_start else 0)
                candidates.append((score, i, j + 1))
                break
            if j - i >= 4:
                break
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return candidates[0][1], candidates[0][2]


def _words_right_of(
    words: list[dict], start: int, end: int, y_ref: dict, x_end: float
) -> list[dict]:
    """Words on the same visual row as y_ref, to the right of x_end."""
    return [
        w for k, w in enumerate(words)
        if not (start <= k < end)
        and w["x0"] >= x_end - 2
        and _y_overlap(y_ref, w) >= 0.5
    ]


def _rows_below(
    words: list[dict], start: int, end: int, y_ref: dict, n_rows: int, lh: float
) -> list[list[dict]]:
    """Up to ``n_rows`` visual rows immediately below y_ref."""
    below = [
        w for k, w in enumerate(words)
        if not (start <= k < end)
        and w["top"] > y_ref["bottom"] - lh * 0.2
        and w["top"] < y_ref["bottom"] + (n_rows + 1) * lh
    ]
    return _group_rows(below, tol=lh * 0.6)[:n_rows]


# =====================================================================
# Value pickers
# =====================================================================

def _pick_inline(
    compiled_labels: list[re.Pattern], words: list[dict]
) -> Optional[tuple[str, dict]]:
    """Find label, return (value_text, label_word_ref) from same-row right-of."""
    for pat in compiled_labels:
        span = _find_label_span(pat, words)
        if span is None:
            continue
        start, end = span
        label_ref = words[start]
        x_end = max(w["x1"] for w in words[start:end])
        right = _words_right_of(words, start, end, label_ref, x_end)
        if not right:
            continue
        right.sort(key=lambda w: w["x0"])
        out: list[str] = []
        for w in right:
            # Stop when we hit another labeled field (e.g. "Date of issue:")
            if out and w["text"].endswith(":"):
                break
            out.append(w["text"])
        if out:
            return " ".join(out).strip(), label_ref
    return None


def _label_positions(
    compiled_labels: list[re.Pattern], words: list[dict]
) -> list[tuple[int, int, dict]]:
    """Return every (start, end, label_ref) for labels matched in words."""
    positions: list[tuple[int, int, dict]] = []
    for pat in compiled_labels:
        span = _find_label_span(pat, words)
        if span is not None:
            start, end = span
            positions.append((start, end, words[start]))
    return positions


def _pick_value(
    compiled_labels: list[re.Pattern],
    words: list[dict],
    validate,
    col_tol: Optional[float] = None,
) -> Optional[str]:
    """General picker: try inline-right, then column-aligned below, then any below,
    applying ``validate(text)`` to each candidate until one passes.
    """
    if not words:
        return None
    lh = _line_height(words)
    # Scale by line height so the same ratio holds for pdfplumber points
    # (lh ~12) and tesseract pixels (lh ~30–60). 60pt / 12pt ≈ 5.
    if col_tol is None:
        col_tol = max(60.0, lh * 5)
    for pat in compiled_labels:
        span = _find_label_span(pat, words)
        if span is None:
            continue
        start, end = span
        label_ref = words[start]
        x_end = max(w["x1"] for w in words[start:end])
        x_anchor = min(w["x0"] for w in words[start:end])

        # 1) Same row, right of label.
        right = _words_right_of(words, start, end, label_ref, x_end)
        if right:
            right.sort(key=lambda w: w["x0"])
            pieces: list[str] = []
            for w in right:
                if pieces and w["text"].endswith(":"):
                    break
                pieces.append(w["text"])
            candidate = " ".join(pieces).strip()
            v = validate(candidate)
            if v:
                return v

        # 2) Row directly below, aligned to label column.
        rows = _rows_below(words, start, end, label_ref, n_rows=4, lh=lh)
        for row in rows:
            in_col = [w for w in row if abs(w["x0"] - x_anchor) <= col_tol]
            if in_col:
                in_col.sort(key=lambda w: w["x0"])
                candidate = " ".join(w["text"] for w in in_col).strip()
                v = validate(candidate)
                if v:
                    return v
            # 3) Any word on that row as fallback.
            candidate = " ".join(w["text"] for w in row).strip()
            v = validate(candidate)
            if v:
                return v
    return None


def _pick_block_below(
    compiled_labels: list[re.Pattern],
    words: list[dict],
    col_tol: Optional[float] = None,
) -> Optional[str]:
    """For block labels (Seller:/Client:), return the first non-stop row below
    that is aligned with the label's x-column.
    """
    lh = _line_height(words)
    if col_tol is None:
        col_tol = max(50.0, lh * 4)
    for pat in compiled_labels:
        span = _find_label_span(pat, words)
        if span is None:
            continue
        start, end = span
        label_ref = words[start]
        x_anchor = min(w["x0"] for w in words[start:end])
        x_right_boundary = x_anchor + 300
        rows = _rows_below(words, start, end, label_ref, n_rows=6, lh=lh)
        for row in rows:
            in_col = [
                w for w in row
                if abs(w["x0"] - x_anchor) <= col_tol
                or (x_anchor - col_tol <= w["x0"] <= x_right_boundary)
            ]
            in_col = [w for w in in_col if w["x0"] <= x_right_boundary]
            if not in_col:
                continue
            in_col.sort(key=lambda w: w["x0"])
            text = " ".join(w["text"] for w in in_col).strip()
            if _STOP_BLOCK_RE.match(text):
                break
            cleaned = _clean_block_value(text)
            if cleaned:
                return cleaned
    return None


def _clean_block_value(text: str) -> Optional[str]:
    text = text.strip(" :,-")
    if not text:
        return None
    # Strip trailing label-like fragment
    m = re.search(r"\b(tax\s*id|vat|iban|phone|tel|email|bank)\b", text, re.IGNORECASE)
    if m:
        text = text[: m.start()].strip(" :,-")
    if not text or len(text) < 2:
        return None
    # Reject if mostly digits (an address line)
    letters = sum(1 for c in text if c.isalpha())
    if letters < 2:
        return None
    return text


# =====================================================================
# Per-field validators
# =====================================================================

def _validate_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    m = DATE_RE.search(s)
    if not m:
        return None
    return next((g for g in m.groups() if g), None)


def _validate_invoice_number(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    for tok in s.split():
        cand = tok.rstrip(":,.;")
        if _YEAR_ONLY_RE.fullmatch(cand):
            continue
        if _INV_NUMBER_VALUE_RE.match(cand):
            return cand
    return None


def _is_monetary(tok: str) -> bool:
    t = tok.strip()
    if _YEAR_ONLY_RE.fullmatch(t):
        return False
    if re.search(r"[\$€£]|RM|USD|EUR|GBP", t):
        return True
    # Has a decimal separator with 2-digit cents
    if re.search(r"[.,]\d{2}\b", t):
        return True
    digits = re.sub(r"[^\d]", "", t)
    return len(digits) >= 3


def _amount_as_float(tok: str) -> float:
    t = re.sub(r"^(?:[\$€£]|RM|USD|EUR|GBP)\s*", "", tok).strip()
    t = t.replace(" ", "").replace("\u00A0", "")
    try:
        # EU: 1.234.567,89 — dot thousands, comma cents
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d{2}", t):
            return float(t.replace(".", "").replace(",", "."))
        # EU simple: 1234,56
        if re.fullmatch(r"\d+,\d{2}", t):
            return float(t.replace(",", "."))
        # US: 1,234,567.89 or plain
        return float(t.replace(",", ""))
    except ValueError:
        return -1.0


def _strip_currency(s: str) -> str:
    return re.sub(r"^(?:[\$€£]|RM|USD|EUR|GBP)\s*", "", s).strip()


# =====================================================================
# Layout-path extractors (per field)
# =====================================================================

def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def _extract_invoice_number(words: list[dict]) -> Optional[str]:
    val = _pick_value(
        _compile(_LABELS["invoice_number"]), words, _validate_invoice_number,
    )
    if val:
        return val
    # Fallback: a bare "#ABC-123" token (common on Canva-style invoices).
    for w in words:
        m = re.fullmatch(r"#([A-Z0-9][A-Z0-9\-/]{2,19})", w["text"], re.IGNORECASE)
        if m:
            return f"#{m.group(1)}"
    return None


def _extract_date(words: list[dict], key: str) -> Optional[str]:
    val = _pick_value(_compile(_LABELS[key]), words, _validate_date)
    if val or key != "invoice_date":
        return val
    # Fallback: first parseable date on any row that isn't next to a
    # "due date" / "payment due" label.
    due_positions = _label_positions(_compile(_LABELS["due_date"]), words)
    due_tops = {round(words[s]["top"]) for s, _, _ in due_positions}
    lh = _line_height(words)
    rows = _group_rows(words, tol=lh * 0.6)
    for row in rows:
        if round(row[0]["top"]) in due_tops:
            continue
        line = " ".join(w["text"] for w in row)
        v = _validate_date(line)
        if v:
            return v
    return None


def _extract_issuer(words: list[dict]) -> Optional[str]:
    compiled = _compile(_ISSUER_LABELS)
    # Block-below first (common Seller:\nCompany pattern).
    val = _pick_block_below(compiled, words)
    if val:
        return val
    # Same-row right of a "From:" / "Vendor:" label.
    val = _pick_value(compiled, words, _clean_block_value)
    if val:
        return val
    return _header_company_guess(words)


def _extract_recipient(words: list[dict]) -> Optional[str]:
    compiled = _compile(_RECIPIENT_LABELS)
    # Same-row right ("Bill To: Acme Corp") is common too.
    val = _pick_value(compiled, words, _clean_block_value)
    if val:
        return val
    return _pick_block_below(compiled, words)


def _header_company_guess(words: list[dict]) -> Optional[str]:
    """Fallback: pick the most prominent text line in the top ~20% of page as
    the issuer when no 'Seller:' anchor is found."""
    if not words:
        return None
    lh = _line_height(words)
    y_min = min(w["top"] for w in words)
    y_max = max(w["bottom"] for w in words)
    cutoff = y_min + (y_max - y_min) * 0.25
    head = [w for w in words if w["bottom"] <= cutoff]
    if not head:
        return None
    rows = _group_rows(head, tol=lh * 0.6)
    for row in rows:
        text = " ".join(w["text"] for w in row).strip(" :,-")
        text = _trim_generic_suffix(text)
        if len(text) < 3:
            continue
        if re.fullmatch(r"(invoice|receipt|bill|statement|tax\s*invoice)",
                        text, re.IGNORECASE):
            continue
        if _STOP_BLOCK_RE.match(text):
            continue
        if re.search(r"[\$€£]|\d[.,]\d{2}", text):
            continue
        letters = sum(1 for c in text if c.isalpha())
        if letters < 3:
            continue
        return text
    return None


def _trim_generic_suffix(text: str) -> str:
    """Strip trailing doc-type words that bled into a header row, e.g.
    ``ACME WIDGETS, INC. INVOICE`` → ``ACME WIDGETS, INC.``"""
    return re.sub(
        r"\s+(INVOICE|RECEIPT|BILL|STATEMENT|TAX\s*INVOICE|QUOTATION|QUOTE)\s*$",
        "", text, flags=re.IGNORECASE,
    ).strip()


def _extract_total(words: list[dict]) -> Optional[str]:
    """Find the total amount.

    Strategy: locate every ``Total`` / ``Amount due`` label. For each, gather
    the same visual row to the right and also the row directly below; take the
    rightmost monetary token. Prefer the bottom-most label on the page (summary
    section). ``Sub Total`` is excluded.
    """
    if not words:
        return None
    label_patterns = _compile(_TOTAL_LABELS)
    lh = _line_height(words)

    # Collect all label matches (pattern-ordered, most specific first).
    all_hits: list[tuple[int, int, dict]] = []
    for pat in label_patterns:
        # Walk the whole word list; _find_label_span returns one match so iterate manually.
        for i in range(len(words)):
            pieces = [words[i]["text"]]
            for j in range(i, len(words)):
                if j > i and _y_overlap(words[i], words[j]) < 0.5:
                    break
                if j > i:
                    pieces.append(words[j]["text"])
                raw = " ".join(pieces).strip()
                joined = raw.lower().rstrip(":").strip()
                if pat.fullmatch(joined):
                    if re.search(r"sub\s*total", joined):
                        break
                    all_hits.append((i, j + 1, words[i]))
                    break
                if j - i >= 3:
                    break

    if not all_hits:
        return None

    # Prefer bottom-most (last in reading order).
    all_hits.sort(key=lambda t: t[2]["top"], reverse=True)

    for start, end, label_ref in all_hits:
        x_end = max(w["x1"] for w in words[start:end])
        same_row = [
            x for k, x in enumerate(words)
            if not (start <= k < end)
            and _y_overlap(label_ref, x) >= 0.5
            and x["x0"] >= label_ref["x1"] - 2
        ]
        same_row.sort(key=lambda x: x["x0"])
        row_text = " ".join(x["text"] for x in same_row)
        amt = _pick_rightmost_monetary(row_text)
        if amt is not None:
            return _strip_currency(amt)
        # Try row below.
        rows = _rows_below(words, start, end, label_ref, n_rows=2, lh=lh)
        for row in rows:
            text = " ".join(w["text"] for w in row)
            amt = _pick_rightmost_monetary(text)
            if amt is not None:
                return _strip_currency(amt)
    return None


def _pick_rightmost_monetary(text: str) -> Optional[str]:
    best: Optional[str] = None
    best_val = -1.0
    for m in _AMOUNT_RE.finditer(text):
        tok = m.group(0).strip()
        if not _is_monetary(tok):
            continue
        v = _amount_as_float(tok)
        if v > best_val:
            best_val = v
            best = tok
    return best


# =====================================================================
# Page / document aggregation
# =====================================================================

_FIELDS = ("invoice_number", "invoice_date", "due_date",
           "issuer", "recipient", "total")


def _all_none() -> dict[str, Optional[str]]:
    return {k: None for k in _FIELDS}


def extract_from_words(words: list[dict]) -> dict[str, Optional[str]]:
    """Run the full field extraction against one page's word boxes."""
    result = _all_none()
    if not words:
        return result
    result["invoice_number"] = _extract_invoice_number(words)
    result["invoice_date"]   = _extract_date(words, "invoice_date")
    result["due_date"]       = _extract_date(words, "due_date")
    result["issuer"]         = _extract_issuer(words)
    result["recipient"]      = _extract_recipient(words)
    result["total"]          = _extract_total(words)
    return result


def _extract_from_pages(pages) -> dict[str, Optional[str]]:
    result = _all_none()
    for page in pages:
        words = page.extract_words(extra_attrs=["fontname", "size"]) or []
        page_result = extract_from_words(words)
        for k, v in page_result.items():
            if result[k] is None and v is not None:
                result[k] = v
    return result


def _extract_from_pdf_bytes(pdf_bytes: bytes) -> dict[str, Optional[str]]:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = list(pdf.pages)
            if not pages:
                return _all_none()
            return _extract_from_pages(pages)
    except Exception:
        return _all_none()


# =====================================================================
# Text-path fallback (1D, no bbox information)
# =====================================================================

_INV_NUMBER_TEXT_LABEL = re.compile(
    r"(?:invoice\s*(?:no|number|num|#)|inv\s*[#:]|bill\s*no)\.?\s*[:#]?\s*",
    re.IGNORECASE,
)
_INV_NUMBER_TEXT_VALUE = re.compile(r"(#?[A-Z0-9][A-Z0-9\-/]{2,19})")
_DUE_LABEL_TEXT = re.compile(r"\b(due\s*date|payment\s*due|pay\s*by)\b", re.IGNORECASE)
_ISSUE_LABEL_TEXT = re.compile(
    r"\b(invoice\s*date|date\s*of\s*issue|issue\s*date|bill\s*date|date)\b",
    re.IGNORECASE,
)
_RECIPIENT_LABEL_TEXT = re.compile(
    r"(?:bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|issued\s*to"
    r"|client|customer|recipient)\s*[:\-]?\s*(.*)",
    re.IGNORECASE,
)
_SELLER_LABEL_TEXT = re.compile(
    r"^\s*(seller|from|vendor|bill\s*from|issued\s*by)\s*[:\-]?\s*(.*)",
    re.IGNORECASE,
)
_TOTAL_LABEL_TEXT = re.compile(
    r"\b(grand\s*total|total\s*due|amount\s*due|balance\s*due|total)\b",
    re.IGNORECASE,
)


def _find_label_value_on_lines(
    lines: Sequence[str], label_re: re.Pattern, validator
) -> Optional[str]:
    for i, line in enumerate(lines):
        m = label_re.search(line)
        if not m:
            continue
        tail = line[m.end():]
        v = validator(tail)
        if v:
            return v
        for j in range(i + 1, min(i + 4, len(lines))):
            nxt = lines[j].strip()
            if not nxt:
                continue
            v = validator(nxt)
            if v:
                return v
            break
    return None


def _text_invoice_number(text: str) -> Optional[str]:
    def _val(s: str) -> Optional[str]:
        m = _INV_NUMBER_TEXT_VALUE.search(s)
        return _validate_invoice_number(m.group(1)) if m else None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _INV_NUMBER_TEXT_LABEL.search(line)
        if m:
            tail = line[m.end():]
            v = _val(tail)
            if v:
                return v
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j].strip()
                if not nxt:
                    continue
                v = _val(nxt)
                if v:
                    return v
                break
    m = re.search(r"(?<![A-Za-z0-9])#(\d{3,})\b", text)
    return f"#{m.group(1)}" if m else None


def _text_date(text: str, label_re: re.Pattern) -> Optional[str]:
    return _find_label_value_on_lines(
        text.splitlines(), label_re, _validate_date,
    )


def _text_recipient(text: str) -> Optional[str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _RECIPIENT_LABEL_TEXT.search(line)
        if not m:
            continue
        tail = _clean_block_value(m.group(1) or "")
        if tail:
            return tail
        for j in range(i + 1, min(i + 4, len(lines))):
            nxt = lines[j].strip()
            if nxt:
                return _clean_block_value(nxt)
    return None


def _text_issuer(text: str) -> Optional[str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _SELLER_LABEL_TEXT.match(line)
        if not m:
            continue
        tail = _clean_block_value(m.group(2) or "")
        if tail:
            return tail
        for j in range(i + 1, min(i + 5, len(lines))):
            nxt = lines[j].strip()
            if nxt:
                return _clean_block_value(nxt)
    # First substantive line that looks like a company name
    for line in lines[:15]:
        cleaned = _clean_block_value(line)
        if cleaned and len(cleaned) >= 4 and not re.fullmatch(
            r"(invoice|receipt|bill|statement|tax\s*invoice)", cleaned,
            re.IGNORECASE,
        ):
            return cleaned
    return None


def _text_total(text: str) -> Optional[str]:
    lines = text.splitlines()
    for line in reversed(lines):
        if not _TOTAL_LABEL_TEXT.search(line):
            continue
        if re.search(r"sub\s*total", line, re.IGNORECASE):
            continue
        amt = _pick_rightmost_monetary(line)
        if amt is not None:
            return _strip_currency(amt)
    return None


def extract_from_text(text: str) -> dict[str, Optional[str]]:
    if not text:
        return _all_none()
    return {
        "invoice_number": _text_invoice_number(text),
        "invoice_date":   _text_date(text, _ISSUE_LABEL_TEXT),
        "due_date":       _text_date(text, _DUE_LABEL_TEXT),
        "issuer":         _text_issuer(text),
        "recipient":      _text_recipient(text),
        "total":          _text_total(text),
    }


# =====================================================================
# Public API
# =====================================================================

@dataclass
class InvoiceFields:
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    issuer: Optional[str] = None
    recipient: Optional[str] = None
    total: Optional[str] = None


def extract_invoice_fields(
    text: str,
    pdf_bytes: Optional[bytes] = None,
    image_bytes: Optional[bytes] = None,
    words: Optional[list[dict]] = None,
) -> dict:
    """Extract six invoice fields from text and/or bbox data.

    Priority per field: layout result (when word boxes available) → text result.
    Pass ``words`` directly to skip the internal OCR/parse step when the caller
    already has word-box data.
    """
    layout: Optional[dict] = None
    if words is not None:
        layout = extract_from_words(words)
    elif pdf_bytes is not None:
        layout = _extract_from_pdf_bytes(pdf_bytes)
    elif image_bytes is not None:
        try:
            from src.pdf_loader import image_to_words
            ocr_words = image_to_words(image_bytes)
            layout = extract_from_words(ocr_words)
        except Exception:
            layout = None

    text_result = extract_from_text(text or "")
    if layout is None:
        return asdict(InvoiceFields(**text_result))
    return _merge(text_result, layout)


def _merge(text_result: dict, layout_result: dict) -> dict:
    out = dict(text_result)
    for k, lv in layout_result.items():
        if lv is not None:
            out[k] = lv
    return out
