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


def _y_overlap(a: dict, b: dict) -> float:
    """Fractional y-band overlap between two words (0..1)."""
    top = max(a["top"], b["top"])
    bot = min(a["bottom"], b["bottom"])
    if bot <= top:
        return 0.0
    span = min(a["bottom"] - a["top"], b["bottom"] - b["top"])
    return (bot - top) / span if span > 0 else 0.0


def _line_height(words: list[dict]) -> float:
    if not words:
        return 12.0
    heights = [w["bottom"] - w["top"] for w in words]
    return sum(heights) / len(heights)


def _find_label_span(label_pattern: "re.Pattern", words: list[dict]) -> Optional[tuple[int, int]]:
    """Return (start, end_exclusive) word-index range whose concatenated text
    FULLY matches ``label_pattern`` (case-insensitive).

    A label candidate is "real" when it behaves like a label, not a random
    occurrence of the same word inside a product description (``Thin Client
    Computer`` contains the word ``Client`` but isn't a label). We score
    candidates by two signals and return the best:
      - trailing colon on the last token (strongest label signal)
      - row-start: no substantive word to the immediate left in the same y-band
    """
    n = len(words)
    candidates: list[tuple[tuple[int, int], int, int]] = []
    for i in range(n):
        pieces = [words[i]["text"]]
        for j in range(i, n):
            if j > i and _y_overlap(words[i], words[j]) < 0.5:
                break
            if j > i:
                pieces.append(words[j]["text"])
            raw = " ".join(pieces).strip()
            joined = raw.lower().rstrip(":").strip()
            if label_pattern.fullmatch(joined):
                has_colon = raw.endswith(":")
                left = words[i]
                is_row_start = not any(
                    k != i
                    and w["x1"] <= left["x0"] - 5
                    and _y_overlap(left, w) >= 0.5
                    for k, w in enumerate(words)
                )
                score = (2 if has_colon else 0) + (1 if is_row_start else 0)
                candidates.append(((score, -i), i, j + 1))
                break
            if j - i >= 5:
                break
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def find_value_for_label(
    label_patterns: list[str],
    words: list[dict],
    prefer_below: bool = False,
) -> Optional[str]:
    """Locate a label among ``words`` and return the nearest value string.

    Default search order: right-of-label in the same y-band, then below-label.
    Set ``prefer_below=True`` for block-style labels (``Seller:`` / ``Client:``)
    whose value is typically on the next row.
    """
    if not words:
        return None
    compiled = [re.compile(p, re.IGNORECASE) for p in label_patterns]
    lh = _line_height(words)

    def _right(start: int, end: int, y_ref: dict, x_end: float) -> Optional[str]:
        cands = [
            w for k, w in enumerate(words)
            if not (start <= k < end)
            and w["x0"] >= x_end - 2
            and _y_overlap(y_ref, w) >= 0.5
        ]
        if not cands:
            return None
        cands.sort(key=lambda w: w["x0"])
        # Stop at the next label-like token to avoid eating into a neighbouring column.
        out = []
        for w in cands:
            if re.search(r":$", w["text"]) and out:
                break
            out.append(w["text"])
        return " ".join(out).strip() or None

    def _below(start: int, end: int, y_ref: dict, x_start: float) -> Optional[str]:
        below = [
            w for k, w in enumerate(words)
            if not (start <= k < end)
            and w["top"] > y_ref["bottom"]
            and w["top"] < y_ref["bottom"] + 3 * lh
        ]
        if not below:
            return None
        below.sort(key=lambda w: (w["top"], w["x0"]))
        # Group into visual rows, tolerating a small vertical jitter.
        rows: list[list[dict]] = []
        for w in below:
            if rows and abs(w["top"] - rows[-1][0]["top"]) < lh / 2:
                rows[-1].append(w)
            else:
                rows.append([w])
        # Find the first row containing a word aligned with the label's column.
        # Walk rightward from that word; a big horizontal gap means column break.
        for row in rows:
            row.sort(key=lambda w: w["x0"])
            aligned = [w for w in row if abs(w["x0"] - x_start) <= 50]
            if not aligned:
                continue
            first = min(aligned, key=lambda w: w["x0"])
            idx = row.index(first)
            out: list[dict] = [first]
            for w in row[idx + 1:]:
                if w["text"].endswith(":"):
                    break
                gap = w["x0"] - out[-1]["x1"]
                if gap > 3 * lh:   # ~3x line-height => column break
                    break
                out.append(w)
            return " ".join(w["text"] for w in out).strip() or None
        return None

    for pat in compiled:
        span = _find_label_span(pat, words)
        if span is None:
            continue
        start, end = span
        label_words = words[start:end]
        x_end   = max(w["x1"] for w in label_words)
        x_start = min(w["x0"] for w in label_words)
        y_ref   = label_words[0]

        strategies = (_below, _right) if prefer_below else (_right, _below)
        for strategy in strategies:
            if strategy is _right:
                val = _right(start, end, y_ref, x_end)
            else:
                val = _below(start, end, y_ref, x_start)
            if val:
                return val
    return None


_AMOUNT_TOKEN_RE = re.compile(
    r"(?:[\$€£]|RM|USD|EUR|GBP)?\s*"
    # US style: 1,234 or 1,234.56 | EU style: 1 234 or 1 234,56
    # | plain decimal: 164.97 / 164,56 | plain integer (≥3 digits for money-ness)
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d{2})?"
    r"|\d{1,3}(?:\s\d{3})+(?:,\d{2})?"
    r"|\d+[,.]\d{2}"
    r"|\d+(?:\.\d{2})?)"
)

_INVOICE_NO_VALUE_RE = re.compile(r"^[#A-Z0-9][A-Z0-9\-/]{2,19}$", re.IGNORECASE)


def _validate_date(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    m = DATE_RE.search(s)
    return next((g for g in m.groups() if g), None) if m else None


def _validate_invoice_number(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    first = s.split()[0].rstrip(":,.;")
    return first if _INVOICE_NO_VALUE_RE.match(first) else None


def _strip_currency(s: str) -> str:
    return re.sub(r"^(?:[\$€£]|RM|USD|EUR|GBP)\s*", "", s).strip()


_YEAR_LIKE = re.compile(r"^(?:19|20)\d{2}$")


def _looks_like_money(tok: str) -> bool:
    """A genuine monetary amount has a decimal separator, a currency symbol,
    a thousands separator, or is at least 3 digits. Days (1-31), months
    (1-12), and percentages rarely look like totals — filter them out."""
    t = tok.strip()
    if _YEAR_LIKE.fullmatch(t):
        return False
    if re.search(r"[\$€£]|RM|USD|EUR|GBP", t):
        return True
    if re.search(r"[.,]\d{2}\b", t):
        return True
    digits = re.sub(r"[^\d]", "", t)
    return len(digits) >= 3


def _validate_amount(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    matches = list(re.finditer(_AMOUNT_TOKEN_RE, s))
    if not matches:
        return None
    # Prefer the rightmost match that looks like real money.
    for m in reversed(matches):
        tok = m.group(0).strip()
        if _looks_like_money(tok):
            return tok
    return None


def _amount_as_float(tok: str) -> float:
    """Parse a money token supporting US (1,234.56) and EU (1 234,56) styles."""
    t = re.sub(r"^(?:[\$€£]|RM|USD|EUR|GBP)\s*", "", tok).strip()
    t = t.replace(" ", "")
    if re.fullmatch(r"\d+,\d{2}", t):
        t = t.replace(",", ".")
    else:
        t = t.replace(",", "")
    try:
        return float(t)
    except ValueError:
        return -1.0


_LABELS = {
    "invoice_number": [r"invoice\s*(?:no|number|num|#)\.?", r"bill\s*no\.?", r"inv\s*[#:]?"],
    "invoice_date":   [r"date\s*of\s*issue", r"invoice\s*date", r"issue\s*date", r"date"],
    "due_date":       [r"due\s*date", r"payment\s*due", r"pay\s*by"],
}

_ISSUER_LABELS    = [r"seller", r"from", r"vendor", r"bill\s*from"]
_RECIPIENT_LABELS = [r"client", r"bill\s*to", r"billed\s*to", r"sold\s*to",
                     r"ship\s*to", r"invoice\s*to", r"issued\s*to", r"customer"]

_TOTAL_LABELS = [r"grand\s*total", r"total\s*due", r"amount\s*due",
                 r"balance\s*due", r"total"]


def extract_fields_from_words(words: list[dict]) -> dict[str, Optional[str]]:
    out = _all_none()
    out["invoice_number"] = _validate_invoice_number(find_value_for_label(_LABELS["invoice_number"], words))
    out["invoice_date"]   = _validate_date(find_value_for_label(_LABELS["invoice_date"], words))
    out["due_date"]       = _validate_date(find_value_for_label(_LABELS["due_date"], words))
    return out


def extract_issuer_recipient_from_words(words: list[dict]) -> dict[str, Optional[str]]:
    return {
        "issuer":    find_value_for_label(_ISSUER_LABELS, words, prefer_below=True),
        "recipient": find_value_for_label(_RECIPIENT_LABELS, words, prefer_below=True),
    }


def extract_total_from_words(words: list[dict]) -> Optional[str]:
    """Return the grand-total amount from a list of OCR word dicts.

    Invoices commonly present a VAT breakdown table: Net | VAT | Gross on one
    row, with a ``Total`` label row directly below. Both rows contain money,
    and the Gross column is the true grand total. OCR sometimes truncates the
    Gross value on the Total row itself, so we collect money tokens from both
    the label's row and the row directly above, then take the maximum — the
    grand total is (almost) always the largest figure in that region.
    """
    if not words:
        return None
    label_pat = re.compile("|".join(f"(?:{p})" for p in _TOTAL_LABELS), re.IGNORECASE)
    span = _find_label_span(label_pat, words)
    if span is not None:
        start, end = span
        y_ref = words[start]
        x_end = max(w["x1"] for w in words[start:end])
        lh = _line_height(words)

        rows_text: list[str] = []
        # Same-row words to the right of the label.
        same = [
            w for k, w in enumerate(words)
            if not (start <= k < end)
            and w["x0"] >= x_end - 2
            and _y_overlap(y_ref, w) >= 0.5
        ]
        if same:
            same.sort(key=lambda w: w["x0"])
            rows_text.append(" ".join(w["text"] for w in same))
        # Row directly above the label (VAT summary row), same x region.
        above = [
            w for w in words
            if w["bottom"] <= y_ref["top"]
            and w["bottom"] > y_ref["top"] - 1.8 * lh
            and w["x1"] >= x_end - 2
        ]
        if above:
            above.sort(key=lambda w: (w["top"], w["x0"]))
            # take the closest (last) row only
            rows: list[list[dict]] = []
            for w in above:
                if rows and abs(w["top"] - rows[-1][0]["top"]) < lh / 2:
                    rows[-1].append(w)
                else:
                    rows.append([w])
            last = sorted(rows[-1], key=lambda w: w["x0"])
            rows_text.append(" ".join(w["text"] for w in last))

        best: Optional[str] = None
        best_val = -1.0
        for line in rows_text:
            for m in re.finditer(_AMOUNT_TOKEN_RE, line):
                tok = m.group(0).strip()
                if not _looks_like_money(tok):
                    continue
                v = _amount_as_float(tok)
                if v > best_val:
                    best_val = v
                    best = tok
        if best is not None:
            return _strip_currency(best)

    raw = find_value_for_label(_TOTAL_LABELS, words)
    amt = _validate_amount(raw) if raw else None
    return _strip_currency(amt) if amt else None


def extract_total_from_table(rows: list[list[str]]) -> Optional[str]:
    for row in rows:
        if not row:
            continue
        first = (row[0] or "").lower().strip()
        if re.search(r"\btotal\b", first) and not re.search(r"sub\s*total", first):
            for cell in reversed(row):
                amt = _validate_amount(cell or "")
                if amt:
                    return _strip_currency(amt)
    return None


def extract_from_words(words: list[dict]) -> dict[str, Optional[str]]:
    """Run the full layout extraction against a single page of words.
    Used by the image pipeline (tesseract) and as the core of the PDF path."""
    result = _all_none()
    if not words:
        return result
    header = extract_fields_from_words(words)
    for k in ("invoice_number", "invoice_date", "due_date"):
        result[k] = header[k]
    ir = extract_issuer_recipient_from_words(words)
    result["issuer"]    = ir["issuer"]
    result["recipient"] = ir["recipient"]
    result["total"]     = extract_total_from_words(words)
    return result


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

    result = _all_none()
    for words in pages_words:
        header = extract_fields_from_words(words)
        for k in ("invoice_number", "invoice_date", "due_date"):
            if result[k] is None:
                result[k] = header[k]
        ir = extract_issuer_recipient_from_words(words)
        if result["issuer"] is None:
            result["issuer"] = ir["issuer"]
        if result["recipient"] is None:
            result["recipient"] = ir["recipient"]

    for page, words in zip(reversed(pages), reversed(pages_words)):
        tables = page.extract_tables() or []
        for table in tables:
            t = extract_total_from_table(table)
            if t is not None:
                result["total"] = t
                break
        if result["total"] is not None:
            break
        t = extract_total_from_words(words)
        if t is not None:
            result["total"] = t
            break

    return result
