"""Invoice field extraction — rules + layout + CRF in a single module.

Public entry point: ``extract_invoice_fields(text, pdf_bytes=None, image_bytes=None)``
returns the six target fields (invoice_number, invoice_date, due_date, issuer,
recipient, total), with missing values as None.

Three extraction paths combine here:
  - Rule-based regex over flat text (always run).
  - Layout-aware extraction over pdfplumber/tesseract word boxes (run when
    ``pdf_bytes`` or ``image_bytes`` is provided). Layout wins per field
    because bbox-aware reading respects columns.
  - Optional CRF model for ISSUER / RECIPIENT, loaded lazily from
    ``models/crf_invoice.pkl``. Trained via ``scripts/train_crf.py``.

Set ``INVOICE_EXTRACT_LOG_FALLBACK=1`` to log per-field events where the
layout path returned None but the text path produced a value — useful for
tuning layout tolerances against real data.
"""

from __future__ import annotations

import io
import logging
import os
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pdfplumber

try:
    import sklearn_crfsuite
except ImportError:  # pragma: no cover
    sklearn_crfsuite = None

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None


_FALLBACK_LOG = logging.getLogger("invoice_extraction.fallback")
_LOG_FALLBACK = os.getenv("INVOICE_EXTRACT_LOG_FALLBACK", "") not in ("", "0", "false", "False")

_CRF_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "crf_invoice.pkl"
_crf = None
_crf_tried = False


# =====================================================================
# Dates
# =====================================================================

_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"

_DATE_PATTERNS = [
    # DD/MM/YYYY (day-first wins on ambiguous inputs — matches SROIE layout)
    r"\b((?:0?[1-9]|[12]\d|3[01])[/\-\.](?:0?[1-9]|1[0-2])[/\-\.](?:\d{4}|\d{2}))\b",
    # MM/DD/YYYY (US-style, checked second)
    r"\b((?:0?[1-9]|1[0-2])[/\-\.](?:0?[1-9]|[12]\d|3[01])[/\-\.](?:\d{4}|\d{2}))\b",
    # YYYY/MM/DD
    r"\b(\d{4}[/\-\.](?:0?[1-9]|1[0-2])[/\-\.](?:0?[1-9]|[12]\d|3[01]))\b",
    # DD-MON-YYYY
    r"\b((?:0?[1-9]|[12]\d|3[01])[/\-\.]" + _MONTHS + r"[/\-\.](?:\d{4}|\d{2}))\b",
    # Compact DDMMYYYY / YYYYMMDD — year must start 19/20 so 8-digit invoice
    # numbers don't falsely parse as dates.
    r"\b((?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])(?:19|20)\d{2})\b",
    r"\b((?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))\b",
    # "15 March 2024" / "March 15, 2024"
    r"\b((?:0?[1-9]|[12]\d|3[01])\s+" + _MONTHS + r"\.?\s+\d{2,4})\b",
    r"\b(" + _MONTHS + r"\.?\s+(?:0?[1-9]|[12]\d|3[01]),?\s+\d{2,4})\b",
    # "MARCH.06.2024" / "March-06-2024"
    r"\b(" + _MONTHS + r"[\./\-](?:0?[1-9]|[12]\d|3[01])[\./\-]\d{2,4})\b",
]

DATE_RE = re.compile("|".join(_DATE_PATTERNS), re.IGNORECASE)
_DATE_RE = DATE_RE  # internal alias used below


def find_dates(text: str) -> List[str]:
    return [next(g for g in m.groups() if g) for m in DATE_RE.finditer(text)]


_find_dates = find_dates  # internal alias


# =====================================================================
# CRF (issuer / recipient)
# =====================================================================

_TOKEN_RE = re.compile(r"\S+")
_LABEL_LINE_RE = re.compile(
    r"\b(bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|issued\s*to"
    r"|customer|recipient|^\s*to\s*:)",
    re.IGNORECASE,
)
_COMPANY_SUFFIXES = {
    "BHD", "BND", "SDN", "LLC", "PLLC", "LTD", "INC", "CORP", "CO", "CO.",
    "PTE", "GMBH", "AG", "KG", "PLC", "ENTERPRISE", "TRADING",
    "SA", "S.A.", "SL", "S.L.", "SARL", "SAS", "SASU", "EURL",
    "SRL", "SPA", "BV", "NV", "OY", "AB", "AS", "LLP", "LP",
}


def tokenize(text: str) -> List[List[str]]:
    """Return tokens grouped by line. Empty lines are dropped."""
    lines = []
    for raw in text.splitlines():
        toks = _TOKEN_RE.findall(raw)
        if toks:
            lines.append(toks)
    return lines


def _shape(word: str) -> str:
    out = []
    for ch in word:
        if ch.isupper():
            out.append("X")
        elif ch.islower():
            out.append("x")
        elif ch.isdigit():
            out.append("d")
        else:
            out.append(ch)
    return "".join(out)


def token_features(
    lines: Sequence[Sequence[str]],
    line_idx: int,
    tok_idx: int,
    prev_line_has_label: bool,
) -> dict:
    tokens = lines[line_idx]
    word = tokens[tok_idx]
    feats = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word.shape": _shape(word)[:8],
        "suffix3": word[-3:].lower(),
        "prefix3": word[:3].lower(),
        "is_upper": word.isupper(),
        "is_title": word.istitle(),
        "is_digit": word.isdigit(),
        "has_currency": bool(re.search(r"[$€£]|\bRM\b|\bUSD\b", word)),
        "has_at": "@" in word,
        "line_idx": min(line_idx, 10),
        "line_pos": tok_idx,
        "line_len": len(tokens),
        "is_first_in_line": tok_idx == 0,
        "is_last_in_line": tok_idx == len(tokens) - 1,
        "in_company_suffix": word.strip(".,").upper() in _COMPANY_SUFFIXES,
        "prev_line_label": prev_line_has_label,
    }
    if tok_idx > 0:
        prev = tokens[tok_idx - 1]
        feats["-1:word.lower"] = prev.lower()
        feats["-1:is_title"] = prev.istitle()
    else:
        feats["BOL"] = True
    if tok_idx < len(tokens) - 1:
        nxt = tokens[tok_idx + 1]
        feats["+1:word.lower"] = nxt.lower()
        feats["+1:is_title"] = nxt.istitle()
    else:
        feats["EOL"] = True
    if line_idx == 0:
        feats["BOS"] = True
    return feats


def sent_features(lines: Sequence[Sequence[str]]) -> List[dict]:
    """Flatten tokenized lines into a single token sequence with features.

    The CRF sees the whole document as one sequence — line structure is
    captured via features (line_idx, prev_line_has_label, etc.).
    """
    feats = []
    for li, line in enumerate(lines):
        prev_line_has_label = li > 0 and bool(
            _LABEL_LINE_RE.search(" ".join(lines[li - 1]))
        )
        for ti in range(len(line)):
            feats.append(token_features(lines, li, ti, prev_line_has_label))
    return feats


def flat_tokens(lines: Sequence[Sequence[str]]) -> List[str]:
    return [tok for line in lines for tok in line]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def align_entity(
    tokens: Sequence[str], value: str, min_score: float = 85.0
) -> Optional[Tuple[int, int]]:
    """Find (start, end) token span matching ``value`` in ``tokens``.

    Exact join-and-substring first, falls back to fuzzy window search.
    """
    if not value:
        return None
    target = _normalize(value)
    target_ntoks = max(1, len(target.split()))

    joined = " ".join(t.lower() for t in tokens)
    if target in joined:
        start_char = joined.index(target)
        prefix_tokens = joined[:start_char].split()
        start = len(prefix_tokens)
        end = start + target_ntoks
        if end <= len(tokens):
            return start, end

    if fuzz is None:
        return None
    best = (0.0, None)
    window_sizes = {target_ntoks, max(1, target_ntoks - 1), target_ntoks + 1}
    for w in window_sizes:
        for i in range(0, len(tokens) - w + 1):
            cand = " ".join(t.lower() for t in tokens[i : i + w])
            score = fuzz.ratio(cand, target)
            if score > best[0]:
                best = (score, (i, i + w))
    if best[0] >= min_score and best[1] is not None:
        return best[1]
    return None


def build_bio(
    text: str, entities: dict
) -> Tuple[List[List[str]], List[str], dict]:
    """Return (lines, bio_tags, alignment_report).

    ``entities`` maps label → string value. Tags are BIO at flat-token
    granularity matching ``flat_tokens``.
    """
    lines = tokenize(text)
    tokens = flat_tokens(lines)
    tags = ["O"] * len(tokens)
    report = {"aligned": [], "missed": []}
    for label, value in entities.items():
        if not value:
            continue
        span = align_entity(tokens, value)
        if span is None:
            report["missed"].append(label)
            continue
        start, end = span
        tags[start] = f"B-{label}"
        for k in range(start + 1, end):
            tags[k] = f"I-{label}"
        report["aligned"].append(label)
    return lines, tags, report


class CRFInvoiceExtractor:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def load(cls, path: str | Path) -> "CRFInvoiceExtractor":
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        return cls(model=model)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.model, fh)

    def train(
        self,
        train_lines: Iterable[Sequence[Sequence[str]]],
        train_tags: Iterable[Sequence[str]],
        c1: float = 0.1,
        c2: float = 0.1,
        max_iter: int = 100,
    ) -> None:
        if sklearn_crfsuite is None:
            raise RuntimeError("sklearn-crfsuite not installed")
        X = [sent_features(lines) for lines in train_lines]
        y = [list(t) for t in train_tags]
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iter,
            all_possible_transitions=True,
        )
        self.model.fit(X, y)

    def predict_tags(self, text: str) -> Tuple[List[str], List[str]]:
        if self.model is None:
            return [], []
        lines = tokenize(text)
        tokens = flat_tokens(lines)
        if not tokens:
            return [], []
        feats = sent_features(lines)
        tags = self.model.predict([feats])[0]
        return tokens, tags

    def predict(self, text: str) -> dict:
        """Return {entity_label: extracted_string or None}."""
        tokens, tags = self.predict_tags(text)
        out: dict = {}
        current_label = None
        current_toks: List[str] = []

        def flush():
            nonlocal current_label, current_toks
            if current_label and current_toks:
                existing = out.get(current_label)
                candidate = " ".join(current_toks)
                if existing is None or len(candidate) > len(existing):
                    out[current_label] = candidate
            current_label = None
            current_toks = []

        for tok, tag in zip(tokens, tags):
            if tag == "O":
                flush()
                continue
            prefix, _, label = tag.partition("-")
            if prefix == "B":
                flush()
                current_label = label
                current_toks = [tok]
            elif prefix == "I" and label == current_label:
                current_toks.append(tok)
            else:
                flush()
                current_label = label
                current_toks = [tok]
        flush()
        return out


def _get_crf() -> Optional[CRFInvoiceExtractor]:
    global _crf, _crf_tried
    if _crf_tried:
        return _crf
    _crf_tried = True
    if not _CRF_MODEL_PATH.exists():
        return None
    try:
        _crf = CRFInvoiceExtractor.load(_CRF_MODEL_PATH)
    except Exception:
        _crf = None
    return _crf


# =====================================================================
# Layout-aware extraction (bbox-based)
# =====================================================================

_LAYOUT_FIELDS = ("invoice_number", "invoice_date", "due_date",
                  "issuer", "recipient", "total")


def _all_none() -> dict[str, Optional[str]]:
    return {k: None for k in _LAYOUT_FIELDS}


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


def _find_label_span(
    label_pattern: "re.Pattern", words: list[dict]
) -> Optional[tuple[int, int]]:
    """Return (start, end_exclusive) word-index range whose joined text fully
    matches ``label_pattern``. Best candidate by colon + row-start signal.
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

    Default: right-of-label same-row, then below. ``prefer_below=True`` for
    block-style labels (``Seller:`` / ``Client:``) whose value is on the next row.
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
        rows: list[list[dict]] = []
        for w in below:
            if rows and abs(w["top"] - rows[-1][0]["top"]) < lh / 2:
                rows[-1].append(w)
            else:
                rows.append([w])
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
                if gap > 3 * lh:
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
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d{2})?"
    r"|\d{1,3}(?:\s\d{3})+(?:,\d{2})?"
    r"|\d+[,.]\d{2}"
    r"|\d+(?:\.\d{2})?)"
)

_INVOICE_NO_VALUE_RE = re.compile(r"^[#A-Z0-9][A-Z0-9\-/]{2,19}$", re.IGNORECASE)
_YEAR_LIKE = re.compile(r"^(?:19|20)\d{2}$")


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


def _looks_like_money(tok: str) -> bool:
    """A genuine monetary amount has a decimal separator, currency symbol,
    thousands separator, or is at least 3 digits — filters out days/months/years."""
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
    out["invoice_number"] = _validate_invoice_number(
        find_value_for_label(_LABELS["invoice_number"], words)
    )
    out["invoice_date"]   = _validate_date(find_value_for_label(_LABELS["invoice_date"], words))
    out["due_date"]       = _validate_date(find_value_for_label(_LABELS["due_date"], words))
    return out


def extract_issuer_recipient_from_words(words: list[dict]) -> dict[str, Optional[str]]:
    return {
        "issuer":    find_value_for_label(_ISSUER_LABELS, words, prefer_below=True),
        "recipient": find_value_for_label(_RECIPIENT_LABELS, words, prefer_below=True),
    }


def extract_total_from_words(words: list[dict]) -> Optional[str]:
    """Return the grand-total amount from a list of word dicts.

    Invoices commonly print Net | VAT | Gross above a ``Total`` label row.
    Both rows contain money but Gross is the true total. We collect money
    tokens from the label row and the row directly above, then take the max.
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
        same = [
            w for k, w in enumerate(words)
            if not (start <= k < end)
            and w["x0"] >= x_end - 2
            and _y_overlap(y_ref, w) >= 0.5
        ]
        if same:
            same.sort(key=lambda w: w["x0"])
            rows_text.append(" ".join(w["text"] for w in same))
        above = [
            w for w in words
            if w["bottom"] <= y_ref["top"]
            and w["bottom"] > y_ref["top"] - 1.8 * lh
            and w["x1"] >= x_end - 2
        ]
        if above:
            above.sort(key=lambda w: (w["top"], w["x0"]))
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
    """Run the full layout extraction against a single page of words."""
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


def extract_layout(pdf_bytes: bytes) -> dict[str, Optional[str]]:
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


# =====================================================================
# Rule-based per-field extractors (text path)
# =====================================================================

def extract_invoice_date(text: str) -> Optional[str]:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.search(r'\b(invoice\s*date|date\s*of\s*issue|date)\b', line, re.IGNORECASE):
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
    m = re.search(
        r'\b(?:payment|due|pay)\b[^.\n]*?\bwithin\s+(\d{1,3})\s*days?\b',
        text, re.IGNORECASE,
    )
    if m:
        return f'NET{m.group(1)}'
    return None


_INVOICE_NO_LABEL = re.compile(
    r'(?:invoice\s*(?:no|number|num|#)\.?|inv\s*[#:]|bill\s*no\.?)'
    r'\s*[:#]?\s*',
    re.IGNORECASE,
)
_INVOICE_NO_VALUE = re.compile(r'(#?[A-Z0-9][A-Z0-9\-/]{2,})')
_HASH_NUM_RE = re.compile(r'^\s*(#\d{3,})\s*$', re.MULTILINE)


def extract_invoice_number(text: str) -> Optional[str]:
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


_AMOUNT_RE = re.compile(
    r'(?:RM|MYR|USD|EUR|GBP|NOK|SEK|DKK|\$|€|£)?\s*'
    r'(\d{1,3}(?:,\d{3})+(?:\.\d{2})?'
    r'|\d{1,3}(?:\s\d{3})+(?:,\d{2})?'
    r'|\d+,\d{2}(?!\d)'
    r'|\d+(?:\.\d{2})?)'
    r'\s*(?:RM|MYR|USD|EUR|GBP|NOK|SEK|DKK)?'
)
_TOTAL_KEYWORDS = re.compile(
    r'\b(grand\s*total|total\s*amount|total\s*due|totals?|amount\s*due|balance\s*due)\b',
    re.IGNORECASE,
)


def _as_float(s: str) -> float:
    s = s.strip().replace(' ', '')
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
       wins. Otherwise fall back to max across total-keyword lines.
    """
    lines = text.splitlines()
    candidates: list[tuple[str, bool]] = []

    def _amounts_in(line: str) -> list[str]:
        if _DATE_RE.search(line):
            return []
        amounts = _AMOUNT_RE.findall(line)
        return [a for a in amounts if not _YEAR_LIKE.fullmatch(a.strip())]

    for i, line in enumerate(lines):
        if not _TOTAL_KEYWORDS.search(line):
            continue
        if re.search(r'sub\s*total', line, re.IGNORECASE):
            continue
        is_round = bool(re.search(r'round', line, re.IGNORECASE))
        amounts = _amounts_in(line)
        if amounts:
            candidates.append((amounts[-1], is_round))
            continue
        seen = 0
        for j in range(i + 1, min(i + 5, len(lines))):
            nxt = lines[j].strip()
            if not nxt:
                continue
            seen += 1
            nxt_amounts = _amounts_in(nxt)
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


_COMPANY_SUFFIX = re.compile(
    r'\b(SDN\.?\s*B[HN]D|BHD|BND|LLC|PLLC|LTD|INC|CORP|CO\.?|ENTERPRISE|TRADING'
    r'|PTE|GMBH|AG|KG|PLC|S\.?A\.?|S\.?L\.?|SARL|SAS|SASU|EURL|SRL|SPA|BV|NV'
    r'|OY|AB|AS|LLP|LP)\b',
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
_DOC_TYPE_RE = re.compile(
    r'^(invoice|receipt|bill|statement|quotation|quote|tax\s*invoice'
    r'|invoice\s*/\s*tax\s*receipt)$',
    re.IGNORECASE,
)
_TEMPLATE_TITLE_RE = re.compile(
    r'\b(invoice|receipt|bill|statement|quotation)\s*$',
    re.IGNORECASE,
)
_TABLE_HEADER_RE = re.compile(
    r'^(description|item|items|qty|quantity|rate|price|amount|total|tax|subtotal'
    r'|hours|um|net\s+price|net\s+worth|gross\s+worth|vat|no\.?|id|#)\b',
    re.IGNORECASE,
)
_TABLE_HEADER_WORDS = frozenset({
    "id", "no", "no.", "#", "description", "item", "items", "qty", "quantity",
    "rate", "price", "amount", "total", "tax", "subtotal", "hours", "um",
    "net", "gross", "vat", "worth",
})


def _is_table_header_line(line: str) -> bool:
    tokens = re.findall(r"[A-Za-z#]+\.?", line.lower())
    if len(tokens) < 3:
        return False
    hits = sum(1 for t in tokens if t in _TABLE_HEADER_WORDS)
    return hits >= 3


def extract_issuer(text: str) -> Optional[str]:
    crf = _get_crf()
    if crf is not None:
        pred = crf.predict(text).get("ISSUER")
        if pred:
            return pred
    return _rule_based_issuer(text)


def _rule_based_issuer(text: str) -> Optional[str]:
    lines = text.splitlines()

    for i, ln in enumerate(lines):
        if _SELLER_LABEL_RE.match(ln):
            for j in range(i + 1, min(i + 5, len(lines))):
                nxt = lines[j].strip()
                if nxt:
                    return nxt
            break

    head_lines: list[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if _RECIPIENT_MARKER.search(ln):
            break
        if _DOC_TYPE_RE.match(ln):
            continue
        if _TEMPLATE_TITLE_RE.search(ln) and len(ln.split()) >= 3:
            continue
        if _TABLE_HEADER_RE.match(ln) or _is_table_header_line(ln):
            continue
        if _INVOICE_NO_LABEL.match(ln):
            continue
        if re.search(r'[\$€£]\s*\d|\d[\.,]\d{2}\b', ln):
            continue
        if _DATE_RE.fullmatch(ln.strip()):
            continue
        letters = sum(1 for ch in ln if ch.isalpha())
        if letters < 3 or letters < len(ln.replace(" ", "")) * 0.4:
            continue
        head_lines.append(ln)
        if len(head_lines) >= 8:
            break
    with_suffix = [ln for ln in head_lines if _COMPANY_SUFFIX.search(ln)]
    if with_suffix:
        return max(with_suffix, key=len)
    return head_lines[0] if head_lines else None


_RECIPIENT_LABEL_RE = re.compile(
    r'(?:bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|issued\s*to'
    r'|customer|recipient|client)\s*[:\-]?\s*(.*)',
    re.IGNORECASE,
)
_TO_LABEL_LINE_RE = re.compile(r'^\s*to\s*[:\-]\s*(.*)$', re.IGNORECASE)
_NON_NAME_RE = re.compile(
    r'(\$|€|£|\bRM\b|\bUSD\b|\bEUR\b|\bGBP\b|\d{2,}|:)',
    re.IGNORECASE,
)
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
) -> dict:
    text_result = asdict(InvoiceFields(
        invoice_number=extract_invoice_number(text),
        invoice_date=extract_invoice_date(text),
        due_date=extract_due_date(text),
        issuer=extract_issuer(text),
        recipient=extract_recipient(text),
        total=extract_total(text),
    ))

    layout_result: Optional[dict] = None
    if pdf_bytes is not None:
        layout_result = extract_layout(pdf_bytes)
    elif image_bytes is not None:
        from .pdf_loader import image_to_words
        try:
            words = image_to_words(image_bytes)
            layout_result = extract_from_words(words)
        except Exception:
            layout_result = None

    if layout_result is None:
        return text_result
    return _merge(text_result, layout_result)


def _merge(text_result: dict, layout_result: dict) -> dict:
    """Per-field merge of text path and layout path. Layout wins where present."""
    out = dict(text_result)
    for k, lv in layout_result.items():
        if lv is not None:
            out[k] = lv
        elif _LOG_FALLBACK and text_result.get(k):
            _FALLBACK_LOG.info(
                "fallback field=%s text=%r", k, text_result[k],
            )
    return out
