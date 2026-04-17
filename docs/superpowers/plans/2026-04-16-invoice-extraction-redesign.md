# Invoice Information Extraction Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve invoice information extraction accuracy on modern templated PDFs without regressing SROIE scores, by adding a layout-aware extraction path and patching the existing text-based rules.

**Architecture:** Two parallel extractors (text-based + layout-aware). The public entry point `extract_invoice_fields(text, pdf_bytes=None)` runs both when PDF bytes are available and merges results per-field (layout wins, text fills gaps). Layout extractor uses `pdfplumber` word bounding boxes and table detection. CRF stays unchanged.

**Tech Stack:** Python, pdfplumber, regex, pytest (new), existing CRF/sklearn stack.

**Spec:** `docs/superpowers/specs/2026-04-16-invoice-extraction-redesign-design.md`

---

## File Structure

**Create:**
- `src/date_utils.py` — shared date regex/parser, extracted from `information_extraction.py` for reuse by layout extractor.
- `src/layout_extractor.py` — path B. `extract(pdf_bytes) -> dict[str, Optional[str]]`.
- `tests/__init__.py` — empty, makes pytest discover package.
- `tests/test_extraction.py` — unit tests for path A fixes + path B integration.
- `tests/fixtures/wood_kim.pdf` — sample PDF (user-provided, the failing invoice).
- `scripts/eval_invoices.py` — qualitative eval across the Kaggle image set.

**Modify:**
- `src/information_extraction.py` — import dates from `date_utils`, apply the five rule fixes, change public signature to `extract_invoice_fields(text, pdf_bytes=None)`, wire in path B merge.
- `src/pdf_loader.py` — add `image_to_text(path_or_bytes)` helper for `.jpg`/`.png`.
- `src/service.py` — pass `pdf_bytes` through to `extract_invoice_fields`.
- `requirements.txt` — add `pytest`.

**Do not touch:**
- `src/crf_extractor.py`
- `scripts/train_crf.py`
- `models/crf_invoice.pkl`
- Notebooks (except regression check in §10)
- `frontend/`

---

## Task 1: Add pytest and set up tests/ directory

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/` (empty dir, just `mkdir`)

- [ ] **Step 1: Add pytest to requirements**

Edit `requirements.txt`, add a new line at the end:

```
pytest>=7.0
```

- [ ] **Step 2: Install it**

Run: `pip install pytest`
Expected: `Successfully installed pytest-...`

- [ ] **Step 3: Create tests package**

```bash
mkdir -p tests/fixtures
touch tests/__init__.py
```

- [ ] **Step 4: Verify pytest discovers nothing yet**

Run: `pytest tests/ -v`
Expected: `collected 0 items` (no error).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/__init__.py
git commit -m "chore: add pytest and tests package skeleton"
```

---

## Task 2: Extract date parser into src/date_utils.py

Pure mechanical refactor — no behavior change. Needed so `layout_extractor.py` can reuse the date regex without importing `information_extraction.py` (circular).

**Files:**
- Create: `src/date_utils.py`
- Modify: `src/information_extraction.py` (lines 50–73 approximately, the date section)
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_extraction.py`:

```python
from src.date_utils import find_dates, DATE_RE


def test_find_dates_basic():
    assert find_dates("Invoice date: 15/03/2024") == ["15/03/2024"]


def test_find_dates_named_month():
    assert find_dates("Issued March 15, 2024") == ["March 15, 2024"]


def test_find_dates_multiple():
    got = find_dates("From 01/01/2024 to 31/12/2024")
    assert got == ["01/01/2024", "31/12/2024"]


def test_date_re_exported():
    assert DATE_RE.search("15/03/2024") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extraction.py -v`
Expected: `ModuleNotFoundError: No module named 'src.date_utils'`.

- [ ] **Step 3: Create src/date_utils.py with the extracted code**

Create `src/date_utils.py`:

```python
"""Shared date regex and helper. Extracted from information_extraction.py
so layout_extractor.py can reuse it without a circular import."""

from __future__ import annotations

import re
from typing import List

_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"

_DATE_PATTERNS = [
    # DD/MM/YYYY, MM-DD-YYYY, etc. (day-first — matches SROIE layout)
    r"\b((?:0?[1-9]|[12]\d|3[01])[/\-\.](?:0?[1-9]|1[0-2])[/\-\.](?:\d{4}|\d{2}))\b",
    # YYYY/MM/DD or YYYY-MM-DD
    r"\b(\d{4}[/\-\.](?:0?[1-9]|1[0-2])[/\-\.](?:0?[1-9]|[12]\d|3[01]))\b",
    # DD-MON-YYYY or DD-MON-YY
    r"\b((?:0?[1-9]|[12]\d|3[01])[/\-\.]" + _MONTHS + r"[/\-\.](?:\d{4}|\d{2}))\b",
    # Compact DDMMYYYY or YYYYMMDD
    r"\b((?:0[1-9]|[12]\d|3[01])(?:0[1-9]|1[0-2])\d{4})\b",
    r"\b(\d{4}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))\b",
    # "15 March 2024" or "March 15, 2024"
    r"\b((?:0?[1-9]|[12]\d|3[01])\s+" + _MONTHS + r"\.?\s+\d{2,4})\b",
    r"\b(" + _MONTHS + r"\.?\s+(?:0?[1-9]|[12]\d|3[01]),?\s+\d{2,4})\b",
    # "MARCH.06.2024" / "March-06-2024"
    r"\b(" + _MONTHS + r"[\./\-](?:0?[1-9]|[12]\d|3[01])[\./\-]\d{2,4})\b",
]

DATE_RE = re.compile("|".join(_DATE_PATTERNS), re.IGNORECASE)


def find_dates(text: str) -> List[str]:
    return [next(g for g in m.groups() if g) for m in DATE_RE.finditer(text)]
```

- [ ] **Step 4: Update src/information_extraction.py to import from date_utils**

Replace the date section (roughly lines 50–73 — the `_MONTHS`, `_DATE_PATTERNS`, `_DATE_RE`, `_find_dates` block) with:

```python
from .date_utils import DATE_RE as _DATE_RE, find_dates as _find_dates  # noqa: F401
```

(The leading `from __future__ import annotations` and existing imports stay.)

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: 4 tests pass.

- [ ] **Step 6: Smoke-check information_extraction still imports**

Run: `python -c "from src.information_extraction import extract_invoice_fields; print(extract_invoice_fields('Invoice date: 15/03/2024'))"`
Expected: dict with `invoice_date='15/03/2024'`, other fields None or best-effort.

- [ ] **Step 7: Commit**

```bash
git add src/date_utils.py src/information_extraction.py tests/test_extraction.py
git commit -m "refactor: extract date parser into src/date_utils.py"
```

---

## Task 3: Add US date format (MM/DD/YYYY) to date_utils

**Files:**
- Modify: `src/date_utils.py` (the `_DATE_PATTERNS` list)
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extraction.py`:

```python
def test_us_date_format():
    # MM/DD/YYYY — day > 12 makes this unambiguous US-style
    assert find_dates("Date of issue: 07/29/2011") == ["07/29/2011"]


def test_day_first_still_preferred_on_ambiguous():
    # 03/04/2024 is ambiguous — day-first should win (SROIE behavior)
    # We check that parsing succeeds; exact interpretation doesn't matter
    # here because we return the raw string.
    assert find_dates("Date: 03/04/2024") == ["03/04/2024"]
```

- [ ] **Step 2: Run test to verify first one fails**

Run: `pytest tests/test_extraction.py::test_us_date_format -v`
Expected: `FAIL` — assertion gets `[]` because `29` > `12` so DD/MM pattern rejects it.

- [ ] **Step 3: Add the US pattern to src/date_utils.py**

In `_DATE_PATTERNS`, add as a NEW entry immediately AFTER the existing DD/MM/YYYY pattern (index 1 — so day-first still has first shot):

```python
    # MM/DD/YYYY — month-first alternative (US-style). Day-first above still
    # wins on ambiguous inputs because it's checked first.
    r"\b((?:0?[1-9]|1[0-2])[/\-\.](?:0?[1-9]|[12]\d|3[01])[/\-\.](?:\d{4}|\d{2}))\b",
```

The final list order should be: [day-first DD/MM/YYYY, **new MM/DD/YYYY**, YYYY/MM/DD, DD-MON-YYYY, ...].

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: all date tests pass (including the original 4).

- [ ] **Step 5: Commit**

```bash
git add src/date_utils.py tests/test_extraction.py
git commit -m "feat: support MM/DD/YYYY (US) date format in extractor"
```

---

## Task 4: Add `Seller:` / `Client:` handling to path A

**Files:**
- Modify: `src/information_extraction.py` (regex definitions + `_rule_based_issuer` + `_rule_based_recipient`)
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_extraction.py`:

```python
from src.information_extraction import extract_invoice_fields


def _extract(text):
    return extract_invoice_fields(text)


def test_client_label_as_recipient():
    text = (
        "Some Header Line\n"
        "Client:\n"
        "Thompson PLC\n"
        "810 Adkins Canyon\n"
    )
    assert _extract(text)["recipient"] == "Thompson PLC"


def test_seller_label_as_issuer():
    text = (
        "Date of issue: 07/29/2011\n"
        "\n"
        "Seller:\n"
        "Wood-Kim\n"
        "8881 Nicholas Grove\n"
        "\n"
        "Client:\n"
        "Thompson PLC\n"
    )
    assert _extract(text)["issuer"] == "Wood-Kim"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extraction.py::test_client_label_as_recipient tests/test_extraction.py::test_seller_label_as_issuer -v`
Expected: FAIL — recipient is None, issuer is "Date of issue:" or wrong.

- [ ] **Step 3: Update `_RECIPIENT_MARKER` and `_RECIPIENT_LABEL_RE` in src/information_extraction.py**

Find:

```python
_RECIPIENT_MARKER = re.compile(
    r'\b(bill(?:ed)?\s*to|sold\s*to|ship\s*to|customer|recipient|invoice\s*to|issued\s*to)\b',
    re.IGNORECASE,
)
```

Replace with:

```python
_RECIPIENT_MARKER = re.compile(
    r'\b(bill(?:ed)?\s*to|sold\s*to|ship\s*to|customer|recipient|invoice\s*to'
    r'|issued\s*to|client)\b',
    re.IGNORECASE,
)
```

Find:

```python
_RECIPIENT_LABEL_RE = re.compile(
    r'(?:bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|issued\s*to'
    r'|customer|recipient)\s*[:\-]?\s*(.*)',
    re.IGNORECASE,
)
```

Replace with:

```python
_RECIPIENT_LABEL_RE = re.compile(
    r'(?:bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|issued\s*to'
    r'|customer|recipient|client)\s*[:\-]?\s*(.*)',
    re.IGNORECASE,
)
```

- [ ] **Step 4: Add `Seller:` handling to `_rule_based_issuer`**

Add a new module-level regex near the other issuer regexes:

```python
_SELLER_LABEL_RE = re.compile(r'^\s*(seller|from|vendor|bill\s*from)\s*[:\-]?\s*$',
                              re.IGNORECASE)
```

Replace the body of `_rule_based_issuer` with:

```python
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
        if _TEMPLATE_TITLE_RE.search(ln) and len(ln.split()) >= 3:
            continue
        head_lines.append(ln)
        if len(head_lines) >= 8:
            break
    with_suffix = [ln for ln in head_lines if _COMPANY_SUFFIX.search(ln)]
    if with_suffix:
        return max(with_suffix, key=len)
    return head_lines[0] if head_lines else None
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: `test_client_label_as_recipient` and `test_seller_label_as_issuer` now PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/information_extraction.py tests/test_extraction.py
git commit -m "feat: recognize Seller:/Client: labels for issuer/recipient"
```

---

## Task 5: Label-only-line fallthrough for date fields

**Files:**
- Modify: `src/information_extraction.py` (`extract_invoice_date`, `extract_due_date`)
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_extraction.py`:

```python
def test_date_on_separate_line_from_label():
    text = "Date of issue:\n07/29/2011\nSeller:\nFoo Co\n"
    assert _extract(text)["invoice_date"] == "07/29/2011"


def test_due_date_on_separate_line():
    text = "Due date:\n15/04/2024\n"
    assert _extract(text)["due_date"] == "15/04/2024"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extraction.py::test_date_on_separate_line_from_label tests/test_extraction.py::test_due_date_on_separate_line -v`
Expected: FAIL — both return None because the label line has no date on it.

- [ ] **Step 3: Update `extract_invoice_date` in src/information_extraction.py**

Replace:

```python
def extract_invoice_date(text: str) -> Optional[str]:
    for line in text.splitlines():
        if re.search(r'\b(invoice\s*date|date)\b', line, re.IGNORECASE):
            hits = _find_dates(line)
            if hits:
                return hits[0]
    hits = _find_dates(text)
    return hits[0] if hits else None
```

With:

```python
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
```

- [ ] **Step 4: Update `extract_due_date` similarly**

Replace:

```python
def extract_due_date(text: str) -> Optional[str]:
    for line in text.splitlines():
        if _DUE_DATE_LABEL.search(line):
            hits = _find_dates(line)
            if hits:
                return hits[0]
    m = re.search(r'\bnet\s*(\d{1,3})\b', text, re.IGNORECASE)
    if m:
        return f'NET{m.group(1)}'
    return None
```

With:

```python
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
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: both new tests pass, no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/information_extraction.py tests/test_extraction.py
git commit -m "feat: fall through to next line for label-only date lines"
```

---

## Task 6: European decimal comma in amounts + rightmost-amount-on-total-line

**Files:**
- Modify: `src/information_extraction.py` (`_AMOUNT_RE`, `_as_float`, `extract_total`)
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_extraction.py`:

```python
def test_european_decimal_total():
    text = "TOTAL: $ 164,97\n"
    assert _extract(text)["total"] == "164,97"


def test_net_vat_gross_row_picks_rightmost():
    text = (
        "           VAT [%]   Net worth   VAT    Gross worth\n"
        "              10%     149,97    15,00    164,97\n"
        "Total                 $ 149,97  $ 15,00  $ 164,97\n"
    )
    assert _extract(text)["total"] == "164,97"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extraction.py::test_european_decimal_total tests/test_extraction.py::test_net_vat_gross_row_picks_rightmost -v`
Expected: FAIL.

- [ ] **Step 3: Update `_AMOUNT_RE` to support European decimal comma**

Find:

```python
_AMOUNT_RE = re.compile(
    r'(?:RM|MYR|USD|EUR|GBP|NOK|SEK|DKK|\$|€|£)?\s*'
    r'(\d{1,3}(?:[,\s]\d{3})+(?:\.\d{2})?|\d+(?:\.\d{2})?)'
    r'\s*(?:RM|MYR|USD|EUR|GBP|NOK|SEK|DKK)?'
)
```

Replace with:

```python
_AMOUNT_RE = re.compile(
    r'(?:RM|MYR|USD|EUR|GBP|NOK|SEK|DKK|\$|€|£)?\s*'
    # 1) Thousands-grouped with optional dot-decimal: 1,234 or 1,234.56
    # 2) European decimal comma: 149,97 (exactly 2 digits after comma,
    #    nothing right after — avoids eating into "1,234")
    # 3) Plain: 164 or 164.97
    r'(\d{1,3}(?:[,\s]\d{3})+(?:\.\d{2})?|\d+,\d{2}(?!\d)|\d+(?:\.\d{2})?)'
    r'\s*(?:RM|MYR|USD|EUR|GBP|NOK|SEK|DKK)?'
)
```

- [ ] **Step 4: Update `_as_float` to canonicalize European decimals**

Find:

```python
def _as_float(s: str) -> float:
    try:
        return float(s.replace(',', '').replace(' ', ''))
    except ValueError:
        return 0.0
```

Replace with:

```python
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
```

- [ ] **Step 5: Update `extract_total` to prefer rightmost amount on total line**

Find the candidate-collection block in `extract_total`:

```python
        amounts = _AMOUNT_RE.findall(line)
        if amounts:
            candidates.append((amounts[-1], is_round))
            continue
```

This already takes `amounts[-1]` (rightmost). Verify visually that it still does after the regex change — it does, because `findall` returns matches in order. No edit needed here; the behavior fix comes from the regex accepting `164,97` as a single token.

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: both new total tests pass, no regressions.

- [ ] **Step 7: Commit**

```bash
git add src/information_extraction.py tests/test_extraction.py
git commit -m "feat: handle European decimal commas in amount regex"
```

---

## Task 7: Change `extract_invoice_fields` signature to accept optional `pdf_bytes`

Signature change only; no layout logic yet. This unblocks path B wiring later.

**Files:**
- Modify: `src/information_extraction.py` (`extract_invoice_fields`)
- Modify: `src/service.py` (pass `pdf_bytes` even though it's unused today — keeps call sites consistent)
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extraction.py`:

```python
def test_extract_invoice_fields_accepts_pdf_bytes_kwarg():
    # pdf_bytes=None should behave identically to text-only call.
    text = "Invoice date: 15/03/2024\n"
    a = extract_invoice_fields(text)
    b = extract_invoice_fields(text, pdf_bytes=None)
    assert a == b
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_extraction.py::test_extract_invoice_fields_accepts_pdf_bytes_kwarg -v`
Expected: FAIL — `TypeError: got an unexpected keyword argument 'pdf_bytes'`.

- [ ] **Step 3: Update signature in src/information_extraction.py**

Find:

```python
def extract_invoice_fields(text: str) -> dict:
    return asdict(InvoiceFields(
        invoice_number=extract_invoice_number(text),
        invoice_date=extract_invoice_date(text),
        due_date=extract_due_date(text),
        issuer=extract_issuer(text),
        recipient=extract_recipient(text),
        total=extract_total(text),
    ))
```

Replace with:

```python
def extract_invoice_fields(text: str, pdf_bytes: bytes | None = None) -> dict:
    text_result = asdict(InvoiceFields(
        invoice_number=extract_invoice_number(text),
        invoice_date=extract_invoice_date(text),
        due_date=extract_due_date(text),
        issuer=extract_issuer(text),
        recipient=extract_recipient(text),
        total=extract_total(text),
    ))
    # pdf_bytes is unused for now; path B is wired in a later task.
    return text_result
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Update src/service.py to pass pdf_bytes through**

In `/classify`, find:

```python
    if label == 'invoice':
        response['invoice_fields'] = extract_invoice_fields(raw_text)
```

Replace with:

```python
    if label == 'invoice':
        response['invoice_fields'] = extract_invoice_fields(raw_text, pdf_bytes=pdf_bytes)
```

In `/extract`, find:

```python
    return {'filename': file.filename, 'fields': extract_invoice_fields(text)}
```

Replace with:

```python
    return {'filename': file.filename, 'fields': extract_invoice_fields(text, pdf_bytes=pdf_bytes)}
```

- [ ] **Step 6: Smoke-test service imports**

Run: `python -c "from src.service import app; print('ok')"`
Expected: `ok`.

- [ ] **Step 7: Commit**

```bash
git add src/information_extraction.py src/service.py tests/test_extraction.py
git commit -m "feat: extract_invoice_fields accepts optional pdf_bytes"
```

---

## Task 8: Create layout_extractor.py skeleton

**Files:**
- Create: `src/layout_extractor.py`
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extraction.py`:

```python
from src.layout_extractor import extract as layout_extract


def test_layout_extract_empty_bytes_returns_all_none():
    # Zero-byte PDF — pdfplumber will fail gracefully; we must return all-None.
    result = layout_extract(b"")
    assert result == {
        "invoice_number": None,
        "invoice_date":   None,
        "due_date":       None,
        "issuer":         None,
        "recipient":      None,
        "total":          None,
    }
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_extraction.py::test_layout_extract_empty_bytes_returns_all_none -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.layout_extractor'`.

- [ ] **Step 3: Create src/layout_extractor.py**

```python
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
        # Malformed PDF / pdfplumber error — fall back to all-None so text
        # path can fill in via merge.
        return _all_none()


def _extract_from_pages(pages) -> dict[str, Optional[str]]:
    # Collect all words up front, tagged with page index.
    pages_words: list[list[dict]] = []
    for page in pages:
        words = page.extract_words(extra_attrs=["fontname", "size"]) or []
        pages_words.append(words)

    # If no page has any words, this is a scan — return all-None.
    if not any(pages_words):
        return _all_none()

    # Implementation added in later tasks.
    return _all_none()
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/layout_extractor.py tests/test_extraction.py
git commit -m "feat: scaffold layout_extractor module"
```

---

## Task 9: Implement `find_value_for_label` primitive

**Files:**
- Modify: `src/layout_extractor.py`
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extraction.py`:

```python
from src.layout_extractor import find_value_for_label


def test_find_value_right_of_label():
    # Label "Date of issue:" at x=100..200 on line y=50.
    # Value "07/29/2011" at x=400 on same line.
    words = [
        {"text": "Date",   "x0": 100, "x1": 130, "top": 50, "bottom": 60},
        {"text": "of",     "x0": 135, "x1": 150, "top": 50, "bottom": 60},
        {"text": "issue:", "x0": 155, "x1": 200, "top": 50, "bottom": 60},
        {"text": "07/29/2011", "x0": 400, "x1": 470, "top": 50, "bottom": 60},
    ]
    got = find_value_for_label([r"date of issue"], words)
    assert got == "07/29/2011"


def test_find_value_below_label():
    # Label "Seller:" at x=100..150 on line y=50.
    # Value "Wood-Kim" at x=100..160 on line y=80 (same column).
    words = [
        {"text": "Seller:",  "x0": 100, "x1": 150, "top": 50, "bottom": 60},
        {"text": "Wood-Kim", "x0": 100, "x1": 160, "top": 80, "bottom": 90},
    ]
    got = find_value_for_label([r"seller"], words)
    assert got == "Wood-Kim"


def test_find_value_returns_none_when_label_absent():
    words = [{"text": "Hello", "x0": 0, "x1": 50, "top": 0, "bottom": 10}]
    assert find_value_for_label([r"date"], words) is None
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_extraction.py -k find_value -v`
Expected: FAIL — `find_value_for_label` not exported yet.

- [ ] **Step 3: Implement `find_value_for_label` in src/layout_extractor.py**

Add these helpers and the function ABOVE `extract()`:

```python
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


def _find_label_span(label_pattern: re.Pattern, words: list[dict]) -> Optional[tuple[int, int]]:
    """Return (start, end_exclusive) word-index range whose concatenated text
    matches ``label_pattern`` (case-insensitive), preferring the shortest match.
    Words must be in the same y-band (overlap ≥ 50%)."""
    n = len(words)
    for i in range(n):
        pieces = [words[i]["text"]]
        for j in range(i, n):
            if j > i and _y_overlap(words[i], words[j]) < 0.5:
                break
            if j > i:
                pieces.append(words[j]["text"])
            joined = " ".join(pieces).lower().rstrip(":").strip()
            if label_pattern.search(joined):
                return i, j + 1
            if j - i >= 5:  # label longer than 6 tokens is not realistic
                break
    return None


def find_value_for_label(label_patterns: list[str], words: list[dict]) -> Optional[str]:
    """Locate a label among ``words`` and return the nearest value string.

    Search order: right-of-label in the same y-band, then below-label in the
    same column. Returns None if no label matches or no value found.
    """
    if not words:
        return None
    compiled = [re.compile(p, re.IGNORECASE) for p in label_patterns]
    lh = _line_height(words)

    for pat in compiled:
        span = _find_label_span(pat, words)
        if span is None:
            continue
        start, end = span
        label_words = words[start:end]
        x_end = max(w["x1"] for w in label_words)
        x_start = min(w["x0"] for w in label_words)
        y_ref = label_words[0]

        # 1) Right of label, same y-band.
        right = [
            w for k, w in enumerate(words)
            if not (start <= k < end)
            and w["x0"] >= x_end - 2
            and _y_overlap(y_ref, w) >= 0.5
        ]
        if right:
            right.sort(key=lambda w: w["x0"])
            return " ".join(w["text"] for w in right).strip()

        # 2) Below label, same column (x within ±50pt of label start).
        below = [
            w for k, w in enumerate(words)
            if not (start <= k < end)
            and w["top"] > y_ref["bottom"]
            and w["top"] < y_ref["bottom"] + 3 * lh
            and abs(w["x0"] - x_start) < 50
        ]
        if below:
            # Group by line (top within lh/2).
            below.sort(key=lambda w: (w["top"], w["x0"]))
            first_top = below[0]["top"]
            first_line = [w for w in below if abs(w["top"] - first_top) < lh / 2]
            first_line.sort(key=lambda w: w["x0"])
            return " ".join(w["text"] for w in first_line).strip()

    return None
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/layout_extractor.py tests/test_extraction.py
git commit -m "feat: add find_value_for_label primitive to layout_extractor"
```

---

## Task 10: Implement field-specific layout extraction for invoice_number / invoice_date / due_date

**Files:**
- Modify: `src/layout_extractor.py`
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extraction.py`:

```python
def test_layout_extracts_invoice_date_right_of_label():
    words = [
        {"text": "Date",       "x0": 100, "x1": 130, "top": 50, "bottom": 60},
        {"text": "of",         "x0": 135, "x1": 150, "top": 50, "bottom": 60},
        {"text": "issue:",     "x0": 155, "x1": 200, "top": 50, "bottom": 60},
        {"text": "07/29/2011", "x0": 400, "x1": 470, "top": 50, "bottom": 60},
    ]
    from src.layout_extractor import extract_fields_from_words
    got = extract_fields_from_words(words)
    assert got["invoice_date"] == "07/29/2011"
    assert got["invoice_number"] is None
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_extraction.py::test_layout_extracts_invoice_date_right_of_label -v`
Expected: FAIL — `extract_fields_from_words` not exported.

- [ ] **Step 3: Add validators and `extract_fields_from_words` to src/layout_extractor.py**

Above `extract()`:

```python
_AMOUNT_TOKEN_RE = re.compile(
    r"(?:[\$€£]|RM|USD|EUR|GBP)?\s*"
    r"\d{1,3}(?:[,\s]\d{3})+(?:\.\d{2})?|\d+[,.]\d{2}|\d+(?:\.\d{2})?"
)

_INVOICE_NO_VALUE_RE = re.compile(r"^[#A-Z0-9][A-Z0-9\-/]{2,19}$", re.IGNORECASE)


def _validate_date(s: str) -> Optional[str]:
    if not s:
        return None
    m = DATE_RE.search(s)
    return next((g for g in m.groups() if g), None) if m else None


def _validate_invoice_number(s: str) -> Optional[str]:
    if not s:
        return None
    first = s.split()[0].rstrip(":,.;")
    return first if _INVOICE_NO_VALUE_RE.match(first) else None


def _validate_amount(s: str) -> Optional[str]:
    if not s:
        return None
    matches = list(re.finditer(_AMOUNT_TOKEN_RE, s))
    if not matches:
        return None
    return matches[-1].group(0).strip()


_LABELS = {
    "invoice_number": [r"invoice\s*(?:no|number|num|#)", r"bill\s*no", r"\binv\s*[#:]"],
    "invoice_date":   [r"date\s*of\s*issue", r"invoice\s*date", r"issue\s*date", r"\bdate\b"],
    "due_date":       [r"due\s*date", r"payment\s*due", r"pay\s*by"],
}


def extract_fields_from_words(words: list[dict]) -> dict[str, Optional[str]]:
    """Extract the header fields (number / date / due_date) from a word list."""
    out = _all_none()
    raw = find_value_for_label(_LABELS["invoice_number"], words)
    out["invoice_number"] = _validate_invoice_number(raw)

    raw = find_value_for_label(_LABELS["invoice_date"], words)
    out["invoice_date"] = _validate_date(raw)

    raw = find_value_for_label(_LABELS["due_date"], words)
    out["due_date"] = _validate_date(raw)

    return out
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/layout_extractor.py tests/test_extraction.py
git commit -m "feat: layout extraction for invoice_number/date/due_date"
```

---

## Task 11: Implement issuer/recipient block extraction in layout_extractor

**Files:**
- Modify: `src/layout_extractor.py`
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extraction.py`:

```python
def test_layout_extracts_issuer_below_seller_label():
    # Seller: (y=50), Wood-Kim (y=80), 8881 Nicholas (y=100)
    words = [
        {"text": "Seller:",   "x0": 100, "x1": 150, "top": 50,  "bottom": 60},
        {"text": "Wood-Kim",  "x0": 100, "x1": 160, "top": 80,  "bottom": 90},
        {"text": "8881",      "x0": 100, "x1": 130, "top": 100, "bottom": 110},
        {"text": "Nicholas",  "x0": 135, "x1": 190, "top": 100, "bottom": 110},
        {"text": "Grove",     "x0": 195, "x1": 230, "top": 100, "bottom": 110},
        {"text": "Client:",   "x0": 400, "x1": 450, "top": 50,  "bottom": 60},
        {"text": "Thompson",  "x0": 400, "x1": 470, "top": 80,  "bottom": 90},
        {"text": "PLC",       "x0": 475, "x1": 500, "top": 80,  "bottom": 90},
    ]
    from src.layout_extractor import extract_issuer_recipient_from_words
    got = extract_issuer_recipient_from_words(words)
    assert got["issuer"] == "Wood-Kim"
    assert got["recipient"] == "Thompson PLC"
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_extraction.py::test_layout_extracts_issuer_below_seller_label -v`
Expected: FAIL.

- [ ] **Step 3: Add `extract_issuer_recipient_from_words` to src/layout_extractor.py**

Append after `extract_fields_from_words`:

```python
_ISSUER_LABELS    = [r"seller", r"\bfrom\b", r"vendor", r"bill\s*from"]
_RECIPIENT_LABELS = [r"client", r"bill\s*to", r"billed\s*to", r"sold\s*to",
                     r"ship\s*to", r"invoice\s*to", r"issued\s*to", r"customer"]


def extract_issuer_recipient_from_words(words: list[dict]) -> dict[str, Optional[str]]:
    out = {"issuer": None, "recipient": None}
    out["issuer"]    = find_value_for_label(_ISSUER_LABELS, words)
    out["recipient"] = find_value_for_label(_RECIPIENT_LABELS, words)
    return out
```

Note: `find_value_for_label` already returns the first below-label line, which is the name — exactly what we want.

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/layout_extractor.py tests/test_extraction.py
git commit -m "feat: layout extraction for issuer and recipient"
```

---

## Task 12: Implement total extraction (table-based + label-based)

**Files:**
- Modify: `src/layout_extractor.py`
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extraction.py`:

```python
def test_layout_total_from_label_words():
    # "Total: $ 164,97" spread across words
    words = [
        {"text": "Total:",  "x0": 100, "x1": 140, "top": 50, "bottom": 60},
        {"text": "$",       "x0": 150, "x1": 160, "top": 50, "bottom": 60},
        {"text": "164,97",  "x0": 165, "x1": 210, "top": 50, "bottom": 60},
    ]
    from src.layout_extractor import extract_total_from_words
    assert extract_total_from_words(words) == "164,97"


def test_layout_total_picks_rightmost_amount_on_label_line():
    # Three amounts on a "Total" line — gross (rightmost) wins.
    words = [
        {"text": "Total",   "x0": 100, "x1": 140, "top": 50, "bottom": 60},
        {"text": "$149,97", "x0": 300, "x1": 360, "top": 50, "bottom": 60},
        {"text": "$15,00",  "x0": 400, "x1": 450, "top": 50, "bottom": 60},
        {"text": "$164,97", "x0": 500, "x1": 560, "top": 50, "bottom": 60},
    ]
    from src.layout_extractor import extract_total_from_words
    assert extract_total_from_words(words) == "164,97"


def test_layout_total_from_table():
    from src.layout_extractor import extract_total_from_table
    # First cell "Total", last cell "$164,97" is the rightmost amount.
    rows = [
        ["VAT [%]", "Net worth", "VAT",    "Gross worth"],
        ["10%",     "149,97",    "15,00",  "164,97"],
        ["Total",   "$ 149,97",  "$ 15,00", "$ 164,97"],
    ]
    assert extract_total_from_table(rows) == "164,97"
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_extraction.py -k total -v`
Expected: FAIL.

- [ ] **Step 3: Add total extraction helpers to src/layout_extractor.py**

Append:

```python
_TOTAL_LABELS = [r"grand\s*total", r"total\s*due", r"amount\s*due",
                 r"balance\s*due", r"\btotal\b"]


def extract_total_from_words(words: list[dict]) -> Optional[str]:
    """Find a 'total'-ish label and return the rightmost amount on the same line
    (or below, via find_value_for_label fallback)."""
    if not words:
        return None
    # Locate the label's y-band; then collect all amounts in that y-band to the
    # right of the label, take the rightmost.
    label_pat = re.compile("|".join(f"(?:{p})" for p in _TOTAL_LABELS), re.IGNORECASE)
    span = _find_label_span(label_pat, words)
    if span is not None:
        start, end = span
        y_ref = words[start]
        x_end = max(w["x1"] for w in words[start:end])
        right = [
            w for k, w in enumerate(words)
            if not (start <= k < end)
            and w["x0"] >= x_end - 2
            and _y_overlap(y_ref, w) >= 0.5
        ]
        if right:
            right.sort(key=lambda w: w["x0"])
            joined = " ".join(w["text"] for w in right)
            amt = _validate_amount(joined)
            if amt is not None:
                return _strip_currency(amt)

    # Fallback: label-based below-line search.
    raw = find_value_for_label(_TOTAL_LABELS, words)
    amt = _validate_amount(raw) if raw else None
    return _strip_currency(amt) if amt else None


def _strip_currency(s: str) -> str:
    return re.sub(r"^(?:[\$€£]|RM|USD|EUR|GBP)\s*", "", s).strip()


def extract_total_from_table(rows: list[list[str]]) -> Optional[str]:
    for row in rows:
        if not row:
            continue
        first = (row[0] or "").lower().strip()
        if re.search(r"\btotal\b", first) and not re.search(r"sub\s*total", first):
            # Rightmost cell with an amount-shaped token wins.
            for cell in reversed(row):
                amt = _validate_amount(cell or "")
                if amt:
                    return _strip_currency(amt)
    return None
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/layout_extractor.py tests/test_extraction.py
git commit -m "feat: total extraction from summary tables and label lines"
```

---

## Task 13: Wire multi-page iteration in `_extract_from_pages`

**Files:**
- Modify: `src/layout_extractor.py` (`_extract_from_pages`)
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing test**

We need a real PDF for this. Use a fixture generator at test time (no big binary in git). Append to `tests/test_extraction.py`:

```python
def _make_pdf(text_blocks):
    """Create a simple single-page PDF from (x, y, text) tuples. Returns bytes."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
    except ImportError:
        import pytest
        pytest.skip("reportlab not installed")
    from io import BytesIO
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for x, y, t in text_blocks:
        c.drawString(x, y, t)
    c.showPage()
    c.save()
    return buf.getvalue()


def test_layout_extract_end_to_end_two_column_invoice():
    pdf_bytes = _make_pdf([
        (72, 720, "Date of issue:"),
        (300, 720, "07/29/2011"),
        (72, 640, "Seller:"),
        (72, 620, "Wood-Kim"),
        (300, 640, "Client:"),
        (300, 620, "Thompson PLC"),
        (72, 400, "Total"),
        (500, 400, "$164.97"),
    ])
    from src.layout_extractor import extract as layout_extract
    got = layout_extract(pdf_bytes)
    assert got["invoice_date"] == "07/29/2011"
    assert got["issuer"] == "Wood-Kim"
    assert got["recipient"] == "Thompson PLC"
    assert got["total"] == "164.97"
```

Also add reportlab to requirements (dev only, used only in tests):

Edit `requirements.txt`, append:

```
reportlab>=4.0
```

Then: `pip install reportlab`.

- [ ] **Step 2: Run test**

Run: `pytest tests/test_extraction.py::test_layout_extract_end_to_end_two_column_invoice -v`
Expected: FAIL — `_extract_from_pages` currently returns `_all_none()`.

- [ ] **Step 3: Implement `_extract_from_pages` in src/layout_extractor.py**

Replace the stub:

```python
def _extract_from_pages(pages) -> dict[str, Optional[str]]:
    pages_words: list[list[dict]] = []
    for page in pages:
        words = page.extract_words(extra_attrs=["fontname", "size"]) or []
        pages_words.append(words)

    if not any(pages_words):
        return _all_none()

    # Header fields + issuer/recipient — first non-None wins across pages.
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

    # Total — prefer last page first, try tables then label search.
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
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/layout_extractor.py tests/test_extraction.py requirements.txt
git commit -m "feat: wire layout extractor end-to-end across pages"
```

---

## Task 14: Merge path B into `extract_invoice_fields`

**Files:**
- Modify: `src/information_extraction.py` (`extract_invoice_fields`)
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extraction.py`:

```python
def test_extract_invoice_fields_merges_layout_and_text():
    # Layout will catch date + issuer + recipient + total.
    # Text path catches invoice_number (unique to the text, layout won't have it).
    pdf_bytes = _make_pdf([
        (72, 720, "Date of issue:"),
        (300, 720, "07/29/2011"),
        (72, 700, "Invoice number: INV-2011-42"),
        (72, 640, "Seller:"),
        (72, 620, "Wood-Kim"),
        (300, 640, "Client:"),
        (300, 620, "Thompson PLC"),
        (72, 400, "Total"),
        (500, 400, "$164.97"),
    ])
    # Provide a text that loosely matches what pdfplumber would extract.
    text = (
        "Date of issue:        07/29/2011\n"
        "Invoice number: INV-2011-42\n"
        "Seller:\nWood-Kim\n"
        "Client:\nThompson PLC\n"
        "Total  $164.97\n"
    )
    got = extract_invoice_fields(text, pdf_bytes=pdf_bytes)
    assert got["invoice_date"]    == "07/29/2011"
    assert got["issuer"]          == "Wood-Kim"
    assert got["recipient"]       == "Thompson PLC"
    assert got["total"]           == "164.97"
    assert got["invoice_number"]  == "INV-2011-42"
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_extraction.py::test_extract_invoice_fields_merges_layout_and_text -v`
Expected: FAIL — layout values aren't merged yet (pdf_bytes is ignored).

- [ ] **Step 3: Wire layout merge into src/information_extraction.py**

Add at the top of the file:

```python
from . import layout_extractor
```

Replace the `extract_invoice_fields` body:

```python
def extract_invoice_fields(text: str, pdf_bytes: bytes | None = None) -> dict:
    text_result = asdict(InvoiceFields(
        invoice_number=extract_invoice_number(text),
        invoice_date=extract_invoice_date(text),
        due_date=extract_due_date(text),
        issuer=extract_issuer(text),
        recipient=extract_recipient(text),
        total=extract_total(text),
    ))
    if pdf_bytes is None:
        return text_result
    layout_result = layout_extractor.extract(pdf_bytes)
    return {
        k: (layout_result[k] if layout_result.get(k) is not None else text_result.get(k))
        for k in text_result
    }
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/information_extraction.py tests/test_extraction.py
git commit -m "feat: merge layout-aware and text-based extraction per-field"
```

---

## Task 15: Add `image_to_text` helper to pdf_loader.py

**Files:**
- Modify: `src/pdf_loader.py`
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extraction.py`:

```python
def test_image_to_text_smoke(tmp_path):
    # Create a tiny PNG with known text via PIL and ensure OCR produces some text.
    try:
        from PIL import Image, ImageDraw, ImageFont
        import pytesseract  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("PIL/pytesseract not installed")
    img = Image.new("RGB", (400, 80), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 20), "INVOICE 12345", fill="black")
    p = tmp_path / "inv.png"
    img.save(p)

    from src.pdf_loader import image_to_text
    text = image_to_text(str(p))
    assert "INVOICE" in text.upper()
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_extraction.py::test_image_to_text_smoke -v`
Expected: FAIL — `image_to_text` not defined.

- [ ] **Step 3: Add `image_to_text` to src/pdf_loader.py**

Append at the end of the file:

```python
def image_to_text(path_or_bytes: PathLike) -> str:
    """Run pytesseract on a .jpg/.png/image bytes. Returns extracted text."""
    import pytesseract
    from PIL import Image

    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(path_or_bytes)))
    else:
        img = Image.open(str(path_or_bytes))
    return pytesseract.image_to_string(img)
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: all pass (or `test_image_to_text_smoke` skipped if tesseract missing).

- [ ] **Step 5: Commit**

```bash
git add src/pdf_loader.py tests/test_extraction.py
git commit -m "feat: add image_to_text helper for .jpg/.png inputs"
```

---

## Task 16: Add `scripts/eval_invoices.py`

**Files:**
- Create: `scripts/eval_invoices.py`

- [ ] **Step 1: Create the script**

```python
"""Run information extraction across data/raw/invoices/high_quality_images/
and write per-file results to data/processed/invoice_extraction_eval.csv.

Usage: python scripts/eval_invoices.py

No ground truth is used — this is a qualitative eval to surface systematic
failures across diverse invoice templates.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.information_extraction import extract_invoice_fields
from src.pdf_loader import image_to_text

IMG_DIR = ROOT / "data/raw/invoices/high_quality_images"
OUT_CSV = ROOT / "data/processed/invoice_extraction_eval.csv"
FIELDS = ["invoice_number", "invoice_date", "due_date", "issuer", "recipient", "total"]


def main() -> None:
    if not IMG_DIR.exists():
        print(f"✗ {IMG_DIR} not found — run notebook 01_data_collection first.")
        return

    images = sorted(
        p for p in IMG_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        print(f"✗ No images in {IMG_DIR}")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(images)} images...")

    rows = []
    for i, img in enumerate(images, 1):
        try:
            text = image_to_text(str(img))
            fields = extract_invoice_fields(text)
        except Exception as e:
            print(f"  [{i}/{len(images)}] {img.name}: ERROR {e}")
            fields = {k: None for k in FIELDS}
        non_null = sum(1 for k in FIELDS if fields.get(k))
        rows.append({"filename": img.name, **fields, "non_null_count": non_null})
        if i % 25 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}]")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename"] + FIELDS + ["non_null_count"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Wrote {OUT_CSV}")
    total = len(rows)
    print("\nPer-field non-null rate:")
    for k in FIELDS:
        n = sum(1 for r in rows if r[k])
        print(f"  {k:16s} : {n:4d}/{total} ({100*n/total:5.1f}%)")

    print("\nWorst 10 (lowest non_null_count) — review these for systematic failures:")
    worst = sorted(rows, key=lambda r: r["non_null_count"])[:10]
    for r in worst:
        print(f"  {r['filename']}  (got {r['non_null_count']}/{len(FIELDS)} fields)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on a 5-image sample**

Quick sanity check without processing the whole set. Run:

```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, '.')
from src.pdf_loader import image_to_text
from src.information_extraction import extract_invoice_fields
imgs = sorted(Path('data/raw/invoices/high_quality_images').iterdir())[:3]
for img in imgs:
    if img.suffix.lower() not in {'.jpg', '.jpeg', '.png'}: continue
    t = image_to_text(str(img))
    print(img.name, '→', extract_invoice_fields(t))
"
```

Expected: 3 dicts printed, at least some fields populated. (If tesseract is missing, install via `brew install tesseract`.)

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_invoices.py
git commit -m "feat: add scripts/eval_invoices.py for qualitative diversity eval"
```

---

## Task 17: Add Wood-Kim fixture + end-to-end acceptance test

**Files:**
- Create: `tests/fixtures/wood_kim.pdf` (you provide — save the failing invoice PDF here)
- Test: `tests/test_extraction.py`

- [ ] **Step 1: Save the fixture**

Save the Wood-Kim invoice PDF (the one that produced the original failure) to `tests/fixtures/wood_kim.pdf`. If the file is not available, use the `_make_pdf` helper to build an equivalent synthetic one inside the test.

- [ ] **Step 2: Write the acceptance test**

Append to `tests/test_extraction.py`:

```python
def test_wood_kim_full_extraction():
    fixture = Path(__file__).parent / "fixtures" / "wood_kim.pdf"
    if not fixture.exists():
        import pytest
        pytest.skip("wood_kim.pdf fixture not committed")
    pdf_bytes = fixture.read_bytes()
    from src.pdf_loader import pdf_to_text
    text = pdf_to_text(pdf_bytes)
    got = extract_invoice_fields(text, pdf_bytes=pdf_bytes)
    assert got["invoice_date"] == "07/29/2011"
    assert got["issuer"] == "Wood-Kim"
    assert got["recipient"] == "Thompson PLC"
    # total: either "164,97" (EU comma preserved) or "164.97" (normalized)
    assert got["total"] in {"164,97", "164.97"}
```

Add Path import at top of test file if missing:

```python
from pathlib import Path
```

- [ ] **Step 3: Run test**

Run: `pytest tests/test_extraction.py::test_wood_kim_full_extraction -v`
Expected: PASS (or SKIP if no fixture).

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/wood_kim.pdf tests/test_extraction.py
git commit -m "test: Wood-Kim invoice end-to-end acceptance test"
```

---

## Task 18: Run SROIE regression check

No code changes. This is a verification step.

- [ ] **Step 1: Run notebook 03 end-to-end**

Open `notebooks/03_information_extraction.ipynb` in Jupyter and run all cells. The final cells report per-field SROIE scores.

- [ ] **Step 2: Compare to baseline**

Baseline (per spec):
- `invoice_date` ≥ 94.9% exact
- `issuer` ≥ 60.5% exact, ≥ 89.1% relaxed
- `total` ≥ 63.4% exact, ≥ 70.0% relaxed

- [ ] **Step 3: Decide**

If any metric drops > 1 pp below baseline: file it as a follow-up task (likely the new MM/DD/YYYY pattern picking up DD/MM dates from SROIE). Fix before merge.

If within tolerance: done. Commit any non-code changes (e.g., updated notebook output) as:

```bash
git add notebooks/03_information_extraction.ipynb
git commit -m "chore: refresh notebook 03 outputs after extraction changes"
```

---

## Task 19: Run qualitative eval and spot-check worst failures

- [ ] **Step 1: Make sure the Kaggle image dataset is downloaded**

Run the new cell in `notebooks/01_data_collection.ipynb` (section 5) if not already done.

- [ ] **Step 2: Run the eval**

```bash
python scripts/eval_invoices.py
```

Expected output: a per-field non-null rate table and a list of the 10 worst files.

- [ ] **Step 3: Open the 10 worst files**

For each of the worst 10 PDF/images, open the image and compare to the extracted fields (in `data/processed/invoice_extraction_eval.csv`). Look for patterns:
- A label/value pair the current rules don't recognize.
- A currency or date format the regex doesn't handle.
- A layout the label-right/label-below heuristic doesn't cover.

- [ ] **Step 4: If systematic failures found — file follow-up tasks**

Do not add them to this plan. Capture them as a new plan / spec if the fixes are non-trivial, or as a small patch commit if each is a one-line regex addition.

- [ ] **Step 5: Commit the eval output**

```bash
git add data/processed/invoice_extraction_eval.csv
git commit -m "chore: commit initial qualitative eval output"
```

---

## Task 20: Manual demo verification

- [ ] **Step 1: Start the service**

```bash
uvicorn src.service:app --reload --port 8000
```

- [ ] **Step 2: Upload the Wood-Kim PDF via Swagger**

Open `http://localhost:8000/docs`, expand `POST /extract`, click **Try it out**, upload the Wood-Kim PDF.

- [ ] **Step 3: Verify the response**

Expected:

```json
{
  "filename": "...",
  "fields": {
    "invoice_number": null,
    "invoice_date":   "07/29/2011",
    "due_date":       null,
    "issuer":         "Wood-Kim",
    "recipient":      "Thompson PLC",
    "total":          "164,97"
  }
}
```

(invoice_number is null because this PDF has no invoice-number field; due_date is null because it has no due date. Both are correct per the source document.)

- [ ] **Step 4: Upload 2–3 other invoices from the Kaggle image set**

Pick 2–3 images, convert them to PDF (simple: drag into Preview.app → File → Export as PDF), upload via `/extract`. Spot-check extraction correctness.

- [ ] **Step 5: If all pass — done**

Merge the branch via your usual flow. If the user flow has any surprising failures, file a follow-up.

---

## Notes for the executing engineer

- **Keep commits small.** One commit per task step 5 (not per task group). Every step is atomic — test fails → test passes → commit.
- **Do not modify the CRF or its training.** If you think CRF needs changes, stop and ask.
- **If a test fixture file is missing** (e.g., `wood_kim.pdf`), skip with `pytest.skip(...)` rather than failing — the fixture is user-provided.
- **Do not use `git add -A` or `git add .`.** Add files by explicit path (the instructions above always do).
- **If any task reveals a spec ambiguity**, pause and ask. Don't invent behavior.
