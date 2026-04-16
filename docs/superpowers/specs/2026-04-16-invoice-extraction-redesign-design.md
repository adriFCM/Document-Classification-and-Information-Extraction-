# Invoice Information Extraction — Redesign

**Date:** 2026-04-16
**Branch:** `feat/hybrid-crf-extraction`
**Status:** Design approved; ready for implementation plan

## Problem

The current rule-based invoice extractor (`src/information_extraction.py`) fails on modern templated PDFs. Reproducer: the Wood-Kim sample invoice returns:

```json
{
  "invoice_number": null,
  "invoice_date":   null,
  "due_date":       null,
  "issuer":         "Date of issue:",
  "recipient":      null,
  "total":          "00"
}
```

Root causes for this invoice:

1. **US date format** (`07/29/2011`) is not in `_DATE_PATTERNS` (day-first only).
2. **European decimal comma** (`$ 149,97`) is not parsed by `_AMOUNT_RE` (only thousand-separator commas).
3. **`Seller:` / `Client:` labels** are missing from `_RECIPIENT_MARKER`, so the issuer heuristic has no stop-boundary and picks `"Date of issue:"` as the first non-empty line.
4. **Label-only lines** — when a label sits alone on a text line (columnar layouts flattened by pdfplumber), the current regex finds no value on that line and gives up.
5. **Summary-table totals** — three amounts appear on the `Total` row (`Net / VAT / Gross`); the current logic takes the last amount found on the line, which mis-parses under European decimals.

The user reports that every new invoice layout surfaces new failure modes — a structural generalization problem, not a one-off bug.

## Goals

- Keep SROIE 2019 scores within 1 pp of current baseline (`invoice_date` 95.9% exact / `issuer` 61.5% exact / 90.1% relaxed / `total` 64.4% exact / 71.0% relaxed).
- Materially improve extraction on modern templated PDF invoices (Wood-Kim-style).
- Avoid regressing existing rule patterns that already work.
- Stay within the assignment constraint: no generative AI. Classical ML (CRF) is allowed.

## Non-goals

- CRF retraining. Current model stays as-is; revisit after a held-out test set exists.
- Refactoring `information_extraction.py` into smaller modules beyond the minimal `date_utils.py` extraction needed for reuse.
- New API endpoints. Existing `/classify` and `/extract` continue to work with the same request shapes.
- LayoutLM or transformer-based document-understanding models.

## Architecture

Two extraction paths, chosen at the entry point based on whether PDF bytes are available:

```
extract_invoice_fields(text, pdf_bytes=None)
  ├─ pdf_bytes provided  → run path B (layout), then path A (text), merge
  └─ pdf_bytes is None   → run path A only (unchanged contract)

Path B output merges field-by-field with path A:
  out[field] = layout[field] if layout[field] is not None else text[field]
```

- **Path A (text-based):** the existing regex + CRF stack in `src/information_extraction.py`, patched to fix the five failure modes above. Used for OCR'd scans, SROIE eval, and as a per-field fallback when path B returns None.
- **Path B (layout-aware):** new module `src/layout_extractor.py` that takes PDF bytes, uses `pdfplumber` word bounding boxes and table detection to locate label→value pairs spatially. Handles two-column layouts and summary tables natively.
- **CRF** is unchanged. Called inside path A for `issuer`/`recipient`. Model file `models/crf_invoice.pkl` stays in place.
- **Merge policy:** layout wins per field (higher precision when it fires), text fills gaps (higher recall). False positives from layout are worse than None, so layout must return None rather than guess.

## Components

### Path A fixes — `src/information_extraction.py`

**A.1 US date format (`MM/DD/YYYY`)**
Add a month-first alternative to `_DATE_PATTERNS`, validating month ≤ 12 and day ≤ 31 in the correct positions. Day-first patterns remain first in the alternation, so ambiguous dates (`03/04/2024`) parse as day-first — preserving SROIE behavior. New pattern matches `07/29/2011`.

**A.2 European decimal comma for amounts**
Extend `_AMOUNT_RE` with a branch for `\d+,\d{2}(?!\d)` so `149,97` captures as a single amount. Add a canonicalization step in `_as_float` that detects the comma-as-decimal case (comma followed by exactly 2 digits at end-of-number) and normalizes to dot-decimal for magnitude comparison in `extract_total`.

**A.3 Summary-table total detection**
In `extract_total`, when a `_TOTAL_KEYWORDS`-matching line contains ≥ 2 amounts, take the **rightmost** amount (gross-total convention for Net/VAT/Gross columns). Existing sub-total exclusion and `round` priority are preserved.

**A.4 `Seller:` / `Client:` labels**
- Add `client` to `_RECIPIENT_MARKER` and `_RECIPIENT_LABEL_RE`.
- Add `seller` as a new issuer-block marker: when `Seller:` appears, the issuer is the first non-empty line below it (within 3 lines), not the document's first non-empty line.
- `Seller:` also acts as a stop-boundary for the current head-lines heuristic, so template cover titles above `Seller:` are still skipped.

**A.5 Label-only-line fallthrough**
In `extract_invoice_date`, `extract_invoice_number`, `extract_due_date`: when a label line matches but contains no value, scan the next 1–2 non-empty lines for the value before falling through. `extract_invoice_number` already does this partially; extend to the date fields.

**A.6 Shared date parser — `src/date_utils.py`**
Extract `_DATE_PATTERNS`, `_DATE_RE`, and `_find_dates` into a new small module so `layout_extractor.py` can reuse them without circular imports. `information_extraction.py` re-imports from there. Pure mechanical extraction — no behavior change.

### Path B — `src/layout_extractor.py` (new)

Public API:

```python
def extract(pdf_bytes: bytes) -> dict[str, Optional[str]]:
    """Return {invoice_number, invoice_date, due_date, issuer, recipient, total}.
    Any field we cannot confidently identify is returned as None."""
```

**B.1 Word extraction**
Use `pdfplumber.open(BytesIO(pdf_bytes))`. For each page, call `page.extract_words(extra_attrs=['fontname', 'size'])` to get `[{text, x0, x1, top, bottom, ...}]`. If a page yields zero words (scanned image), treat the whole doc as non-layout-extractable and return all-None.

**B.2 Label → value primitive**

```python
def find_value_for_label(label_patterns, words, page) -> Optional[str]
```

- Scan `words` for contiguous spans whose concatenated lowercased text matches any `label_patterns` regex. Multi-word labels (`date of issue`) must match across adjacent words in the same y-band.
- For the matched label span, compute `x_end` (right edge) and `y_band = [top, bottom]`.
- **Right-of-label candidates:** words with `x0 ≥ x_end - tolerance` and y-band overlap ≥ 50%. Concatenate in x-order.
- **Below-label candidates:** words with `top > bottom` and `top < bottom + 3 * line_height`, with `abs(x0 - label_x0) < 50pt` (same column). Group by line and return the first non-empty line.
- Return first non-empty candidate that passes the field's value validator.

**B.3 Field-specific logic**

| Field | Label patterns | Validator |
|---|---|---|
| `invoice_number` | `invoice #/no/number`, `bill no`, `inv` | ≥1 digit, length 3–20 |
| `invoice_date` | `date`, `invoice date`, `date of issue`, `issue date` | parses via `date_utils._DATE_RE` |
| `due_date` | `due date`, `payment due`, `pay by` | parses via `date_utils._DATE_RE` |
| `issuer` | `seller`, `from`, `vendor`, `bill from` | non-empty text block (below-label only) |
| `recipient` | `client`, `bill to`, `billed to`, `sold to`, `ship to`, `invoice to`, `issued to`, `customer` | same |
| `total` | (see B.4) | amount-shaped |

**B.4 Total — two strategies, first hit wins**
1. **Table-based:** `page.extract_tables()`. For each table, find a row whose first cell matches `total` (not `sub total`). Return the rightmost amount-shaped cell in that row (handles Net/VAT/Gross columns).
2. **Label-based:** `find_value_for_label(["total", "amount due", "balance due", "grand total"])`, then extract the rightmost amount-shaped token from the captured value string.

**B.5 Issuer / recipient block capture**
When the label is `Seller:` or `Client:`, the value is a multi-line text block below the label. Extract lines below until one of:
- blank line (no words within 1.5 * line_height below the previous line),
- another known label fires (e.g., `Tax Id:`, `IBAN:`, `Items`),
- 5 lines captured (safety cap).

Return the first line of the block (the name), dropping address/tax-id lines.

**B.6 Multi-page ordering**
Iterate pages in natural order for most fields. For `total`, prefer the last page first (summary tables live there). First non-None value per field wins.

### Merge in `information_extraction.py`

```python
def extract_invoice_fields(text: str, pdf_bytes: bytes | None = None) -> dict:
    text_result = _extract_text_based(text)
    if pdf_bytes is None:
        return text_result
    layout_result = layout_extractor.extract(pdf_bytes)
    return _merge(layout_result, text_result)

def _merge(layout, text):
    return {k: (layout[k] if layout.get(k) is not None else text.get(k))
            for k in ("invoice_number", "invoice_date", "due_date",
                      "issuer", "recipient", "total")}
```

### `src/service.py`

Both `/classify` (when label is `invoice`) and `/extract` already have `pdf_bytes` in scope. Pass through:

```python
extract_invoice_fields(raw_text, pdf_bytes=pdf_bytes)
```

No other changes to the service.

### Notebook 03

SROIE eval passes pre-extracted text, no PDF bytes → path B is never invoked on SROIE. SROIE results can only move because of path A changes (A.1–A.5). Re-run notebook 03 after implementation to confirm.

## Testing

### Unit tests — `tests/test_extraction.py` (new)

Fast tests, no model load, no OCR:
- `test_us_date_format` — `"Date of issue: 07/29/2011"` → `invoice_date == "07/29/2011"`.
- `test_european_decimal` — `"Total: $ 164,97"` → `total == "164,97"`.
- `test_seller_client_labels` — synthetic text with `Seller:\nWood-Kim\n` / `Client:\nThompson PLC` → correct issuer/recipient.
- `test_label_only_line_fallthrough` — label line with no value, value on next non-empty line.
- `test_layout_wood_kim_pdf` — calls `extract_invoice_fields(text, pdf_bytes=...)` on `tests/fixtures/wood_kim.pdf` (anonymized copy of the failing invoice). Asserts all six fields match expected.

Run: `pytest tests/ -v`.

### SROIE regression

Re-run `notebooks/03_information_extraction.ipynb`. Accept if every metric stays within 1 pp of baseline:
- `invoice_date` ≥ 94.9% exact
- `issuer` ≥ 60.5% exact, ≥ 89.1% relaxed
- `total` ≥ 63.4% exact, ≥ 70.0% relaxed

### Qualitative eval on Kaggle image set

New script: `scripts/eval_invoices.py`.
1. Iterate `data/raw/invoices/high_quality_images/`, OCR each image.
2. Run `extract_invoice_fields(text)` (text-only path — layout doesn't apply to raw image OCR).
3. Write `data/processed/invoice_extraction_eval.csv` with per-file results and `non_null_count`.
4. Print per-field non-null rate. Spot-check the 10 files with lowest `non_null_count` to find systematic failure patterns.

Supporting change: extend `src/pdf_loader.py` (or add `src/image_loader.py`) to accept `.jpg`/`.png` paths by going straight to the pytesseract path. Small change — existing OCR fallback already uses pytesseract.

### Manual demo check

Upload Wood-Kim PDF through Swagger UI at `/extract`. Expected: all six fields populated correctly.

## Risks & mitigations

- **Risk:** Layout extraction over-confident on ambiguous labels (e.g., `Date:` appearing in a footer). **Mitigation:** validators reject non-parsing dates/amounts; first-page-first ordering prefers header labels.
- **Risk:** `pdfplumber.extract_tables()` hallucinates tables on non-tabular layouts. **Mitigation:** table-based total only triggers when a row has a cell matching `total` and ≥ 1 amount-shaped cell; else fall through to label-based.
- **Risk:** Adding MM/DD/YYYY regex causes SROIE dates to re-parse incorrectly. **Mitigation:** DD/MM/YYYY remains first in alternation order, so SROIE (Malaysian, day-first) stays on its existing branch. Regression check via notebook 03 confirms.
- **Risk:** Text-based `Seller:` heuristic picks wrong line when the block has a logo/image. **Mitigation:** pdfplumber skips non-text elements; we take the first non-empty text line below `Seller:`. Layout path handles this more reliably via column-based y-below-label search.

## Out of scope for this spec

- Retraining CRF with layout features or larger synthetic dataset — deferred until a labeled held-out test set exists.
- Switching to transformer-based document understanding models.
- Classifier changes.
- Frontend changes (Swagger UI and existing frontend continue to work unchanged).
