# Notebook 03 — Information Extraction (Invoices)

Companion write-up for `03_information_extraction.ipynb`. Describes what the
notebook does, the datasets and ground truth used, the rule-based extractors,
the challenges we hit along the way, and the iterative fixes that got us to
the final scores.

---

## 1. Goal

Extract six structured fields from documents the classifier has labeled as
`invoice`:

| Field            | Source of ground truth        |
| ---------------- | ----------------------------- |
| `invoice_number` | (no SROIE labels — eyeballed) |
| `invoice_date`   | SROIE `date`                  |
| `due_date`       | (no SROIE labels — eyeballed) |
| `issuer`         | SROIE `company`               |
| `recipient`      | (no SROIE labels — eyeballed) |
| `total`          | SROIE `total`                 |

The assignment disallows generative AI, so the extractor is pure regex + small
heuristics, with no ML model.

## 2. Data choices and why we did not reuse `dataset.csv`

Notebook 02 builds `data/labeled/dataset.csv` for the classifier. That
preprocessing step lowercases text, strips numbers, and removes dates — all
fine for bag-of-words classification, but fatal for information extraction,
which needs the original characters. So notebook 03 bypasses the cleaned CSV
and reads **raw SROIE box files** directly from
`data/raw/invoices/SROIE2019/{train,test}/box/*.txt`.

Each box file is `x1,y1,x2,y2,x3,y3,x4,y4,text`. We keep the original casing
and join tokens in file order, which approximately follows reading order, then
keep the filename stem as the join key with the ground truth.

### Why SROIE isn't enough on its own

SROIE labels only four keys per document (`company`, `date`, `address`,
`total`). Our target schema has six fields. **Three fields
(`invoice_number`, `due_date`, `recipient`) cannot be scored on SROIE at
all** — we validate them qualitatively on a small set of real invoice PDFs
collected for the live demo. This is the "other issue" that made us stop
treating SROIE as our single source of truth and plan the PDF demo set.

## 3. Notebook structure

| Cell | What it does                                                                    |
| ---- | ------------------------------------------------------------------------------- |
| 0    | Header + target fields + input rationale (markdown)                             |
| 2    | Imports, `chdir` to repo root, paths                                            |
| 4    | `load_sroie_box`, `load_all_sroie` — builds `df_inv` (973 rows)                 |
| 6    | `load_sroie_ground_truth` — builds `df_gt` (973 rows) with renamed columns     |
| 8    | **All six extractors** + `InvoiceFields` dataclass + `extract_invoice_fields`   |
| 10   | Applies the extractor to every SROIE doc → `df_pred`                            |
| 12   | `evaluate(df_pred, df_gt)` — exact-match and relaxed (substring) per field     |
| 14   | `show_mistakes` — prints the first 5 mismatches per field for error analysis   |
| 16   | Placeholder for `src.pdf_loader.pdf_to_text` demo wiring                        |
| 17   | "Next steps" markdown                                                           |

## 4. The extractors (final state)

All six return `Optional[str]`. Public entry point is
`extract_invoice_fields(text) -> dict` which always returns all six keys.

### `invoice_date` / `due_date`

A union regex over the four common date shapes (`dd/mm/yyyy`, `yyyy-mm-dd`,
`dd Mon yyyy`, `Mon dd, yyyy`). We first scan only lines containing a `date`
or `due date` keyword, and only fall back to the first date anywhere in the
document if the keyword search fails. `extract_due_date` also understands the
relative `NET 30` / `NET 45` terms.

### `invoice_number`

A single regex tied to an explicit label: `invoice no/number/num/#`, `inv#`,
or `bill no`. See §5 for the false-match bug this had to fix.

### `total`

Easily the gnarliest. See §5.

### `issuer`

Header heuristic: among the first ~8 non-empty lines (stopping early at the
first Bill-To / Sold-To / Ship-To / Invoice-To marker), pick the **longest**
line containing a company suffix (`SDN BHD`, `BHD`, `LLC`, `LTD`, `INC`,
`CORP`, `CO`, `ENTERPRISE`, `TRADING`, `PTE`, `GMBH`, plus the OCR-corrupt
`BND` variant). Fall back to the first header line if no suffix is found.

### `recipient`

Regex on a `Bill To` / `Sold To` / `Customer` / `Recipient` prefix. SROIE
has no recipient labels so this is unscored.

## 5. Challenges and the fixes we shipped

The initial version (all six extractors written in one pass) ran end-to-end on
973 SROIE receipts. The first numbers were:

| Field        | Exact  | Relaxed |
| ------------ | ------ | ------- |
| issuer       | 61.9 % | 92.7 %  |
| invoice_date | 95.9 % | 96.0 %  |
| **total**    | **5.3 %** | **6.5 %** |

`invoice_date` was basically done on the first try. The other fields had three
distinct bugs that took several iterations to shake out:

### Bug 1 — `total` scored ~5 % because labels and amounts sit on different lines

A typical SROIE box file looks like this:

```
TOTAL:
9.00
ROUND D TOTAL (RM):
9.00
```

The first version of `extract_total` only scanned within a single line, so for
documents like this it found no amount at all.

**Fix:** when a `TOTAL`-keyword line has no amount on it, look ahead up to two
non-empty lines and accept the first amount found there.

**Result:** `total` jumped from **5.3 %** → **63.8 %** exact.

### Bug 2 — `invoice_number` captured `OICE-xxxxx`

The regex had `inv\s*[:#]?\s*` as one of its label alternatives. On input
`INVOICE-123`, the `inv` alternative matched the first three characters of
the word `INVOICE`, and the value-capture group then started from the `o`,
producing garbage like `OICE-123` and `OICE`.

**Fix:** require a literal `#` or `:` *right after* the `inv` prefix
(`inv\s*[#:]\s*`). The `invoice\s*(no|number|num|#)` alternative still handles
the `INVOICE No: 123` case cleanly.

**Result:** `invoice_number` no longer emits `OICE` false positives. Values
are `None` on SROIE (no explicit `invoice no:` label on most receipts, which
is expected).

### Bug 3 — `total` was systematically **1–2 cents high**

Once bug 1 was fixed, a lot of the predictions looked like:

```
pred='60.31'   gt='60.30'
pred='80.91'   gt='80.90'
pred='33.92'   gt='33.90'
```

Every miss was off by a penny or two — clearly not random OCR noise. The
cause: SROIE receipts print several totals in sequence:

```
TOTAL               60.31       ← pre-rounding
ROUNDING ADJ        -0.01
ROUND D TOTAL       60.30       ← ground truth
```

Our aggregation was `max()` over all total-keyword lines, so it always picked
the pre-rounding figure.

**Attempted fix:** "prefer the last total-keyword amount, not the max."
That actually **regressed** the score (63.8 % → 51.3 %). On receipts without
a `ROUND D TOTAL` line, "last" picked up post-total junk (cash, change,
tax-inclusive recapitulations), turning near-misses into far-misses like
`pred='11.00' gt='26.60'` and `pred='0.85' gt='15.00'`.

**Final fix:** two-tier scoring of total candidates:

1. If any total-keyword line contains the word `round`, return the amount
   from that line (the ground-truth post-rounding figure).
2. Otherwise fall back to `max()` — the grand total is typically the largest
   amount on a total line.

**Result:** `total` landed at **64.4 %** exact / **71.0 %** relaxed. Not
every SROIE receipt spells `round` in the label line, so rows 1 and 3 of the
preview still show `60.31` / `80.91` — but this is the least-bad trade-off
we found without hand-tuning for every store chain's layout.

### Bug 4 — `issuer` picks the cashier name instead of the company

First pass of `extract_issuer` returned the first header line containing a
company suffix. On receipts like this:

```
TAN WOON YANN                  ← line 1 (owner/cashier, no suffix)
BOOK TA .K(TAMAN DAYA) SDN BND  ← line 2 (real company, with suffix)
789417-W
```

...the function short-circuited on line 1 (no suffix match, so it returned
`head_lines[0]`). Also, SROIE's OCR occasionally mangles `BHD` into `BND`,
which our suffix regex didn't accept.

**Fix:** (a) accept `B[HN]D` / `BND` in the suffix regex and broaden the
suffix list, and (b) instead of first-wins, pick the **longest** header line
containing a suffix. This fixes the `TAN WOON YANN` class of misses.

**Result:** `issuer` exact match is essentially unchanged at 61.5 %, but
this is misleading — the 38.5 % that "fail" are mostly OCR-level differences
(`BND` vs `BHD`, spacing, punctuation) that relaxed matching catches. The
relaxed score is **90.1 %**, which is where the real signal is.

### Bug 5 — `evaluate` threw a pandas FutureWarning

The relaxed-match column was being built by `and`ing a `Series` with a raw
list comprehension, which pandas flagged as deprecated.

**Fix:** wrap the list comp in `pd.Series(..., index=df.index)` before the
logical `&`.

### Bug 6 — real-PDF smoke test exposed three more failures

After the SROIE numbers stabilized, we lifted the extractors into `src/` and
ran a real invoice PDF (Madrid → Oslo, issuer = individual, total in NOK)
through the FastAPI microservice. First response:

```
issuer:    'INVOICE'
recipient: None
total:     '0'
```

Three independent problems surfaced at once:

**(a) `_AMOUNT_RE` chopped whole-number totals.** The pattern was
`\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?` — the `{1,3}` head is greedy but capped
at three digits, and the group repeat is `*` (zero or more). On input
`4000 NOK`, `findall` matched `400` first (three digits, zero groups, no
decimal), then restarted and matched the leftover `0`. Last-in-line picked
`0`.
**Fix:** require at least one grouping block in the grouped alternative
(`+` instead of `*`) and add a separate `\d+(?:\.\d{2})?` alternative for
plain whole or decimal amounts. `4000` now matches as one token. We also
added `NOK`, `SEK`, `DKK` to the currency list (optional) on both sides of
the amount.

**(b) `extract_recipient` needed a label on the same line as the name.** The
PDF has `Bill To` on one line and `Fortifai` on the next. Our regex
`(?:bill\s*to|...)\s*[:\-]?\s*(.+)` required the payload on the same line.
**Fix:** if the capture group is empty, fall through to the next non-empty
line (same pattern we used for `extract_total`).

**(c) `extract_issuer` fell back to the word `INVOICE`.** The PDF's
letterhead is the big styled word "INVOICE" on line 1. No suffix match, so
the first-line fallback returned the document type instead of the actual
issuer.
**Fix:** skip standalone document-type words (`INVOICE`, `RECEIPT`, `BILL`,
`STATEMENT`, `QUOTATION`, `QUOTE`, `TAX INVOICE`) when building the header
line pool. Now the fallback lands on `Ryan Muenker` (the actual issuer, an
individual with no company suffix) — still best-effort, but correct on this
kind of personal-services invoice.

**Result on the real PDF:**

```
invoice_number: None           (no label on the document — correct)
invoice_date:   04/03/2026     ✓
due_date:       31/03/2026     ✓
issuer:         Ryan Muenker   ✓
recipient:      Fortifai       ✓
total:          4000           ✓
```

5/6 populated, 1/6 correctly `None` (the invoice has no `Invoice No:` label
anywhere on it, so `None` is the honest answer — the extractor is
label-anchored by design to avoid the `OICE-` false-positive class from Bug 2).

## 6. Final scores

973 SROIE receipts, evaluated against the 3 fields SROIE labels:

| Field          | Exact  | Relaxed |
| -------------- | ------ | ------- |
| `issuer`       | 61.5 % | 90.1 %  |
| `invoice_date` | 95.9 % | 96.0 %  |
| `total`        | 64.4 % | 71.0 %  |

The remaining `issuer` and `total` failures are dominated by OCR artefacts
and unlabeled store-chain-specific layouts. We chose to stop tuning SROIE
here — the demo set (real invoice PDFs parsed via `pdfplumber` with
`pytesseract` fallback) has cleaner headers and explicit labels, so these
numbers are a pessimistic floor, not a ceiling.

## 7. What couldn't be scored on SROIE

`invoice_number`, `due_date`, and `recipient` have no SROIE ground truth.
They are validated qualitatively against the real invoice PDFs collected for
the live demo:

- `invoice_number` — the fixed label regex catches `INVOICE No: …`,
  `INV#: …`, `BILL NO: …` reliably on structured PDFs.
- `due_date` — label-anchored, with a `NET N` fallback for relative terms.
- `recipient` — `Bill To` / `Sold To` label extraction.

## 8. Handoff to the microservice

At the end of the notebook, `extract_invoice_fields` is the single public
entry point. The `src/` package lifts it out of the notebook so it can be
served from a FastAPI microservice:

```
src/
├── information_extraction.py   # extract_invoice_fields(text) -> dict
├── pdf_loader.py               # pdfplumber → pytesseract fallback
└── service.py                  # FastAPI: GET /health, POST /extract
```

Run locally:

```bash
pip install fastapi uvicorn python-multipart
brew install tesseract     # macOS; required by pytesseract
uvicorn src.service:app --reload --port 8000
```

`POST /extract` accepts a multipart PDF upload and returns:

```json
{
  "filename": "invoice.pdf",
  "fields": {
    "invoice_number": "INV-2024-001",
    "invoice_date":   "15/03/2024",
    "due_date":       "15/04/2024",
    "issuer":         "ACME TRADING SDN BHD",
    "recipient":      "Foo Customer Ltd",
    "total":          "106.00"
  }
}
```

The classification step is upstream — the service trusts that whatever is
POSTed is already an invoice. Wiring in the classifier is a one-liner in
`service.extract()` once that model is packaged.
