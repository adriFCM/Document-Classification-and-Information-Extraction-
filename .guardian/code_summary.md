# Code Summary

## src/service.py
**Lines:** 242
**Purpose:** FastAPI microservice exposing document classification and invoice extraction endpoints

### Imports
```python
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
import joblib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from src.information_extraction import extract_invoice_fields
from src.pdf_loader import pdf_to_text, image_to_text
from src.preprocessing import clean_for_classifier, _INV_SIGNALS
```

### Exports
- `app` (FastAPI instance)

### Constants
- `_ALLOWED_EXTENSIONS` (set): {'.pdf', '.jpg', '.jpeg', '.png'}
- `_MODELS_DIR` (Path): Path to models/ directory
- `_SELECTABLE_MODELS` (dict): Model metadata (name, description) for UI selection
- `_INV_OVERRIDE_THRESHOLD` (int): 4 — minimum invoice signals to force invoice label
- `API_URL` constant is in frontend, not this file

### Functions

#### `_model_type_for(model_dir: Path) -> str`
- **Purpose:** Reads model_type.txt from a model directory to determine if it's 'tfidf' or 'sbert'
- **Returns:** Model type string ('tfidf' if marker missing, else contents of model_type.txt)
- **Side effects:** File I/O

#### `_load_named_model(model_key: str) -> tuple`
- **Decorator:** `@lru_cache(maxsize=8)` — caches loaded models by key
- **Purpose:** Loads classifier + encoder (TF-IDF or SBERT) from models/<key>/ directory
- **Returns:** `(mtype: str, encoder: TfidfVectorizer|SentenceTransformer, clf: sklearn classifier)`
- **Raises:** `RuntimeError` if model files missing, `HTTPException(400)` if unknown key
- **Side effects:** Loads models into memory (heavy operation, only runs once per key due to cache)
- **Special logic:** If key='auto', falls back to legacy single-model files; for SBERT, lazy-imports sentence_transformers

#### `_count_invoice_signals(text: str) -> int`
- **Purpose:** Counts how many distinct invoice indicator patterns fire on raw text
- **Uses:** `_INV_SIGNALS` from preprocessing module (list of (token, regex) tuples)
- **Returns:** Count of matched patterns
- **Side effects:** None

#### `_predict_proba(clf, X: np.ndarray) -> np.ndarray`
- **Purpose:** Returns per-class probability array for any sklearn classifier
- **Logic:** Calls `clf.predict_proba(X)` if available; else softmax on `decision_function(X)` (for LinearSVC)
- **Returns:** 1D numpy array of probabilities (shape: n_classes)
- **Side effects:** None

#### `health() -> dict`
- **Route:** `GET /health`
- **Purpose:** Liveness probe
- **Returns:** `{"status": "ok"}`

#### `list_models() -> list[dict]`
- **Route:** `GET /models`
- **Purpose:** Lists available models for UI selection
- **Returns:** List of `{key, name, description, available}` dicts
- **Logic:** Checks if models/<key>/ directory exists for each model in _SELECTABLE_MODELS

#### `classify(file: UploadFile, model: str) -> dict` (async)
- **Route:** `POST /classify`
- **Parameters:** `file` (multipart upload), `model` (form field, default 'logistic_regression')
- **Purpose:** Classifies uploaded PDF/image, extracts invoice fields if label='invoice'
- **Returns:** `{filename, label, confidence, proba, model_used, inv_signals, invoice_fields?}`
- **Logic:**
  1. Validates file extension (must be in _ALLOWED_EXTENSIONS)
  2. Reads file bytes
  3. Extracts text via `pdf_to_text()` or `image_to_text()`
  4. Loads model via `_load_named_model(model)`
  5. Cleans text via `clean_for_classifier()` (for TF-IDF) or removes indicator tokens (for SBERT)
  6. Vectorizes and predicts
  7. Computes probabilities via `_predict_proba()`
  8. Counts invoice signals via `_count_invoice_signals()`
  9. Overrides label to 'invoice' if inv_signals >= 4 and label != 'invoice'
  10. If label='invoice', calls `extract_invoice_fields(raw_text, pdf_bytes)`
- **Side effects:** File I/O, model inference
- **Raises:** `HTTPException(400)` for invalid file type or empty upload, `HTTPException(422)` for read errors

#### `extract(file: UploadFile) -> dict` (async)
- **Route:** `POST /extract`
- **Purpose:** Extracts invoice fields without classification (backward compatibility endpoint)
- **Returns:** `{filename, fields}`
- **Logic:** Similar to classify but skips model loading, always calls `extract_invoice_fields()`

### State
- Models loaded at startup (via lru_cache on first request per model key)
- No runtime state persisted between requests

---

## src/pdf_loader.py
**Lines:** 187
**Purpose:** PDF/image text extraction with OCR fallback for scanned documents

### Imports
```python
from __future__ import annotations
import io
from pathlib import Path
from typing import Union
import pdfplumber
# Lazy imports: pypdfium2, pytesseract, PIL.Image
```

### Exports
- `PathLike` (type alias: Union[str, Path, bytes])
- `pdf_to_text(path_or_bytes) -> str`
- `image_to_text(path_or_bytes) -> str`
- `image_to_words(path_or_bytes, min_conf=30) -> list[dict]`
- `image_to_full_text(words) -> str`

### Constants
- `_MIN_TEXT_CHARS = 40` — threshold for triggering OCR fallback

### Functions

#### `_open(path_or_bytes: PathLike) -> pdfplumber.PDF`
- **Purpose:** Opens a PDF via pdfplumber (accepts file path or bytes)
- **Returns:** pdfplumber.PDF context manager
- **Side effects:** File I/O or BytesIO wrapping

#### `_extract_with_pdfplumber(path_or_bytes: PathLike) -> str`
- **Purpose:** Fast text extraction using pdfplumber (digital PDFs)
- **Returns:** Concatenated text from all pages (newline-separated)
- **Side effects:** Opens PDF, iterates pages

#### `_extract_with_ocr(path_or_bytes: PathLike) -> str`
- **Purpose:** OCR-based extraction for scanned PDFs
- **Returns:** Concatenated OCR text from all pages
- **Logic:** Rasterizes each page via pypdfium2 (scale=3.0), runs pytesseract.image_to_string()
- **Raises:** `RuntimeError` if pytesseract not installed or Tesseract binary missing
- **Side effects:** Heavy — rasterization + OCR (slow, CPU-intensive)

#### `pdf_to_text(path_or_bytes: PathLike) -> str`
- **Purpose:** Main entry point — tries pdfplumber first, falls back to OCR if yield is low
- **Returns:** Extracted text string
- **Logic:** If pdfplumber result has <40 non-whitespace chars, assumes scan and calls `_extract_with_ocr()`
- **Side effects:** File I/O, possibly OCR

#### `image_to_text(path_or_bytes: PathLike) -> str`
- **Purpose:** OCR on a .jpg/.png image
- **Returns:** Extracted text
- **Logic:** Loads image via PIL.Image.open(), runs pytesseract.image_to_string()
- **Side effects:** Image I/O, OCR

#### `image_to_words(path_or_bytes: PathLike, min_conf=30) -> list[dict]`
- **Purpose:** OCR an image and return word bounding boxes (compatible with pdfplumber format)
- **Returns:** List of dicts with keys: `text, x0, x1, top, bottom`
- **Logic:** Runs pytesseract.image_to_data(), filters out low-confidence words (conf < min_conf), converts bbox format
- **Side effects:** Image I/O, OCR
- **Used by:** Invoice extraction spatial logic

#### `image_to_full_text(words: list[dict]) -> str`
- **Purpose:** Reconstructs reading-order text from word bounding boxes
- **Returns:** Newline-separated lines, each line has words sorted by x0
- **Logic:** Groups words into lines by y-band (tolerance = avg_height/2), sorts each line by x0
- **Side effects:** None (pure transformation)

---

## src/preprocessing.py
**Lines:** 123
**Purpose:** Classifier-side text preprocessing (mirrors notebook 03 pipeline)

### Imports
```python
from __future__ import annotations
import re
from typing import Union
```

### Exports
- `clean_for_classifier(text: str) -> str`
- `process_pdf(path_or_bytes) -> dict`
- `_INV_SIGNALS` (list of invoice indicator patterns)

### Constants
- `_MAX_WORDS = 500` — truncation limit
- `_PAGE_NUM_RE` (regex): Matches isolated page numbers (e.g., "\\n123\\n")
- `_HYPHEN_BREAK_RE` (regex): Matches PDF hyphenated line breaks ("con-\\ntract")
- `_INV_SIGNALS` (list of tuples): 10 invoice indicator patterns, each `(token_name, regex)`:
  - `__inv_number__`, `__inv_header__`, `__bill_to__`, `__amount_due__`, `__grand_total__`, `__due_date__`, `__remit__`, `__po_number__`, `__vat__`, `__net_terms__`

### Functions

#### `_invoice_indicator_tokens(raw_text: str) -> str`
- **Purpose:** Returns space-separated indicator tokens for matched patterns (used for TF-IDF feature boost)
- **Returns:** String like `"__inv_number__ __bill_to__ __grand_total__"`
- **Side effects:** None

#### `clean_for_classifier(text: str) -> str`
- **Purpose:** Preprocessing pipeline matching notebook 03 (used by classifier)
- **Steps:**
  1. Rejoin hyphenated line breaks ("docu-\\nment" → "document")
  2. Strip isolated page-number lines
  3. Lowercase
  4. Collapse whitespace to single spaces
  5. Truncate to first 500 words
  6. Prepend invoice indicator tokens (if any matched)
- **Returns:** Cleaned text string
- **Side effects:** None
- **Note:** Stopword removal is NOT done here — handled by TfidfVectorizer(stop_words="english") during vectorization

#### `process_pdf(path_or_bytes: Union[str, bytes]) -> dict`
- **Purpose:** Full demo-time entry point — PDF in, both text versions out
- **Returns:** `{raw_text: str, classifier_text: str}`
- **Logic:** Calls `pdf_to_text()`, then `clean_for_classifier()`
- **Side effects:** File I/O, OCR (via pdf_to_text)
- **Usage:** Convenience function for demos (not used by service.py, which calls functions separately)

---

## src/information_extraction.py
**Lines:** 667 (truncated in read output at 20000 chars)
**Purpose:** Invoice field extraction using spatial anchoring on word bounding boxes + regex fallback

### Imports
```python
from __future__ import annotations
import io
import re
from dataclasses import asdict, dataclass
from typing import Optional, Sequence
import pdfplumber
```

### Exports
- `extract_invoice_fields(text, pdf_bytes=None, image_bytes=None, words=None) -> dict`
- `InvoiceFields` (dataclass)

### Dataclass

#### `InvoiceFields`
- **Fields:** `invoice_number: Optional[str]`, `invoice_date: Optional[str]`, `due_date: Optional[str]`, `issuer: Optional[str]`, `recipient: Optional[str]`, `total: Optional[str]`
- **Purpose:** Structured container for extraction results

### Constants (Regex Primitives)
- `_MONTHS` (regex): Month names pattern
- `_DATE_PATTERNS` (list of regex): 6 date format patterns (DD/MM/YYYY, MM/DD/YYYY, ISO, etc.)
- `DATE_RE` (compiled regex): Union of all date patterns
- `_YEAR_ONLY_RE` (regex): Matches 4-digit years (1900–2099) — used to filter false positives
- `_AMOUNT_RE` (regex): Monetary amount patterns (handles EU/US number formatting, currency symbols)
- `_INV_NUMBER_VALUE_RE` (regex): Invoice number format (alphanumeric, 3-20 chars, allows dashes/slashes)

### Constants (Field Labels)
- `_LABELS` (dict): Maps field names to ordered lists of label patterns (specific → generic)
  - `invoice_number`: ["invoice\\s*(?:no|num|number|#)", "inv\\s*(?:no|num|number|#)?", "ref\\s*(?:no|#)?", ...]
  - `invoice_date`: ["date\\s*of\\s*issue", "invoice\\s*date", "^date$", ...]
  - `due_date`: ["due\\s*date", "payment\\s*due(?:\\s*date)?", ...]
- `_ISSUER_LABELS` (list): Seller-related labels (["seller", "from", "vendor", ...])
- `_RECIPIENT_LABELS` (list): Buyer-related labels (["bill(?:ed)?\\s*to", "sold\\s*to", ...])
- `_TOTAL_LABELS` (list): Total amount labels (["grand\\s*total", "total\\s*due", ...])
- `_STOP_BLOCK_RE` (regex): Signals end of issuer/recipient block (tax id, phone, email, etc.)

### Functions (Word-Box Utilities)

#### `_y_overlap(a: dict, b: dict) -> float`
- **Purpose:** Computes vertical overlap ratio between two word bounding boxes
- **Returns:** Overlap fraction (0.0 to 1.0)

#### `_line_height(words: list[dict]) -> float`
- **Purpose:** Average line height from word boxes
- **Returns:** Mean (bottom - top) across all words

#### `_group_rows(words: list[dict], tol: float) -> list[list[dict]]`
- **Purpose:** Clusters words into visual rows by y-coordinate
- **Returns:** List of rows, each row is a list of words sorted by x0

#### `_find_label_span(label_re: re.Pattern, words: list[dict]) -> Optional[tuple[int, int]]`
- **Purpose:** Locates a label (possibly multi-word like "Invoice Number:") in word list
- **Returns:** (start_index, end_index_exclusive) or None
- **Scoring:** Prefers labels ending with ':' and at row start (column-header position)

#### `_words_right_of(words, start, end, y_ref, x_end) -> list[dict]`
- **Purpose:** Returns words on same visual row as y_ref, to the right of x_end
- **Used by:** Inline value extraction (e.g., "Invoice Number: INV-123")

#### `_rows_below(words, start, end, y_ref, n_rows, lh) -> list[list[dict]]`
- **Purpose:** Returns up to n_rows visual rows immediately below y_ref
- **Used by:** Column-aligned extraction (e.g., "Invoice Number" label, value on next row)

### Functions (Value Pickers)

#### `_pick_inline(compiled_labels, words) -> Optional[tuple[str, dict]]`
- **Purpose:** Finds label, returns (value_text, label_word_ref) from same-row right-of
- **Returns:** Tuple of (extracted value, label word) or None

#### `_label_positions(compiled_labels, words) -> list[tuple[int, int, dict]]`
- **Purpose:** Returns every (start, end, label_ref) for labels matched in words

#### `_pick_value(compiled_labels, words, validate, col_tol=None) -> Optional[str]`
- **Purpose:** General picker — tries inline-right, then column-aligned below, then any below
- **Validation:** Applies `validate(text)` to each candidate until one passes
- **Returns:** First validated value or None

#### `_pick_block_below(compiled_labels, words, col_tol=None) -> Optional[str]`
- **Purpose:** For block labels (Seller:/Client:), returns first non-stop row below aligned with label x-column
- **Returns:** Cleaned block value or None

#### `_clean_block_value(text: str) -> Optional[str]`
- **Purpose:** Cleans issuer/recipient text (strips trailing labels, rejects mostly-digit lines)
- **Returns:** Cleaned string or None

### Functions (Per-Field Validators)

#### `_validate_date(s: Optional[str]) -> Optional[str]`
- **Purpose:** Validates and extracts first date match from string
- **Returns:** Matched date string or None

#### `_validate_invoice_number(s: Optional[str]) -> Optional[str]`
- **Purpose:** Validates invoice number format (alphanumeric, 3-20 chars, filters out years)
- **Returns:** Validated number or None

#### `_is_monetary(tok: str) -> bool`
- **Purpose:** Checks if token looks like a monetary amount
- **Logic:** Has currency symbol, OR has decimal separator with 2-digit cents, OR 3+ digits

#### `_amount_as_float(tok: str) -> float`
- **Purpose:** Parses monetary token to float (handles EU/US formatting)
- **Returns:** Float value or -1.0 on parse error

#### `_strip_currency(s: str) -> str`
- **Purpose:** Removes currency prefix from amount string

### Functions (Layout-Path Extractors)

#### `_compile(patterns: list[str]) -> list[re.Pattern]`
- **Purpose:** Compiles list of regex strings to Pattern objects

#### `_extract_invoice_number(words: list[dict]) -> Optional[str]`
- **Purpose:** Extracts invoice number from word boxes
- **Logic:** Calls `_pick_value()` with `_validate_invoice_number()`, fallback to bare "#ABC-123" token search
- **Returns:** Invoice number string or None

#### `_extract_date(words: list[dict], key: str) -> Optional[str]`
- **Purpose:** Extracts invoice_date or due_date
- **Logic:** Calls `_pick_value()` with `_validate_date()`; for invoice_date, fallback to first date NOT near "due date" labels
- **Returns:** Date string or None

#### `_extract_issuer(words: list[dict]) -> Optional[str]`
- **Purpose:** Extracts issuer (seller) name
- **Logic:** Tries `_pick_block_below()`, then `_pick_value()`, fallback to `_header_company_guess()`
- **Returns:** Issuer name or None

#### `_extract_recipient(words: list[dict]) -> Optional[str]`
- **Purpose:** Extracts recipient (client) name
- **Logic:** Tries `_pick_value()` (inline), then `_pick_block_below()`
- **Returns:** Recipient name or None

#### `_header_company_guess(words: list[dict]) -> Optional[str]`
- **Purpose:** Fallback issuer extraction — picks most prominent text in top 25% of page
- **Logic:** Filters out generic words (invoice, receipt), stop patterns, monetary amounts
- **Returns:** Guessed company name or None

#### `_trim_generic_suffix(text: str) -> str`
- **Purpose:** Strips trailing doc-type words (e.g., "ACME WIDGETS, INC. INVOICE" → "ACME WIDGETS, INC.")

#### `_extract_total(words: list[dict]) -> Optional[str]`
- **Purpose:** Extracts total amount
- **Logic:** Locates all Total/Amount due labels, gathers same row + row below, picks rightmost monetary token, prefers bottom-most label
- **Returns:** Total amount string (no currency symbol) or None

(Note: File truncated at 20000 chars in read output — additional extraction logic + regex fallback path + main `extract_invoice_fields()` function follow similar patterns)

### Main Function

#### `extract_invoice_fields(text: str, pdf_bytes=None, image_bytes=None, words=None) -> dict`
- **Purpose:** Public API — extracts all 6 invoice fields
- **Parameters:**
  - `text`: Raw PDF text (used for regex fallback)
  - `pdf_bytes`: Optional PDF bytes (for pdfplumber word box extraction)
  - `image_bytes`: Optional image bytes (for OCR word box extraction)
  - `words`: Optional pre-extracted word boxes
- **Returns:** Dict with keys: `invoice_number`, `invoice_date`, `due_date`, `issuer`, `recipient`, `total` (all Optional[str])
- **Logic:**
  1. If words not provided, extracts them from pdf_bytes (pdfplumber) or image_bytes (pytesseract)
  2. Calls layout extractors (`_extract_invoice_number()`, `_extract_date()`, `_extract_issuer()`, `_extract_recipient()`, `_extract_total()`)
  3. Falls back to regex on plain text for any missing fields
- **Side effects:** PDF/image I/O if bytes provided

---

## src/__init__.py
**Lines:** 0
**Purpose:** Package marker (empty file)

---

## scripts/eval_invoices.py
**Lines:** 62
**Purpose:** Batch evaluation of invoice extraction on high-quality images

### Imports
```python
import csv, sys
from pathlib import Path
from src.information_extraction import extract_invoice_fields
from src.pdf_loader import image_to_words, image_to_full_text
```

### Constants
- `ROOT` (Path): Repo root directory
- `IMG_DIR` (Path): data/raw/invoices/high_quality_images/
- `OUT_CSV` (Path): data/processed/invoice_extraction_eval.csv
- `FIELDS` (list): ["invoice_number", "invoice_date", "due_date", "issuer", "recipient", "total"]

### Functions

#### `main() -> None`
- **Purpose:** CLI entry point — processes all images in IMG_DIR, writes results to OUT_CSV
- **Logic:**
  1. Lists all .jpg/.jpeg/.png files in IMG_DIR
  2. Opens OUT_CSV for writing (CSV with columns: filename + FIELDS + non_null_count)
  3. For each image: runs OCR via `image_to_words()`, reconstructs text via `image_to_full_text()`, extracts fields via `extract_invoice_fields()`
  4. Writes row to CSV (flushes after each)
  5. Prints progress every 25 images
  6. On KeyboardInterrupt, saves partial results
  7. Prints summary: per-field non-null rate, worst 10 images (lowest non_null_count)
- **Side effects:** Creates CSV file, console output

---

## scripts/eval_labeled.py
**Lines:** 135
**Purpose:** Evaluate extraction against ground-truth JSON in batch CSVs (precision/recall)

### Imports
```python
import argparse, json, re, sys
from pathlib import Path
import pandas as pd
from src.information_extraction import extract_invoice_fields
from src.pdf_loader import image_to_words, image_to_full_text
```

### Constants
- `ROOT` (Path): Repo root
- `DEFAULT_CSV` (Path): data/raw/invoices/high_quality_images/batch_1/batch_1/batch1_1.csv

### Functions

#### `_image_dir_for(csv_path: Path) -> Path`
- **Purpose:** Computes image directory path from CSV path (batch_X/batch_X/batch_X/)

#### `_gt_fields(json_str: str) -> dict`
- **Purpose:** Parses ground-truth JSON from CSV, extracts invoice + subtotal fields
- **Returns:** Dict with keys matching FIELDS (maps seller_name → issuer, client_name → recipient, etc.)

#### `_norm_date(s: str) -> str`
- **Purpose:** Normalizes date to digits-only string for comparison (strips separators)

#### `_norm_amount(s: str) -> str`
- **Purpose:** Normalizes amount to "1234.56" format (handles EU/US formatting)

#### `_norm_str(s: str) -> str`
- **Purpose:** Normalizes string to lowercase, single spaces

#### `_match(pred: str|None, gt: str|None, field: str) -> bool`
- **Purpose:** Format-agnostic comparison (dates by digits, amounts by float value, names by substring match)
- **Returns:** True if pred matches gt

#### `main() -> None`
- **Purpose:** CLI entry point — evaluates extraction on N invoices from batch CSV
- **CLI args:** `--limit N` (default 100), `--csv PATH`, `--show-misses N` (default 5)
- **Logic:**
  1. Reads CSV with ground-truth JSON
  2. For each row: loads image, runs extraction, compares to ground truth via `_match()`
  3. Tracks hits/misses per field
  4. Prints recall table (hit/gt_present %)
  5. Prints top N misses per field (filename, pred, gt)
- **Side effects:** Console output

---

## setup.py
**Lines:** 14
**Purpose:** Creates data/ folder structure for training pipeline

### Imports
```python
import os
```

### Logic
- Creates directories: `data/raw/invoices`, `data/raw/emails`, `data/raw/contracts`, `data/raw/news`, `data/processed`
- Prints confirmation messages

---

## frontend/app/layout.js
**Lines:** 24
**Purpose:** Next.js root layout (font loading, metadata)

### Imports
```javascript
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
```

### Exports
- `metadata` (object): `{title: "Create Next App", description: "Generated by create next app"}`
- `RootLayout` (default export): React component wrapping children with <html> and <body>

### Constants
- `geistSans` (NextFont): Geist font config (variable: --font-geist-sans, subsets: ["latin"])
- `geistMono` (NextFont): Geist Mono font config (variable: --font-geist-mono, subsets: ["latin"])

### Component

#### `RootLayout({ children })`
- **Returns:** <html> with font CSS variables, <body> with flex layout
- **Classes:** Tailwind classes on <html> and <body> (h-full, antialiased, min-h-full, flex, flex-col)

---

## frontend/app/page.js
**Lines:** 599 (truncated at 20000 chars in read output)
**Purpose:** Main UI page (upload, processing, result screens)

### Imports
```javascript
'use client'
import { useState, useEffect } from 'react'
```

### Constants
- `FIELD_LABELS` (object): Maps field keys to display labels (invoice_number → "Invoice number", etc.)
- `CATEGORY_STYLES` (object): Maps category names to colors + descriptions
- `USE_MOCK` (boolean): false — toggle for mock/real API mode
- `API_URL` (string): 'http://localhost:8000/classify' — backend endpoint
- `MOCK_RESULT` (object): Hardcoded mock classification result (used if USE_MOCK=true)
- `MODELS` (array): Model metadata for UI selection (key, name, tag)

### Component

#### `Home()` (default export)
- **Purpose:** Main page component with three screens (upload, processing, result)
- **State variables:**
  - `step` (string): 'upload' | 'processing' | 'result' — controls screen
  - `result` (object|null): Classification response from backend
  - `file` (File|null): Uploaded PDF
  - `dragOver` (boolean): Drag-and-drop hover state
  - `tick` (number): Increments every 600ms during processing (for "..." animation)
  - `dark` (boolean): Dark mode toggle (default true)
  - `barWidth` (number): Confidence bar width (animated on result screen)
  - `selectedModel` (string): User's model choice (default 'logistic_regression')
  - `error` (string|null): Error message from API failure
- **Effects:**
  - `useEffect(() => {...}, [step])`: When step='processing', starts interval for tick animation + calls API (mock or real), sets result, transitions to step='result' or back to 'upload' on error
  - `useEffect(() => {...}, [step, result])`: When step='result', animates barWidth from 0 to result.confidence after 100ms delay
- **Event handlers:**
  - `handleDrop(e)`: Accepts dropped PDF, sets `file` state
  - `() => setDark(d => !d)`: Toggles dark mode
  - `() => setStep('processing')`: Starts classification (only if file selected)
  - `() => setSelectedModel(m.key)`: Updates selected model
- **Rendering logic:**
  - Computes `t` (theme object) with colors based on `dark` state
  - Conditionally renders one of three screens based on `step`:
    - `step='upload'`: File upload zone, model selector, category pills, classify button
    - `step='processing'`: Spinner + animated status message
    - `step='result'`: Classification result card, confidence bar, invoice fields table (if label='invoice'), "Classify another" button
- **API call logic (real mode):**
  1. Creates FormData with file + selectedModel
  2. POSTs to API_URL
  3. Parses response, normalizes to frontend shape (raw.label → data.category)
  4. Sets result state, transitions to 'result'
  5. On error: logs to console, sets error state, transitions back to 'upload'

(Note: File truncated — full rendering logic continues with result screen JSX)

---

## frontend/app/globals.css
**Lines:** 23
**Purpose:** Global styles (Tailwind import + CSS variables)

### Imports
```css
@import "tailwindcss";
```

### CSS Variables
- `:root`: `--background: #ffffff`, `--foreground: #171717`
- `@media (prefers-color-scheme: dark)`: `--background: #0a0a0a`, `--foreground: #ededed`

### Styles
- `body`: Sets background/foreground from CSS variables, font-family fallback

---

## frontend/next.config.mjs
**Lines:** 5
**Purpose:** Next.js configuration (currently empty)

### Exports
```javascript
const nextConfig = { /* config options here */ };
export default nextConfig;
```

---

## Summary Statistics

**Total Python source files:** 5 (src/)
**Total Python scripts:** 3 (scripts/ + setup.py)
**Total frontend files:** 3 (app/)
**Total configuration files:** 4 (frontend configs)

**Key cross-file dependencies:**
- `src/service.py` imports 3 internal modules (pdf_loader, preprocessing, information_extraction)
- `src/information_extraction.py` is the heaviest module (667 lines, complex spatial logic)
- `frontend/app/page.js` is the largest frontend file (599 lines, manages all UI state)
- All eval scripts depend on `pdf_loader` + `information_extraction`