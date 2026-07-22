# Code Summary

## Python Backend Source Files

### src/__init__.py
- **Lines**: 0 (empty file)
- **Purpose**: Package marker to make `src/` importable
- **Imports**: None
- **Exports**: None

---

### src/service.py
- **Lines**: 218 (estimated)
- **Purpose**: FastAPI REST API microservice exposing document classification and invoice extraction endpoints

**Imports**:
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
# Lazy import in _load_named_model():
from sentence_transformers import SentenceTransformer
```

**Exports**:
- `app: FastAPI` — Main application instance

**Module-level constants**:
- `_ALLOWED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png'}` — Accepted file types
- `_MODELS_DIR = Path(__file__).parent.parent / 'models'` — Path to serialized models
- `_SELECTABLE_MODELS: dict[str, dict]` — Model registry with metadata:
  ```python
  {
      'linear_svc': {'name': 'Linear SVM', 'description': 'Highest F1 on training data. Fast, sparse features.'},
      'logistic_regression': {'name': 'Logistic Regression', 'description': 'Calibrated probabilities. Robust generalisation.'},
      'sbert_logreg': {'name': 'SBERT + LogReg', 'description': 'Semantic embeddings. Most robust on unseen vocabulary.'}
  }
  ```
- `_INV_OVERRIDE_THRESHOLD = 4` — Min invoice signals to override classifier prediction

**Functions**:

1. **`_model_type_for(model_dir: Path) -> str`**
   - Reads `model_type.txt` from model directory
   - Returns `'tfidf'` or `'sbert'` (defaults to `'tfidf'` if file missing)
   - No external calls, pure I/O

2. **`_load_named_model(model_key: str) -> tuple`**
   - **Signature**: `(model_key: str) -> tuple[str, Any, Any]`
   - **Decorator**: `@lru_cache(maxsize=8)` — Caches loaded models per key
   - **Purpose**: Loads classifier and vectorizer/encoder from `models/<model_key>/`
   - **Returns**: `(model_type, encoder, classifier)`
     - If `model_type == 'sbert'`: encoder is `SentenceTransformer`, classifier is scikit-learn model
     - If `model_type == 'tfidf'`: encoder is `TfidfVectorizer`, classifier is scikit-learn model
   - **Side effects**: Loads files from disk via `joblib.load()`, downloads SBERT model on first use
   - **Raises**: `HTTPException(400)` if model_key unknown, `HTTPException(503)` if model dir not found, `RuntimeError` if files missing
   - **Calls**: `_model_type_for()`, `joblib.load()`, `SentenceTransformer()` (if SBERT)

3. **`_count_invoice_signals(text: str) -> int`**
   - Counts how many distinct invoice regex patterns from `_INV_SIGNALS` match the raw text
   - Returns integer count
   - Called by `classify()` for override logic

4. **`_predict_proba(clf, X) -> np.ndarray`**
   - **Purpose**: Unified probability extraction for any scikit-learn classifier
   - **Logic**: 
     - If classifier has `predict_proba()` → use it directly
     - Else (LinearSVC with only `decision_function()`) → apply softmax to decision scores
   - **Returns**: 1D numpy array of per-class probabilities (sums to 1.0)
   - No external calls, pure numpy math

5. **`health() -> dict`**
   - **Route**: `GET /health`
   - **Returns**: `{"status": "ok"}`
   - **Purpose**: Liveness probe for container orchestration

6. **`list_models() -> list[dict]`**
   - **Route**: `GET /models`
   - **Returns**: List of model metadata dicts:
     ```python
     [{"key": str, "name": str, "description": str, "available": bool}, ...]
     ```
   - **Purpose**: UI can query available models dynamically
   - Checks filesystem for model directory existence

7. **`classify(file: UploadFile, model: str = 'logistic_regression') -> dict`**
   - **Route**: `POST /classify`
   - **Parameters**:
     - `file` (multipart): PDF or image file
     - `model` (form field, default `'logistic_regression'`): Model key to use
   - **Returns**: 
     ```python
     {
         "filename": str,
         "label": str,  # 'invoice' | 'contract' | 'email' | 'news'
         "confidence": float,  # 0-1
         "proba": {label: float, ...},
         "model_used": str,
         "inv_signals": int,
         "invoice_fields": dict | None  # Only if label == 'invoice'
     }
     ```
   - **Process**:
     1. Validate file extension
     2. Read file bytes (`await file.read()`)
     3. Extract text (`pdf_to_text()` or `image_to_text()`)
     4. Load model (`_load_named_model()`)
     5. Preprocess text (`clean_for_classifier()` for TF-IDF, or strip indicators for SBERT)
     6. Vectorize (TF-IDF transform or SBERT encode)
     7. Predict class and probabilities (`clf.predict()`, `_predict_proba()`)
     8. Count invoice signals (`_count_invoice_signals()`)
     9. **Override logic**: If `label != 'invoice'` and `inv_signals >= 4` → force `label = 'invoice'`
     10. If `label == 'invoice'` → call `extract_invoice_fields(raw_text, pdf_bytes)`
   - **Side effects**: Loads models (cached), reads uploaded file
   - **Raises**: `HTTPException(400)` for invalid file type, `HTTPException(422)` for read errors, `HTTPException(503)` for model errors
   - **Calls**: `pdf_to_text()`, `image_to_text()`, `_load_named_model()`, `clean_for_classifier()`, `_count_invoice_signals()`, `_predict_proba()`, `extract_invoice_fields()`

8. **`extract(file: UploadFile) -> dict`**
   - **Route**: `POST /extract`
   - **Purpose**: Direct invoice extraction without classification (backward compatibility / testing)
   - **Returns**: `{"filename": str, "fields": dict}`
   - **Process**:
     1. Validate file extension
     2. Read file bytes
     3. Extract text (`pdf_to_text()` or `image_to_text()`)
     4. Extract fields (`extract_invoice_fields()`)
   - **Calls**: `pdf_to_text()`, `image_to_text()`, `extract_invoice_fields()`

**CORS middleware**:
- Allows `http://localhost:3000` origin
- All methods and headers permitted

**Control flow notes**:
- Invoice override logic bypasses classifier when strong signals present (catches borderline cases)
- SBERT model uses regex to strip indicator tokens before encoding (they're TF-IDF-specific features)
- Model caching prevents reloading on every request (performance optimization)

---

### src/pdf_loader.py
- **Lines**: 188 (estimated)
- **Purpose**: PDF and image text extraction with OCR fallback

**Imports**:
```python
from __future__ import annotations
import io
from pathlib import Path
from typing import Union
import pdfplumber
# Lazy imports in functions:
import pypdfium2 as pdfium
import pytesseract
from PIL import Image
```

**Exports**:
- `pdf_to_text(path_or_bytes: PathLike) -> str`
- `image_to_text(path_or_bytes: PathLike) -> str`
- `image_to_words(path_or_bytes: PathLike, min_conf: int = 30) -> list[dict]`
- `image_to_full_text(words: list[dict]) -> str`

**Type alias**:
- `PathLike = Union[str, Path, bytes]`

**Module-level constants**:
- `_MIN_TEXT_CHARS = 40` — Threshold to trigger OCR fallback (if pdfplumber extracts <40 chars)

**Functions**:

1. **`_open(path_or_bytes: PathLike) -> pdfplumber.PDF`**
   - **Private helper**
   - Opens PDF from path or bytes using pdfplumber
   - Returns pdfplumber PDF object (context manager)

2. **`_extract_with_pdfplumber(path_or_bytes: PathLike) -> str`**
   - **Private helper**
   - Extracts text from all pages using pdfplumber (fast path)
   - **Process**: Open PDF → iterate pages → `page.extract_text()` → join with `\n`
   - Returns full text string
   - No OCR, digital-only

3. **`_extract_with_ocr(path_or_bytes: PathLike) -> str`**
   - **Private helper**
   - OCR-based extraction for scanned PDFs
   - **Process**:
     1. Open PDF with pypdfium2
     2. Rasterize each page to PIL image (`page.render(scale=3.0).to_pil()`)
     3. Run Tesseract OCR (`pytesseract.image_to_string(pil_image)`)
     4. Join page texts with `\n`
   - **Side effects**: Calls system `tesseract` binary
   - **Raises**: `RuntimeError` if pytesseract not installed or Tesseract binary not found
   - Returns full text string

4. **`pdf_to_text(path_or_bytes: PathLike) -> str`**
   - **Main entry point for PDF extraction**
   - **Strategy**: Try pdfplumber first, fallback to OCR if text too short
   - **Process**:
     1. Extract text with `_extract_with_pdfplumber()`
     2. If `len(text.strip()) < _MIN_TEXT_CHARS` → call `_extract_with_ocr()`
     3. Return text
   - **Rationale**: Digital PDFs are fast to extract; scanned PDFs yield empty text → triggers OCR
   - **Calls**: `_extract_with_pdfplumber()`, `_extract_with_ocr()` (conditional)

5. **`image_to_text(path_or_bytes: PathLike) -> str`**
   - **Purpose**: OCR on standalone image files (JPG/PNG)
   - Opens image with PIL, calls `pytesseract.image_to_string()`
   - Returns extracted text
   - **Raises**: `RuntimeError` if Tesseract not installed

6. **`image_to_words(path_or_bytes: PathLike, min_conf: int = 30) -> list[dict]`**
   - **Purpose**: OCR with bounding box extraction for spatial analysis
   - **Process**:
     1. Open image with PIL
     2. Run `pytesseract.image_to_data(img, output_type=Output.DICT)` → gets word-level bounding boxes
     3. Filter out empty text and low-confidence words (conf < `min_conf`, except -1 which means structural element)
     4. Convert to list of dicts with pdfplumber-compatible keys
   - **Returns**: 
     ```python
     [
         {"text": str, "x0": float, "x1": float, "top": float, "bottom": float},
         ...
     ]
     ```
   - **Use case**: Enables spatial extraction on scanned invoices

7. **`image_to_full_text(words: list[dict]) -> str`**
   - **Purpose**: Reconstruct reading-order text from word bounding boxes
   - **Process**:
     1. Group words into visual rows by y-coordinate (tolerance = avg height / 2)
     2. Sort each row left-to-right by x-coordinate
     3. Join words in each row with spaces, rows with `\n`
   - **Returns**: Plain text string
   - **Rationale**: Better preserves adjacency than `image_to_string()` for regex matching

**Control flow notes**:
- OCR fallback is automatic and transparent to caller
- Bounding box extraction enables column-aware spatial parsing
- Scale factor of 3.0 in rasterization balances OCR accuracy vs memory usage

---

### src/preprocessing.py
- **Lines**: 106 (estimated)
- **Purpose**: Text cleaning pipeline for classifier (mirrors training pipeline in notebooks)

**Imports**:
```python
from __future__ import annotations
import re
from typing import Union
# Lazy import in process_pdf():
from src.pdf_loader import pdf_to_text
```

**Exports**:
- `clean_for_classifier(text: str) -> str` — Main preprocessing function
- `process_pdf(path_or_bytes: Union[str, bytes]) -> dict` — Demo helper (not used in service)
- `_INV_SIGNALS: list[tuple[str, re.Pattern]]` — Invoice indicator patterns (exported for service.py)

**Module-level constants**:
- `_MAX_WORDS = 500` — Truncation limit (matches training pipeline)
- `_PAGE_NUM_RE = re.compile(r'(?m)^\s*\d+\s*$')` — Isolated page number pattern
- `_HYPHEN_BREAK_RE = re.compile(r'-\n')` — PDF line-break hyphenation
- `_INV_SIGNALS: list[tuple[str, re.Pattern]]` — 10 invoice indicator patterns:
  1. `__inv_number__` — Matches "Invoice No.", "Inv #", etc.
  2. `__inv_header__` — Matches "Tax Invoice", "Invoice"
  3. `__bill_to__` — Matches "Bill To", "Sold To", etc.
  4. `__amount_due__` — Matches "Amount Due", "Balance Due"
  5. `__grand_total__` — Matches "Grand Total", "Subtotal"
  6. `__due_date__` — Matches "Due Date", "Payment Due"
  7. `__remit__` — Matches "Remit Payment", "Please Pay"
  8. `__po_number__` — Matches "Purchase Order", "PO Number"
  9. `__vat__` — Matches "VAT", "GST", "Sales Tax"
  10. `__net_terms__` — Matches "Net 30", "Net 60"

**Functions**:

1. **`_invoice_indicator_tokens(raw_text: str) -> str`**
   - **Private helper**
   - Checks which invoice patterns match the raw text
   - Returns space-separated string of indicator tokens (e.g., `"__inv_number__ __bill_to__"`)
   - Used to prepend synthetic tokens to cleaned text for TF-IDF

2. **`clean_for_classifier(text: str) -> str`**
   - **Main preprocessing function**
   - **Purpose**: Transform raw document text into classifier-ready input (matches training pipeline)
   - **Process**:
     1. Collect invoice indicator tokens from original text (before lowercasing)
     2. Rejoin PDF hyphenated line-breaks (`docu-\nment` → `document`)
     3. Strip isolated page-number lines
     4. Lowercase
     5. Collapse all whitespace to single spaces
     6. Truncate to first 500 words
     7. Prepend indicator tokens if any matched
   - **Returns**: Cleaned text string
   - **Notes**: 
     - Does NOT remove stopwords (handled by TfidfVectorizer)
     - Does NOT remove punctuation (kept for classification)
     - Indicator tokens amplify weak signals for minority invoice class

3. **`process_pdf(path_or_bytes: Union[str, bytes]) -> dict`**
   - **Demo entry point** (not used in service.py)
   - **Purpose**: One-call PDF → both raw and cleaned text
   - **Returns**: `{"raw_text": str, "classifier_text": str}`
   - **Use case**: Notebooks or scripts that need both versions
   - **Calls**: `pdf_to_text()`, `clean_for_classifier()`

**State management**:
- No state, pure functions
- Indicator tokens collected fresh on every call (no caching)

**Control flow notes**:
- Indicator tokens are extracted BEFORE lowercasing to preserve case-sensitive patterns
- Truncation to 500 words prevents memory issues and matches training data distribution
- Preprocessing is idempotent (applying twice has same effect as once)

---

### src/information_extraction.py
- **Lines**: 664 (estimated, file was truncated in read output)
- **Purpose**: Rule-based invoice field extraction using spatial layout analysis and regex

**Imports**:
```python
from __future__ import annotations
import io
import re
from dataclasses import asdict, dataclass
from typing import Optional, Sequence
import pdfplumber
```

**Exports**:
- `extract_invoice_fields(text: str, pdf_bytes=None, image_bytes=None, words=None) -> dict`
- Many private helper functions (callable but not part of public API)

**Regex constants**:
- `_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"` — Month names
- `_DATE_PATTERNS: list[str]` — 6 date format patterns (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD, "15 Jan 2024", etc.)
- `DATE_RE: re.Pattern` — Compiled union of date patterns
- `_YEAR_ONLY_RE: re.Pattern` — Matches 4-digit years (to exclude from invoice numbers)
- `_AMOUNT_RE: re.Pattern` — Monetary amounts (handles EU/US formats, currency symbols)
- `_INV_NUMBER_VALUE_RE: re.Pattern` — Valid invoice number pattern (`#?[A-Z0-9][A-Z0-9\-/]{2,19}`)
- `_STOP_BLOCK_RE: re.Pattern` — Stop words signaling end of issuer/recipient blocks (tax ID, VAT, email, etc.)

**Label dictionaries** (field name → regex patterns, ordered by specificity):
- `_LABELS: dict[str, list[str]]` — Patterns for invoice_number, invoice_date, due_date
- `_ISSUER_LABELS: list[str]` — Patterns for issuer field (Seller, From, Vendor, etc.)
- `_RECIPIENT_LABELS: list[str]` — Patterns for recipient (Bill To, Sold To, Client, etc.)
- `_TOTAL_LABELS: list[str]` — Patterns for total amount (Grand Total, Amount Due, etc.)

**Key functions** (condensed, file was truncated):

1. **`extract_invoice_fields(text: str, pdf_bytes=None, image_bytes=None, words=None) -> dict`**
   - **Main entry point**
   - **Returns**: `{"invoice_number": str|None, "invoice_date": str|None, "due_date": str|None, "issuer": str|None, "recipient": str|None, "total": str|None}`
   - **Strategy**:
     - If word bounding boxes available (from pdfplumber or image_to_words) → use spatial extraction
     - Else → fallback to 1D regex on plain text
   - **Process**:
     1. Try extracting word boxes from PDF (if pdf_bytes) or image (if image_bytes)
     2. For each field, call corresponding `_extract_<field>()` function
     3. Return dict with all fields (None for missing)
   - **Calls**: `_extract_invoice_number()`, `_extract_date()`, `_extract_issuer()`, `_extract_recipient()`, `_extract_total()`

2. **`_find_label_span(label_re: re.Pattern, words: list[dict]) -> Optional[tuple[int, int]]`**
   - **Purpose**: Locate a label (possibly multi-word) in word list
   - **Returns**: `(start_index, end_index_exclusive)` or None
   - **Scoring**: Prefers labels ending with `:` and at row start (column headers)

3. **`_words_right_of(words, start, end, y_ref, x_end) -> list[dict]`**
   - **Purpose**: Find words on same visual row as label, to the right of label
   - **Use case**: Extract value inline with label (e.g., "Invoice No: 12345")

4. **`_rows_below(words, start, end, y_ref, n_rows, lh) -> list[list[dict]]`**
   - **Purpose**: Get up to `n_rows` visual rows immediately below label
   - **Use case**: Extract value in column below label (e.g., "Invoice No:\n12345")

5. **`_pick_value(compiled_labels, words, validate, col_tol=None) -> Optional[str]`**
   - **General picker**: Try inline-right, then column-aligned below, then any below
   - **Process**:
     1. Find label with `_find_label_span()`
     2. Try same-row right of label → validate → return if valid
     3. Try rows below, column-aligned → validate → return if valid
     4. Try any word on below rows → validate → return if valid
   - **Returns**: First valid value, or None
   - **Calls**: Label-specific validator function

6. **`_pick_block_below(compiled_labels, words, col_tol=None) -> Optional[str]`**
   - **Purpose**: Extract multi-word blocks (issuer/recipient names)
   - **Process**: Find label, return first non-stop row below that's column-aligned
   - **Stops at**: Tax ID, VAT, phone, email (via `_STOP_BLOCK_RE`)

7. **`_extract_invoice_number(words: list[dict]) -> Optional[str]`**
   - Uses `_pick_value()` with `_validate_invoice_number()` validator
   - **Fallback**: Searches for bare `#ABC-123` tokens if no label found

8. **`_extract_date(words: list[dict], key: str) -> Optional[str]`**
   - Uses `_pick_value()` with `_validate_date()` validator
   - **Fallback** (for invoice_date only): First parseable date not near "due date" label

9. **`_extract_issuer(words: list[dict]) -> Optional[str]`**
   - Uses `_pick_block_below()` for "Seller:" labels
   - **Fallback**: `_header_company_guess()` — picks most prominent text in top 25% of page

10. **`_extract_recipient(words: list[dict]) -> Optional[str]`**
    - Uses `_pick_value()` for inline "Bill To: Company" patterns
    - Falls back to `_pick_block_below()`

11. **`_extract_total(words: list[dict]) -> Optional[str]`**
    - Locates "Total" / "Amount Due" labels
    - Gathers same row and row below
    - Takes rightmost monetary token (via `_pick_rightmost_monetary()`)
    - Prefers bottom-most label (summary section)
    - **Excludes**: "Sub Total" labels

**Validator functions**:
- `_validate_date(s) -> Optional[str]` — Extracts first date match from string
- `_validate_invoice_number(s) -> Optional[str]` — Checks against `_INV_NUMBER_VALUE_RE`
- `_is_monetary(tok) -> bool` — Detects currency symbols or decimal separator with 2 cents
- `_amount_as_float(tok) -> float` — Parses EU/US number formats to float

**Spatial utilities**:
- `_y_overlap(a, b) -> float` — Vertical overlap ratio between two word boxes
- `_line_height(words) -> float` — Average word height (used for row grouping tolerance)
- `_group_rows(words, tol) -> list[list[dict]]` — Clusters words into visual rows

**Control flow notes**:
- Spatial extraction is preferred over regex (recovers 2-column layout)
- Fallback to 1D regex when no bounding boxes available
- Label specificity ordering prevents false matches (e.g., "Invoice Date" before "Date")
- Stop-word detection prevents bleeding contact info into company names
- Total extraction prefers bottom-most label (invoices often repeat "Total" in item rows)

---

## Frontend Source Files

### frontend/app/layout.js
- **Lines**: 24
- **Purpose**: Root layout component for Next.js app

**Imports**:
```javascript
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
```

**Exports**:
- `export const metadata = {title: "Create Next App", description: "Generated by create next app"}`
- `export default function RootLayout({ children })`

**Constants**:
- `geistSans` — Geist Sans font loader (Google Fonts)
- `geistMono` — Geist Mono font loader (Google Fonts)

**Component: RootLayout({ children })**:
- **Props**: `{ children: ReactNode }`
- **Returns**: HTML structure with `<html>` and `<body>` tags
- **Styling**: Applies font CSS variables, Tailwind classes (`h-full`, `antialiased`, `min-h-full`, `flex`, `flex-col`)
- **Purpose**: Wraps all pages, sets up fonts and global layout structure

---

### frontend/app/page.js
- **Lines**: 476 (estimated, file was truncated)
- **Purpose**: Main single-page application UI for document classification

**Directive**:
```javascript
'use client'  // Marks component as client-side rendered (required for hooks)
```

**Imports**:
```javascript
import { useState, useEffect } from 'react'
```

**Exports**:
- `export default function Home()`

**Constants**:

1. **`FIELD_LABELS: object`** — Display names for invoice fields
   ```javascript
   {
       invoice_number: 'Invoice number',
       invoice_date:   'Invoice date',
       due_date:       'Due date',
       issuer:         'Issuer',
       recipient:      'Recipient',
       total:          'Total amount'
   }
   ```

2. **`CATEGORY_STYLES: object`** — Visual styles per document category
   ```javascript
   {
       invoice:  { color: '#a3e635', colorLight: '#3f6212', label: 'Invoice',  description: 'Field extraction triggered' },
       contract: { color: '#86efac', colorLight: '#16a34a', label: 'Contract', description: 'No extraction for contracts' },
       email:    { color: '#4ade80', colorLight: '#15803d', label: 'Email',    description: 'No extraction for emails' },
       news:     { color: '#34d399', colorLight: '#047857', label: 'News',     description: 'No extraction for news' }
   }
   ```

3. **`USE_MOCK = false`** — Feature flag for mock API mode (development)
4. **`API_URL = 'http://localhost:8000/classify'`** — Backend endpoint
5. **`MOCK_RESULT`** — Sample response for UI testing
6. **`MODELS: array`** — Model selector options
   ```javascript
   [
       { key: 'linear_svc',          name: 'Linear SVM',        tag: 'Highest F1' },
       { key: 'logistic_regression', name: 'Logistic Regression', tag: 'Calibrated' },
       { key: 'sbert_logreg',        name: 'SBERT + LogReg',    tag: 'Most Robust' }
   ]
   ```

**Component: Home()**:

**State variables** (via `useState`):
- `step: string` — Current UI screen: `'upload'` | `'processing'` | `'result'`
- `result: object | null` — Classification response from backend
- `file: File | null` — Selected PDF file
- `dragOver: boolean` — Drag-and-drop hover state
- `tick: number` — Animation counter for processing screen dots
- `dark: boolean` — Dark mode toggle (default true)
- `barWidth: number` — Confidence bar animation width (0-100)
- `selectedModel: string` — Active model key (default `'logistic_regression'`)
- `error: string | null` — Error message

**Effects** (via `useEffect`):

1. **Processing effect** (dependency: `[step]`):
   - **Trigger**: When `step` changes to `'processing'`
   - **Actions**:
     - Starts interval timer to increment `tick` (animates "Extracting text… Running classifier… Extracting fields…" dots)
     - Calls async `classify()` function:
       - If `USE_MOCK === true` → wait 2.5s, set `result` to `MOCK_RESULT`
       - Else → POST to API_URL with FormData (file + model), parse JSON, normalize shape
       - On success → `setResult(data)`, `setStep('result')`
       - On error → `setError(err.message)`, `setStep('upload')`
     - Cleanup: Clears interval on unmount
   - **Normalization**: Backend `{label, confidence, invoice_fields}` → Frontend `{category, confidence, fields}`

2. **Bar animation effect** (dependency: `[step, result]`):
   - **Trigger**: When `step` changes to `'result'`
   - **Action**: Sets `barWidth` to confidence value after 100ms delay (triggers CSS animation)

**Event handlers**:

1. **`handleDrop(e)`**:
   - Prevents default browser behavior
   - Reads first file from `e.dataTransfer.files`
   - Sets `file` state if file is PDF
   - Clears `dragOver` state

**Derived values**:
- `t: object` — Theme color palette (changes based on `dark` state)
- `cat: object` — Active category style from `CATEGORY_STYLES[result.category]`
- `catColor: string` — Active category color (dark or light based on theme)
- `dots: array` — Processing screen status messages
- `pills: array` — Category badge data with colors

**Render structure**:
- **Upload screen** (`step === 'upload'`):
  - Left column: Title, description, error message (if any)
  - Right column: 
    - Drag-and-drop zone (clickable, triggers file input)
    - Category pills (invoice/contract/email/news)
    - Model selector (radio-style buttons)
    - "Run classification" button (disabled if no file)
  - Handlers: `onDragOver`, `onDragLeave`, `onDrop`, `onClick` (triggers file picker)
- **Processing screen** (`step === 'processing'`):
  - Centered: Spinning loader, animated status text, filename
- **Result screen** (`step === 'result'`):
  - Classification result card:
    - Category label (large, colored)
    - Confidence badge
    - Filename, model used
    - Confidence bar (animated)
  - Invoice fields card (if `result.category === 'invoice'` and `result.fields`):
    - 2-column grid of field labels and values
    - "None extracted" fallback for null values
  - "Classify another document" button → resets to upload

**Styling approach**:
- Inline `style={{}}` objects with CSS variables from `t` theme palette
- Dynamic styles based on state (`dark`, `dragOver`, `selectedModel`, `result.category`)
- CSS animations defined in `<style>` tag (spin, fadeIn)
- Tailwind utility classes in layout.js but not extensively used in page.js

**API contract**:
- **Request**: `FormData { file: Blob, model: string }`
- **Response**: `{ label, confidence, proba, invoice_fields?, model_used, inv_signals }`
- **Frontend normalization**: `label → category`, `confidence * 100`, `invoice_fields → fields`

**Control flow**:
- Linear progression: upload → processing → result
- No back button in processing (can't cancel)
- Result screen has "Classify another" button to restart
- Error state resets to upload screen with error message displayed
- Dark mode toggle persists within session (resets on page reload)

---

### frontend/app/globals.css
- **Lines**: 24
- **Purpose**: Global styles and Tailwind CSS configuration

**Content**:
```css
@import "tailwindcss";

:root {
  --background: #ffffff;
  --foreground: #171717;
}

@theme inline {
  --color-background: var(--background);
  --color-foreground: var(--foreground);
  --font-sans: var(--font-geist-sans);
  --font-mono: var(--font-geist-mono);
}

@media (prefers-color-scheme: dark) {
  :root {
    --background: #0a0a0a;
    --foreground: #ededed;
  }
}

body {
  background: var(--background);
  color: var(--foreground);
  font-family: Arial, Helvetica, sans-serif;
}
```

**CSS variables**:
- `--background` — Background color (light: white, dark: near-black)
- `--foreground` — Text color (light: dark gray, dark: off-white)
- `--font-sans`, `--font-mono` — Font family variables (set by layout.js font loaders)

**Media query**:
- Detects system dark mode preference
- Overrides CSS variables for dark theme
- **Note**: Not used by page.js (which uses JS-controlled dark state)

---

### frontend/next.config.mjs
- **Lines**: 5
- **Purpose**: Next.js configuration file

**Content**:
```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  /* config options here */
};

export default nextConfig;
```

**Configuration**: Empty (uses Next.js defaults)

---

## Python Script Files

### scripts/eval_invoices.py
- **Lines**: 71
- **Purpose**: Batch evaluation of invoice extraction on unlabeled images

**Imports**:
```python
from __future__ import annotations
import csv
import sys
from pathlib import Path
# Adds project root to path:
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.information_extraction import extract_invoice_fields
from src.pdf_loader import image_to_words, image_to_full_text
```

**Constants**:
- `IMG_DIR = ROOT / "data/raw/invoices/high_quality_images"` — Image directory
- `OUT_CSV = ROOT / "data/processed/invoice_extraction_eval.csv"` — Output CSV
- `FIELDS = ["invoice_number", "invoice_date", "due_date", "issuer", "recipient", "total"]`

**Function: main()**:
- **Purpose**: Process all images in IMG_DIR, extract fields, write CSV with results
- **Process**:
  1. Check if IMG_DIR exists
  2. Glob all `.jpg`, `.jpeg`, `.png` files recursively
  3. Open output CSV with writer
  4. For each image:
     - Read bytes
     - Extract words with `image_to_words()`
     - Reconstruct text with `image_to_full_text()`
     - Extract fields with `extract_invoice_fields()`
     - Count non-null fields
     - Write row to CSV
     - Flush after each row (progress saved if interrupted)
  5. Print summary stats (per-field non-null rate, worst 10 files)
- **Error handling**: Try/except per image, continues on error
- **Ctrl+C handling**: Partial results saved on interrupt
- **Output columns**: filename, 6 field columns, non_null_count

**Usage**: `python scripts/eval_invoices.py`

---

### scripts/eval_labeled.py
- **Lines**: 128
- **Purpose**: Evaluate extraction accuracy against ground-truth JSON labels

**Imports**:
```python
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
import pandas as pd
# Adds project root to path:
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.information_extraction import extract_invoice_fields
from src.pdf_loader import image_to_words, image_to_full_text
```

**Constants**:
- `DEFAULT_CSV = ROOT / "data/raw/invoices/high_quality_images/batch_1/batch_1/batch1_1.csv"`

**Functions**:

1. **`_image_dir_for(csv_path: Path) -> Path`**
   - Returns image directory path derived from CSV path (parent / stem)

2. **`_gt_fields(json_str: str) -> dict`**
   - Parses ground-truth JSON string from CSV
   - Extracts 6 fields from nested structure
   - Returns dict with same keys as extraction output

3. **`_norm_date(s: str) -> str`**
   - Strips all non-digits from date string
   - Used for fuzzy date matching

4. **`_norm_amount(s: str) -> str`**
   - Normalizes monetary amount to `"123.45"` format
   - Handles EU (comma decimal) and US (dot decimal) formats

5. **`_norm_str(s: str) -> str`**
   - Lowercases, collapses whitespace, strips
   - Used for text field normalization

6. **`_match(pred: str | None, gt: str | None, field: str) -> bool`**
   - **Purpose**: Field-specific fuzzy matching
   - **Logic**:
     - Dates: Match if digit sequences equal
     - Amounts: Match if normalized floats equal
     - Invoice number: Match if normalized strings equal (ignoring leading `#`)
     - Issuer/recipient: Match if either is substring of other
   - **Returns**: True if predicted value matches ground truth

7. **`main()`**:
   - **CLI arguments**: `--limit N`, `--csv PATH`, `--show-misses N`
   - **Process**:
     1. Read CSV with ground-truth JSON
     2. For each row (up to limit):
        - Locate image file
        - Extract fields
        - Compare to ground truth with `_match()`
        - Track hits/total per field
     3. Print per-field recall (hits / ground-truth present)
     4. Print worst misses (predicted vs GT)
   - **Output**: Table of recall percentages, sample mismatches

**Usage**: `python scripts/eval_labeled.py --limit 100 --csv path/to/batch.csv`

---

## Other Files

### setup.py
- **Lines**: 14
- **Purpose**: Creates data directory structure for first-time setup

**Content**:
```python
import os

folders = [
    "data/raw/invoices",
    "data/raw/emails",
    "data/raw/contracts",
    "data/raw/news",
    "data/processed"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ Created {folder}")

print("\n✅ Setup complete! Now run 01_data_collection.ipynb")
```

**Execution**: `python setup.py`

---

### .gitignore (root)
- **Purpose**: Prevents committing temporary files and large datasets

**Patterns**:
```
venv/
__pycache__/
*.pyc
.DS_Store
*.egg-info/
.env
data/raw/
data/processed/
data/labeled/
.venv/
justfile
DEV_NOTES.md
EXTRACTION_NOTES.md
PIPELINE.md
```

**Key exclusions**:
- Virtual environments
- Python cache
- Data directories (too large, regenerated from notebooks)
- Local dev notes

---

### frontend/.gitignore
- **Purpose**: Prevents committing Node.js build artifacts

**Patterns** (standard Next.js):
```
/node_modules
/.next/
/out/
/build
.DS_Store
*.pem
npm-debug.log*
.env*
.vercel
*.tsbuildinfo
next-env.d.ts
```

**Key exclusions**:
- Dependencies
- Build output
- Environment files
- Next.js cache
