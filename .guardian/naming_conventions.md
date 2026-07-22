# Naming Conventions & Code Patterns

*This document records the established patterns observed across the codebase. Use this as the reference for how code should be written in this project.*

---

## Component / Class Naming

### React Components (Frontend)
- **Pattern:** PascalCase, descriptive noun
- **Examples:**
  - `RootLayout` (frontend/app/layout.js:16)
  - `Home` (frontend/app/page.js:33)
- **Consistency:** ✅ All React components follow this pattern

### Python Classes
- **Pattern:** PascalCase (when present)
- **Example:**
  - `InvoiceFields` dataclass (src/information_extraction.py)
- **Note:** Very few classes in codebase — mostly functional style

---

## File & Directory Naming

### Python Source Files
- **Pattern:** snake_case.py
- **Examples:**
  - `src/service.py`
  - `src/pdf_loader.py`
  - `src/preprocessing.py`
  - `src/information_extraction.py`
  - `scripts/eval_invoices.py`
  - `scripts/eval_labeled.py`
- **Consistency:** ✅ 100% adherence

### Frontend Files
- **Pattern:** kebab-case for config files, PascalCase for React components converted to .js
- **Examples:**
  - `app/layout.js` (lowercase, Next.js convention)
  - `app/page.js` (lowercase, Next.js convention)
  - `app/globals.css`
  - `next.config.mjs`
  - `postcss.config.mjs`
  - `eslint.config.mjs`
- **Consistency:** ✅ Follows Next.js App Router conventions

### Directories
- **Pattern:** lowercase, underscore or hyphen
- **Examples:**
  - `src/`, `scripts/`, `notebooks/`, `models/`
  - `data/raw/`, `data/processed/`
  - `frontend/app/`, `frontend/public/`
- **Consistency:** ✅ All lowercase

---

## Variable & Constant Naming

### Python Variables (Local)
- **Pattern:** snake_case
- **Examples:**
  - `raw_text` (src/service.py:166)
  - `pdf_bytes` (src/service.py:162)
  - `model_dir` (src/service.py:72)
  - `line_height` (src/information_extraction.py)
- **Consistency:** ✅ Universal

### Python Constants (Module-Level)
- **Pattern:** SCREAMING_SNAKE_CASE or _PRIVATE_SCREAMING_SNAKE_CASE
- **Examples:**
  - `_ALLOWED_EXTENSIONS` (src/service.py:23)
  - `_MODELS_DIR` (src/service.py:25)
  - `_MIN_TEXT_CHARS` (src/pdf_loader.py:16)
  - `_MAX_WORDS` (src/preprocessing.py:24)
  - `_INV_SIGNALS` (src/preprocessing.py:35-45)
  - `DATE_RE` (src/information_extraction.py:40, public constant)
  - `ROOT`, `IMG_DIR`, `OUT_CSV`, `FIELDS` (scripts/eval_invoices.py:13-16, public)
- **Pattern variation:**
  - Private constants: Prefix with underscore (most common)
  - Public constants: No underscore (rare, only in scripts or regex exports)
- **Consistency:** ✅ Consistent within files

### JavaScript/React Variables
- **Pattern:** camelCase (local state, props, functions)
- **Examples:**
  - `geistSans`, `geistMono` (frontend/app/layout.js:3-9)
  - `step`, `result`, `file`, `dragOver` (frontend/app/page.js:34-40, state vars)
  - `selectedModel`, `barWidth`, `dark` (frontend/app/page.js)
- **Consistency:** ✅ Universal

### JavaScript Constants
- **Pattern:** SCREAMING_SNAKE_CASE (module-level config), camelCase (derived/computed)
- **Examples:**
  - `FIELD_LABELS`, `CATEGORY_STYLES`, `USE_MOCK`, `API_URL`, `MOCK_RESULT`, `MODELS` (frontend/app/page.js:3-31)
- **Consistency:** ⚠️ Mixed — configuration constants use SCREAMING_SNAKE_CASE, but some are arrays/objects with PascalCase keys (e.g., `CATEGORY_STYLES.invoice.label`). This is intentional for API-like data structures.

---

## Function / Method Naming

### Python Public Functions
- **Pattern:** snake_case, verb-noun or verb
- **Examples:**
  - `pdf_to_text()` (src/pdf_loader.py:60)
  - `image_to_text()` (src/pdf_loader.py:68)
  - `clean_for_classifier()` (src/preprocessing.py:55)
  - `extract_invoice_fields()` (src/information_extraction.py, main API)
- **Consistency:** ✅ All public functions follow this

### Python Private/Internal Functions
- **Pattern:** _snake_case (leading underscore)
- **Examples:**
  - `_open()` (src/pdf_loader.py:19)
  - `_extract_with_pdfplumber()` (src/pdf_loader.py:24)
  - `_model_type_for()` (src/service.py:41)
  - `_load_named_model()` (src/service.py:46)
  - `_count_invoice_signals()` (src/service.py:83)
  - `_validate_date()` (src/information_extraction.py)
  - `_extract_invoice_number()` (src/information_extraction.py)
- **Consistency:** ✅ 100% adherence (all internal helpers are prefixed)

### React/JavaScript Functions
- **Pattern:** camelCase, verb-noun
- **Examples:**
  - `handleDrop` (frontend/app/page.js:56, event handler)
  - `setStep`, `setFile`, `setDark` (React setState functions, convention)
- **Consistency:** ✅ All follow camelCase

### Async Functions
- **Pattern:** Same as sync (snake_case in Python, camelCase in JS), marked with `async` keyword
- **Examples:**
  - `async def classify(...)` (src/service.py:118)
  - `async def extract(...)` (src/service.py:187)
  - `async function classify() { ... }` (frontend/app/page.js:65, inside useEffect)
- **Consistency:** ✅ No special naming — relies on `async` keyword

### CRUD Verb Patterns
- **Extract:** `extract_invoice_fields()`, `_extract_with_pdfplumber()`, `_extract_with_ocr()`, `_extract_invoice_number()`, `_extract_date()`, `_extract_issuer()`, `_extract_recipient()`, `_extract_total()`
- **Load:** `_load_named_model()`
- **Clean:** `clean_for_classifier()`, `_clean_block_value()`
- **Validate:** `_validate_date()`, `_validate_invoice_number()`
- **Pick:** `_pick_inline()`, `_pick_value()`, `_pick_block_below()`
- **Count:** `_count_invoice_signals()`
- **Predict:** `_predict_proba()`
- **Norm/Normalize:** `_norm_date()`, `_norm_amount()`, `_norm_str()` (scripts/eval_labeled.py)
- **Match:** `_match()` (scripts/eval_labeled.py)

---

## Type / Interface Naming

### Python Type Aliases
- **Pattern:** PascalCase
- **Examples:**
  - `PathLike = Union[str, Path, bytes]` (src/pdf_loader.py:13)
- **Consistency:** ✅ Single example, follows convention

### Dataclasses
- **Pattern:** PascalCase, noun
- **Example:**
  - `InvoiceFields` (src/information_extraction.py)
- **Consistency:** ✅ Only dataclass in codebase

### Type Hints
- **Pattern:** Standard typing module names (Optional, Union, Sequence, etc.)
- **Usage:** Present in all Python source files (modern Python 3.10+ style with `from __future__ import annotations`)
- **Consistency:** ✅ Comprehensive — all function signatures have type hints

---

## Error Handling Patterns

### Python (FastAPI)
- **Pattern:** Raise `HTTPException` with status code + detail message
- **Examples:**
  - `raise HTTPException(status_code=400, detail='file must be a .pdf, .jpg, or .png')` (src/service.py:130)
  - `raise HTTPException(status_code=400, detail='empty upload')` (src/service.py:133)
  - `raise HTTPException(status_code=422, detail=f'could not read file: {e}')` (src/service.py:138)
  - `raise HTTPException(status_code=503, detail=f'Model "{model_key}" not found...')` (src/service.py:62)
- **Consistency:** ✅ All API errors use HTTPException

### Python (Internal)
- **Pattern:** Raise `RuntimeError` with descriptive message
- **Examples:**
  - `raise RuntimeError('No model files found in...')` (src/service.py:57)
  - `raise RuntimeError('pytesseract is not installed...')` (src/pdf_loader.py:49)
  - `raise RuntimeError('Tesseract OCR binary not found...')` (src/pdf_loader.py:63)
- **Consistency:** ✅ Internal errors use RuntimeError

### Python (Scripts)
- **Pattern:** Try-except with error logging, graceful degradation
- **Examples:**
  - `except Exception as e: print(f"  [{i}/{len(images)}] {img.name}: ERROR {e}")` (scripts/eval_invoices.py:37)
  - `except KeyboardInterrupt: print(f"\nInterrupted after {len(rows)} images...")` (scripts/eval_invoices.py:43)
- **Consistency:** ✅ Scripts log errors, continue processing

### JavaScript (Frontend)
- **Pattern:** Try-catch, set error state, log to console
- **Example:**
  - `catch (err) { console.error('Classification failed:', err); setError(err.message || 'Something went wrong...'); setStep('upload'); }` (frontend/app/page.js:96)
- **Consistency:** ✅ Single error boundary in classify effect

### Gap Identified
- **Issue:** No structured logging in backend (errors only printed to console via uvicorn)
- **Risk:** Low (development project), but production deployment would need proper logging

---

## Import Organization

### Python
- **Pattern (Standard):**
  1. `from __future__ import annotations` (always first if present)
  2. Python stdlib imports (grouped)
  3. External package imports (grouped)
  4. Internal project imports (grouped, using `from src.module import ...`)
- **Example (src/service.py:10-20):**
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
- **Consistency:** ✅ All files follow this order

### JavaScript/React
- **Pattern:**
  1. React directives (`'use client'`) first
  2. React imports (`from 'react'`)
  3. Next.js imports (`from 'next/...'`)
  4. Local imports (CSS files)
- **Example (frontend/app/page.js:1-2, frontend/app/layout.js:1-2):**
  ```javascript
  'use client'
  import { useState, useEffect } from 'react'
  ```
  ```javascript
  import { Geist, Geist_Mono } from "next/font/google";
  import "./globals.css";
  ```
- **Consistency:** ✅ All files follow this order

---

## State Management Patterns

### React (Frontend)
- **Pattern:** `useState` for local component state, no global state
- **Example (frontend/app/page.js:34-42):**
  ```javascript
  const [step, setStep] = useState('upload')
  const [result, setResult] = useState(null)
  const [file, setFile] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [tick, setTick] = useState(0)
  const [dark, setDark] = useState(true)
  const [barWidth, setBarWidth] = useState(0)
  const [selectedModel, setSelectedModel] = useState('logistic_regression')
  const [error, setError] = useState(null)
  ```
- **Naming:** State variables use camelCase nouns, setters use `set` + PascalCase
- **Consistency:** ✅ All state follows this pattern

### Python (Backend)
- **Pattern:** Stateless (no in-memory state persisted between requests)
- **Caching:** `@lru_cache` decorator for model loading
- **Example (src/service.py:46):**
  ```python
  @lru_cache(maxsize=8)
  def _load_named_model(model_key: str):
      ...
  ```
- **Consistency:** ✅ Only cached data is loaded models

---

## Event Handling Patterns

### React Event Handlers
- **Pattern:** `handleEventName` for custom handlers, inline lambdas for simple state updates
- **Examples:**
  - Custom: `handleDrop = (e) => { e.preventDefault(); ... }` (frontend/app/page.js:56)
  - Inline: `onClick={() => setDark(d => !d)}` (frontend/app/page.js:89)
  - Inline: `onClick={() => { if (file) setStep('processing') }}` (frontend/app/page.js:244)
- **Consistency:** ✅ Simple handlers inline, complex handlers named with `handle` prefix

---

## CSS / Styling Approach

### Frontend
- **Pattern:** Inline styles (JavaScript objects) + Tailwind utility classes
- **Inline styles:**
  - Computed theme object `t` with color tokens (src/page.js:44-54)
  - Per-element style objects: `style={{ minHeight: '100vh', backgroundColor: t.bg, ... }}`
- **Tailwind classes:**
  - Layout: `flex`, `flex-col`, `min-h-full`, `h-full`
  - Spacing: `gap-4`, `p-2`, `mt-4`
  - Typography: `font-bold`, `text-lg`, `antialiased`
- **Dark mode:** Computed dynamically via `dark` state, not CSS classes
- **Consistency:** ✅ All components use inline styles for theming, Tailwind for layout

---

## External Integration / Side-Effect Patterns

### API Calls (Frontend)
- **Pattern:** Fetch API in `useEffect` hook, FormData for file uploads
- **Example (frontend/app/page.js:77-95):**
  ```javascript
  const formData = new FormData()
  formData.append('file', file)
  formData.append('model', selectedModel)
  const response = await fetch(API_URL, { method: 'POST', body: formData })
  if (!response.ok) { ... }
  const raw = await response.json()
  ```
- **Consistency:** ✅ Single API call pattern

### File I/O (Python)
- **Pattern:** Pathlib for paths, context managers for file operations
- **Examples:**
  - `with _open(path_or_bytes) as pdf:` (src/pdf_loader.py:26)
  - `marker.read_text().strip()` (src/service.py:42)
  - `img_path.read_bytes()` (scripts/eval_invoices.py:34)
- **Consistency:** ✅ All file I/O uses pathlib + context managers

### Model Loading (Python)
- **Pattern:** `joblib.load()` for .joblib files, lazy imports for heavy libraries
- **Example (src/service.py:79):**
  ```python
  from sentence_transformers import SentenceTransformer  # lazy import
  ```
- **Consistency:** ✅ All model loading cached via `@lru_cache`

---

## Internal Inconsistencies Flagged

### 1. Frontend Model List Duplication
- **Location:** `frontend/app/page.js:25-29` defines `MODELS` array, but backend also has `_SELECTABLE_MODELS` (src/service.py:27-39) and `/models` endpoint
- **Issue:** Frontend hardcodes model metadata instead of fetching from `/models`
- **Risk:** MEDIUM — if notebooks add a new model, frontend won't show it until `MODELS` constant updated
- **Recommendation:** Frontend should fetch `/models` on mount and use that response (keeps frontend/backend in sync)

### 2. Legacy Model File Fallback
- **Location:** `src/service.py:54-59` has `if model_key == 'auto'` branch that loads `lr_classifier.joblib` and `tfidf_vectorizer.joblib` from models/ root
- **Issue:** Unclear when this path is used (notebooks write to subdirectories, frontend never passes 'auto')
- **Risk:** LOW — likely dead code from refactor
- **Recommendation:** Remove 'auto' fallback or document when it's used

### 3. Mixed Mock/Real API Toggle
- **Location:** `frontend/app/page.js:17` has `USE_MOCK = false` constant
- **Issue:** Mock code still present in production bundle (lines 70-73)
- **Risk:** LOW (just dead code), but increases bundle size
- **Recommendation:** Use build-time environment variable (process.env.NEXT_PUBLIC_USE_MOCK) and remove mock branch in production builds

### 4. Inconsistent Error Messages
- **Pattern 1:** Backend uses lowercase, no trailing period: `'file must be a .pdf, .jpg, or .png'` (src/service.py:130)
- **Pattern 2:** Backend uses sentence case, period: `'No model files found in {_MODELS_DIR}.'` (src/service.py:57)
- **Pattern 3:** Frontend uses sentence case, period: `'Something went wrong. Please try again.'` (frontend/app/page.js:98)
- **Risk:** LOW (cosmetic)
- **Recommendation:** Standardize on sentence case + period (aligns with user-facing messages)

### 5. Magic Numbers
- **Locations:**
  - `_INV_OVERRIDE_THRESHOLD = 4` (src/service.py:29) — hardcoded threshold for invoice override
  - `_MIN_TEXT_CHARS = 40` (src/pdf_loader.py:16) — OCR fallback trigger
  - `_MAX_WORDS = 500` (src/preprocessing.py:24) — truncation limit
  - `col_tol = max(60.0, lh * 5)` (src/information_extraction.py) — spatial tolerance for column alignment
- **Issue:** No documentation for how these values were chosen
- **Risk:** LOW (project-specific tuning), but makes future tuning harder
- **Recommendation:** Add comments explaining rationale (e.g., "4 signals = precision 95% on validation set")

---

## Recommendations for New Code

### Python
1. ✅ **Use snake_case for all variables/functions, _snake_case for private, SCREAMING_SNAKE_CASE for constants**
2. ✅ **Always use type hints (leverage `from __future__ import annotations`)**
3. ✅ **Organize imports: stdlib → external → internal, alphabetical within groups**
4. ✅ **Use pathlib.Path for file paths, not strings**
5. ✅ **Use context managers (`with`) for file I/O**
6. ⚠️ **Add docstrings for public functions** (currently missing in most files)
7. ✅ **Raise HTTPException in FastAPI routes, RuntimeError for internal errors**
8. ⚠️ **Document magic numbers with inline comments**

### JavaScript/React
1. ✅ **Use camelCase for variables/functions, PascalCase for components**
2. ✅ **Use `useState` for component state, avoid global state**
3. ✅ **Name event handlers with `handle` prefix**
4. ✅ **Use inline styles for theme-specific properties, Tailwind for layout**
5. ⚠️ **Fetch configuration from API instead of hardcoding** (e.g., model list)
6. ✅ **Use Fetch API for HTTP requests, FormData for file uploads**

### Cross-Language
1. ⚠️ **Keep API contracts in sync** — if backend changes response shape, update frontend normalization logic
2. ⚠️ **Version API responses** — consider adding `version` field to `/classify` response for future-proofing
3. ✅ **Use consistent field names across layers** (e.g., `invoice_number` in extraction → service → frontend)