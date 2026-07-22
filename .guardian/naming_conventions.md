# Naming Patterns & Code Conventions

This document captures the established naming patterns and code style conventions used throughout the project. All examples reference actual code locations.

## Component & Class Naming

**Pattern**: PascalCase for React components and dataclasses

**Examples**:
- `RootLayout` — React component (frontend/app/layout.js:15)
- `Home` — React component (frontend/app/page.js:46)
- No Python classes defined in backend (functional style preferred)

**Consistency**: ✅ Uniform across codebase

---

## File & Directory Naming

**Pattern**: snake_case for Python modules, kebab-case for docs, descriptive names

**Python modules**:
- `src/service.py` — FastAPI app
- `src/pdf_loader.py` — PDF extraction utilities
- `src/preprocessing.py` — Text cleaning
- `src/information_extraction.py` — Invoice field parsing
- `scripts/eval_invoices.py` — Batch evaluation script
- `scripts/eval_labeled.py` — Accuracy evaluation script

**JavaScript/React**:
- `frontend/app/page.js` — Main page component
- `frontend/app/layout.js` — Root layout
- `frontend/app/globals.css` — Global styles

**Documentation**:
- `docs/superpowers/plans/2026-04-16-invoice-extraction-redesign.md` — Plan document with date prefix
- `docs/superpowers/specs/2026-04-16-invoice-extraction-redesign-design.md` — Spec with date prefix

**Data directories**:
- `data/raw/` — Unprocessed datasets
- `data/processed/` — Cleaned datasets
- `models/linear_svc/` — Model artifacts with descriptive names

**Consistency**: ✅ Clear separation: snake_case for Python, kebab-case for docs, descriptive folder names

---

## Variable & Constant Naming

**Variables**: snake_case for local variables and function parameters

**Examples**:
- `pdf_bytes` (src/service.py:170)
- `raw_text` (src/service.py:176)
- `model_dir` (src/service.py:58)
- `file_path` (scripts/eval_invoices.py:14)
- `cleaned` (src/preprocessing.py:102)

**Constants**: UPPER_SNAKE_CASE for module-level constants

**Examples**:
- `_ALLOWED_EXTENSIONS` (src/service.py:23)
- `_MODELS_DIR` (src/service.py:25)
- `_INV_OVERRIDE_THRESHOLD` (src/service.py:43)
- `_MIN_TEXT_CHARS` (src/pdf_loader.py:20)
- `_MAX_WORDS` (src/preprocessing.py:19)
- `DATE_RE` (src/information_extraction.py:28)
- `FIELD_LABELS` (frontend/app/page.js:3)
- `API_URL` (frontend/app/page.js:21)

**Private constants** (module-internal): Prefixed with single underscore

**Examples**:
- `_PAGE_NUM_RE` (src/preprocessing.py:23)
- `_HYPHEN_BREAK_RE` (src/preprocessing.py:26)
- `_INV_SIGNALS` (src/preprocessing.py:29)
- `_MONTHS` (src/information_extraction.py:17)
- `_DATE_PATTERNS` (src/information_extraction.py:19)

**JavaScript constants**: UPPER_SNAKE_CASE or PascalCase for config objects

**Examples**:
- `FIELD_LABELS` (frontend/app/page.js:3)
- `CATEGORY_STYLES` (frontend/app/page.js:11)
- `USE_MOCK` (frontend/app/page.js:19)
- `API_URL` (frontend/app/page.js:22)
- `MODELS` (frontend/app/page.js:38)

**React state variables**: camelCase

**Examples**:
- `step`, `result`, `file`, `dragOver`, `tick`, `dark`, `barWidth`, `selectedModel`, `error` (frontend/app/page.js:47-55)

**Consistency**: ✅ Clear distinction between constants (UPPER) and variables (lower)

---

## Function & Method Naming

**Public functions**: snake_case, descriptive verb phrases

**Examples**:
- `pdf_to_text()` (src/pdf_loader.py:71)
- `image_to_text()` (src/pdf_loader.py:77)
- `image_to_words()` (src/pdf_loader.py:83)
- `clean_for_classifier()` (src/preprocessing.py:86)
- `extract_invoice_fields()` (src/information_extraction.py:main function)

**Private helper functions**: Prefixed with single underscore

**Examples**:
- `_open()` (src/pdf_loader.py:24)
- `_extract_with_pdfplumber()` (src/pdf_loader.py:28)
- `_extract_with_ocr()` (src/pdf_loader.py:39)
- `_model_type_for()` (src/service.py:47)
- `_load_named_model()` (src/service.py:51)
- `_count_invoice_signals()` (src/service.py:82)
- `_predict_proba()` (src/service.py:87)
- `_invoice_indicator_tokens()` (src/preprocessing.py:65)
- `_find_label_span()` (src/information_extraction.py)
- `_words_right_of()` (src/information_extraction.py)
- `_extract_invoice_number()` (src/information_extraction.py)
- `_validate_date()` (src/information_extraction.py)
- `_norm_date()` (scripts/eval_labeled.py:36)

**React components**: PascalCase

**Examples**:
- `RootLayout({ children })` (frontend/app/layout.js:15)
- `Home()` (frontend/app/page.js:46)

**Event handlers**: camelCase with `handle` prefix

**Examples**:
- `handleDrop(e)` (frontend/app/page.js:95)

**API endpoint handlers**: snake_case, noun-based

**Examples**:
- `health()` (src/service.py:117)
- `list_models()` (src/service.py:123)
- `classify()` (src/service.py:137)
- `extract()` (src/service.py:204)

**CRUD verb patterns**: Not applicable (no database operations)

**Async functions**: JavaScript async functions named like sync functions (no special prefix)

**Example**:
```javascript
async function classify() { ... }  // (frontend/app/page.js:76)
```

**Consistency**: ✅ Clear public/private distinction, descriptive verbs

---

## Type & Interface Naming

**TypeScript**: Not used (frontend is plain JavaScript)

**Python type hints**: Descriptive type aliases

**Example**:
- `PathLike = Union[str, Path, bytes]` (src/pdf_loader.py:22)

**Dataclass naming**: Not used (dictionaries preferred for data structures)

**Function signatures with type hints**:
```python
def pdf_to_text(path_or_bytes: PathLike) -> str:
def image_to_words(path_or_bytes: PathLike, min_conf: int = 30) -> list[dict]:
def clean_for_classifier(text: str) -> str:
def _load_named_model(model_key: str) -> tuple[str, Any, Any]:
```

**Consistency**: ✅ Type hints present but not exhaustive (mainly for public APIs)

---

## Error Handling Patterns

**FastAPI error handling**: Raise `HTTPException` with status codes

**Examples**:
```python
# 400 Bad Request
raise HTTPException(status_code=400, detail='file must be a .pdf, .jpg, or .png')
raise HTTPException(status_code=400, detail='empty upload')
raise HTTPException(status_code=400, detail=f'Unknown model: {model_key}')

# 422 Unprocessable Entity
raise HTTPException(status_code=422, detail=f'could not read file: {e}')

# 503 Service Unavailable
raise HTTPException(status_code=503, detail=f'Model "{model_key}" not found. Run notebook 04...')
```
(src/service.py:149, 152, 174, 181, 62, 69)

**RuntimeError for setup issues**:
```python
raise RuntimeError('pytesseract is not installed. Run: pip install pytesseract')
raise RuntimeError('Tesseract OCR binary not found on PATH. Install it: ...')
raise RuntimeError(f'No model files found in {_MODELS_DIR}. Run notebook 04...')
```
(src/pdf_loader.py:49, 61, src/service.py:56)

**Try/except in scripts**: Catch per-item, continue processing

**Example**:
```python
try:
    # process image
except Exception as e:
    print(f"  ERR {fname}: {e}")
    continue
```
(scripts/eval_labeled.py:95-99)

**React error handling**: Try/catch in async functions, set error state

**Example**:
```javascript
try {
    const response = await fetch(API_URL, {...})
    if (!response.ok) { throw new Error(detail) }
} catch (err) {
    setError(err.message || 'Something went wrong. Please try again.')
    setStep('upload')
}
```
(frontend/app/page.js:81-93)

**Gap**: No structured error logging (errors printed to console/stdout only)

**Consistency**: ✅ HTTP status codes appropriate, user-facing error messages clear

---

## Import Organization

**Python import order** (per PEP 8):
1. Future imports (`from __future__ import annotations`)
2. Standard library imports
3. External package imports
4. Internal project imports
5. Lazy imports (in functions when needed)

**Example (src/service.py)**:
```python
from __future__ import annotations        # 1. Future
from functools import lru_cache           # 2. Standard library
from pathlib import Path
import numpy as np                        # 3. External
import joblib
from fastapi import FastAPI, ...
from src.information_extraction import ...  # 4. Internal
from src.pdf_loader import ...
```

**Example (src/pdf_loader.py)**:
```python
from __future__ import annotations
import io
from pathlib import Path
from typing import Union
import pdfplumber
# Lazy imports in functions:
# import pypdfium2 as pdfium
# import pytesseract
# from PIL import Image
```

**JavaScript import order**:
- Directives first (`'use client'`)
- External packages (React)
- Internal modules (CSS)

**Example (frontend/app/page.js)**:
```javascript
'use client'
import { useState, useEffect } from 'react'
```

**Example (frontend/app/layout.js)**:
```javascript
import { Geist, Geist_Mono } from "next/font/google"
import "./globals.css"
```

**Consistency**: ✅ Standard order followed, lazy imports for heavy dependencies

---

## State Management Patterns

**Backend**: Stateless (no session state)

**Model caching**:
```python
@lru_cache(maxsize=8)
def _load_named_model(model_key: str):
    # Cached per model_key to avoid reloading on every request
```
(src/service.py:51)

**Frontend React state**: `useState` hooks for component-local state

**Example**:
```javascript
const [step, setStep] = useState('upload')
const [result, setResult] = useState(null)
const [file, setFile] = useState(null)
const [dark, setDark] = useState(true)
```
(frontend/app/page.js:47-55)

**State updates**: Functional updates for derived state

**Example**:
```javascript
setTick(n => n + 1)  // Incremental update
setDark(d => !d)     // Toggle
```
(frontend/app/page.js:67, 260)

**Derived state**: Computed values, not stored in state

**Example**:
```javascript
const cat = result ? (CATEGORY_STYLES[result.category] || CATEGORY_STYLES.invoice) : null
const catColor = cat ? (dark ? cat.color : cat.colorLight) : '#a3e635'
```
(frontend/app/page.js:103-104)

**No global state**: No Redux, Context API, or Zustand (single-component app)

**Consistency**: ✅ Clear state ownership, no prop drilling issues (single component)

---

## Event Handling Patterns

**React event handlers**: Inline arrow functions or named handlers

**Inline**:
```javascript
onClick={() => document.getElementById('file-input').click()}
onClick={() => setDark(d => !d)}
onClick={() => { if (file) setStep('processing') }}
```
(frontend/app/page.js:312, 260, 422)

**Named handler**:
```javascript
const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped?.type === 'application/pdf') setFile(dropped)
}
```
(frontend/app/page.js:95)

**Drag-and-drop pattern**:
```javascript
onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
onDragLeave={() => setDragOver(false)}
onDrop={handleDrop}
```
(frontend/app/page.js:308-310)

**File input trigger pattern**:
```javascript
// Hidden file input
<input id="file-input" type="file" accept=".pdf,.jpg,.jpeg,.png" style={{ display: 'none' }}
  onChange={(e) => { setFile(e.target.files[0]); setError(null) }} />
// Trigger on div click
onClick={() => document.getElementById('file-input').click()}
```
(frontend/app/page.js:343, 312)

**Consistency**: ✅ Standard React event handling, no jQuery-style manual listeners

---

## CSS & Styling Approach

**Frontend approach**: Tailwind CSS utility classes + CSS-in-JS inline styles

**Tailwind usage** (in layout.js):
```javascript
className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
className="min-h-full flex flex-col"
```
(frontend/app/layout.js:18, 20)

**CSS-in-JS** (in page.js):
- Extensive use of inline `style={{}}` objects
- Dynamic styles based on state (`dark`, `dragOver`, `selectedModel`)
- **Example**:
  ```javascript
  style={{
      backgroundColor: dark ? '#0a0a0a' : '#f5f5f5',
      color: dark ? '#ffffff' : '#0a0a0a',
      transition: 'all 0.2s'
  }}
  ```
  (frontend/app/page.js:111)

**Theme palette approach**:
```javascript
const t = {
    bg:         dark ? '#0a0a0a'  : '#f5f5f5',
    surface:    dark ? '#0f0f0f'  : '#ffffff',
    border:     dark ? '#1f1f1f'  : '#e0e0e0',
    text:       dark ? '#ffffff'  : '#0a0a0a',
    // ...
}
```
(frontend/app/page.js:58)

**CSS animations** (in `<style>` tag):
```css
@keyframes spin { to { transform: rotate(360deg) } }
@keyframes fadeIn { from { opacity: 0 } to { opacity: 1 } }
```
(frontend/app/page.js:115-119)

**CSS variables** (in globals.css):
```css
:root {
  --background: #ffffff;
  --foreground: #171717;
}
```
(frontend/app/globals.css:3-6)

**Consistency**: ⚠️ Mixed approach (Tailwind in layout, CSS-in-JS in page) — not problematic for small app, but inconsistent pattern. Recommendation: Prefer Tailwind for consistency, use CSS-in-JS only for highly dynamic styles.

---

## External Integration & Side-Effect Patterns

**Backend external integrations**:

**1. System binary calls** (Tesseract OCR):
```python
pytesseract.image_to_string(pil_image)
```
(src/pdf_loader.py:59)

**2. File I/O**:
```python
joblib.load(clf_path)  # Model loading
marker.read_text().strip()  # Config file reading
```
(src/service.py:75, 48)

**3. HTTP requests** (frontend → backend):
```javascript
const response = await fetch(API_URL, {
    method: 'POST',
    body: formData
})
```
(frontend/app/page.js:81)

**Lazy loading pattern** (defer imports until needed):
```python
# In function, not at module level
import pypdfium2 as pdfium
import pytesseract
from PIL import Image
```
(src/pdf_loader.py:41, 43, 48)

**Rationale**: Avoids startup delays if OCR not needed

**Consistency**: ✅ Side effects isolated in dedicated modules, lazy imports for heavy dependencies

---

## Internal Inconsistencies & Recommendations

### Inconsistency 1: Model file naming — legacy vs new

**Pattern A (legacy)**: Single-model files at root of `models/`
- `models/lr_classifier.joblib`
- `models/tfidf_vectorizer.joblib`
- `models/model_type.txt`

**Pattern B (current)**: Per-model subdirectories
- `models/logistic_regression/clf.joblib`
- `models/logistic_regression/tfidf.joblib`
- `models/logistic_regression/model_type.txt`

**Evidence**: src/service.py:54-57 has fallback logic for legacy files

**Risk**: LOW — Backward compatibility preserved, but dead code path once all notebooks updated

**Recommendation**: Remove legacy path after confirming all models migrated to subdirectory structure (deprecation notice in next release)

---

### Inconsistency 2: Date format in doc filenames

**Pattern**: `YYYY-MM-DD` prefix in design docs
- `docs/superpowers/plans/2026-04-16-invoice-extraction-redesign.md`
- `docs/superpowers/specs/2026-04-16-invoice-extraction-redesign-design.md`

**Issue**: Future date (2026) suggests placeholder or error

**Risk**: LOW — Doesn't affect functionality, but confusing

**Recommendation**: Use actual creation date or remove date prefix if not tracking doc versions

---

### Inconsistency 3: Frontend styling — Tailwind vs CSS-in-JS

**Pattern A**: Tailwind utility classes (frontend/app/layout.js)
```javascript
className="min-h-full flex flex-col"
```

**Pattern B**: Inline CSS-in-JS (frontend/app/page.js)
```javascript
style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}
```

**Risk**: MEDIUM — Mixed approach makes styling harder to maintain, no single source of truth

**Recommendation**: Standardize on Tailwind for static styles, reserve CSS-in-JS for dynamic (state-dependent) styles only. Create Tailwind utility classes for repeated patterns.

---

### Inconsistency 4: Error message format

**Pattern A**: User-facing messages (src/service.py:149)
```python
raise HTTPException(status_code=400, detail='file must be a .pdf, .jpg, or .png')
```

**Pattern B**: Technical messages (src/service.py:56)
```python
raise RuntimeError(f'No model files found in {_MODELS_DIR}. Run notebook 04_classification.ipynb first.')
```

**Issue**: RuntimeError exposes internal paths to end users (if propagated to API)

**Risk**: LOW — RuntimeError raised at startup, not during request handling

**Recommendation**: Consistent error message format: user-facing errors use HTTPException, setup errors use RuntimeError (current pattern is acceptable)

---

### Inconsistency 5: File extension validation

**Backend**: Checks file extension via string suffix (src/service.py:146)
```python
ext = Path(file.filename or '').suffix.lower()
if ext not in _ALLOWED_EXTENSIONS:
```

**Issue**: Extension-only check doesn't verify actual file format (user could rename .png to .pdf)

**Risk**: MEDIUM — Could cause parse errors downstream, but caught by try/except

**Recommendation**: Add MIME type validation or magic number check for production use (current approach acceptable for demo/academic project)

---

## Summary

**Strengths**:
- ✅ Consistent snake_case for Python, PascalCase for React components
- ✅ Clear public/private distinction with underscore prefix
- ✅ Standard import ordering (PEP 8)
- ✅ Descriptive variable and function names
- ✅ Type hints on public APIs
- ✅ Lazy imports for heavy dependencies

**Areas for improvement**:
- ⚠️ Remove legacy model loading path (technical debt)
- ⚠️ Standardize frontend styling (prefer Tailwind over CSS-in-JS for static styles)
- ⚠️ Fix future date in doc filenames (minor)

**Overall consistency rating**: 🟢 HIGH — Conventions are well-established and followed consistently across modules
