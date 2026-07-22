# Dependencies Map

## External Dependencies (Python Backend)

From `requirements.txt`:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.26,<3 | Numerical array operations for ML |
| pandas | >=2.2,<3 | Dataset loading and manipulation |
| scikit-learn | >=1.4,<2 | ML classifier training (TF-IDF, LogReg, SVM) |
| scipy | >=1.11,<2 | Scientific computing (dependency of scikit-learn) |
| joblib | >=1.3,<2 | Model serialization/deserialization |
| xgboost | >=2.0,<4 | Gradient boosting classifier (optional, not currently used) |
| sentence-transformers | >=3.0,<6 | SBERT embeddings for semantic classification |
| matplotlib | >=3.8,<4 | Data visualization in notebooks |
| openpyxl | >=3.1,<4 | Excel file reading (invoice dataset) |
| pdfplumber | >=0.11,<1 | Fast PDF text extraction |
| pypdfium2 | >=4.30,<6 | PDF page rasterization for OCR |
| pytesseract | >=0.3.10,<0.4 | Tesseract OCR Python wrapper |
| Pillow | >=10,<13 | Image processing |
| fastapi | >=0.110,<1 | REST API framework |
| uvicorn[standard] | >=0.30,<1 | ASGI web server |
| python-multipart | >=0.0.9,<1 | Multipart form data parsing for file uploads |
| kaggle | >=1.6,<3 | Kaggle dataset download API |
| kagglehub[pandas-datasets] | >=0.3,<1 | Kaggle dataset interface |
| jupyterlab | >=4.0,<5 | Notebook IDE |
| ipykernel | >=6.0,<7 | Jupyter kernel for Python |
| pytest | >=7,<9 | Testing framework |

**External system dependencies**:
- Tesseract OCR binary (not a Python package)

## External Dependencies (Frontend)

From `frontend/package.json`:

| Package | Version | Purpose |
|---------|---------|---------|
| next | 16.2.3 | React framework with SSR |
| react | 19.2.4 | UI component library |
| react-dom | 19.2.4 | React DOM rendering |
| @tailwindcss/postcss | ^4 (dev) | Tailwind CSS PostCSS plugin |
| tailwindcss | ^4 (dev) | Utility-first CSS framework |
| eslint | ^9 (dev) | JavaScript linter |
| eslint-config-next | 16.2.3 (dev) | Next.js ESLint rules |

## Internal Import Graph

### Backend (`src/`)

**src/__init__.py** (0 imports, 0 exports)
- Empty package marker

**src/service.py**
- **Imports**:
  - Standard library: `from __future__ import annotations`, `from functools import lru_cache`, `from pathlib import Path`
  - External: `import numpy as np`, `import joblib`, `from fastapi import FastAPI, File, Form, HTTPException, UploadFile`, `from fastapi.middleware.cors import CORSMiddleware`
  - Internal: `from src.information_extraction import extract_invoice_fields`, `from src.pdf_loader import pdf_to_text, image_to_text`, `from src.preprocessing import clean_for_classifier, _INV_SIGNALS`
  - Conditional: `from sentence_transformers import SentenceTransformer` (lazy import in `_load_named_model()`)
- **Exports**: `app` (FastAPI instance), functions: `health()`, `list_models()`, `classify()`, `extract()`
- **Depends on**: `src.information_extraction`, `src.pdf_loader`, `src.preprocessing`

**src/pdf_loader.py**
- **Imports**:
  - Standard library: `from __future__ import annotations`, `import io`, `from pathlib import Path`, `from typing import Union`
  - External: `import pdfplumber`, `import pypdfium2 as pdfium` (lazy), `import pytesseract` (lazy), `from PIL import Image` (lazy)
- **Exports**: Functions: `pdf_to_text(path_or_bytes)`, `image_to_text(path_or_bytes)`, `image_to_words(path_or_bytes, min_conf)`, `image_to_full_text(words)`
- **Type alias**: `PathLike = Union[str, Path, bytes]`
- **Depends on**: No internal imports (leaf module)

**src/preprocessing.py**
- **Imports**:
  - Standard library: `from __future__ import annotations`, `import re`, `from typing import Union`
  - Internal (lazy): `from src.pdf_loader import pdf_to_text` (only in `process_pdf()`)
- **Exports**: 
  - Functions: `clean_for_classifier(text)`, `process_pdf(path_or_bytes)` (demo helper, not used in service)
  - Module-level constants: `_INV_SIGNALS` (list of tuple[str, re.Pattern])
- **Depends on**: `src.pdf_loader` (lazy import)

**src/information_extraction.py**
- **Imports**:
  - Standard library: `from __future__ import annotations`, `import io`, `import re`, `from dataclasses import asdict, dataclass`, `from typing import Optional, Sequence`
  - External: `import pdfplumber`
- **Exports**: 
  - Main function: `extract_invoice_fields(text, pdf_bytes=None, image_bytes=None, words=None) -> dict`
  - Helper functions (private, but callable): `_extract_invoice_number()`, `_extract_date()`, `_extract_issuer()`, `_extract_recipient()`, `_extract_total()`, etc.
  - Regex constants: `DATE_RE`, `_AMOUNT_RE`, `_INV_NUMBER_VALUE_RE`, `_STOP_BLOCK_RE`
- **Depends on**: No internal imports (leaf module)

### Frontend (`frontend/app/`)

**frontend/app/layout.js**
- **Imports**: `import { Geist, Geist_Mono } from "next/font/google"`, `import "./globals.css"`
- **Exports**: `export const metadata = {...}`, `export default function RootLayout({ children })`
- **Depends on**: `./globals.css`

**frontend/app/page.js**
- **Imports**: `'use client'`, `import { useState, useEffect } from 'react'`
- **Exports**: `export default function Home()`
- **Depends on**: No internal imports (leaf component)

**frontend/app/globals.css**
- **Imports**: `@import "tailwindcss"`
- **Exports**: CSS variables and styles
- **Depends on**: Tailwind CSS package

## Dependency Direction

**Backend**:

**Leaf modules** (import nothing internal):
- `src/pdf_loader.py` — Pure utility, no internal dependencies
- `src/information_extraction.py` — Pure utility, no internal dependencies

**Hub modules** (imported by multiple files):
- `src/pdf_loader.py` — Imported by `src.service` and `src.preprocessing`
- `src.preprocessing.py` — Imports from `src.pdf_loader`, imported by `src.service`
- `src.information_extraction.py` — Imported by `src.service`

**Entry point**:
- `src/service.py` — Imports all other `src/` modules, defines FastAPI app (never imported by others)

**Import hierarchy**:
```
src/service.py (entry point)
    ├── src/pdf_loader.py (leaf)
    ├── src/preprocessing.py
    │   └── src/pdf_loader.py (leaf)
    └── src/information_extraction.py (leaf)
```

**Frontend**:

**Leaf modules**:
- `frontend/app/page.js` — Main component, no internal imports
- `frontend/app/globals.css` — Styles only

**Hub modules**:
- `frontend/app/layout.js` — Root layout, imported by Next.js app router
- `frontend/app/globals.css` — Imported by layout.js

**Import hierarchy**:
```
frontend/app/layout.js (root layout)
    └── frontend/app/globals.css
Next.js App Router renders layout.js with page.js as children
```

## Shared Interfaces

**Cross-module types** (critical for consistency):

### Backend

**1. Invoice field dictionary shape** (shared contract):
- **Defined implicitly in**: `src/information_extraction.py` (returned by `extract_invoice_fields()`)
- **Consumed by**: `src/service.py` (in `/classify` and `/extract` endpoints)
- **Shape**:
  ```python
  {
      "invoice_number": str | None,
      "invoice_date": str | None,
      "due_date": str | None,
      "issuer": str | None,
      "recipient": str | None,
      "total": str | None
  }
  ```
- **Risk**: If extraction function changes field names or adds fields, service.py response shape changes

**2. Word bounding box dictionary** (spatial extraction):
- **Defined in**: `src/pdf_loader.py` (`image_to_words()` return type)
- **Consumed by**: `src/information_extraction.py` (all spatial extraction functions)
- **Shape**:
  ```python
  {
      "text": str,
      "x0": float,
      "x1": float,
      "top": float,
      "bottom": float
  }
  ```
- **Also produced by**: `pdfplumber.Page.extract_words()` (external contract)
- **Risk**: If pdfplumber changes word dict keys, spatial extraction breaks

**3. Model directory structure contract**:
- **Defined implicitly by**: `notebooks/05_classification.ipynb` (writes model files)
- **Consumed by**: `src/service.py` (`_load_named_model()` function)
- **Structure**:
  ```
  models/<model_key>/
      ├── clf.joblib          (scikit-learn classifier)
      ├── tfidf.joblib        (TF-IDF vectorizer, if model_type == 'tfidf')
      ├── model_type.txt      ('tfidf' or 'sbert')
      └── sbert_model_name.txt (only if model_type == 'sbert')
  ```
- **Risk**: If notebook changes file naming, service cannot load models

**4. Cleaned text format** (classifier input):
- **Produced by**: `src/preprocessing.py` (`clean_for_classifier()`)
- **Consumed by**: `src/service.py` (passes to TF-IDF vectorizer or SBERT encoder)
- **Contract**: 
  - Lowercased
  - Whitespace-normalized
  - Page numbers removed
  - Truncated to 500 words
  - Prefixed with indicator tokens (`__inv_number__`, etc.) if invoice patterns detected
- **Risk**: If preprocessing changes, must retrain models with same pipeline

**5. `_INV_SIGNALS` constant** (invoice pattern list):
- **Defined in**: `src/preprocessing.py`
- **Imported by**: `src/service.py` (for `_count_invoice_signals()` override logic)
- **Type**: `list[tuple[str, re.Pattern]]`
- **Risk**: If preprocessing module refactors this, service.py breaks

### Frontend

**6. Backend API response shape** (`/classify` endpoint):
- **Produced by**: `src/service.py` (`classify()` function)
- **Consumed by**: `frontend/app/page.js` (in `classify()` async function)
- **Shape**:
  ```javascript
  {
      filename: string,
      label: string,  // 'invoice' | 'contract' | 'email' | 'news'
      confidence: number,  // 0.0 to 1.0
      proba: { [label: string]: number },
      model_used: string,
      inv_signals: number,
      invoice_fields?: {  // Only present if label === 'invoice'
          invoice_number: string | null,
          invoice_date: string | null,
          due_date: string | null,
          issuer: string | null,
          recipient: string | null,
          total: string | null
      }
  }
  ```
- **Frontend normalization**: Raw response is transformed:
  ```javascript
  data = {
      category: raw.label,
      confidence: Math.round((raw.confidence ?? 0) * 100),
      fields: raw.invoice_fields ?? null,
      modelUsed: raw.model_used ?? selectedModel,
  }
  ```
- **Risk**: If backend changes `label` → `category` or `invoice_fields` → `fields`, frontend must update

**7. Model metadata shape** (`/models` endpoint):
- **Produced by**: `src/service.py` (`list_models()`)
- **Consumed by**: `frontend/app/page.js` (not currently used, future feature)
- **Shape**:
  ```javascript
  [
      {
          key: string,
          name: string,
          description: string,
          available: boolean
      }
  ]
  ```

## Notes

- **No TypeScript**: Frontend has no type checking; API contract violations discovered at runtime
- **Model files**: Must exist before service starts, or service raises RuntimeError
- **Field extraction independence**: Invoice extraction can be called standalone via `/extract` endpoint, bypassing classification
- **CORS dependency**: Frontend must run on `http://localhost:3000` or backend rejects requests
- **File extension contract**: Backend only accepts `.pdf`, `.jpg`, `.jpeg`, `.png` (checked via file extension)
