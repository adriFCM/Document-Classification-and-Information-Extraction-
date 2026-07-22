# Architecture & Design Strategy

## Project Purpose

This project is a **Document Classification and Information Extraction pipeline** for academic/research purposes (IE University — AI: Statistical Learning and Prediction course). It classifies PDF documents into four categories (invoice, contract, email, news) using classical machine learning models (no LLMs) and automatically extracts structured invoice fields from documents classified as invoices using rule-based parsing, OCR, and spatial analysis.

**Problem solved**: Automated document classification and invoice data extraction without requiring expensive LLM APIs, suitable for educational environments and research on classical ML approaches.

## Target Users

- **Primary users**: Students, researchers, and instructors analyzing document classification pipelines
- **Access method**: 
  - Web browser (Next.js frontend at http://localhost:3000)
  - Direct API calls (FastAPI backend at http://localhost:8000)
  - Jupyter notebooks for training and experimentation
- **Deployment**: Local development only (no production deployment configured)

## How It's Built

**Backend (Python FastAPI)**:
- Build: No build step required (Python interpreted)
- Run: `uvicorn src.service:app --reload --port 8000`
- Dev server: Hot-reload enabled via `--reload` flag
- Output: REST API endpoints

**Frontend (Next.js)**:
- Build: `npm run build` (in frontend/ directory)
- Dev server: `npm run dev` (runs on port 3000)
- Output: Static/SSR pages served by Next.js
- Production: `npm run start`

**Machine Learning Models**:
- Training: Via Jupyter notebooks (notebooks/05_classification.ipynb)
- Output: Serialized .joblib files in models/ directory

## Project Type

**Hybrid system**:
- **Backend**: REST API microservice (FastAPI)
- **Frontend**: Server-side rendered web application (Next.js 16.2.3 with React 19.2.4)
- **ML Pipeline**: Jupyter notebook-based training pipeline
- **Data Processing**: Batch processing scripts for dataset preparation and model evaluation

## Framework Stack

| Dependency | Version | Role |
|------------|---------|------|
| **Backend (Python)** |
| fastapi | >=0.110,<1 | REST API framework |
| uvicorn[standard] | >=0.30,<1 | ASGI server with hot-reload |
| python-multipart | >=0.0.9,<1 | File upload handling |
| pdfplumber | >=0.11,<1 | Fast PDF text extraction |
| pypdfium2 | >=4.30,<6 | PDF rasterization for OCR fallback |
| pytesseract | >=0.3.10,<0.4 | OCR wrapper (requires system Tesseract binary) |
| Pillow | >=10,<13 | Image processing |
| scikit-learn | >=1.4,<2 | ML classifier training and inference |
| sentence-transformers | >=3.0,<6 | SBERT embeddings for semantic model |
| joblib | >=1.3,<2 | Model serialization |
| pandas | >=2.2,<3 | Dataset manipulation |
| numpy | >=1.26,<3 | Numerical operations |
| xgboost | >=2.0,<4 | Optional classifier (not currently used in production) |
| **Frontend (JavaScript/Node.js)** |
| next | 16.2.3 | React framework with SSR |
| react | 19.2.4 | UI library |
| react-dom | 19.2.4 | React DOM rendering |
| tailwindcss | ^4 | Utility-first CSS framework |
| @tailwindcss/postcss | ^4 | Tailwind PostCSS integration |
| eslint | ^9 | Code linting |
| eslint-config-next | 16.2.3 | Next.js ESLint configuration |
| **Development Tools** |
| jupyterlab | >=4.0,<5 | Notebook environment |
| ipykernel | >=6.0,<7 | Jupyter kernel |
| kaggle | >=1.6,<3 | Dataset download API |
| kagglehub | >=0.3,<1 | Kaggle dataset interface |
| pytest | >=7,<9 | Testing framework |
| matplotlib | >=3.8,<4 | Data visualization |
| openpyxl | >=3.1,<4 | Excel file handling |

**External system dependencies**:
- Tesseract OCR binary (brew install tesseract / apt-get install tesseract-ocr)
- Kaggle API credentials (~/.kaggle/kaggle.json)

## Module Structure

```
├── data/                        [Not committed - created by setup.py]
│   ├── raw/                     ← Downloaded datasets (BBC news, Enron emails, CUAD contracts, invoices)
│   └── processed/               ← Cleaned, merged datasets + evaluation results
├── docs/
│   └── superpowers/             ← Design/spec documents for invoice extraction redesign
│       ├── plans/2026-04-16-invoice-extraction-redesign.md
│       └── specs/2026-04-16-invoice-extraction-redesign-design.md
├── frontend/                    [Next.js web application]
│   ├── app/
│   │   ├── layout.js            ← Root layout with Google Fonts (Geist Sans/Mono)
│   │   ├── page.js              ← Main UI: upload, model selector, result display
│   │   ├── globals.css          ← Tailwind imports + CSS variables
│   │   └── favicon.ico          ← Browser icon
│   ├── public/                  ← Static assets (SVG icons)
│   ├── package.json             ← Frontend dependencies
│   ├── next.config.mjs          ← Next.js configuration (empty)
│   ├── postcss.config.mjs       ← PostCSS configuration
│   ├── jsconfig.json            ← JavaScript path aliases
│   ├── eslint.config.mjs        ← ESLint configuration
│   └── .gitignore               ← Node/Next.js ignore patterns
├── models/                      [Created by notebook 05 - serialized ML models]
│   ├── linear_svc/              ← Linear SVM classifier
│   │   ├── clf.joblib
│   │   ├── tfidf.joblib
│   │   └── model_type.txt
│   ├── logistic_regression/     ← Logistic Regression classifier (default)
│   │   ├── clf.joblib
│   │   ├── tfidf.joblib
│   │   └── model_type.txt
│   ├── sbert_logreg/            ← SBERT + Logistic Regression (semantic)
│   │   ├── clf.joblib
│   │   ├── model_type.txt
│   │   └── sbert_model_name.txt
│   └── [Legacy files: lr_classifier.joblib, model_type.txt]
├── notebooks/                   [Jupyter notebooks - executed in order]
│   ├── 01_data_collection.ipynb ← Download BBC news, Enron emails, CUAD contracts, invoice datasets
│   ├── 02_EDA.ipynb             ← Exploratory data analysis
│   ├── 03_preprocessing.ipynb   ← Text cleaning, dataset merging (→ full_dataset_preprocessed.csv)
│   ├── 04_information_extraction.ipynb ← Invoice field extraction evaluation
│   └── 05_classification.ipynb  ← Train classifiers, save to models/
├── scripts/                     [Batch evaluation tools]
│   ├── eval_invoices.py         ← Batch extraction on unlabeled invoice images
│   └── eval_labeled.py          ← Extraction accuracy vs ground-truth JSON
├── src/                         [Python backend source]
│   ├── __init__.py              ← Empty package marker
│   ├── service.py               ← FastAPI app: /health, /models, /classify, /extract endpoints
│   ├── pdf_loader.py            ← PDF text extraction (pdfplumber + OCR fallback)
│   ├── preprocessing.py         ← Text cleaning for classifier (matches training pipeline)
│   └── information_extraction.py ← Invoice field extraction (spatial + regex)
├── requirements.txt             ← Python dependencies
├── setup.py                     ← Creates data/ folder structure
├── .gitignore                   ← Python/data ignore patterns
└── README.md                    ← Project documentation
```

## State Management

**Frontend (Next.js/React)**:
- **Approach**: React hooks (`useState`, `useEffect`) — no external state library
- **State location**: Local component state in `frontend/app/page.js`
- **Key state variables**:
  - `step` (string): 'upload' | 'processing' | 'result' (controls UI screen)
  - `result` (object | null): Classification response from backend
  - `file` (File | null): Selected PDF file
  - `dragOver` (boolean): Drag-and-drop visual feedback
  - `dark` (boolean): Dark mode toggle
  - `selectedModel` (string): Active classifier ('linear_svc', 'logistic_regression', 'sbert_logreg')
  - `error` (string | null): Error message display
  - `barWidth` (number): Confidence bar animation width
- **State flow**: User action → setState → re-render → API call (if needed) → setState with response
- **No persistence**: State resets on page reload (intentional for demo)

**Backend (FastAPI)**:
- **Stateless**: Each request is independent
- **Model caching**: `@lru_cache(maxsize=8)` on `_load_named_model()` prevents re-loading models between requests
- **No session state**: No user sessions or request history

## Data Persistence

**Training Data**:
- **Location**: `data/raw/` (not committed to Git)
- **Format**: 
  - BBC News: Plain text files grouped by topic
  - Enron Emails: CSV
  - CUAD Contracts: CSV
  - Invoices: XLSX (converted_invoice_dataset.xlsx)
- **Read mechanism**: Pandas in Jupyter notebooks
- **Write mechanism**: Notebooks write to `data/processed/full_dataset_preprocessed.csv`

**Trained Models**:
- **Location**: `models/` directory (committed to Git)
- **Format**: joblib-serialized scikit-learn objects (.joblib files)
- **Read mechanism**: `joblib.load()` in `src/service.py` via `_load_named_model()`
- **Write mechanism**: `joblib.dump()` in notebook 05
- **Failure behavior**: Server returns 503 HTTP error if model files missing

**Uploaded Files**:
- **Location**: In-memory only (bytes read via `await file.read()`)
- **No disk persistence**: Files are not saved; processed and discarded per request

**Limitations**:
- No database
- No file storage beyond local filesystem
- No user accounts or authentication
- Training data must be manually downloaded via notebooks

## Routing

**Backend (FastAPI)**:
- **Mechanism**: Decorator-based route registration (`@app.get`, `@app.post`)
- **Routes**:
  - `GET /health` → `health()` — Liveness check, returns `{"status": "ok"}`
  - `GET /models` → `list_models()` — Returns available model metadata
  - `POST /classify` → `classify(file, model)` — Classify PDF + extract if invoice
  - `POST /extract` → `extract(file)` — Direct invoice extraction (skips classification)
- **CORS**: Enabled for http://localhost:3000 (frontend origin)

**Frontend (Next.js)**:
- **Mechanism**: App Router (Next.js 16.x) with file-system routing
- **Routes**:
  - `/` → `frontend/app/page.js` (single-page app, no sub-routes)
- **Client-side navigation**: None (single page only)

## Styling

**Frontend**:
- **Approach**: Tailwind CSS v4 utility classes + inline CSS-in-JS
- **Evidence**:
  - `frontend/package.json` includes `"tailwindcss": "^4"` and `"@tailwindcss/postcss": "^4"`
  - `frontend/app/globals.css` imports `@import "tailwindcss"`
  - `frontend/app/layout.js` applies Tailwind utility classes (`h-full`, `antialiased`, `min-h-full`, `flex`, `flex-col`)
  - `frontend/app/page.js` uses extensive inline `style={{}}` objects for component-specific styling (gradients, animations, responsive layouts)
- **Theme**: Dynamic light/dark mode toggle stored in component state (`dark` boolean), CSS variables derived from state
- **Fonts**: Google Fonts (Geist Sans, Geist Mono) loaded via `next/font/google`

## API/External Communication

**Frontend → Backend**:
- **Endpoint**: `POST http://localhost:8000/classify`
- **Method**: `fetch()` with FormData (multipart/form-data)
- **Payload**: `file` (PDF bytes), `model` (string)
- **Response**: JSON with `{label, confidence, proba, invoice_fields, model_used, inv_signals}`
- **Error handling**: Try/catch → sets `error` state → displays in UI
- **Mock mode**: `USE_MOCK = false` flag in `page.js` can simulate backend for UI development

**Backend → External Services**:
- **None at runtime** — Backend is fully self-contained
- **Training-time external access**:
  - Kaggle API: `notebooks/01_data_collection.ipynb` downloads datasets via `kaggle` and `kagglehub` packages
  - Requires `~/.kaggle/kaggle.json` credentials

**Backend → System Binaries**:
- **Tesseract OCR**: `pytesseract.image_to_string()` calls system `tesseract` binary for scanned PDF extraction
- **Failure mode**: Raises `RuntimeError` if Tesseract not found on PATH

## Key Architectural Decisions

1. **Classical ML over LLMs**: Explicitly avoids large language models to focus on traditional ML techniques (TF-IDF, logistic regression, SVM) suitable for academic study and low-resource environments.

2. **Two-stage pipeline**: Classification first, then conditional extraction. Invoice extraction is only triggered when document is classified as "invoice", saving compute.

3. **OCR fallback strategy**: Fast path (pdfplumber) for digital PDFs, automatic fallback to OCR (pypdfium2 + Tesseract) for scanned documents when text extraction yields <40 characters.

4. **Spatial extraction over pure regex**: Invoice fields are located using 2D bounding-box analysis (column-aware anchoring) rather than flat text regex, recovering layout structure lost in text-only extraction.

5. **Model hot-swapping**: Backend supports multiple pre-trained classifiers (Linear SVM, Logistic Regression, SBERT+LogReg) selected at request time via `model` parameter, enabling A/B testing without redeployment.

6. **Invoice signal override**: If 4+ invoice-specific patterns fire (e.g., "Invoice No.", "Bill To:", "Due Date"), the classifier's prediction is overridden to "invoice" even if confidence is low, catching edge cases.

7. **Indicator token injection**: Text preprocessing prepends synthetic tokens (`__inv_number__`, `__bill_to__`) to cleaned text before TF-IDF vectorization, amplifying weak signals for the minority invoice class.

8. **Stateless API design**: No user sessions, no request history, every call is independent. Models are cached per process, not per user.

9. **Separation of frontend and backend**: Next.js frontend and FastAPI backend are fully decoupled services communicating via HTTP, deployable independently.

10. **Notebook-driven training**: ML models are trained interactively in Jupyter notebooks (not automated scripts), prioritizing explorability and reproducibility for educational contexts.

## Constraints & Limitations

**What the project does NOT do**:
- **No production deployment**: No Docker, no cloud infrastructure, no CI/CD
- **No authentication/authorization**: Anyone with access to localhost can use the service
- **No data persistence**: Uploaded files are not saved, no database, no user history
- **No multi-page document handling**: Each PDF is treated as a single document
- **No batch processing API**: Only single-file uploads (batch processing exists in scripts, not API)
- **No real-time collaboration**: Single-user experience only
- **No model retraining API**: Models must be retrained via notebooks, not exposed as API
- **No field validation**: Extracted invoice fields are not validated against business rules
- **No multi-language support**: English-only (regex patterns, stopwords, training data)
- **No mobile app**: Web-only interface
- **No document editing/annotation**: Read-only document processing
- **No export formats**: Results displayed in UI only, no CSV/JSON download
- **No search/indexing**: No document database or search functionality
- **No version control for models**: Models overwrite previous versions when retrained
- **No A/B testing framework**: Model comparison must be done manually
- **No monitoring/logging**: No request logging, error tracking, or performance metrics
- **No rate limiting**: No protection against abuse
- **No HTTPS**: Local HTTP only
- **No email notifications**: No alerts or results delivery
- **Contract/email/news extraction**: Only invoices have field extraction; other document types are classified but not parsed
- **No LLM integration**: Explicitly excludes GPT/Claude/etc.
- **No incremental learning**: Models must be fully retrained on entire dataset
- **No explainability**: No SHAP/LIME explanations for predictions
- **Tesseract dependency**: OCR requires manual installation of system binary
- **Kaggle API dependency**: Dataset download requires Kaggle account and API token

## Build & Deployment

**Setup (First Time)**:
```bash
# 1. Clone repository
git clone <repo-url>
cd Document-Classification-and-Information-Extraction-

# 2. Create Python virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Create data folder structure
python setup.py

# 5. Install Tesseract OCR (system dependency)
brew install tesseract  # macOS
# sudo apt-get install tesseract-ocr  # Ubuntu/Debian
# choco install tesseract  # Windows

# 6. Set up Kaggle API credentials
# Place kaggle.json in ~/.kaggle/ (Linux/macOS) or %USERPROFILE%\.kaggle\ (Windows)

# 7. Run notebooks in order (01 → 05) to download data and train models
# Start JupyterLab: jupyter lab

# 8. Install frontend dependencies
cd frontend
npm install
```

**Running the Application**:
```bash
# Terminal 1: Start backend
source venv/bin/activate
uvicorn src.service:app --reload --port 8000

# Terminal 2: Start frontend
cd frontend
npm run dev
# Opens at http://localhost:3000
```

**Building for Production** (not configured for deployment):
```bash
# Backend: No build step (Python interpreted)
# Frontend:
cd frontend
npm run build
npm run start  # Runs optimized production server
```

**Testing**:
```bash
# Python tests (framework installed, no tests written yet)
pytest

# Frontend linting
cd frontend
npm run lint
```

**Scripts** (from package.json):
- `npm run dev` — Development server with hot reload
- `npm run build` — Production build
- `npm run start` — Serve production build
- `npm run lint` — Run ESLint

**Data Pipeline** (manual, via notebooks):
1. `01_data_collection.ipynb` — Download datasets → `data/raw/`
2. `02_EDA.ipynb` — Explore data distributions
3. `03_preprocessing.ipynb` — Clean and merge → `data/processed/full_dataset_preprocessed.csv`
4. `04_information_extraction.ipynb` — Evaluate invoice extraction
5. `05_classification.ipynb` — Train models → `models/*/clf.joblib` + `tfidf.joblib`
