# Document-Classification-and-Information-Extraction-

**IE University — AI: Statistical Learning and Prediction**

This project classifies documents into four categories and extracts invoice fields using classic machine learning, rule based parsing, and OCR fallback. It does not use LLM models.

## Project Overview

The pipeline has four parts:

1. Collect the raw datasets.
2. Clean and preprocess the text.
3. Train and evaluate classification models.
4. Extract invoice fields from PDFs and expose the workflow through a FastAPI backend and a Next.js frontend.

The project currently includes these document classes:

- news
- email
- contract
- invoice

## Project Structure

```
├── data/
│   ├── raw/          ← downloaded source datasets
│   ├── processed/    ← cleaned datasets

├── docs/
│   └── superpowers/   ← design/spec notes for the invoice extraction redesign
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_classification.ipynb
│   ├── 05_information_extraction.ipynb
│   └── ...
├── scripts/
│   ├── eval_invoices.py
│   └── eval_labeled.py
├── src/
│   ├── pdf_loader.py
│   ├── preprocessing.py
│   ├── information_extraction.py
│   └── service.py
├── frontend/
│   ├── app/
│   ├── package.json
│   └── ...
├── models/
├── requirements.txt
├── setup.py
└── README.md
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR-USERNAME/Document-Classification-and-Information-Extraction-.git
cd Document-Classification-and-Information-Extraction-
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it based on your OS/shell:

```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Create the folder structure

```bash
python setup.py
```

### 5. Install the OCR binary (Tesseract)

The invoice OCR fallback uses Tesseract. Install it for your OS:

```bash
# macOS (Homebrew)
brew install tesseract

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr

# Windows (Chocolatey)
choco install tesseract
```

If you install it manually on Windows, make sure the Tesseract install path is added to your PATH.

### 6. Set up Kaggle access

Some dataset downloads in [notebooks/01_data_collection.ipynb](notebooks/01_data_collection.ipynb) use the Kaggle API or `kagglehub`.

- Create a Kaggle API token in your Kaggle account settings.
- Windows path for `kaggle.json`: `%USERPROFILE%\\.kaggle\\kaggle.json`
- macOS/Linux path for `kaggle.json`: `~/.kaggle/kaggle.json`

## How to Run

### Notebook order

Run the notebooks in this order:

1. [notebooks/01_data_collection.ipynb](notebooks/01_data_collection.ipynb) — download raw datasets.
2. [notebooks/02_EDA.ipynb](notebooks/02_EDA.ipynb) — inspect the datasets and basic patterns.
3. [notebooks/03_preprocessing.ipynb](notebooks/03_preprocessing.ipynb) — clean the data and build the merged labeled dataset.
4. [notebooks/04_classification.ipynb](notebooks/04_classification.ipynb) — train and evaluate classifiers.
5. [notebooks/05_information_extraction.ipynb](notebooks/05_information_extraction.ipynb) — evaluate invoice field extraction.

### Backend service

Start the FastAPI backend with:

```bash
uvicorn src.service:app --reload --port 8000
```

The backend exposes:

- `GET /health`
- `GET /models`
- `POST /classify`
- `POST /extract`

`POST /classify` accepts a PDF, predicts one of the four document classes, and returns invoice fields when the document is classified as an invoice.

`POST /extract` skips classification and extracts invoice fields directly.

### Frontend app

The frontend is a separate Next.js app in [frontend](frontend).

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` after the backend is running on port 8000.

## Quick Start (Cross-Platform)

1. Create and activate your virtual environment.
2. Install Python dependencies with `pip install -r requirements.txt`.
3. Run `python setup.py`.
4. Run the notebooks in order.
5. Start backend: `uvicorn src.service:app --reload --port 8000`.
6. Start frontend from [frontend](frontend): `npm install && npm run dev`.

## Datasets

### BBC News

- Source: University College Dublin
- Format: plain text files grouped by topic
- Use: news classification

### Enron Emails

- Source: Kaggle `marcelwiechmann/enron-spam-data`
- Format: CSV
- Use: email classification

### CUAD Contracts

- Source: Kaggle `konradb/atticus-open-contract-dataset-aok-beta`
- Format: CSV
- Use: contract classification

### Invoice data

- Source: Kaggle invoice datasets and local invoice extracts
- Format: the current preprocessing notebook reads `data/raw/invoices/converted_invoice_dataset.xlsx`
- Use: invoice classification and field extraction

## Final Training Dataset

The merged labeled dataset used by the classifier is written to `data/processed/full_dataset_preprocessed.csv`.

It contains four labels:

| Label | Count |
| --- | ---: |
| news | 2,119 |
| email | 1,222 |
| invoice | 946 |
| contract | 496 |
| Total | 4,783 |

## Invoice Extraction

The invoice extractor is rule-based and uses:

- raw PDF text
- `pdfplumber` for the fast path
- `pypdfium2` + `pytesseract` for scanned PDFs
- regex and layout heuristics for field extraction

It extracts these fields:

- `invoice_number`
- `invoice_date`
- `due_date`
- `issuer`
- `recipient`
- `total`

The extractor is evaluated on SROIE-style receipts, with only the overlapping fields reported against the benchmark ground truth.

## Notes

- `requirements.txt` covers the Python stack used by the notebooks, scripts, and backend.
- Frontend dependencies are managed separately in `frontend/package.json`.
- OCR requires the Tesseract system binary in addition to Python packages.

---
