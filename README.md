# Document-Classification-and-Information-Extraction-
**IE University — AI: Statistical Learning and Prediction**

The goal of this project is to develop a technical solution capable of classifying documents into different categories and extracting specific information from invoices, using traditional Artificial Intelligence techniques and data processing methods, without employing generative AI models.

---

## 📁 Project Structure
```
├── data/
│   ├── raw/          ← downloaded datasets (not pushed to GitHub)
│   ├── processed/    ← cleaned text after preprocessing (one CSV per dataset)
│   └── labeled/      ← final combined CSV for model training
├── notebooks/
│   ├── 01_data_collection.ipynb        ← download all datasets
│   ├── 02_preprocessing.ipynb          ← clean and normalize all datasets
│   └── 03_information_extraction.ipynb ← invoice field extraction + SROIE eval
├── src/
│   ├── information_extraction.py ← extract_invoice_fields(text) -> dict
│   ├── pdf_loader.py             ← PDF → text, pdfplumber with OCR fallback
│   └── service.py                ← FastAPI microservice (GET /health, POST /extract)
├── setup.py          ← creates folder structure automatically
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR-USERNAME/Document-Classification-and-Information-Extraction-.git
cd Document-Classification-and-Information-Extraction-
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
#.\venv\Scripts\Activate.ps1    # Windows PowerShell
#.\venv\Scripts\activate        # Windows CMD
# source venv/bin/activate     # Mac/Linux     
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up Kaggle API
- Go to [kaggle.com](https://www.kaggle.com) → Settings → API → Create New Token
- This downloads `kaggle.json`
- Move it to the right place:
```bash
mkdir $env:USERPROFILE\.kaggle
move $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
```

### 5. Create folder structure
```bash
python setup.py
```

### 6. Run the notebooks in order
```
notebooks/01_data_collection.ipynb        ← downloads all 4 datasets automatically
notebooks/02_preprocessing.ipynb          ← cleans and generates data/labeled/dataset.csv
notebooks/03_information_extraction.ipynb ← invoice field extraction + SROIE eval
```

### 7. Run the extraction microservice (live demo)
```bash
# macOS only — required by pytesseract for the OCR fallback path
brew install tesseract

uvicorn src.service:app --port 8000
```
Then open `http://localhost:8000/docs` in a browser, expand `POST /extract`,
click **Try it out**, and upload any invoice PDF. The response is the six
extracted fields as JSON:
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
The same code handles digitally-generated PDFs (fast path via `pdfplumber`)
and scanned PDFs (OCR fallback via `pypdfium2` + `pytesseract`) without any
configuration change.

---

## 📊 Datasets

### 1. 📰 BBC News — 2,119 articles (after preprocessing)
- **Source:** University College Dublin (direct download)
- **Format:** Plain text files organized by subcategory
- **Why we chose it:** Clean, well-structured, plain text with no parsing issues. Covers business, entertainment, politics, sport and tech — all clearly distinguishable from the other document categories. Industry standard dataset for text classification.

### 2. 📧 Enron Emails — 1,222 emails (after preprocessing)
- **Source:** Kaggle (`marcelwiechmann/enron-spam-data`)
- **Format:** CSV with Subject, Message, Spam/Ham, Date columns
- **Why we chose it:** Emails have a very distinct structure (subject line, sender, body) that makes them easy to differentiate from other categories. The Enron dataset is the most well-known email dataset in the field. Sampled and filtered from 33,716 raw emails removing system generated and duplicate emails.

### 3. 📄 CUAD Contracts — 496 contracts (after preprocessing)
- **Source:** Kaggle (`konradb/atticus-open-contract-dataset-aok-beta`)
- **Format:** CSV with contract text and clause labels
- **Why we chose it:** Legal contracts have very distinctive language (numbered clauses, legal terminology, party definitions) that separates them clearly from emails, news and invoices. CUAD is the most widely used contract dataset in NLP research.

### 4. 🧾 SROIE Invoices — 946 invoices (after preprocessing)
- **Source:** Kaggle (`ryanznie/sroie-datasetv2-with-labels`)
- **Format:** Images (JPG) with pre-extracted text in box files and labeled entities (invoice number, date, total, issuer, recipient)
- **Why we chose it:** Highest usability rating (0.9375) among all SROIE versions on Kaggle. Already split into train/test sets and comes with ground truth labels for all extraction fields — making it ideal for both the classification and information extraction phases of the project. Uses real scanned receipts which reflects real-world invoice processing scenarios.

---

## 📊 Final Dataset
The final labeled dataset is at `data/labeled/dataset.csv` with **4,783 documents** and two columns:
- `text` — cleaned document text
- `label` — category (news, email, contract, invoice)

| Label | Count |
|---|---|
| news | 2,119 |
| email | 1,222 |
| invoice | 946 |
| contract | 496 |
| **Total** | **4,783** |

---

## 🧾 Information Extraction (Invoices)

Notebook 03 and the `src/` package implement rule-based extraction of six
fields from invoices classified upstream as `invoice`:

`invoice_number` · `invoice_date` · `due_date` · `issuer` · `recipient` · `total`

The extractor is pure regex + small heuristics (no generative AI, no
pretrained document-understanding models), per the assignment rules.

### Scores on SROIE 2019 (973 receipts)

SROIE only labels 3 of our 6 fields (`company` → `issuer`, `date` →
`invoice_date`, `total`). The other three fields (`invoice_number`,
`due_date`, `recipient`) have no SROIE ground truth and are validated
qualitatively on real invoice PDFs via the microservice.

| Field          | Exact  | Relaxed |
| -------------- | ------ | ------- |
| `invoice_date` | 95.9 % | 96.0 %  |
| `issuer`       | 61.5 % | 90.1 %  |
| `total`        | 64.4 % | 71.0 %  |

---
