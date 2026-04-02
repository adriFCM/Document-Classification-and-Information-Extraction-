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
│   ├── 01_data_collection.ipynb      ← download all datasets
│   └── 02_preprocessing.ipynb        ← clean and normalize all datasets
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
.\venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate         # Mac/Linux
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
notebooks/01_data_collection.ipynb   ← downloads all 4 datasets automatically
notebooks/02_preprocessing.ipynb     ← cleans and generates data/labeled/dataset.csv
```

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
