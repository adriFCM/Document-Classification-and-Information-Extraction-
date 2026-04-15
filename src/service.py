"""FastAPI microservice: PDF in, classification + optional extraction out.

Run with:
    uvicorn src.service:app --reload --port 8000

Endpoints:
    GET  /health      → liveness probe
    POST /classify    → multipart file=@doc.pdf
                        returns label, confidence, per-class probabilities,
                        and (if label == "invoice") extracted fields
    POST /extract     → multipart file=@invoice.pdf
                        skips classification, always runs extraction
                        (kept for backward compatibility / direct testing)

Models are loaded once at startup from models/ (created by notebook 04).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.information_extraction import extract_invoice_fields
from src.pdf_loader import pdf_to_text
from src.preprocessing import clean_for_classifier, _INV_SIGNALS

_MODELS_DIR = Path(__file__).parent.parent / 'models'


@lru_cache(maxsize=1)
def _load_models():
    tfidf_path = _MODELS_DIR / 'tfidf_vectorizer.joblib'
    clf_path   = _MODELS_DIR / 'lr_classifier.joblib'
    if not tfidf_path.exists() or not clf_path.exists():
        raise RuntimeError(
            f'Model files not found in {_MODELS_DIR}. '
            'Run notebook 04_classification.ipynb first to train and save the models.'
        )
    return joblib.load(tfidf_path), joblib.load(clf_path)


_INV_OVERRIDE_THRESHOLD = 4   # number of distinct invoice signals needed to override

def _count_invoice_signals(text: str) -> int:
    """Count how many distinct invoice indicator patterns fire on raw text."""
    return sum(1 for _, pattern in _INV_SIGNALS if pattern.search(text))


def _predict_proba(clf, X) -> np.ndarray:
    """Return per-class probability array for any sklearn classifier.

    Models with predict_proba (LR, NB, RF, CalibratedSVC) use it directly.
    LinearSVC only exposes decision_function; we apply softmax to get a
    calibrated-enough confidence score suitable for display.
    """
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)[0]
    scores = clf.decision_function(X)[0]          # shape (n_classes,)
    scores = scores - scores.max()                # numerical stability
    exp    = np.exp(scores)
    return exp / exp.sum()


app = FastAPI(title='Document Classification & Information Extraction', version='0.2.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/classify')
async def classify(file: UploadFile = File(...)):
    """Classify an uploaded PDF and extract fields if it is an invoice."""
    if not (file.filename or '').lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='file must be a .pdf')
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail='empty upload')

    try:
        raw_text = pdf_to_text(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f'could not read pdf: {e}')

    tfidf, clf = _load_models()
    cleaned    = clean_for_classifier(raw_text)
    X          = tfidf.transform([cleaned])
    label      = clf.predict(X)[0]
    proba_arr  = _predict_proba(clf, X)
    proba      = {cls: round(float(p), 4) for cls, p in zip(clf.classes_, proba_arr)}
    confidence = round(float(proba[label]), 4)

    # ── Invoice keyword override ──────────────────────────────────────────────
    # If the model is uncertain AND the raw text fires enough hard invoice
    # signals, override to invoice. This catches borderline cases (e.g. an
    # invoice that reads like an email) where the model under-predicts invoice
    # due to the small training class.
    inv_signals = _count_invoice_signals(raw_text)
    if label != 'invoice' and inv_signals >= _INV_OVERRIDE_THRESHOLD:
        label      = 'invoice'
        confidence = round(float(proba.get('invoice', 0.0)), 4)

    response = {
        'filename':    file.filename,
        'label':       label,
        'confidence':  confidence,
        'proba':       proba,
        'inv_signals': inv_signals,   # useful for debugging during demo
    }

    if label == 'invoice':
        response['invoice_fields'] = extract_invoice_fields(raw_text)

    return response


@app.post('/extract')
async def extract(file: UploadFile = File(...)):
    """Extract invoice fields from a PDF without running classification."""
    if not (file.filename or '').lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='file must be a .pdf')
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail='empty upload')
    try:
        text = pdf_to_text(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f'could not read pdf: {e}')
    return {'filename': file.filename, 'fields': extract_invoice_fields(text)}
