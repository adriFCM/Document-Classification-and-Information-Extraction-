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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.information_extraction import extract_invoice_fields
from src.pdf_loader import pdf_to_text
from src.preprocessing import clean_for_classifier, _INV_SIGNALS

_MODELS_DIR = Path(__file__).parent.parent / 'models'

# Models available for selection in the UI.
# Each entry: display name, description, subdirectory name under models/
_SELECTABLE_MODELS = {
    'linear_svc': {
        'name':        'Linear SVM',
        'description': 'Highest F1 on training data. Fast, sparse features.',
    },
    'logistic_regression': {
        'name':        'Logistic Regression',
        'description': 'Calibrated probabilities. Robust generalisation.',
    },
    'sbert_logreg': {
        'name':        'SBERT + LogReg',
        'description': 'Semantic embeddings. Most robust on unseen vocabulary.',
    },
}


def _model_type_for(model_dir: Path) -> str:
    marker = model_dir / 'model_type.txt'
    return marker.read_text().strip() if marker.exists() else 'tfidf'


@lru_cache(maxsize=8)
def _load_named_model(model_key: str):
    """Load a named model from models/<model_key>/. Cached per key."""
    if model_key == 'auto':
        # Fall back to legacy single-model files
        clf_path = _MODELS_DIR / 'lr_classifier.joblib'
        if not clf_path.exists():
            raise RuntimeError(
                f'No model files found in {_MODELS_DIR}. '
                'Run notebook 04_classification.ipynb first.'
            )
        mtype = _model_type_for(_MODELS_DIR)
        model_dir = _MODELS_DIR
    else:
        if model_key not in _SELECTABLE_MODELS:
            raise HTTPException(status_code=400, detail=f'Unknown model: {model_key}')
        model_dir = _MODELS_DIR / model_key
        if not model_dir.exists():
            raise HTTPException(
                status_code=503,
                detail=f'Model "{model_key}" not found. Run notebook 04_classification.ipynb first.'
            )
        mtype = _model_type_for(model_dir)
        clf_path = model_dir / 'clf.joblib'

    if mtype == 'sbert':
        from sentence_transformers import SentenceTransformer
        name_file = model_dir / 'sbert_model_name.txt'
        sbert_name = name_file.read_text().strip() if name_file.exists() else 'all-MiniLM-L6-v2'
        return mtype, SentenceTransformer(sbert_name), joblib.load(clf_path)
    else:
        tfidf_path = model_dir / ('tfidf_vectorizer.joblib' if model_key == 'auto' else 'tfidf.joblib')
        if not tfidf_path.exists():
            raise RuntimeError(f'TF-IDF vectorizer not found: {tfidf_path}')
        return mtype, joblib.load(tfidf_path), joblib.load(clf_path)


_INV_OVERRIDE_THRESHOLD = 4   # distinct invoice signals needed to override prediction

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


@app.get('/models')
def list_models():
    """Return the models available for selection in the UI."""
    result = []
    for key, meta in _SELECTABLE_MODELS.items():
        available = (_MODELS_DIR / key).exists()
        result.append({
            'key':         key,
            'name':        meta['name'],
            'description': meta['description'],
            'available':   available,
        })
    return result


@app.post('/classify')
async def classify(
    file:  UploadFile = File(...),
    model: str        = Form('logistic_regression'),
):
    """Classify an uploaded PDF and extract fields if it is an invoice.

    Args:
        file:  The PDF to classify.
        model: One of 'linear_svc', 'logistic_regression', 'sbert_logreg', or 'auto'.
    """
    if not (file.filename or '').lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='file must be a .pdf')
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail='empty upload')

    try:
        raw_text = pdf_to_text(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f'could not read pdf: {e}')

    mtype, encoder, clf = _load_named_model(model)

    if mtype == 'sbert':
        import re as _re
        _ind_re = _re.compile(r'\b__\w+__\b')
        text_for_model = _ind_re.sub('', clean_for_classifier(raw_text)).strip()
        X = encoder.encode([text_for_model], show_progress_bar=False)
    else:
        cleaned = clean_for_classifier(raw_text)
        X       = encoder.transform([cleaned])
    label      = clf.predict(X)[0]
    proba_arr  = _predict_proba(clf, X)
    proba      = {cls: round(float(p), 4) for cls, p in zip(clf.classes_, proba_arr)}
    confidence = round(float(proba[label]), 4)

    # ── Invoice keyword override ──────────────────────────────────────────────
    # If 4+ distinct invoice signals fire on the raw text, force invoice even if
    # the model is uncertain (catches email-style invoices on borderline models).
    inv_signals = _count_invoice_signals(raw_text)
    if label != 'invoice' and inv_signals >= _INV_OVERRIDE_THRESHOLD:
        label      = 'invoice'
        confidence = round(float(proba.get('invoice', 0.0)), 4)

    model_meta = _SELECTABLE_MODELS.get(model, {'name': model})
    response = {
        'filename':    file.filename,
        'label':       label,
        'confidence':  confidence,
        'proba':       proba,
        'model_used':  model_meta['name'],
        'inv_signals': inv_signals,
    }

    if label == 'invoice':
        response['invoice_fields'] = extract_invoice_fields(raw_text, pdf_bytes=pdf_bytes)

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
    return {'filename': file.filename, 'fields': extract_invoice_fields(text, pdf_bytes=pdf_bytes)}
