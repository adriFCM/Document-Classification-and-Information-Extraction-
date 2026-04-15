"""FastAPI microservice: PDF in, invoice fields out.

Run with:
    uvicorn src.service:app --reload --port 8000

Endpoints:
    GET  /health      → liveness probe
    POST /extract     → multipart `file=@invoice.pdf`, returns the 6 fields

This service is currently extraction-only; classification is expected to be
done upstream (the assignment only requires extraction from docs already
labeled as invoices). Wiring in the classifier is a one-liner once that model
is packaged — add a guard in `extract()` that rejects non-invoice PDFs.
"""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.information_extraction import extract_invoice_fields
from src.pdf_loader import pdf_to_text

app = FastAPI(title='Invoice Information Extraction', version='0.1.0')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/extract')
async def extract(file: UploadFile = File(...)):
    if not (file.filename or '').lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='file must be a .pdf')
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail='empty upload')
    try:
        text = pdf_to_text(pdf_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f'could not read pdf: {e}')
    fields = extract_invoice_fields(text)
    return {'filename': file.filename, 'fields': fields}
