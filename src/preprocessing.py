"""Classifier-side text preprocessing.

Mirrors the base cleaning pipeline used in `notebooks/03_preprocessing.ipynb`
and `notebooks/04_classification.ipynb` so that arbitrary PDFs passed to the
classifier at demo time see text with the same token distribution as the
training corpus.

Training pipeline (per notebook 03):
    lowercase → whitespace-normalise → artifact removal → 500-word truncate

Stopwords are removed by TfidfVectorizer(stop_words="english") at vectorisation
time, NOT by text preprocessing.  Punctuation, digits, and single-char tokens
are all kept — stripping them was done only in the old notebook 02 pipeline
and was never part of the current training run.

For the extractor (`src.information_extraction`) always use the *raw* text
before calling this function — cleaning lowercases and collapses whitespace,
which is harmless, but you want dates, amounts, and invoice numbers intact.
"""

from __future__ import annotations

import re
from typing import Union

_MAX_WORDS = 500

# Isolated page-number pattern: a line that is *only* digits (possibly
# surrounded by whitespace) — typical of PDF page footers.
_PAGE_NUM_RE = re.compile(r'(?m)^\s*\d+\s*$')

# PDF line-break hyphenation: "con-\ntract" → "contract"
_HYPHEN_BREAK_RE = re.compile(r'-\n')

# ── Invoice indicator patterns ────────────────────────────────────────────────
# These generate explicit tokens prepended to the cleaned text so TF-IDF can
# use them as high-signal features, compensating for the small invoice class.
_INV_SIGNALS: list[tuple[str, re.Pattern]] = [
    ('__inv_number__',   re.compile(r'\b(invoice\s*n[o0]\.?|inv[- #]|invoice\s*#|invoice\s*number|bill\s*no)\b', re.I)),
    ('__inv_header__',   re.compile(r'\b(tax\s*invoice|commercial\s*invoice|invoice\b)', re.I)),
    ('__bill_to__',      re.compile(r'\b(bill\s*to|billed\s*to|sold\s*to|issued\s*to|invoiced\s*to)\b', re.I)),
    ('__amount_due__',   re.compile(r'\b(amount\s*due|balance\s*due|total\s*amount\s*due|total\s*due)\b', re.I)),
    ('__grand_total__',  re.compile(r'\b(grand\s*total|total\s*amount|subtotal|sub\s*total)\b', re.I)),
    ('__due_date__',     re.compile(r'\b(due\s*date|payment\s*due|pay\s*by|payment\s*terms)\b', re.I)),
    ('__remit__',        re.compile(r'\b(remit\s*payment|please\s*pay|please\s*remit|kindly\s*remit)\b', re.I)),
    ('__po_number__',    re.compile(r'\b(purchase\s*order|po\s*number|po\s*#|po\s*no)\b', re.I)),
    ('__vat__',          re.compile(r'\b(vat|gst|sales\s*tax)\b', re.I)),
    ('__net_terms__',    re.compile(r'\bnet\s*\d{1,3}\b', re.I)),
]


def _invoice_indicator_tokens(raw_text: str) -> str:
    """Return a space-separated string of indicator tokens for matched patterns."""
    return ' '.join(token for token, pattern in _INV_SIGNALS if pattern.search(raw_text))


def clean_for_classifier(text: str) -> str:
    """Turn raw document text into classifier-ready input.

    Matches the training pipeline in notebook 03:
        1. Rejoin PDF hyphenated line-breaks ("docu-\\nment" → "document")
        2. Strip isolated page-number lines (PDF artefact)
        3. Lowercase
        4. Collapse all whitespace to single spaces
        5. Truncate to the first 500 words

    Stopword removal is intentionally omitted — it is handled by
    TfidfVectorizer(stop_words="english") during vectorisation, exactly as
    during training.
    """
    raw  = str(text)
    # Collect invoice indicator tokens from original text (before lowercasing)
    indicators = _invoice_indicator_tokens(raw)

    text = _HYPHEN_BREAK_RE.sub('', raw)
    text = _PAGE_NUM_RE.sub(' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()[:_MAX_WORDS]
    cleaned = ' '.join(words)

    # Prepend indicators so TF-IDF sees them as high-frequency signals for invoice
    if indicators:
        cleaned = indicators + ' ' + cleaned
    return cleaned


def process_pdf(path_or_bytes: Union[str, bytes]) -> dict:
    """Full demo-time entry point: PDF in, both text versions out.

    Returns a dict with:
        raw_text        — pdfplumber / OCR output, unchanged.
                          Feed this to `extract_invoice_fields` if the
                          classifier labels the document as `invoice`.
        classifier_text — the same text after `clean_for_classifier`.
                          Feed this to your trained classifier.

    Example:
        from src.preprocessing import process_pdf
        from src.information_extraction import extract_invoice_fields

        out = process_pdf('some_invoice.pdf')
        label = classifier.predict([out['classifier_text']])[0]
        if label == 'invoice':
            fields = extract_invoice_fields(out['raw_text'])
    """
    from src.pdf_loader import pdf_to_text  # local import to keep deps lazy
    raw = pdf_to_text(path_or_bytes)
    return {
        'raw_text': raw,
        'classifier_text': clean_for_classifier(raw),
    }
