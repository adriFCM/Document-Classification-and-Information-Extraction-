"""Classifier-side text preprocessing.

Mirrors the base cleaning pipeline used in `notebooks/02_preprocessing.ipynb`
so that arbitrary PDFs passed through the classifier at demo time see text
with the same token distribution as the training corpus in
`data/labeled/dataset.csv`.

The notebook applies category-specific cleaners on top of a shared base
(lowercase → strip punctuation → remove digit-tokens → remove single-char
tokens → remove stopwords → truncate). At inference time we do **not** know
the category yet, so we apply only the shared base — it matches what every
row in `dataset.csv` went through.

For the extractor (`src.information_extraction`) you want the *raw* text,
not this cleaned version — cleaning strips the very digits, dates, and
punctuation the regex extractors depend on. Use `clean_for_classifier`
only on the text you hand to the classifier.
"""

from __future__ import annotations

import re
from typing import Union

try:
    from nltk.corpus import stopwords
    _STOPWORDS = set(stopwords.words('english'))
except LookupError:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    _STOPWORDS = set(stopwords.words('english'))

_MAX_WORDS = 500


def _base_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_for_classifier(text: str) -> str:
    """Turn raw document text into classifier-ready input.

    Exactly matches the shared base cleaning in notebook 02:
    lowercase → strip punctuation → remove digit tokens → remove single-char
    tokens → remove English stopwords → truncate to the first 500 words.
    """
    text = _base_clean(text)
    text = re.sub(r'\b\d+\w*\b', '', text)
    text = re.sub(r'\b\w\b', '', text)
    words = text.split()
    words = [w for w in words if w not in _STOPWORDS]
    words = words[:_MAX_WORDS]
    return ' '.join(words).strip()


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
