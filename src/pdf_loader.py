"""PDF → text with OCR fallback.

Strategy: try `pdfplumber` first (fast, works for digitally-generated PDFs).
If that yields too little text — typical of scanned invoices — rasterize each
page via `pypdfium2` and run `pytesseract` on the resulting images.

`pytesseract` requires the `tesseract` binary on PATH (brew install tesseract).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import pdfplumber

PathLike = Union[str, Path, bytes]

# If pdfplumber extracts fewer than this many non-whitespace chars across all
# pages, we assume the PDF is a scan and fall back to OCR.
_MIN_TEXT_CHARS = 40


def _open(path_or_bytes: PathLike):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        return pdfplumber.open(io.BytesIO(path_or_bytes))
    return pdfplumber.open(str(path_or_bytes))


def _extract_with_pdfplumber(path_or_bytes: PathLike) -> str:
    out = []
    with _open(path_or_bytes) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ''
            if txt:
                out.append(txt)
    return '\n'.join(out)


def _extract_with_ocr(path_or_bytes: PathLike) -> str:
    import pypdfium2 as pdfium
    try:
        import pytesseract
    except ImportError:
        raise RuntimeError(
            'pytesseract is not installed. Run: pip install pytesseract'
        )

    if isinstance(path_or_bytes, (bytes, bytearray)):
        pdf = pdfium.PdfDocument(bytes(path_or_bytes))
    else:
        pdf = pdfium.PdfDocument(str(path_or_bytes))

    pages = []
    for page in pdf:
        pil_image = page.render(scale=2.0).to_pil()
        try:
            pages.append(pytesseract.image_to_string(pil_image))
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError(
                'Tesseract OCR binary not found on PATH. '
                'Install it: https://github.com/tesseract-ocr/tesseract'
            )
    return '\n'.join(pages)


def pdf_to_text(path_or_bytes: PathLike) -> str:
    """Extract text from a PDF, using OCR as a fallback for scanned docs."""
    text = _extract_with_pdfplumber(path_or_bytes)
    if len(text.strip()) >= _MIN_TEXT_CHARS:
        return text
    return _extract_with_ocr(path_or_bytes)
