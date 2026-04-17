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
    import pytesseract

    if isinstance(path_or_bytes, (bytes, bytearray)):
        pdf = pdfium.PdfDocument(bytes(path_or_bytes))
    else:
        pdf = pdfium.PdfDocument(str(path_or_bytes))

    pages = []
    for page in pdf:
        pil_image = page.render(scale=3.0).to_pil()
        pages.append(pytesseract.image_to_string(pil_image))
    return '\n'.join(pages)


def pdf_to_text(path_or_bytes: PathLike) -> str:
    """Extract text from a PDF, using OCR as a fallback for scanned docs."""
    text = _extract_with_pdfplumber(path_or_bytes)
    if len(text.strip()) >= _MIN_TEXT_CHARS:
        return text
    return _extract_with_ocr(path_or_bytes)


def image_to_text(path_or_bytes: PathLike) -> str:
    """Run pytesseract on a .jpg/.png/image bytes. Returns extracted text."""
    import pytesseract
    from PIL import Image

    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(path_or_bytes)))
    else:
        img = Image.open(str(path_or_bytes))
    return pytesseract.image_to_string(img)


def image_to_words(path_or_bytes: PathLike, min_conf: int = 30) -> list[dict]:
    """OCR an image and return words as bounding-box dicts compatible with
    pdfplumber's extract_words() output (keys: text, x0, x1, top, bottom).

    Drops words whose OCR confidence is below ``min_conf`` (tesseract uses
    -1 for non-word rows like page/block headers — those are kept)."""
    import pytesseract
    from pytesseract import Output
    from PIL import Image

    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(path_or_bytes)))
    else:
        img = Image.open(str(path_or_bytes))

    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    words: list[dict] = []
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = int(float(data["conf"][i]))
        except (TypeError, ValueError):
            conf = -1
        if 0 <= conf < min_conf:
            continue
        left = float(data["left"][i])
        top  = float(data["top"][i])
        w    = float(data["width"][i])
        h    = float(data["height"][i])
        words.append({
            "text":   text,
            "x0":     left,
            "x1":     left + w,
            "top":    top,
            "bottom": top + h,
        })
    return words


def image_to_full_text(words: list[dict]) -> str:
    """Reconstruct a reading-order text string from OCR word dicts.

    Groups words into lines by y-band, sorts each line by x, joins with '\\n'.
    Useful for feeding the text-based regex path when OCR bbox data is
    available — preserves adjacency better than image_to_string()."""
    if not words:
        return ""
    avg_h = sum(w["bottom"] - w["top"] for w in words) / len(words)
    tol = max(3.0, avg_h / 2)
    sorted_w = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: list[list[dict]] = []
    for w in sorted_w:
        if lines and abs(w["top"] - lines[-1][0]["top"]) < tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    out = []
    for line in lines:
        line.sort(key=lambda w: w["x0"])
        out.append(" ".join(w["text"] for w in line))
    return "\n".join(out)
