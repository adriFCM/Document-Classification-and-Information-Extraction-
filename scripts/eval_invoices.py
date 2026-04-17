"""Run information extraction across data/raw/invoices/high_quality_images/
and write per-file results to data/processed/invoice_extraction_eval.csv.

Usage: python scripts/eval_invoices.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.information_extraction import extract_invoice_fields
from src.pdf_loader import image_to_words, image_to_full_text

IMG_DIR = ROOT / "data/raw/invoices/high_quality_images"
OUT_CSV = ROOT / "data/processed/invoice_extraction_eval.csv"
FIELDS = ["invoice_number", "invoice_date", "due_date", "issuer", "recipient", "total"]


def main() -> None:
    if not IMG_DIR.exists():
        print(f"{IMG_DIR} not found - run notebook 01_data_collection first.")
        return

    images = sorted(
        p for p in IMG_DIR.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        print(f"No images in {IMG_DIR}")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(images)} images... (Ctrl+C to stop early; progress saved.)")

    rows = []
    fh = OUT_CSV.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=["filename"] + FIELDS + ["non_null_count"])
    writer.writeheader()
    fh.flush()
    try:
        for i, img in enumerate(images, 1):
            try:
                img_bytes = img.read_bytes()
                words = image_to_words(img_bytes)
                text = image_to_full_text(words)
                fields = extract_invoice_fields(text, image_bytes=img_bytes)
            except Exception as e:
                print(f"  [{i}/{len(images)}] {img.name}: ERROR {e}")
                fields = {k: None for k in FIELDS}
            non_null = sum(1 for k in FIELDS if fields.get(k))
            row = {"filename": img.name, **fields, "non_null_count": non_null}
            rows.append(row)
            writer.writerow(row)
            fh.flush()
            if i % 25 == 0 or i == len(images):
                print(f"  [{i}/{len(images)}]")
    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(rows)} images — partial results saved.")
    finally:
        fh.close()

    if not rows:
        print("No rows written.")
        return
    print(f"\nWrote {OUT_CSV}")
    total = len(rows)
    print("\nPer-field non-null rate:")
    for k in FIELDS:
        n = sum(1 for r in rows if r[k])
        print(f"  {k:16s} : {n:4d}/{total} ({100*n/total:5.1f}%)")

    print("\nWorst 10 (lowest non_null_count):")
    worst = sorted(rows, key=lambda r: r["non_null_count"])[:10]
    for r in worst:
        print(f"  {r['filename']}  ({r['non_null_count']}/{len(FIELDS)})")


if __name__ == "__main__":
    main()
