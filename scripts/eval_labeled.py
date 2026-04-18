"""Evaluate invoice extraction against the ground-truth JSON in the batch CSVs.

Usage: python scripts/eval_labeled.py [--limit N] [--csv PATH]

Reads ``File Name`` and ``Json Data`` columns from the batch CSV, locates the
corresponding image under ``batch_X/batch_X/batch_X/<fname>``, runs the
extractor on the image bytes, and compares each predicted field to the
ground-truth value with a format-agnostic comparator.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.information_extraction import extract_invoice_fields
from src.pdf_loader import image_to_words, image_to_full_text


DEFAULT_CSV = ROOT / "data/raw/invoices/high_quality_images/batch_1/batch_1/batch1_1.csv"


def _image_dir_for(csv_path: Path) -> Path:
    return csv_path.parent / csv_path.stem


def _gt_fields(json_str: str) -> dict:
    j = json.loads(json_str)
    inv = j.get("invoice") or {}
    sub = j.get("subtotal") or {}
    return {
        "invoice_number": inv.get("invoice_number") or None,
        "invoice_date":   inv.get("invoice_date") or None,
        "due_date":       inv.get("due_date") or None,
        "issuer":         inv.get("seller_name") or None,
        "recipient":      inv.get("client_name") or None,
        "total":          sub.get("total") or None,
    }


def _norm_date(s: str) -> str:
    if not s:
        return ""
    digits = re.sub(r"[^\d]", "", s)
    return digits


def _norm_amount(s: str) -> str:
    if not s:
        return ""
    t = re.sub(r"[^\d.,]", "", s)
    if re.fullmatch(r"\d+,\d{2}", t):
        t = t.replace(",", ".")
    else:
        t = t.replace(",", "")
    try:
        return f"{float(t):.2f}"
    except ValueError:
        return t


def _norm_str(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip().lower()


def _match(pred: str | None, gt: str | None, field: str) -> bool:
    if not gt:
        return pred is None or pred == ""
    if not pred:
        return False
    if field in ("invoice_date", "due_date"):
        return _norm_date(pred) == _norm_date(gt)
    if field == "total":
        return _norm_amount(pred) == _norm_amount(gt)
    if field == "invoice_number":
        return _norm_str(pred).lstrip("#") == _norm_str(gt).lstrip("#")
    # issuer/recipient — substring match either way
    p, g = _norm_str(pred), _norm_str(gt)
    return g in p or p in g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--show-misses", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    img_dir = _image_dir_for(args.csv)
    fields = ["invoice_number", "invoice_date", "due_date",
              "issuer", "recipient", "total"]
    scored = {k: {"total": 0, "hits": 0, "gt_present": 0} for k in fields}
    misses: dict[str, list[tuple[str, str, str]]] = {k: [] for k in fields}

    n = min(args.limit, len(df))
    print(f"Evaluating {n} invoices from {args.csv.name}")
    for i in range(n):
        row = df.iloc[i]
        fname = row["File Name"]
        img_path = img_dir / fname
        if not img_path.exists():
            continue
        gt = _gt_fields(row["Json Data"])
        try:
            img_bytes = img_path.read_bytes()
            words = image_to_words(img_bytes)
            text = image_to_full_text(words)
            pred = extract_invoice_fields(text, words=words)
        except Exception as e:
            print(f"  ERR {fname}: {e}")
            continue
        for k in fields:
            scored[k]["total"] += 1
            gt_v = gt.get(k) or ""
            if gt_v:
                scored[k]["gt_present"] += 1
            ok = _match(pred.get(k), gt.get(k), k)
            if ok and gt_v:
                scored[k]["hits"] += 1
            elif not ok and gt_v:
                misses[k].append((fname, str(pred.get(k)), str(gt_v)))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n}]")

    print()
    print(f"{'field':18s} {'hit/gt':>10s}  {'recall':>7s}")
    print("-" * 45)
    for k in fields:
        s = scored[k]
        if s["gt_present"] == 0:
            print(f"{k:18s} {'n/a':>10s}  {'n/a':>7s}")
            continue
        rec = s["hits"] / s["gt_present"]
        print(f"{k:18s} {s['hits']:>4d}/{s['gt_present']:<5d} {rec*100:>6.1f}%")

    if args.show_misses > 0:
        print()
        for k in fields:
            if not misses[k]:
                continue
            print(f"--- {k} misses (showing {min(args.show_misses, len(misses[k]))}/{len(misses[k])}) ---")
            for fname, p, g in misses[k][: args.show_misses]:
                print(f"  {fname}: pred={p!r} gt={g!r}")


if __name__ == "__main__":
    main()
