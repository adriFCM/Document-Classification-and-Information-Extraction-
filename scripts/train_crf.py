"""Train CRF for invoice ISSUER / RECIPIENT extraction.

Data sources:
- data/raw/invoices/converted_invoice_dataset.csv : 67 Canva-style invoices
  with gold JSON in `Final_Output`. Provides ISSUER (rare) and RECIPIENT.
- data/raw/invoices/SROIE2019/train/{box,entities} : 626 receipt scans.
  Provides ISSUER via the `company` field. No recipient labels.

Pipeline:
1. Load both datasets, build (text, {label: value}) pairs.
2. Convert each pair to (lines, BIO tags) via crf_extractor.build_bio.
3. 5-fold CV on the combined data, report seqeval F1 per entity.
4. Retrain on all data, persist to models/crf_invoice.pkl.

Usage: python scripts/train_crf.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from seqeval.metrics import classification_report, f1_score
from sklearn.model_selection import KFold, train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from information_extraction import (  # noqa: E402
    CRFInvoiceExtractor,
    build_bio,
    flat_tokens,
    sent_features,
)

NEW_CSV = ROOT / "data/raw/invoices/converted_invoice_dataset.csv"
SROIE_TRAIN = ROOT / "data/raw/invoices/SROIE2019/train"
MODEL_OUT = ROOT / "models/crf_invoice.pkl"

RECIPIENT_KEYS = ("BILLED_TO", "BILL_TO", "ISSUED_TO", "RECIPIENT", "CUSTOMER_NAME")
ISSUER_KEYS = ("ISSUER", "COMPANY", "FROM", "VENDOR")


def _first(d: dict, keys) -> str | None:
    for k in keys:
        if k in d and d[k]:
            return str(d[k])
    return None


def load_new_invoices() -> list[tuple[str, dict]]:
    df = pd.read_csv(NEW_CSV)
    out = []
    for _, row in df.iterrows():
        text = str(row["Input"])
        try:
            gold = json.loads(row["Final_Output"])
        except Exception:
            continue
        ents = {}
        rcp = _first(gold, RECIPIENT_KEYS)
        if rcp:
            ents["RECIPIENT"] = rcp
        iss = _first(gold, ISSUER_KEYS)
        if iss:
            ents["ISSUER"] = iss
        if ents:
            out.append((text, ents))
    return out


def _sroie_text_from_box(box_path: Path) -> str:
    """Reconstruct line-grouped text from a SROIE box file.

    Each line is `x1,y1,x2,y2,x3,y3,x4,y4,text`. Tokens are already roughly
    ordered top-to-bottom, so we just split on the first 8 commas and take the
    rest as text. Group by similar y1 to form lines.
    """
    rows = []
    for raw in box_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.split(",", 8)
        if len(parts) < 9:
            continue
        try:
            y1 = int(parts[1])
        except ValueError:
            continue
        rows.append((y1, parts[8]))
    rows.sort()
    # Collapse rows with very close y into the same line.
    lines: list[list[str]] = []
    last_y = None
    for y, txt in rows:
        if last_y is None or abs(y - last_y) > 8:
            lines.append([txt])
        else:
            lines[-1].append(txt)
        last_y = y
    return "\n".join(" ".join(line) for line in lines)


def load_sroie() -> list[tuple[str, dict]]:
    if not SROIE_TRAIN.exists():
        return []
    out = []
    box_dir = SROIE_TRAIN / "box"
    ent_dir = SROIE_TRAIN / "entities"
    for ent_path in sorted(ent_dir.glob("*.txt")):
        box_path = box_dir / ent_path.name
        if not box_path.exists():
            continue
        try:
            ent = json.loads(ent_path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue
        company = ent.get("company")
        if not company:
            continue
        text = _sroie_text_from_box(box_path)
        if not text.strip():
            continue
        out.append((text, {"ISSUER": company}))
    return out


def to_dataset(pairs):
    """Map (text, entities) pairs to (lines, tags). Drop fully unaligned ones."""
    lines_all, tags_all = [], []
    align_stats = {"aligned": 0, "missed": 0}
    for text, ents in pairs:
        lines, tags, report = build_bio(text, ents)
        align_stats["aligned"] += len(report["aligned"])
        align_stats["missed"] += len(report["missed"])
        if any(t != "O" for t in tags):
            lines_all.append(lines)
            tags_all.append(tags)
    return lines_all, tags_all, align_stats


def cv_report(lines_all, tags_all, n_splits: int = 5) -> None:
    if len(lines_all) < n_splits:
        print(f"  too few examples ({len(lines_all)}) for {n_splits}-fold CV")
        return
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_f1 = []
    all_pred, all_true = [], []
    for fold, (tr, te) in enumerate(kf.split(lines_all)):
        clf = CRFInvoiceExtractor()
        clf.train(
            [lines_all[i] for i in tr],
            [tags_all[i] for i in tr],
        )
        X_te = [sent_features(lines_all[i]) for i in te]
        y_te = [list(tags_all[i]) for i in te]
        y_pred = clf.model.predict(X_te)
        f = f1_score(y_te, y_pred)
        fold_f1.append(f)
        all_pred.extend(y_pred)
        all_true.extend(y_te)
        print(f"  fold {fold + 1}: F1 = {f:.3f}")
    print(f"  mean F1 = {sum(fold_f1) / len(fold_f1):.3f}")
    print()
    print("Aggregated entity-level report:")
    print(classification_report(all_true, all_pred, digits=3))


def holdout_report(
    train_lines, train_tags, test_lines, test_tags
) -> None:
    """Train on train split, score once on held-out test split.

    Separates the overfitting signal from CV: if CV F1 >> test F1 the model
    is memorizing template quirks. Run BEFORE the final "train on all data"
    step so the test set stays truly unseen during hyperparameter choices.
    """
    if not test_lines:
        print("  (empty test split — skipped)")
        return
    clf = CRFInvoiceExtractor()
    clf.train(train_lines, train_tags)
    X_te = [sent_features(l) for l in test_lines]
    y_te = [list(t) for t in test_tags]
    y_pred = clf.model.predict(X_te)
    print(f"  held-out F1 = {f1_score(y_te, y_pred):.3f}")
    print(classification_report(y_te, y_pred, digits=3))


def main() -> None:
    print("Loading new invoice dataset...")
    new_pairs = load_new_invoices()
    print(f"  {len(new_pairs)} examples")

    print("Loading SROIE...")
    sroie_pairs = load_sroie()
    print(f"  {len(sroie_pairs)} examples")

    print("Building BIO datasets...")
    new_lines, new_tags, new_stats = to_dataset(new_pairs)
    sroie_lines, sroie_tags, sroie_stats = to_dataset(sroie_pairs)
    print(f"  new : aligned={new_stats['aligned']} missed={new_stats['missed']}"
          f" usable={len(new_lines)}")
    print(f"  sroie: aligned={sroie_stats['aligned']} missed={sroie_stats['missed']}"
          f" usable={len(sroie_lines)}")

    combined_lines = new_lines + sroie_lines
    combined_tags = new_tags + sroie_tags
    print(f"  combined: {len(combined_lines)} examples")

    # Reproducible train/test split BEFORE any model selection, so test F1 is
    # a clean generalization estimate.
    tr_lines, te_lines, tr_tags, te_tags = train_test_split(
        combined_lines, combined_tags,
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    print(f"  train: {len(tr_lines)}   test: {len(te_lines)}"
          f"   (seed={RANDOM_STATE})")

    print("\n=== 5-fold CV on NEW invoices only (issuer + recipient) ===")
    cv_report(new_lines, new_tags)

    print("\n=== 5-fold CV on TRAIN split of COMBINED ===")
    cv_report(tr_lines, tr_tags)

    print("\n=== Held-out TEST F1 (train on COMBINED train split) ===")
    holdout_report(tr_lines, tr_tags, te_lines, te_tags)

    print("\nTraining final model on all COMBINED data...")
    final = CRFInvoiceExtractor()
    final.train(combined_lines, combined_tags)
    final.save(MODEL_OUT)
    print(f"  saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
