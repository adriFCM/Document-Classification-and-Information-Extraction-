"""CRF-based extractor for semantic invoice fields (issuer, recipient).

Rules in ``information_extraction.py`` cover structured fields (invoice number,
dates, total) well. This module handles the semantic fields where regex is
brittle, using a classical linear-chain CRF over line-tokenized invoice text.

Entities: ISSUER, RECIPIENT (BIO-tagged).
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import sklearn_crfsuite
except ImportError:  # pragma: no cover
    sklearn_crfsuite = None

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None


# --- tokenization ---------------------------------------------------------

_TOKEN_RE = re.compile(r"\S+")
_LABEL_LINE_RE = re.compile(
    r"\b(bill(?:ed)?\s*to|sold\s*to|ship\s*to|invoice\s*to|issued\s*to"
    r"|customer|recipient|^\s*to\s*:)",
    re.IGNORECASE,
)
_COMPANY_SUFFIXES = {
    "BHD", "BND", "SDN", "LLC", "LTD", "INC", "CORP", "CO", "CO.",
    "PTE", "GMBH", "ENTERPRISE", "TRADING",
}


def tokenize(text: str) -> List[List[str]]:
    """Return tokens grouped by line. Empty lines are dropped."""
    lines = []
    for raw in text.splitlines():
        toks = _TOKEN_RE.findall(raw)
        if toks:
            lines.append(toks)
    return lines


# --- features -------------------------------------------------------------

def _shape(word: str) -> str:
    out = []
    for ch in word:
        if ch.isupper():
            out.append("X")
        elif ch.islower():
            out.append("x")
        elif ch.isdigit():
            out.append("d")
        else:
            out.append(ch)
    return "".join(out)


def token_features(
    lines: Sequence[Sequence[str]],
    line_idx: int,
    tok_idx: int,
    prev_line_has_label: bool,
) -> dict:
    tokens = lines[line_idx]
    word = tokens[tok_idx]
    feats = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word.shape": _shape(word)[:8],
        "suffix3": word[-3:].lower(),
        "prefix3": word[:3].lower(),
        "is_upper": word.isupper(),
        "is_title": word.istitle(),
        "is_digit": word.isdigit(),
        "has_currency": bool(re.search(r"[$€£]|\bRM\b|\bUSD\b", word)),
        "has_at": "@" in word,
        "line_idx": min(line_idx, 10),
        "line_pos": tok_idx,
        "line_len": len(tokens),
        "is_first_in_line": tok_idx == 0,
        "is_last_in_line": tok_idx == len(tokens) - 1,
        "in_company_suffix": word.strip(".,").upper() in _COMPANY_SUFFIXES,
        "prev_line_label": prev_line_has_label,
    }
    if tok_idx > 0:
        prev = tokens[tok_idx - 1]
        feats["-1:word.lower"] = prev.lower()
        feats["-1:is_title"] = prev.istitle()
    else:
        feats["BOL"] = True  # beginning of line
    if tok_idx < len(tokens) - 1:
        nxt = tokens[tok_idx + 1]
        feats["+1:word.lower"] = nxt.lower()
        feats["+1:is_title"] = nxt.istitle()
    else:
        feats["EOL"] = True
    if line_idx == 0:
        feats["BOS"] = True
    return feats


def sent_features(lines: Sequence[Sequence[str]]) -> List[dict]:
    """Flatten tokenized lines into a single token sequence with features.

    The CRF sees the whole document as one sequence — line structure is
    captured via features (line_idx, prev_line_has_label, etc.).
    """
    feats = []
    for li, line in enumerate(lines):
        prev_line_has_label = li > 0 and bool(
            _LABEL_LINE_RE.search(" ".join(lines[li - 1]))
        )
        for ti in range(len(line)):
            feats.append(token_features(lines, li, ti, prev_line_has_label))
    return feats


def flat_tokens(lines: Sequence[Sequence[str]]) -> List[str]:
    return [tok for line in lines for tok in line]


# --- BIO alignment --------------------------------------------------------

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def align_entity(
    tokens: Sequence[str], value: str, min_score: float = 85.0
) -> Optional[Tuple[int, int]]:
    """Find (start, end) token span matching ``value`` in ``tokens``.

    Uses exact join-and-substring first, falls back to fuzzy window search.
    Returns None if no good match.
    """
    if not value:
        return None
    target = _normalize(value)
    target_ntoks = max(1, len(target.split()))

    # Exact substring on normalized joined stream.
    joined = " ".join(t.lower() for t in tokens)
    if target in joined:
        start_char = joined.index(target)
        prefix_tokens = joined[:start_char].split()
        start = len(prefix_tokens)
        end = start + target_ntoks
        if end <= len(tokens):
            return start, end

    # Fuzzy window search.
    if fuzz is None:
        return None
    best = (0.0, None)
    window_sizes = {target_ntoks, max(1, target_ntoks - 1), target_ntoks + 1}
    for w in window_sizes:
        for i in range(0, len(tokens) - w + 1):
            cand = " ".join(t.lower() for t in tokens[i : i + w])
            score = fuzz.ratio(cand, target)
            if score > best[0]:
                best = (score, (i, i + w))
    if best[0] >= min_score and best[1] is not None:
        return best[1]
    return None


def build_bio(
    text: str, entities: dict
) -> Tuple[List[List[str]], List[str], dict]:
    """Return (lines, bio_tags, alignment_report).

    ``entities`` maps label → string value, e.g. {"ISSUER": "Borcelle", ...}.
    Tags are in BIO format at flat-token granularity matching ``flat_tokens``.
    """
    lines = tokenize(text)
    tokens = flat_tokens(lines)
    tags = ["O"] * len(tokens)
    report = {"aligned": [], "missed": []}
    for label, value in entities.items():
        if not value:
            continue
        span = align_entity(tokens, value)
        if span is None:
            report["missed"].append(label)
            continue
        start, end = span
        tags[start] = f"B-{label}"
        for k in range(start + 1, end):
            tags[k] = f"I-{label}"
        report["aligned"].append(label)
    return lines, tags, report


# --- model wrapper --------------------------------------------------------

class CRFInvoiceExtractor:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def load(cls, path: str | Path) -> "CRFInvoiceExtractor":
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        return cls(model=model)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.model, fh)

    def train(
        self,
        train_lines: Iterable[Sequence[Sequence[str]]],
        train_tags: Iterable[Sequence[str]],
        c1: float = 0.1,
        c2: float = 0.1,
        max_iter: int = 100,
    ) -> None:
        if sklearn_crfsuite is None:
            raise RuntimeError("sklearn-crfsuite not installed")
        X = [sent_features(lines) for lines in train_lines]
        y = [list(t) for t in train_tags]
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iter,
            all_possible_transitions=True,
        )
        self.model.fit(X, y)

    def predict_tags(self, text: str) -> Tuple[List[str], List[str]]:
        if self.model is None:
            return [], []
        lines = tokenize(text)
        tokens = flat_tokens(lines)
        if not tokens:
            return [], []
        feats = sent_features(lines)
        tags = self.model.predict([feats])[0]
        return tokens, tags

    def predict(self, text: str) -> dict:
        """Return {entity_label: extracted_string or None}."""
        tokens, tags = self.predict_tags(text)
        out: dict = {}
        current_label = None
        current_toks: List[str] = []

        def flush():
            nonlocal current_label, current_toks
            if current_label and current_toks:
                # Keep the longest span we see per label.
                existing = out.get(current_label)
                candidate = " ".join(current_toks)
                if existing is None or len(candidate) > len(existing):
                    out[current_label] = candidate
            current_label = None
            current_toks = []

        for tok, tag in zip(tokens, tags):
            if tag == "O":
                flush()
                continue
            prefix, _, label = tag.partition("-")
            if prefix == "B":
                flush()
                current_label = label
                current_toks = [tok]
            elif prefix == "I" and label == current_label:
                current_toks.append(tok)
            else:
                flush()
                current_label = label
                current_toks = [tok]
        flush()
        return out
