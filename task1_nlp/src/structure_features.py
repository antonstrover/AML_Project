"""Structural / stylometric features for the spam gate (Task 1, my approach).

My approach treats spam as "text that doesn't look like a movie review *in
structure*". The corpus mixes two very different document genres:

  * Real reviews  -> short, single-sentence Rotten-Tomatoes-style snippets.
  * Spam          -> Enron-style emails with headers ("Subject:"), many
                     line-breaks, digits, addresses, longer length.

These genres separate cleanly on a handful of cheap, interpretable
*surface* statistics, independent of sentiment vocabulary. Extracting them
explicitly (rather than relying on a bag-of-words centroid) gives a spam
gate that is (a) transparent, (b) robust to vocabulary drift -- it still
fires on an unseen email even if none of its words were in the training
vocabulary -- and (c) cleanly decoupled from the sentiment model.

This module is deliberately dependency-light (regex + stdlib) so the same
features can be computed identically at train, validation and test time.
"""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

# Pre-compiled regexes (compile once, reuse per document).
_RE_SUBJECT = re.compile(r"^\s*subject\s*:", re.IGNORECASE)
_RE_HEADER = re.compile(r"^\s*(from|to|cc|sent|date|re|fw|fwd)\s*:", re.IGNORECASE)
_RE_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_RE_WS = re.compile(r"\s+")

# The order here defines the feature-vector column order. Keep it stable:
# the GMM and the standardiser are fit against this exact ordering.
FEATURE_NAMES: List[str] = [
    "log_length",          # log(1 + char length) -- emails are far longer
    "line_break_ratio",    # newlines per char -- emails are multi-line
    "digit_ratio",         # fraction of digits -- emails carry meter ids, dates
    "nonalpha_ratio",      # fraction of non-alphanumeric, non-space chars
    "uppercase_ratio",     # caps among letters -- headers/codes raise this
    "avg_token_len",       # mean token length -- emails contain long codes
    "has_subject_header",  # binary: starts with "Subject:"
    "has_any_header",      # binary: any RFC-822-ish header line present
    "has_url",             # binary: contains a URL
    "has_email_addr",      # binary: contains an @-address
]
N_FEATURES = len(FEATURE_NAMES)


def extract_one(text: str) -> np.ndarray:
    """Return the structural feature vector for a single document."""
    t = "" if text is None else str(text)
    n_chars = len(t)
    denom = n_chars + 1e-9

    # Line-break count covers both \n and \r line endings (the raw CSV uses \r\n).
    n_breaks = t.count("\n") + t.count("\r")

    n_digits = sum(c.isdigit() for c in t)
    n_alpha = sum(c.isalpha() for c in t)
    n_upper = sum(c.isupper() for c in t)
    n_nonalpha = sum((not c.isalnum()) and (not c.isspace()) for c in t)

    tokens = _RE_WS.split(t.strip()) if t.strip() else []
    avg_tok = (sum(len(tok) for tok in tokens) / len(tokens)) if tokens else 0.0

    # Header detection looks at the first few lines only (cheap, robust).
    first_lines = t.splitlines()[:6]
    has_subject = 1.0 if (_RE_SUBJECT.match(t) or any(_RE_SUBJECT.match(l) for l in first_lines)) else 0.0
    has_header = 1.0 if any(_RE_HEADER.match(l) for l in first_lines) else 0.0

    return np.array([
        np.log1p(n_chars),
        n_breaks / denom,
        n_digits / denom,
        n_nonalpha / denom,
        n_upper / (n_alpha + 1e-9),
        avg_tok,
        has_subject,
        has_header,
        1.0 if _RE_URL.search(t) else 0.0,
        1.0 if _RE_EMAIL.search(t) else 0.0,
    ], dtype=np.float64)


def extract_matrix(texts) -> np.ndarray:
    """Stack structural feature vectors for an iterable of documents."""
    return np.vstack([extract_one(t) for t in texts])


def high_precision_email_flag(text: str) -> bool:
    """A conservative, near-zero-false-positive rule for "this is an email".

    Used ONLY to build an evaluation gold-standard for the spam gate (we have
    no provided spam labels) and as a sanity anchor -- not as the gate itself.
    An Enron email is identifiable with very high precision by an RFC-822
    header in its opening lines; genuine one-sentence reviews never have these.
    """
    t = "" if text is None else str(text)
    first_lines = t.splitlines()[:6]
    if _RE_SUBJECT.match(t) or any(_RE_SUBJECT.match(l) for l in first_lines):
        return True
    if any(_RE_HEADER.match(l) for l in first_lines):
        return True
    return False


def feature_summary(texts, flags: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Median feature values split by a boolean flag (for the EDA writeup)."""
    M = extract_matrix(texts)
    out = {}
    for name, col in zip(FEATURE_NAMES, M.T):
        out[name] = {
            "spam_median": float(np.median(col[flags])),
            "review_median": float(np.median(col[~flags])),
        }
    return out
