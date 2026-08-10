"""Structural features for the spam gate (Task 1).

This approach defines spam as text with a structure that is different from
the structure of a movie review. The corpus contains two types of document:

  * A real review is a short text of one sentence. It has the style of a
    Rotten Tomatoes review.
  * A spam document is an email of the Enron type. It has headers, for
    example "Subject:". It also has many line breaks, many digits, one or
    more addresses, and more characters.

Some simple statistics of the surface of the text separate the two types.
These statistics do not use the vocabulary of the sentiment. A person can read
and understand each statistic.

This module calculates the statistics directly. It does not use a centroid of
a bag of words. Thus the spam gate has these three properties:

  * A person can see the reason for each decision.
  * The gate operates correctly when the vocabulary changes. It finds a new
    email even if the training vocabulary contains no word of that email.
  * The gate is independent of the sentiment model.

This module uses only a regex and the standard library. Thus the code
calculates the same features at training time, at validation time and at test
time.
"""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

# The code compiles these regexes one time and uses them for each document.
_RE_SUBJECT = re.compile(r"^\s*subject\s*:", re.IGNORECASE)
_RE_HEADER = re.compile(r"^\s*(from|to|cc|sent|date|re|fw|fwd)\s*:", re.IGNORECASE)
_RE_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_RE_WS = re.compile(r"\s+")

# This sequence gives the order of the columns of the feature vector. Do not
# change it. The GMM and the standardiser use this order.
FEATURE_NAMES: List[str] = [
    "log_length",          # log(1 + number of characters). An email is longer.
    "line_break_ratio",    # New lines per character. An email has many lines.
    "digit_ratio",         # The fraction of digits. An email has meter IDs
                           # and dates.
    "nonalpha_ratio",      # The fraction of the characters that are not
                           # alphanumeric and not a space.
    "uppercase_ratio",     # The fraction of the letters in upper case. A
                           # header or a code increases this value.
    "avg_token_len",       # The mean length of a token. An email contains
                           # long codes.
    "has_subject_header",  # 1 if the text starts with "Subject:".
    "has_any_header",      # 1 if the text has a header line of the RFC-822
                           # type.
    "has_url",             # 1 if the text contains a URL.
    "has_email_addr",      # 1 if the text contains an address with an @.
]
N_FEATURES = len(FEATURE_NAMES)


def extract_one(text: str) -> np.ndarray:
    """Calculate the structural feature vector of one document."""
    t = "" if text is None else str(text)
    n_chars = len(t)
    denom = n_chars + 1e-9

    # The count includes the \n character and the \r character, because the raw
    # CSV file uses \r\n at the end of a line.
    n_breaks = t.count("\n") + t.count("\r")

    n_digits = sum(c.isdigit() for c in t)
    n_alpha = sum(c.isalpha() for c in t)
    n_upper = sum(c.isupper() for c in t)
    n_nonalpha = sum((not c.isalnum()) and (not c.isspace()) for c in t)

    tokens = _RE_WS.split(t.strip()) if t.strip() else []
    avg_tok = (sum(len(tok) for tok in tokens) / len(tokens)) if tokens else 0.0

    # The code looks for a header in the first lines only. This method is fast
    # and reliable.
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
    """Calculate the structural feature vector of each document.

    The function puts the vectors together in one matrix.
    """
    return np.vstack([extract_one(t) for t in texts])


def high_precision_email_flag(text: str) -> bool:
    """Tell if the text is an email.

    This rule is careful. It gives almost no incorrect positive result.

    Use this rule ONLY to make the reference labels for the evaluation of the
    spam gate, because the data contains no spam labels. The rule is also a
    check of the gate. The rule is not the gate.

    A header of the RFC-822 type in the first lines identifies an Enron email
    with very high precision. A true review of one sentence has no such
    header.
    """
    t = "" if text is None else str(text)
    first_lines = t.splitlines()[:6]
    if _RE_SUBJECT.match(t) or any(_RE_SUBJECT.match(l) for l in first_lines):
        return True
    if any(_RE_HEADER.match(l) for l in first_lines):
        return True
    return False


def feature_summary(texts, flags: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Calculate the median of each feature for the two groups.

    The argument flags divides the documents into the two groups. The report
    uses these values in the analysis of the data.
    """
    M = extract_matrix(texts)
    out = {}
    for name, col in zip(FEATURE_NAMES, M.T):
        out[name] = {
            "spam_median": float(np.median(col[flags])),
            "review_median": float(np.median(col[~flags])),
        }
    return out
