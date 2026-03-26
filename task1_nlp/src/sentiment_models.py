"""Sentiment models for Task 1 (my approach).

Deliberately *light* preprocessing -- lowercase + whitespace collapse
only -- and let a **combined word + character n-gram TF-IDF** representation
absorb morphology, negation and noise instead of hand-engineering it away.
Character n-grams (3-5) are the key move: they capture "not bad", "wasn't",
sub-word affixes and typos without any stemming/lemmatisation, and they make
the model robust to the messy email vocabulary that leaks past the spam gate.

Headline model: a **calibrated Linear SVM** (hinge loss, the max-margin
discriminative classifier), with probabilities recovered via Platt scaling so
we can apply a confidence threshold for the dummy/spam fallback. We compare it
against the required word-list floor and Multinomial Naive Bayes so the report
has the multi-approach comparison the brief asks for.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

_RE_WS = re.compile(r"\s+")


def light_clean(text: str) -> str:
    """Minimal normalisation: lowercase + whitespace collapse. Nothing else."""
    t = "" if text is None else str(text)
    return _RE_WS.sub(" ", t.lower()).strip()


# --------------------------------------------------------------------------- #
# Required simplest baseline: word-list classifier (lexicon margin).           #
# --------------------------------------------------------------------------- #
@dataclass
class WordListClassifier:
    """Lexicon classifier: score = (#positive-list hits) - (#negative-list hits).

    Lists are built by the greatest-frequency-difference method: the K words
    most over-represented in positive vs negative documents, and vice versa.
    Predict 1 if score > delta, else 0. This is the interpretable floor.
    """
    K: int = 400
    delta: float = 0.0
    pos_set: set = None
    neg_set: set = None

    def fit(self, texts: List[str], y: np.ndarray) -> "WordListClassifier":
        pos_c, neg_c = Counter(), Counter()
        for t, lab in zip(texts, y):
            toks = set(light_clean(t).split())          # binary presence per doc
            (pos_c if lab == 1 else neg_c).update(toks)
        n_pos = max(1, int((y == 1).sum()))
        n_neg = max(1, int((y == 0).sum()))
        vocab = set(pos_c) | set(neg_c)
        diff = {w: pos_c[w] / n_pos - neg_c[w] / n_neg for w in vocab}
        ranked = sorted(diff.items(), key=lambda kv: kv[1])
        self.neg_set = {w for w, _ in ranked[: self.K]}
        self.pos_set = {w for w, _ in ranked[-self.K:]}
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        out = np.empty(len(texts), dtype=int)
        for i, t in enumerate(texts):
            toks = light_clean(t).split()
            score = sum(w in self.pos_set for w in toks) - sum(w in self.neg_set for w in toks)
            out[i] = 1 if score > self.delta else 0
        return out


# --------------------------------------------------------------------------- #
# Shared word+char TF-IDF representation.                                      #
# --------------------------------------------------------------------------- #
def make_vectorizer() -> FeatureUnion:
    """Union of word (1-2) and character (3-5) TF-IDF -- the headline feature set."""
    word = TfidfVectorizer(
        preprocessor=light_clean, analyzer="word", ngram_range=(1, 2),
        sublinear_tf=True, min_df=3, max_df=0.9, strip_accents="unicode",
    )
    char = TfidfVectorizer(
        preprocessor=light_clean, analyzer="char_wb", ngram_range=(3, 5),
        sublinear_tf=True, min_df=3, max_df=0.95,
    )
    return FeatureUnion([("word", word), ("char", char)])


def build_svm(C: float = 1.0, random_state: int = 42) -> Pipeline:
    """Calibrated Linear SVM on the word+char TF-IDF union."""
    base = LinearSVC(C=C, class_weight="balanced", random_state=random_state)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    return Pipeline([("feats", make_vectorizer()), ("clf", clf)])


def build_nb(alpha: float = 0.3) -> Pipeline:
    """Multinomial NB on the same union (the original's best family, for comparison)."""
    return Pipeline([("feats", make_vectorizer()), ("clf", MultinomialNB(alpha=alpha))])


def predict_with_dummy(
    pipe: Pipeline, texts: List[str], spam_mask: np.ndarray,
    conf_threshold: float = 0.0, dummy: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Final 3-way prediction: spam -> dummy; low-confidence -> dummy; else 0/1.

    The structural gate handles the bulk of spam. ``conf_threshold`` adds an
    optional second safety net: any *kept* document the sentiment model is
    unsure about (max class probability below the threshold) is also assigned
    the dummy label, since an ambiguous review is not worth a confident guess.
    """
    proba = pipe.predict_proba(texts)
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    final = pred.copy().astype(int)
    final[spam_mask] = dummy
    if conf_threshold > 0:
        final[(~spam_mask) & (conf < conf_threshold)] = dummy
    return final, conf
