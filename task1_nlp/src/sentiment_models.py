"""Sentiment models for Task 1 (my approach).

Deliberately *light* preprocessing -- lowercase + whitespace collapse
only -- and let a **combined word + character n-gram TF-IDF** representation
absorb morphology, negation and noise instead of hand-engineering it away.
Character n-grams (3-5) are the key move: they capture "not bad", "wasn't",
sub-word affixes and typos without any stemming/lemmatisation, and they make
the model robust to the messy email vocabulary that leaks past the spam gate.

Headline sparse model: a **calibrated Linear SVM** (hinge loss, the max-margin
discriminative classifier), with probabilities recovered via Platt scaling so
we can apply a confidence threshold for the dummy/spam fallback. We compare it
against the required word-list floor and Multinomial Naive Bayes so the report
has the multi-approach comparison the brief asks for.

Second method: a **BiLSTM over pretrained GloVe embeddings** (``GloVeBiLSTM`` below).
Every model above treats a review as a bag of (sub)strings, so it cannot
represent word *order*; the failure cases we mine are overwhelmingly compositional
-- negation scope ("not the worst film I have seen"), and mixed-polarity clauses
that praise the performance while panning the film. A recurrent model reading
the sentence left-to-right and right-to-left is the standard answer to exactly
that, and dense embeddings let rare words share evidence with their neighbours
instead of being pruned by ``min_df``.

Pretrained embeddings are neither additional labelled data nor a library
function that performs sentiment classification, so both of the brief's
restrictions remain satisfied.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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


# --------------------------------------------------------------------------- #
# Second method: BiLSTM over pretrained GloVe embeddings.                      #
# --------------------------------------------------------------------------- #
GLOVE_PATH = os.environ.get(
    "AML_GLOVE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "glove.6B.100d.txt"),
)

_RE_TOK = re.compile(r"[a-z0-9']+|[!?.,;:()\"]")


def tokenise(text: str) -> List[str]:
    """Regex word/punctuation tokeniser over the lightly cleaned text.

    Punctuation is *kept* as tokens: exclamation and question marks carry real
    sentiment signal in one-sentence reviews, and GloVe has vectors for them.
    """
    return _RE_TOK.findall(light_clean(text))


def load_glove(vocab: Dict[str, int], dim: int, path: str = GLOVE_PATH):
    """Build a (V, dim) embedding matrix from a GloVe text file.

    Rows for words absent from GloVe (and the <pad>/<unk> rows) stay at their
    small random initialisation. Returns (matrix, n_found) so the report can
    quote the coverage; if the file is missing, returns (None, 0) and the caller
    falls back to embeddings learned from scratch.
    """
    rng = np.random.default_rng(0)
    emb = rng.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    emb[0] = 0.0                                        # <pad>
    if not os.path.exists(path):
        return None, 0
    found = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word, _, rest = line.partition(" ")
            i = vocab.get(word)
            if i is not None:
                emb[i] = np.fromstring(rest, sep=" ", dtype=np.float32)
                found += 1
    return emb, found


@dataclass
class GloVeBiLSTM:
    """BiLSTM sentiment classifier with a scikit-learn-shaped interface.

    Design choices, each justified in the report:
      * embeddings frozen -- 8.5k short training sentences cannot re-estimate
        400k x 100 parameters without memorising, and freezing keeps the
        semantic geometry GloVe learned from 6B tokens intact;
      * one bidirectional layer, 128 units per direction -- the sentences are
        short, so depth buys nothing and costs overfitting;
      * max-over-time pooling of the hidden states rather than the final state,
        so a decisive clause anywhere in the sentence can carry the prediction;
      * dropout 0.4 before the classifier and early stopping on validation
        accuracy, the two regularisers that matter at this data scale.
    """
    dim: int = 100
    hidden: int = 128
    max_len: int = 60
    min_count: int = 2
    dropout: float = 0.4
    lr: float = 1e-3
    batch: int = 64
    epochs: int = 25
    patience: int = 4
    seed: int = 42
    vocab_: Dict[str, int] = field(default=None, repr=False)
    net_: object = field(default=None, repr=False)
    device_: str = field(default="cpu", repr=False)
    glove_coverage_: float = 0.0

    # -- helpers ----------------------------------------------------------- #
    def _build_vocab(self, texts: List[str]) -> Dict[str, int]:
        c = Counter(t for s in texts for t in tokenise(s))
        words = [w for w, n in c.most_common() if n >= self.min_count]
        return {"<pad>": 0, "<unk>": 1, **{w: i + 2 for i, w in enumerate(words)}}

    def _encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.max_len), dtype=np.int64)
        for i, s in enumerate(texts):
            ids = [self.vocab_.get(t, 1) for t in tokenise(s)][: self.max_len]
            out[i, : len(ids)] = ids
        return out

    @staticmethod
    def _device():
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"

    # -- sklearn-shaped API ------------------------------------------------ #
    def fit(self, texts: List[str], y, val: Optional[Tuple[List[str], np.ndarray]] = None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        self.device_ = self._device()
        self.vocab_ = self._build_vocab(texts)
        emb, found = load_glove(self.vocab_, self.dim)
        self.glove_coverage_ = found / max(1, len(self.vocab_))

        class Net(nn.Module):
            def __init__(s, n_vocab, dim, hidden, dropout, weights):
                super().__init__()
                s.emb = nn.Embedding(n_vocab, dim, padding_idx=0)
                if weights is not None:
                    s.emb.weight.data.copy_(torch.from_numpy(weights))
                    s.emb.weight.requires_grad = False       # frozen: see docstring
                s.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
                s.drop = nn.Dropout(dropout)
                s.fc = nn.Linear(hidden * 2, 2)

            def forward(s, x):
                h, _ = s.lstm(s.emb(x))
                h = h.masked_fill((x == 0).unsqueeze(-1), -1e9)   # ignore padding
                return s.fc(s.drop(h.max(dim=1).values))          # max-over-time

        self.net_ = Net(len(self.vocab_), self.dim, self.hidden, self.dropout, emb)
        self.net_.to(self.device_)

        X = torch.tensor(self._encode(texts))
        Y = torch.tensor(np.asarray(y), dtype=torch.long)
        dl = DataLoader(TensorDataset(X, Y), batch_size=self.batch, shuffle=True)
        params = [p for p in self.net_.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=self.lr)
        lossf = nn.CrossEntropyLoss()

        best, best_state, since = -1.0, None, 0
        for ep in range(self.epochs):
            self.net_.train()
            for xb, yb in dl:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                opt.zero_grad()
                lossf(self.net_(xb), yb).backward()
                opt.step()
            if val is None:
                continue
            acc = float((self.predict(val[0]) == np.asarray(val[1])).mean())
            if acc > best:
                best, since = acc, 0
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                since += 1
                if since >= self.patience:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        import torch
        self.net_.eval()
        X = torch.tensor(self._encode(list(texts)))
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                logits = self.net_(X[i:i + 256].to(self.device_))
                out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2))

    def predict(self, texts: List[str]) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=1)


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
