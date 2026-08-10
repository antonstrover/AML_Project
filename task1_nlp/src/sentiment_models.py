"""Sentiment models for Task 1.

The first hypothesis of this project was that heavy preprocessing is not
necessary. Thus the code starts with light preprocessing: it changes the text
to lower case and removes the unwanted spaces. Then a TF-IDF representation of
word n-grams and character n-grams absorbs the morphology and the noise.

The character n-grams of 3 to 5 characters are the important part. They
contain the data of a word such as "wasn't". They also contain sub-word parts
and spelling errors. Thus the code does not do a stemming operation or a
lemmatisation operation. The character n-grams also make the model reliable
for the email vocabulary that goes through the spam gate.

The ablation in run_task1.representation_ablation tested this hypothesis. Only
one half of the hypothesis was correct:

  * The removal of the stopwords decreases the accuracy. The lemmatisation
    also decreases the accuracy. The hypothesis gives these two results
    correctly.
  * The negation marks increase the accuracy. A mark on the clause after the
    word "not" adds approximately 1.8 points to the SVM, together with the
    character n-grams. Thus the character n-grams find the negation word, but
    they do not find the part of the sentence that the negation changes.

Thus the default preprocessor below is mark_negation. This measurement gives
the default, not the initial hypothesis.

The primary sparse model is a calibrated Linear SVM. It uses the hinge loss
and finds the maximum margin between the two classes. Platt scaling then gives
the probabilities. Thus the code can apply a threshold on the confidence and
give the dummy label. The report compares this model with the word-list model
and with a Multinomial Naive Bayes model. The brief asks for a comparison of
more than one approach.

The second method is a BiLSTM with pretrained GloVe embeddings. The class
GloVeBiLSTM below contains it. Each other model uses a review as a bag of
strings. Thus each other model cannot use the order of the words. Almost all
the failure cases have a structure of this type:

  * The negation changes only one part of the sentence, for example "not the
    worst film I have seen".
  * Different clauses have a different polarity, for example a clause that
    speaks well of the performance and a clause that speaks badly of the film.

A recurrent model reads the sentence from left to right and from right to
left. This is the usual method for these failure cases. Also, a dense
embedding lets a rare word use the data of the words near it. The parameter
min_df does not remove such a word.

A pretrained embedding is not additional labelled data. It is also not a
library function that does sentiment classification. Thus the code obeys the
two limits in the brief.
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
    """Change the text to lower case and remove the unwanted spaces.

    The function does no other operation.
    """
    t = "" if text is None else str(text)
    return _RE_WS.sub(" ", t.lower()).strip()


# --------------------------------------------------------------------------- #
# The preprocessing variants. The ablation compares them.                      #
#                                                                              #
# This project makes a claim: heavy manual preprocessing is not necessary if   #
# the representation contains sub-word data. A test of the claim is necessary. #
# Each variant below is a preprocessor for TfidfVectorizer. Thus the ablation  #
# changes one item only, and the classifier stays the same.                    #
# --------------------------------------------------------------------------- #
_NEG_WORDS = {"not", "no", "never", "none", "cannot", "n't", "without", "hardly",
              "barely", "scarcely"}
_NEG_STOP = re.compile(r"[.,;:!?()\"]")

_lemmatiser = None
_stopwords = None


def _nltk_stopwords():
    """Give the English stopwords, but keep the negation words.

    A negation word such as "not" contains the data that the sentiment task
    needs. If the code removes it, the accuracy decreases.
    """
    global _stopwords
    if _stopwords is None:
        from nltk.corpus import stopwords
        _stopwords = set(stopwords.words("english")) - _NEG_WORDS - {"but", "very"}
    return _stopwords


def remove_stopwords(text: str) -> str:
    """Change the text to lower case and remove the frequent function words.

    The function keeps the negation words.
    """
    sw = _nltk_stopwords()
    return " ".join(w for w in light_clean(text).split() if w not in sw)


def lemmatise(text: str) -> str:
    """Change the text to lower case and apply the WordNet lemmatiser.

    The lemmatiser changes the different forms of a word into one form.
    """
    global _lemmatiser
    if _lemmatiser is None:
        from nltk.stem import WordNetLemmatizer
        _lemmatiser = WordNetLemmatizer()
    return " ".join(_lemmatiser.lemmatize(w) for w in light_clean(text).split())


def mark_negation(text: str) -> str:
    """Add the prefix "not_" to the tokens after a negation word.

    The function adds the prefix to each token between the negation word and
    the next punctuation mark.

    This method is the usual method to find the part of the sentence that a
    negation changes (Das and Chen 2001, Pang 2002). It makes "not good" a
    different token from "good". A bag of single words cannot show this
    difference. The character n-grams show part of this difference. Thus the
    ablation measures the remaining part.
    """
    out, negating = [], False
    for raw in light_clean(text).split():
        # First add the prefix, then change the state. The token with the
        # punctuation mark at its end, for example "good,", is still in the
        # negation. Only the token after it is outside of the negation.
        out.append("not_" + raw if negating else raw)
        if _NEG_STOP.search(raw):
            negating = False
        elif raw in _NEG_WORDS or raw.endswith("n't"):
            negating = True
    return " ".join(out)


PREPROCESSORS = {
    "lowercase only": light_clean,
    "+ stopword removal": remove_stopwords,
    "+ lemmatisation": lemmatise,
    "+ negation marking": mark_negation,
}


# --------------------------------------------------------------------------- #
# The most simple model. The brief makes it necessary. It uses two word lists. #
# --------------------------------------------------------------------------- #
@dataclass
class WordListClassifier:
    """Classify a text with two word lists.

    The score is the number of the words from the positive list minus the
    number of the words from the negative list.

    The code makes the lists from the difference of the frequencies. The
    positive list contains the K words that are most frequent in the positive
    documents. The negative list contains the K words that are most frequent
    in the negative documents.

    The class gives the class 1 if the score is more than delta. If not, it
    gives the class 0. This model is the minimum reference, and a person can
    understand its decisions.
    """
    K: int = 400
    delta: float = 0.0
    pos_set: set = None
    neg_set: set = None

    def fit(self, texts: List[str], y: np.ndarray) -> "WordListClassifier":
        pos_c, neg_c = Counter(), Counter()
        for t, lab in zip(texts, y):
            toks = set(light_clean(t).split())          # count each word one
                                                        # time in each document
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
# The TF-IDF representation of the words and the characters. More than one     #
# model uses it.                                                               #
# --------------------------------------------------------------------------- #
def make_vectorizer(pre=light_clean, kinds=("word", "char")):
    """Make a TF-IDF vectorizer with a given preprocessor.

    The word part uses n-grams of 1 to 2 words. The character part uses
    n-grams of 3 to 5 characters.

    The argument kinds selects the two parts. The default selects both parts,
    which is the primary feature set. The argument pre gives the preprocessor
    for the test. Thus one sweep can test each preprocessor with each
    representation.
    """
    parts = []
    if "word" in kinds:
        parts.append(("word", TfidfVectorizer(
            preprocessor=pre, analyzer="word", ngram_range=(1, 2),
            sublinear_tf=True, min_df=3, max_df=0.9, strip_accents="unicode",
        )))
    if "char" in kinds:
        parts.append(("char", TfidfVectorizer(
            preprocessor=pre, analyzer="char_wb", ngram_range=(3, 5),
            sublinear_tf=True, min_df=3, max_df=0.95,
        )))
    return FeatureUnion(parts)


def build_svm(C: float = 1.0, random_state: int = 42, pre=mark_negation) -> Pipeline:
    """Make a calibrated Linear SVM on the TF-IDF features.

    The features contain the word part and the character part.
    """
    base = LinearSVC(C=C, class_weight="balanced", random_state=random_state)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    return Pipeline([("feats", make_vectorizer(pre)), ("clf", clf)])


def build_nb(alpha: float = 0.3, pre=mark_negation) -> Pipeline:
    """Make a Multinomial Naive Bayes model on the same features.

    This model is the best model of the initial approach. The report uses it
    for the comparison.
    """
    return Pipeline([("feats", make_vectorizer(pre)), ("clf", MultinomialNB(alpha=alpha))])


# --------------------------------------------------------------------------- #
# The second method: a BiLSTM with pretrained GloVe embeddings.                #
# --------------------------------------------------------------------------- #
GLOVE_PATH = os.environ.get(
    "AML_GLOVE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "glove.6B.100d.txt"),
)

_RE_TOK = re.compile(r"[a-z0-9']+|[!?.,;:()\"]")


def tokenise(text: str) -> List[str]:
    """Divide the cleaned text into word tokens and punctuation tokens.

    The function uses a regex.

    The function keeps each punctuation mark as a token. An exclamation mark
    and a question mark contain sentiment data in a review of one sentence.
    GloVe also has a vector for each of these marks.
    """
    return _RE_TOK.findall(light_clean(text))


def load_glove(vocab: Dict[str, int], dim: int, path: str = GLOVE_PATH):
    """Make a (V, dim) embedding matrix from a GloVe text file.

    GloVe does not contain each word. The row of such a word keeps its initial
    small random values. The rows <pad> and <unk> also keep these values.

    The function returns (matrix, n_found). Thus the report can give the
    fraction of the words in GloVe. If the file does not exist, the function
    returns (None, 0). Then the caller learns the embeddings from the training
    data.
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
    """A BiLSTM sentiment classifier with an interface of the scikit-learn type.

    The report gives the reason for each design decision:

      * The code does not train the embeddings. The training set has 8500
        short sentences. This quantity of data cannot estimate 400000 x 100
        parameters, and the model only memorises the data. A frozen embedding
        keeps the semantic structure that GloVe learned from 6 billion tokens.
      * The model has one bidirectional layer with 128 units in each
        direction. The sentences are short. Thus more layers do not increase
        the accuracy, and they increase the overfit.
      * The model takes the maximum of each hidden state across the time
        steps. It does not use the last state only. Thus an important clause
        at any position in the sentence can change the prediction.
      * The model uses a dropout of 0.4 before the classifier. The training
        also stops early when the validation accuracy stops to increase. These
        two methods control the overfit for this quantity of data.
    """
    dim: int = 100
    hidden: int = 128
    max_len: int = 60
    min_count: int = 1     # Keep each word. GloVe gives a good vector for a
                           # word that occurs one time only. A larger value
                           # removes that word and its data.
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

    # -- the internal functions -------------------------------------------- #
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

    # -- the interface of the scikit-learn type ---------------------------- #
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
                    # The training does not change the embeddings. The
                    # docstring of the class gives the reason.
                    s.emb.weight.requires_grad = False
                s.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
                s.drop = nn.Dropout(dropout)
                s.fc = nn.Linear(hidden * 2, 2)

            def forward(s, x):
                h, _ = s.lstm(s.emb(x))
                h = h.masked_fill((x == 0).unsqueeze(-1), -1e9)   # ignore the pad
                return s.fc(s.drop(h.max(dim=1).values))          # the maximum
                                                                  # across the
                                                                  # time steps

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


@dataclass
class SoftVoteEnsemble:
    """Calculate the mean of the calibrated probabilities of the models.

    Each model must be fit before you use this class.

    The accuracy of the BiLSTM is not the only reason for a sequence model.
    The BiLSTM makes different errors from a bag of n-grams, because it is the
    only model that can use the order of the words. The mean of the
    probabilities changes this difference into a better accuracy. Thus the
    ensemble is more accurate than each of its models, and also more accurate
    than the two models with the same individual accuracy.

    Each model must have a function predict_proba. Each model must use the
    same order of the classes. This class fits no model. Thus it adds no
    training time, and its only parameter is the set of the models.
    """
    members: Dict[str, object] = field(default_factory=dict)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        texts = list(texts)
        return np.mean([m.predict_proba(texts) for m in self.members.values()], axis=0)

    def predict(self, texts: List[str]) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=1)


def predict_with_dummy(
    pipe: Pipeline, texts: List[str], spam_mask: np.ndarray,
    conf_threshold: float = 0.0, dummy: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the final prediction, which has three possible values.

    The function gives the dummy label to a spam document. It also gives the
    dummy label to a document with a low confidence. If not, it gives the
    class 0 or the class 1.

    The structural gate finds almost all the spam. The argument conf_threshold
    adds a second protection, and it is optional. The gate keeps some
    documents, and the sentiment model is not sure about some of them. Their
    maximum class probability is less than the threshold. The function gives
    the dummy label to these documents, because a guess about an unclear
    review is not correct sufficiently frequently.
    """
    proba = pipe.predict_proba(texts)
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    final = pred.copy().astype(int)
    final[spam_mask] = dummy
    if conf_threshold > 0:
        final[(~spam_mask) & (conf < conf_threshold)] = dummy
    return final, conf
