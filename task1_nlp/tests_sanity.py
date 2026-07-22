"""Sanity checks for Task 1 logic (fast, no training, no data files needed).

These do not claim sentiment accuracy -- that is what the validation set is for.
They verify the mechanics where text-pipeline bugs actually hide:

  1. light cleaning and tokenisation are stable and keep sentiment-bearing
     punctuation
  2. sequence encoding pads/truncates to a fixed width and maps OOV to <unk>
  3. the GloVe loader degrades gracefully when the vectors are absent
  4. the 3-way routing sends spam to the dummy label and nothing else does
  5. the worksheet's save_as_csv writes 1434 headerless rows and refuses
     anything else

Run:  python tests_sanity.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from sentiment_models import (GloVeBiLSTM, SoftVoteEnsemble, light_clean,
                              load_glove, mark_negation, predict_with_dummy,
                              remove_stopwords, tokenise)
from submission import save_as_csv

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond):
    results.append((name, PASS if cond else FAIL))
    print(f"[{PASS if cond else FAIL}] {name}")


# 1. cleaning + tokenisation ------------------------------------------------ #
check("light_clean lowercases and collapses whitespace",
      light_clean("  A   GREAT\tfilm\n") == "a great film")
check("light_clean maps a missing document to the empty string, not 'none'",
      light_clean(None) == "")
check("tokenise keeps sentiment punctuation",
      tokenise("Awful!! Really?") == ["awful", "!", "!", "really", "?"])
check("tokenise keeps clitics intact", "wasn't" in tokenise("It wasn't bad"))

# 1b. preprocessing variants used in the ablation --------------------------- #
check("negation marking scopes over the following words",
      mark_negation("not a good film") == "not not_a not_good not_film")
check("negation marking stops at punctuation",
      mark_negation("not good, brilliant") == "not not_good, brilliant")
check("negation marking triggers on clitics",
      mark_negation("it wasn't good") == "it wasn't not_good")
check("negation marking leaves unnegated text alone",
      mark_negation("a good film") == "a good film")
check("stopword removal keeps the negations it must not delete",
      set(remove_stopwords("this was not a good film").split()) >= {"not", "good"})

# 2. sequence encoding ------------------------------------------------------ #
m = GloVeBiLSTM(max_len=6, min_count=1)
m.vocab_ = m._build_vocab(["good film", "bad film"])
check("vocab reserves <pad>=0 and <unk>=1",
      m.vocab_["<pad>"] == 0 and m.vocab_["<unk>"] == 1)
enc = m._encode(["good film", "a b c d e f g h", ""])
check("encoding is (N, max_len)", enc.shape == (3, 6))
check("short sequences are right-padded with 0", enc[0, 2:].tolist() == [0, 0, 0, 0])
check("long sequences are truncated, not wrapped", (enc[1] != 0).all())
check("out-of-vocabulary tokens map to <unk>", set(enc[1].tolist()) == {1})
check("empty text encodes to all padding", (enc[2] == 0).all())

# 3. GloVe loader ----------------------------------------------------------- #
emb, found = load_glove(m.vocab_, dim=4, path="/definitely/not/a/file.txt")
check("missing GloVe file returns None so the caller can fall back",
      emb is None and found == 0)
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write("film 1 2 3 4\nzzzz 9 9 9 9\n")
    gpath = f.name
emb, found = load_glove(m.vocab_, dim=4, path=gpath)
check("GloVe matrix is (vocab, dim) with the pad row zeroed",
      emb.shape == (len(m.vocab_), 4) and np.allclose(emb[0], 0))
check("known word gets its pretrained vector",
      np.allclose(emb[m.vocab_["film"]], [1, 2, 3, 4]))
check("only in-vocabulary words are counted as found", found == 1)
os.unlink(gpath)


# 4. 3-way routing ---------------------------------------------------------- #
class _Stub:
    """Returns a fixed probability pair for every document."""
    def __init__(self, p):
        self.p = p

    def predict_proba(self, texts):
        return np.tile(self.p, (len(texts), 1))


# Always confidently positive, so any dummy label must come from routing.
_StubModel = lambda: _Stub([0.1, 0.9])


check("soft vote averages member probabilities",
      np.allclose(SoftVoteEnsemble({"a": _Stub([0.2, 0.8]), "b": _Stub([0.6, 0.4])})
                  .predict_proba(["x"]), [[0.4, 0.6]]))
check("soft vote can outvote a single confident member",
      SoftVoteEnsemble({"a": _Stub([0.9, 0.1]), "b": _Stub([0.45, 0.55]),
                        "c": _Stub([0.4, 0.6])}).predict(["x"])[0] == 0)

spam = np.array([False, True, False, True])
pred, conf = predict_with_dummy(_StubModel(), ["a", "b", "c", "d"], spam, dummy=-1)
check("spam rows get the dummy label", (pred[spam] == -1).all())
check("non-spam rows keep a 0/1 label", set(pred[~spam].tolist()) <= {0, 1})
check("confidence threshold routes unsure documents to the dummy too",
      (predict_with_dummy(_StubModel(), ["a"], np.array([False]),
                          conf_threshold=0.95, dummy=-1)[0] == -1).all())

# 5. submission format ------------------------------------------------------ #
with tempfile.TemporaryDirectory() as d:
    save_as_csv(np.array([0, 1, -1] * 478), d)
    lines = open(os.path.join(d, "results_task1.csv")).read().splitlines()
    check("save_as_csv writes 1434 rows", len(lines) == 1434)
    check("save_as_csv writes no header", lines[0].replace("-", "")[0].isdigit())
    check("the dummy label survives as a third class",
          len({float(x) for x in lines}) == 3)
    try:
        save_as_csv(np.zeros(1000), d)
        ok = False
    except AssertionError:
        ok = True
    check("save_as_csv rejects the wrong number of labels", ok)

# summary -------------------------------------------------------------------- #
n_pass = sum(s == PASS for _, s in results)
print(f"\n{n_pass}/{len(results)} checks passed")
sys.exit(0 if n_pass == len(results) else 1)
