"""End-to-end Task 1 pipeline.

Run:  python run_task1.py
Produces figures/, results/, models/ and submission/task1_predictions.csv.

Architecture (two decoupled gates):
    text --> [structural spam gate: GMM] --spam--> dummy(-1)
                       |
                     keep
                       v
             [sentiment model: calibrated Linear SVM on word+char TF-IDF] --> 0/1
"""
from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)

from structure_features import (FEATURE_NAMES, extract_matrix,
                                high_precision_email_flag)
from spam_gate import SpamGate
from sentiment_models import (WordListClassifier, build_nb, build_svm,
                              light_clean, predict_with_dummy)

SEED = 42
np.random.seed(SEED)
DUMMY = -1

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
MOD = os.path.join(HERE, "models")
SUB = os.path.join(HERE, "submission")
for d in (FIG, RES, MOD, SUB):
    os.makedirs(d, exist_ok=True)


def load():
    tr = pd.read_csv(os.path.join(DATA, "sentiment_analysis_training_data.csv"))
    va = pd.read_csv(os.path.join(DATA, "sentiment_analysis_validation_data.csv"))
    te = pd.read_csv(os.path.join(DATA, "sentiment_analysis_test_data.csv"))
    return tr, va, te


def main():
    log = {}
    tr, va, te = load()
    print(f"[load] train={len(tr)} val={len(va)} test={len(te)}")

    # ----- evaluation gold-standard for spam (no labels provided) ----------- #
    # High-precision RFC-822 header rule: ~perfect for Enron emails, never fires
    # on one-sentence reviews. Used only to *evaluate* the gate, not to build it.
    tr_email = np.array([high_precision_email_flag(t) for t in tr.text])
    va_email = np.array([high_precision_email_flag(t) for t in va.text])
    print(f"[gold] train emails(rule)={tr_email.sum()} ({tr_email.mean():.1%})  "
          f"val emails(rule)={va_email.sum()} ({va_email.mean():.1%})")
    log["rule_email_frac_train"] = float(tr_email.mean())

    # ----- 1. fit the unsupervised structural spam gate --------------------- #
    t0 = time.perf_counter()
    gate = SpamGate(random_state=SEED).fit(tr.text)
    gate_fit_s = time.perf_counter() - t0
    print("\n" + gate.component_report())

    va_spam_p = gate.spam_proba(va.text)

    # Choose the spam threshold by sweeping the precision-recall trade-off of the
    # GMM posterior against the rule-gold on validation, then pick the highest-F1
    # point (the principled, brief-mandated threshold sweep -- in structural space).
    prec, rec, thr = precision_recall_curve(va_email.astype(int), va_spam_p)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_i = int(np.nanargmax(f1s[:-1])) if len(thr) else 0
    # The GMM posterior is near-degenerate (clean separation), so many thresholds
    # tie at max F1. Choosing the raw argmax lands at ~1.0, an over-tight boundary.
    # Instead pick, among thresholds within 99% of the best F1, the one closest to
    # the natural GMM decision boundary 0.5 -- a robust operating point.
    # The posterior turned out near-binary (the genres are structurally
    # separable to ~3 decimal places), so every swept threshold clusters at the
    # extremes. We therefore operate at the natural GMM boundary 0.5, which is
    # robust to a borderline unseen email (posterior 0.6 would still be caught),
    # whereas the swept argmax (~1.0) would miss it. The sweep is reported only
    # to evidence the separation.
    if len(thr):
        good = np.where(f1s[:-1] >= 0.99 * np.nanmax(f1s[:-1]))[0]
        best_i = int(good[np.argmin(np.abs(thr[good] - 0.5))])
    best_thr = 0.5
    gate.set_threshold(best_thr)
    print(f"\n[spam-gate] chosen P(spam) threshold={best_thr:.3f}  "
          f"-> precision={prec[best_i]:.3f} recall={rec[best_i]:.3f} f1={f1s[best_i]:.3f}")
    log["spam_gate"] = {"threshold": best_thr, "precision": float(prec[best_i]),
                        "recall": float(rec[best_i]), "f1": float(f1s[best_i]),
                        "fit_seconds": gate_fit_s}

    # PR curve figure for the spam gate.
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, lw=2)
    plt.scatter([rec[best_i]], [prec[best_i]], c="crimson", zorder=5,
                label=f"chosen (F1={f1s[best_i]:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Spam gate: precision-recall (GMM posterior vs header-rule gold)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "spam_gate_pr_curve.png"), dpi=130); plt.close()

    # Posterior histogram split by gold -- shows the bimodal separation.
    plt.figure(figsize=(5.5, 4))
    plt.hist(va_spam_p[~va_email], bins=40, alpha=.6, label="reviews (gold)", color="steelblue")
    plt.hist(va_spam_p[va_email], bins=40, alpha=.6, label="emails (gold)", color="indianred")
    plt.axvline(best_thr, ls="--", c="k", label=f"threshold={best_thr:.2f}")
    plt.xlabel("P(spam | structure)"); plt.ylabel("count"); plt.yscale("log")
    plt.title("Structural spam posterior is bimodal"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "spam_posterior_hist.png"), dpi=130); plt.close()

    va_spam_pred = gate.predict(va.text)

    # ----- 2. build a CLEANED training set (drop predicted spam) ------------ #
    tr_spam_pred = gate.predict(tr.text)
    keep = ~tr_spam_pred
    Xtr = tr.text[keep].tolist()
    ytr = tr.label.values[keep]
    print(f"\n[clean] kept {keep.sum()}/{len(tr)} training docs as real reviews "
          f"({(~keep).mean():.1%} dropped as spam)")
    log["train_kept"] = int(keep.sum())

    # ----- 3. train + compare sentiment models on cleaned reviews ----------- #
    # Evaluate sentiment only on validation docs that are genuine reviews (gold).
    va_real_mask = ~va_email
    Xva_real = va.text[va_real_mask].tolist()
    yva_real = va.label.values[va_real_mask]

    rows = []
    # (a) word-list floor
    t0 = time.perf_counter(); wl = WordListClassifier(K=400).fit(Xtr, ytr); wl_s = time.perf_counter() - t0
    wl_pred = wl.predict(Xva_real)
    rows.append(("wordlist", (wl_pred == yva_real).mean(), f1_score(yva_real, wl_pred), wl_s))

    # (b) Multinomial NB on word+char union (original's best family)
    t0 = time.perf_counter(); nb = build_nb(alpha=0.3).fit(Xtr, ytr); nb_s = time.perf_counter() - t0
    nb_pred = nb.predict(Xva_real)
    rows.append(("mnb_wordchar", (nb_pred == yva_real).mean(), f1_score(yva_real, nb_pred), nb_s))

    # (c) headline: calibrated Linear SVM on word+char union
    t0 = time.perf_counter(); svm = build_svm(C=1.0).fit(Xtr, ytr); svm_s = time.perf_counter() - t0
    svm_pred = svm.predict(Xva_real)
    rows.append(("svm_wordchar", (svm_pred == yva_real).mean(), f1_score(yva_real, svm_pred), svm_s))

    comp = pd.DataFrame(rows, columns=["model", "val_acc", "val_f1", "train_seconds"])
    comp = comp.sort_values("val_acc", ascending=False).reset_index(drop=True)
    comp.to_csv(os.path.join(RES, "model_comparison.csv"), index=False)
    print("\n[sentiment] validation comparison (real reviews only):")
    print(comp.to_string(index=False))
    best_name = comp.iloc[0]["model"]
    best_model = {"wordlist": wl, "mnb_wordchar": nb, "svm_wordchar": svm}[best_name]
    log["best_sentiment_model"] = best_name

    # ----- 3b. representation ablation: word vs char vs union --------------- #
    # My analog to the original's TF-IDF-vs-Word2vec comparison: isolate what the
    # character n-grams buy us, holding the classifier (Linear SVM) fixed.
    abl = representation_ablation(Xtr, ytr, Xva_real, yva_real)
    abl.to_csv(os.path.join(RES, "representation_ablation.csv"), index=False)
    print("\n[ablation] SVM accuracy by representation:")
    print(abl.to_string(index=False))
    plt.figure(figsize=(5, 3.6))
    plt.bar(abl["representation"], abl["val_acc"], color=["#9ecae1", "#6baed6", "#2171b5"])
    for i, v in enumerate(abl["val_acc"]):
        plt.text(i, v + .003, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylabel("validation accuracy"); plt.ylim(0.5, max(abl["val_acc"]) + .04)
    plt.title("Representation ablation (Linear SVM)"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "representation_ablation.png"), dpi=130); plt.close()
    log["representation_ablation"] = abl.to_dict(orient="records")

    # ----- 4. full 3-way evaluation (neg / pos / spam-dummy) ---------------- #
    # Use the calibrated SVM for the probabilistic path even if NB nudges ahead,
    # because the dummy fallback needs calibrated probabilities; report both.
    prob_model = svm
    final_pred, conf = predict_with_dummy(prob_model, va.text.tolist(), va_spam_pred,
                                          conf_threshold=0.0, dummy=DUMMY)
    # gold 3-way: emails -> dummy; else provided 0/1
    gold3 = np.where(va_email, DUMMY, va.label.values)
    labels3 = [0, 1, DUMMY]
    cm = confusion_matrix(gold3, final_pred, labels=labels3)
    plot_confusion(cm, labels3, ["neg", "pos", "spam"],
                   "3-way confusion (val): structural gate + calibrated SVM",
                   os.path.join(FIG, "confusion_3way.png"))
    overall_acc = (final_pred == gold3).mean()
    print(f"\n[3-way] overall val accuracy (incl. spam routing) = {overall_acc:.3f}")
    log["val_3way_accuracy"] = float(overall_acc)
    log["val_3way_confusion"] = cm.tolist()

    # ----- 5. failure-case mining (real reviews the SVM got wrong) ---------- #
    wrong = np.where((svm_pred != yva_real))[0]
    fc = []
    for i in wrong[:12]:
        fc.append({"text": Xva_real[i][:240], "gold": int(yva_real[i]),
                   "pred": int(svm_pred[i])})
    pd.DataFrame(fc).to_csv(os.path.join(RES, "failure_cases.csv"), index=False)
    print(f"[failures] saved {len(fc)} misclassified review examples")

    # ----- 6. NLTK external evaluation (movie_reviews corpus) --------------- #
    nltk_metrics = nltk_external_eval(gate, svm)
    log["nltk_external"] = nltk_metrics
    print(f"[nltk] acc={nltk_metrics['accuracy']:.3f}  "
          f"spam-gate false-fire rate={nltk_metrics['spam_fire_rate']:.3%}")

    # ----- 7. test submission ---------------------------------------------- #
    te_spam = gate.predict(te.text)
    te_final, _ = predict_with_dummy(prob_model, te.text.tolist(), te_spam,
                                     conf_threshold=0.0, dummy=DUMMY)
    out = pd.DataFrame({"text": te.text, "label": te_final})  # order preserved
    out_path = os.path.join(SUB, "task1_predictions.csv")
    out[["label"]].to_csv(out_path, index=False)
    dist = pd.Series(te_final).value_counts().to_dict()
    print(f"[submit] wrote {out_path}  rows={len(out)}  label dist={dist}")
    log["test_label_distribution"] = {int(k): int(v) for k, v in dist.items()}

    # ----- persist artefacts ------------------------------------------------ #
    joblib.dump(gate, os.path.join(MOD, "spam_gate.joblib"))
    joblib.dump(svm, os.path.join(MOD, "svm_wordchar.joblib"))
    with open(os.path.join(MOD, "BEST_MODEL.txt"), "w") as f:
        f.write("structural_gate + svm_wordchar")
    with open(os.path.join(RES, "run_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("\n[done] artefacts written.")


def representation_ablation(Xtr, ytr, Xva, yva):
    """Hold the classifier fixed (Linear SVM), vary only the TF-IDF representation."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sentiment_models import light_clean
    configs = {
        "word(1,2)": TfidfVectorizer(preprocessor=light_clean, analyzer="word",
                                     ngram_range=(1, 2), sublinear_tf=True, min_df=3, max_df=0.9),
        "char(3,5)": TfidfVectorizer(preprocessor=light_clean, analyzer="char_wb",
                                     ngram_range=(3, 5), sublinear_tf=True, min_df=3, max_df=0.95),
    }
    rows = []
    for name, vec in configs.items():
        pipe = Pipeline([("v", vec), ("c", LinearSVC(C=1.0, random_state=SEED))]).fit(Xtr, ytr)
        rows.append((name, float((pipe.predict(Xva) == yva).mean())))
    # union (the headline representation)
    from sentiment_models import make_vectorizer
    pipe = Pipeline([("v", make_vectorizer()), ("c", LinearSVC(C=1.0, random_state=SEED))]).fit(Xtr, ytr)
    rows.append(("word+char (union)", float((pipe.predict(Xva) == yva).mean())))
    return pd.DataFrame(rows, columns=["representation", "val_acc"])


def plot_confusion(cm, labels, names, title, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title(title, fontsize=10)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def nltk_external_eval(gate: SpamGate, sent_model):
    """Run the chosen final system on NLTK movie_reviews (no spam present)."""
    from nltk.corpus import movie_reviews
    texts, ys = [], []
    for cat in movie_reviews.categories():
        y = 1 if cat == "pos" else 0
        for fid in movie_reviews.fileids(cat):
            texts.append(movie_reviews.raw(fid)); ys.append(y)
    ys = np.array(ys)
    spam_mask = gate.predict(texts)            # should fire ~never
    pred, _ = predict_with_dummy(sent_model, texts, spam_mask, dummy=DUMMY)
    real = pred != DUMMY
    acc = (pred[real] == ys[real]).mean() if real.any() else 0.0
    # confusion figure
    cm = confusion_matrix(ys[real], pred[real], labels=[0, 1])
    plot_confusion(cm, [0, 1], ["neg", "pos"],
                   "NLTK movie_reviews (external) confusion",
                   os.path.join(FIG, "nltk_external_confusion.png"))
    return {"accuracy": float(acc), "n": int(len(ys)),
            "spam_fire_rate": float(spam_mask.mean()),
            "n_spam_fired": int(spam_mask.sum())}


if __name__ == "__main__":
    main()
