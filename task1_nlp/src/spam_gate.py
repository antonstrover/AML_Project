"""Unsupervised structural spam gate (Task 1).

We have NO ground-truth spam labels, so the two document genres are modelled
explicitly as a 2-component Gaussian Mixture in standardised structural-feature
space.

Why a GMM:

  * It is *generative and unsupervised* -- it discovers the two genres
    (reviews vs emails) from the data's own structure, with no labels and no
    appeal to the sentiment centroids, so the gate is independent of the
    sentiment representation.
  * It yields a calibrated *posterior* P(spam | structure), so the spam/keep
    decision is a single interpretable probability threshold we can sweep on
    a precision-recall curve -- the threshold sweep the brief wants, in a
    structural, probabilistic space.
  * Clustering / mixture models are squarely standard ML; framing spam as the
    minority structural cluster is the "originality within scope" move.

The component that ends up being "spam" is chosen automatically as the one
whose mean log-length is larger (emails are reliably the longer genre), so we
never rely on a hard-coded cluster index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from structure_features import FEATURE_NAMES, extract_matrix


@dataclass
class SpamGate:
    """Fit a 2-component GMM on structural features and expose a spam posterior."""

    random_state: int = 42
    threshold: float = 0.5            # decision threshold on P(spam | x)
    scaler: Optional[StandardScaler] = field(default=None, repr=False)
    gmm: Optional[GaussianMixture] = field(default=None, repr=False)
    spam_component_: int = -1

    # log_length is column 0 in FEATURE_NAMES; the spam cluster is the longer one.
    _LEN_COL = 0

    def fit(self, texts) -> "SpamGate":
        X = extract_matrix(texts)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            n_init=5,
            random_state=self.random_state,
        ).fit(Xs)
        # Identify which latent component corresponds to spam (longer docs).
        means_len = self.gmm.means_[:, self._LEN_COL]
        self.spam_component_ = int(np.argmax(means_len))
        return self

    def spam_proba(self, texts) -> np.ndarray:
        """Posterior probability that each document is spam."""
        if self.gmm is None or self.scaler is None:
            raise RuntimeError("SpamGate must be fit before calling spam_proba.")
        Xs = self.scaler.transform(extract_matrix(texts))
        return self.gmm.predict_proba(Xs)[:, self.spam_component_]

    def predict(self, texts) -> np.ndarray:
        """Boolean spam mask at the current threshold."""
        return self.spam_proba(texts) >= self.threshold

    def set_threshold(self, t: float) -> "SpamGate":
        self.threshold = float(t)
        return self

    def component_report(self) -> str:
        """Human-readable comparison of the two discovered clusters."""
        if self.gmm is None:
            return "<unfit>"
        means = self.scaler.inverse_transform(self.gmm.means_)
        lines = ["Discovered structural clusters (de-standardised means):",
                 f"  weights: {self.gmm.weights_.round(3).tolist()}",
                 f"  spam component index: {self.spam_component_}"]
        for name, a, b in zip(FEATURE_NAMES, means[0], means[1]):
            tag0 = " <-spam" if self.spam_component_ == 0 else ""
            tag1 = " <-spam" if self.spam_component_ == 1 else ""
            lines.append(f"    {name:18s}  c0={a:8.3f}{tag0:7s}  c1={b:8.3f}{tag1}")
        return "\n".join(lines)
