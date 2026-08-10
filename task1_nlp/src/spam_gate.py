"""An unsupervised spam gate that uses the structure of the text (Task 1).

The data contains NO ground-truth spam labels. Thus this module models the
two types of document as a Gaussian Mixture with 2 components. The space of
the model is the space of the standardised structural features.

The GMM has these properties:

  * The GMM is generative and unsupervised. It finds the two types of
    document, the review and the email, from the structure of the data. It
    uses no labels. It does not use the centroids of the sentiment model.
    Thus the gate is independent of the sentiment features.
  * The GMM gives a calibrated posterior P(spam | structure). Thus the
    decision to keep a document or to remove it is a threshold on one
    probability. A person can understand this threshold. The code moves the
    threshold along a precision-recall curve. The brief asks for this sweep,
    and this module does it in a structural and probabilistic space.
  * A mixture model is a standard machine-learning method. This module uses
    it in a new way: spam is the smaller structural cluster.

The code selects the spam component automatically. The spam component is the
component with the larger mean log-length, because an email is always the
longer type of document. Thus the code does not use a fixed cluster index.
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
    """Fit a GMM with 2 components to the structural features.

    The class then gives the posterior probability of spam.
    """

    random_state: int = 42
    threshold: float = 0.5            # the decision threshold on P(spam | x)
    scaler: Optional[StandardScaler] = field(default=None, repr=False)
    gmm: Optional[GaussianMixture] = field(default=None, repr=False)
    spam_component_: int = -1

    # The feature log_length is column 0 in FEATURE_NAMES. The spam cluster is
    # the cluster with the longer documents.
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
        # Find the component that shows spam. It has the longer documents.
        means_len = self.gmm.means_[:, self._LEN_COL]
        self.spam_component_ = int(np.argmax(means_len))
        return self

    def spam_proba(self, texts) -> np.ndarray:
        """Calculate the posterior probability of spam for each document."""
        if self.gmm is None or self.scaler is None:
            raise RuntimeError("SpamGate must be fit before calling spam_proba.")
        Xs = self.scaler.transform(extract_matrix(texts))
        return self.gmm.predict_proba(Xs)[:, self.spam_component_]

    def predict(self, texts) -> np.ndarray:
        """Tell for each document if it is spam at the current threshold."""
        return self.spam_proba(texts) >= self.threshold

    def set_threshold(self, t: float) -> "SpamGate":
        self.threshold = float(t)
        return self

    def component_report(self) -> str:
        """Make a text comparison of the two clusters for a person to read."""
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
