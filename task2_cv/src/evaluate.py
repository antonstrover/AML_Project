"""Evaluation metrics and plots for face alignment (Task 2).

Metric (W09_L18, stated as a numbered equation in the report):

    For image i, landmark k, with prediction p_ik and ground truth g_ik,
    the normalised point-to-point error is

        e_ik = || p_ik - g_ik || / d_i      where d_i = || g_i,0 - g_i,1 ||

    i.e. the Euclidean error normalised by the inter-ocular distance (between
    the two eye landmarks, indices 0 and 1). Normalising removes the effect of
    image resolution and face size, exactly as the lecture requires.

We report the mean/median per landmark and the Cumulative Error Distribution
(CED): the fraction of points with normalised error below a sweeping
threshold. One CED line per approach on shared axes is the headline figure.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

LEFT_EYE, RIGHT_EYE = 0, 1


def normalised_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return (N, K) inter-ocular-normalised per-landmark errors."""
    d = np.linalg.norm(gt[:, LEFT_EYE] - gt[:, RIGHT_EYE], axis=1)      # (N,)
    d = np.maximum(d, 1e-6)
    dist = np.linalg.norm(pred - gt, axis=2)                            # (N, K)
    return dist / d[:, None]


def ced(errors_flat: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Cumulative Error Distribution: P(error <= t) for each t."""
    e = errors_flat.ravel()
    return np.array([(e <= t).mean() for t in thresholds])


def auc_ced(errors_flat: np.ndarray, max_thr: float = 0.10) -> float:
    """Area under the CED up to max_thr, normalised to [0,1] (higher = better)."""
    ts = np.linspace(0, max_thr, 100)
    c = ced(errors_flat, ts)
    return float(np.trapezoid(c, ts) / max_thr)


def per_landmark_summary(errors: np.ndarray) -> Dict[str, List[float]]:
    """Per-landmark mean and median normalised error."""
    return {"mean": errors.mean(axis=0).tolist(),
            "median": np.median(errors, axis=0).tolist()}


def plot_ced(curves: Dict[str, np.ndarray], thresholds: np.ndarray, path: str,
             title: str = "Cumulative Error Distribution"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.5, 4))
    for name, errs in curves.items():
        plt.plot(thresholds, ced(errs, thresholds), lw=2,
                 label=f"{name} (AUC={auc_ced(errs):.3f})")
    plt.xlabel("normalised error threshold"); plt.ylabel("fraction of points")
    plt.title(title); plt.legend(); plt.grid(alpha=.3); plt.ylim(0, 1)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_boxplots(curves: Dict[str, np.ndarray], path: str,
                  title: str = "Normalised error by approach"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.5, 4))
    plt.boxplot([e.ravel() for e in curves.values()], labels=list(curves.keys()),
                showfliers=False)
    plt.ylabel("normalised error"); plt.title(title)
    plt.xticks(rotation=15); plt.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
