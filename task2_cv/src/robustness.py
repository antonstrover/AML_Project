"""Robustness analysis harness.

Stress-tests a trained predictor on the validation set under increasing
perturbation and plots how the CED (or a summary error) degrades. The headline
comparison is a model trained WITH augmentation vs WITHOUT: the augmented model
should degrade more gracefully, demonstrating the W10_L20 claim that
augmentation "teaches the network transformations".

Perturbations (W09_L17/W10_L20 vocabulary):
    * Gaussian noise   (sigma sweep)         -- HOG's contrast normalisation
                                                and conv equivariance both help
    * rotation         (degree sweep)        -- HOG ~invariant to small rot
    * scale            (factor sweep)
    * brightness       (offset sweep)

A ``predict_fn(images) -> (N,5,2)`` abstraction lets the same harness drive the
CNN, the shape-model regressor or the mean-face baseline interchangeably.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from augment import affine, photometric
from evaluate import normalised_errors, auc_ced


def _perturb(images, kind: str, level: float, rng):
    out = []
    for im in images:
        if kind == "noise":
            out.append(photometric(im, noise_sigma=level, rng=rng))
        elif kind == "rotation":
            out.append(affine(im, np.zeros((1, 2)), angle_deg=level)[0])
        elif kind == "scale":
            out.append(affine(im, np.zeros((1, 2)), scale=level)[0])
        elif kind == "brightness":
            out.append(photometric(im, brightness=level, rng=rng))
        else:
            raise ValueError(kind)
    return np.array(out)


def _perturb_points(pts, kind, level, hw):
    """Apply the geometric perturbations to the ground-truth points too, so the
    error is measured against the perturbed target (noise/brightness leave
    points unchanged)."""
    if kind in ("noise", "brightness"):
        return pts
    h, w = hw
    out = np.empty_like(pts)
    for i, p in enumerate(pts):
        if kind == "rotation":
            out[i] = affine(np.zeros((h, w), np.float32), p, angle_deg=level)[1]
        elif kind == "scale":
            out[i] = affine(np.zeros((h, w), np.float32), p, scale=level)[1]
    return out


def robustness_curve(predict_fn: Callable, images: np.ndarray, gt: np.ndarray,
                     kind: str, levels: List[float], seed: int = 0) -> Dict[str, list]:
    """Return AUC-CED at each perturbation level for one predictor."""
    rng = np.random.default_rng(seed)
    hw = images.shape[1:3]
    aucs = []
    for lv in levels:
        imgs_p = _perturb(images, kind, lv, rng)
        gt_p = _perturb_points(gt, kind, lv, hw)
        pred = predict_fn(imgs_p)
        aucs.append(auc_ced(normalised_errors(pred, gt_p)))
    return {"levels": list(levels), "auc_ced": aucs}


UNITS = {"noise": "Gaussian noise sigma (intensity in [0,1])",
         "rotation": "rotation (degrees)",
         "scale": "scale factor",
         "brightness": "brightness offset"}


def plot_robustness(curves_by_model: Dict[str, Dict[str, list]], kind: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.0, 4.2))
    for name, c in curves_by_model.items():
        base = c["auc_ced"][0]
        retained = 100 * c["auc_ced"][-1] / base if base else 0.0
        plt.plot(c["levels"], c["auc_ced"], marker="o", lw=2,
                 label=f"{name} ({retained:.0f}% retained)")
    plt.xlabel(UNITS.get(kind, f"{kind} level"))
    plt.ylabel("AUC-CED (higher = better)")
    plt.title(f"Robustness to {kind}: the degradation gap is the argument",
              fontsize=11)
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.ylim(bottom=0)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()
