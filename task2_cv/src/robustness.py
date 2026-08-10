"""Tools for the analysis of the robustness.

This module applies a changed version of the validation set to a trained
model. The change becomes larger at each step. The module then plots the
decrease of the CED or of a summary error.

The primary comparison is a model with augmentation against a model without
augmentation. The error of the augmented model must increase more slowly.
This result shows the statement in W10_L20: an augmentation teaches the
transformations to the network.

The module applies these changes. The words are from W09_L17 and W10_L20:

    * Gaussian noise, with a sweep of sigma. The contrast normalisation of the
      HOG and the equivariance of a convolution both decrease this effect.
    * A rotation, with a sweep of the angle in degrees. The HOG is
      approximately invariant to a small rotation.
    * A scale, with a sweep of the factor.
    * A brightness offset, with a sweep of the offset.

The module calls a function predict_fn(images) that returns an (N,5,2) array.
Thus the same code can test the CNN, the shape-model regressor and the
mean-face baseline.
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
    """Apply the geometric change to the ground-truth landmarks.

    Thus the code measures the error against the changed target. Noise and a
    brightness offset do not move a landmark.
    """
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
    """Calculate the AUC-CED of one model at each level of the change."""
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
