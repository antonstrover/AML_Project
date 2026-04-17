"""Heatmap targets and soft-argmax decoding.

My approach predicts, for each of the 5 landmarks, a low-resolution
**heatmap** whose target is a 2-D Gaussian blob centred on the true
point, and decodes coordinates with **soft-argmax**.

Why this is the more interesting (and more robust) deep approach -- straight
from W09_L18:

  * The lecture notes that "the arg-max is not differentiable, so [we] need to
    create heatmap images as targets or use soft-argmax." Heatmaps make the
    spatial structure of the problem explicit.
  * Errors in heatmap regression are *spatially local*: a confused pixel
    nudges one landmark slightly, whereas a confused unit in a direct-
    regression head can throw a coordinate far across the image. This is the
    mechanism behind heatmap models' empirically better robustness, which I
    test in robustness.py.

This module is pure NumPy so the target-generation and decoding logic can be
unit-tested without a GPU. The CNN itself (model.py) consumes/produces the
same tensors in PyTorch.
"""
from __future__ import annotations

import numpy as np


def make_heatmaps(pts: np.ndarray, out_hw, sigma: float = 1.5) -> np.ndarray:
    """Return (K, Hh, Wh) Gaussian heatmaps for K landmarks.

    pts: (K, 2) landmark coords already scaled to the heatmap resolution.
    """
    Hh, Wh = out_hw
    K = pts.shape[0]
    ys = np.arange(Hh)[:, None]
    xs = np.arange(Wh)[None, :]
    hm = np.zeros((K, Hh, Wh), dtype=np.float32)
    for k, (px, py) in enumerate(pts):
        g = np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma ** 2))
        s = g.sum()
        hm[k] = (g / s) if s > 0 else g          # normalised -> a spatial pmf
    return hm


def soft_argmax(hm: np.ndarray, beta: float = 1.0, from_logits: bool = False) -> np.ndarray:
    """Decode (K, Hh, Wh) heatmaps to (K, 2) coords via spatial expectation.

    Two regimes:
      * ``from_logits=False`` (default): the heatmap is already a non-negative
        map (e.g. a Gaussian target or a ReLU'd prediction). We renormalise it
        to a spatial pmf and take the expected (x, y). This is the correct
        eval-time decode -- applying an extra softmax here would wash a sharp
        peak out toward the image centre.
      * ``from_logits=True``: the heatmap holds raw (possibly negative) network
        logits, so we apply a temperature-``beta`` spatial softmax first. This
        mirrors the differentiable torch decode in model.py.
    """
    K, Hh, Wh = hm.shape
    if from_logits:
        flat = hm.reshape(K, -1) * beta
        flat = flat - flat.max(axis=1, keepdims=True)
        p = np.exp(flat)
        p = (p / p.sum(axis=1, keepdims=True)).reshape(K, Hh, Wh)
    else:
        p = np.clip(hm, 0, None)
        s = p.reshape(K, -1).sum(axis=1).reshape(K, 1, 1)
        p = p / np.maximum(s, 1e-12)
    ex = (p.sum(axis=1) * np.arange(Wh)[None, :]).sum(axis=1)
    ey = (p.sum(axis=2) * np.arange(Hh)[None, :]).sum(axis=1)
    return np.stack([ex, ey], axis=1)


def hard_argmax(hm: np.ndarray) -> np.ndarray:
    """Plain argmax decode (non-differentiable) -- for eval-time comparison."""
    K, Hh, Wh = hm.shape
    out = np.zeros((K, 2))
    for k in range(K):
        iy, ix = np.unravel_index(np.argmax(hm[k]), (Hh, Wh))
        out[k] = [ix, iy]
    return out
