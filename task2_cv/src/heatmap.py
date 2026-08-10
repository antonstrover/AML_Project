"""Heatmap targets and soft-argmax decode.

The model gives one low-resolution heatmap for each of the 5 landmarks. The
target is a 2-D Gaussian blob at the true landmark. This module decodes the
heatmaps to coordinates with soft-argmax.

The lecture W09_L18 gives the reasons for this approach:

  * The arg-max is not differentiable. Thus a heatmap target or a soft-argmax
    is necessary. A heatmap also makes the spatial structure of the problem
    clear.
  * An error in a heatmap is local in space. It moves one landmark a small
    distance. An error in a direct regression head can move a coordinate a
    large distance across the image. This is why a heatmap model is more
    robust in tests. The module robustness.py measures this behaviour.

This module uses only NumPy. Thus you can test the target code and the decode
code without a GPU. The CNN in model.py uses the same tensors in PyTorch.
"""
from __future__ import annotations

import numpy as np


def make_heatmaps(pts: np.ndarray, out_hw, sigma: float = 1.5,
                  normalise: bool = False) -> np.ndarray:
    """Make (K, Hh, Wh) Gaussian heatmaps for K landmarks.

    The argument pts is a (K, 2) array. Scale these coordinates to the heatmap
    resolution before you call this function.

    The default blob has a peak value of 1.0. This is the usual heatmap
    regression target (Tompson 2014, Newell 2016). The training code uses this
    form.

    A sum-to-one target has a peak value of 1/(2*pi*sigma^2), which is
    approximately 0.07. Then the pixel-wise MSE decreases to approximately
    1e-5. Then the coordinate term of the loss is larger by six orders of
    magnitude. Then the model becomes the direct coordinate regressor that this
    approach must replace.

    Set normalise=True to get the sum-to-one form. The function soft_argmax
    makes this form internally when it decodes. The two forms decode to the
    same coordinates.
    """
    Hh, Wh = out_hw
    K = pts.shape[0]
    ys = np.arange(Hh)[:, None]
    xs = np.arange(Wh)[None, :]
    hm = np.zeros((K, Hh, Wh), dtype=np.float32)
    for k, (px, py) in enumerate(pts):
        g = np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma ** 2))
        s = g.sum()
        hm[k] = (g / s) if (normalise and s > 0) else g
    return hm


def soft_argmax(hm: np.ndarray, beta: float = 1.0, from_logits: bool = False) -> np.ndarray:
    """Decode (K, Hh, Wh) heatmaps to (K, 2) coordinates.

    The function calculates the expected position in space.

    There are two modes:
      * from_logits=False is the default. The heatmap is already a
        non-negative map, for example a Gaussian target or a prediction after
        a ReLU. The function normalises the heatmap to a spatial pmf. Then it
        calculates the expected (x, y). Use this mode at evaluation time. An
        additional softmax moves a sharp peak towards the centre of the image.
      * from_logits=True. The heatmap holds network logits, which can be
        negative. The function first applies a spatial softmax with the
        temperature beta. This is equivalent to the differentiable decode in
        model.py.
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
    """Decode the heatmaps with a plain argmax.

    This decode is not differentiable. Use it only for a comparison at
    evaluation time.
    """
    K, Hh, Wh = hm.shape
    out = np.zeros((K, 2))
    for k in range(K):
        iy, ix = np.unravel_index(np.argmax(hm[k]), (Hh, Wh))
        out[k] = [ix, iy]
    return out
