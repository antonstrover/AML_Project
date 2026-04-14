"""Geometric + photometric augmentation for face alignment.

The single most important correctness detail in face-alignment augmentation
(W09_L18 calls it out explicitly) is that a horizontal flip must **also swap
the landmark indices**: the left eye becomes the right eye. For the 5-point
scheme used here:

    0 = left eye, 1 = right eye, 2 = nose, 3 = left mouth, 4 = right mouth

a horizontal flip maps 0<->1 and 3<->4 and leaves 2 in place. Getting this
wrong silently teaches the network a mirror-image identity and wrecks the eye
landmarks -- a classic and hard-to-spot bug. The unit test in
``tests_sanity.py`` checks exactly this round-trip.

Everything is implemented so that image and landmarks are transformed by the
*same* parameters in one call, returning the transformed pair.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import cv2
except Exception:  # cv2 is available in this env; guard anyway
    cv2 = None

# Index permutation applied on a horizontal flip (5-point scheme).
FLIP_PERM = np.array([1, 0, 2, 4, 3])


def hflip(img: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Horizontal flip with the mandatory landmark index swap."""
    h, w = img.shape[:2]
    out_img = img[:, ::-1].copy()
    out_pts = pts.copy()
    out_pts[:, 0] = (w - 1) - out_pts[:, 0]   # mirror x
    out_pts = out_pts[FLIP_PERM]              # swap left/right identities
    return out_img, out_pts


def affine(img, pts, angle_deg=0.0, scale=1.0, tx=0.0, ty=0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate/scale/translate image and landmarks about the image centre."""
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    out_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    ones = np.ones((pts.shape[0], 1))
    out_pts = (np.hstack([pts, ones]) @ M.T)
    return out_img, out_pts


def photometric(img: np.ndarray, brightness=0.0, contrast=1.0, gamma=1.0,
                noise_sigma=0.0, rng=None) -> np.ndarray:
    """Brightness/contrast/gamma/Gaussian-noise on a float image in [0,1]."""
    rng = rng or np.random.default_rng()
    out = img.astype(np.float32)
    out = (out - 0.5) * contrast + 0.5 + brightness         # contrast then brightness
    out = np.clip(out, 0, 1) ** gamma                        # gamma
    if noise_sigma > 0:
        out = out + rng.normal(0, noise_sigma, out.shape).astype(np.float32)
    return np.clip(out, 0, 1)


@dataclass
class AugmentConfig:
    p_flip: float = 0.5
    max_rot: float = 25.0          # degrees
    scale_range: Tuple[float, float] = (0.85, 1.15)
    max_translate_frac: float = 0.08
    brightness: float = 0.12
    contrast: float = 0.20
    gamma_range: Tuple[float, float] = (0.8, 1.25)
    noise_sigma: float = 0.02


def augment_pair(img: np.ndarray, pts: np.ndarray, cfg: AugmentConfig,
                 rng=None) -> Tuple[np.ndarray, np.ndarray]:
    """Sample one random augmentation (the on-the-fly training transform).

    img: HxW float image in [0,1]; pts: (5,2) landmark coords in pixels.
    Returns the transformed (img, pts). This is the W09_L18 augmentation set:
    flips (with annotation swap), rotation, scale, translation, brightness/
    contrast/gamma and Gaussian noise.
    """
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    if rng.random() < cfg.p_flip:
        img, pts = hflip(img, pts)
    angle = rng.uniform(-cfg.max_rot, cfg.max_rot)
    scale = rng.uniform(*cfg.scale_range)
    tx = rng.uniform(-cfg.max_translate_frac, cfg.max_translate_frac) * w
    ty = rng.uniform(-cfg.max_translate_frac, cfg.max_translate_frac) * h
    img, pts = affine(img, pts, angle, scale, tx, ty)
    img = photometric(
        img,
        brightness=rng.uniform(-cfg.brightness, cfg.brightness),
        contrast=rng.uniform(1 - cfg.contrast, 1 + cfg.contrast),
        gamma=rng.uniform(*cfg.gamma_range),
        noise_sigma=cfg.noise_sigma,
        rng=rng,
    )
    return img, pts
