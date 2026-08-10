"""Geometric and photometric augmentation for face alignment.

The most important correctness point is the horizontal flip. W09_L18 gives a
warning about it. A horizontal flip must also change the indices of the
landmarks. The left eye becomes the right eye. This module uses a scheme with
5 landmarks:

    0 = left eye, 1 = right eye, 2 = nose, 3 = left mouth, 4 = right mouth

A horizontal flip changes 0 to 1 and 1 to 0. It changes 3 to 4 and 4 to 3. It
does not move 2. If the code does not do this, the network learns a mirror
image of the face. Then the eye landmarks become incorrect. This is a usual
error, and it is difficult to find. The test in tests_sanity.py makes sure
that this operation is correct.

Each function changes the image and the landmarks with the same parameters in
one call. Each function returns the changed image and the changed landmarks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import cv2
except Exception:  # cv2 is in this environment. This guard is a precaution.
    cv2 = None

# The new sequence of the landmark indices after a horizontal flip.
FLIP_PERM = np.array([1, 0, 2, 4, 3])


def hflip(img: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flip the image horizontally and change the landmark indices.

    The change of the indices is necessary.
    """
    h, w = img.shape[:2]
    out_img = img[:, ::-1].copy()
    out_pts = pts.copy()
    out_pts[:, 0] = (w - 1) - out_pts[:, 0]   # make a mirror image of x
    out_pts = out_pts[FLIP_PERM]              # change left to right
    return out_img, out_pts


def affine(img, pts, angle_deg=0.0, scale=1.0, tx=0.0, ty=0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate, scale and move the image and the landmarks.

    The centre of the image is the centre of the rotation.
    """
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
    """Change the brightness, the contrast, the gamma and the noise.

    The image must be a float image in the range [0,1]. The noise is Gaussian.
    """
    rng = rng or np.random.default_rng()
    out = img.astype(np.float32)
    out = (out - 0.5) * contrast + 0.5 + brightness         # first the contrast,
                                                            # then the brightness
    out = np.clip(out, 0, 1) ** gamma                        # then the gamma
    if noise_sigma > 0:
        out = out + rng.normal(0, noise_sigma, out.shape).astype(np.float32)
    return np.clip(out, 0, 1)


@dataclass
class AugmentConfig:
    p_flip: float = 0.5
    max_rot: float = 25.0          # in degrees
    scale_range: Tuple[float, float] = (0.85, 1.15)
    max_translate_frac: float = 0.08
    brightness: float = 0.12
    contrast: float = 0.20
    gamma_range: Tuple[float, float] = (0.8, 1.25)
    noise_sigma: float = 0.02


def augment_pair(img: np.ndarray, pts: np.ndarray, cfg: AugmentConfig,
                 rng=None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply one random augmentation to an image and its landmarks.

    The training loop calls this function for each image in each epoch.

    The argument img is an HxW float image in the range [0,1]. The argument
    pts is a (5,2) array of landmark coordinates in pixels. The function
    returns the changed image and the changed landmarks.

    W09_L18 gives this set of augmentations:

      * a flip, with the change of the landmark indices
      * a rotation, a scale and a translation
      * a change of the brightness, the contrast and the gamma
      * Gaussian noise
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
