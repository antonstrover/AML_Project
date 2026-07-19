"""Data loading + preprocessing for face alignment (Task 2).

The assignment ships its images and landmarks in a Colab-provided array (the
exact container varies by year -- often an .npz or a pickled dict with keys
like ``images`` and ``points``). This module is written against the canonical
shapes so it works once you point ``load_raw`` at your file:

    images : (N, H, W, 3) uint8   or  (N, H, W) grey
    points : (N, 5, 2) float       landmark coords in ORIGINAL pixel space

Preprocessing pipeline (W06_L12), each step justified in the report:
    1. grayscale            -> 3 channels collapse to 1 (landmark cues are
                               structural, not chromatic)
    2. resize to 64x64      -> the brief says full resolution is unnecessary;
                               CRITICALLY the landmarks are scaled by the same
                               (sx, sy) factors so they stay aligned
    3. intensity to [0,1]   -> float normalisation for stable optimisation
    4. (optional) CLAHE/histogram equalisation -> global contrast; W06_L12
       warns it can amplify noise, so it is a toggle we ablate, off by default

The crucial correctness point mirrored from Task 1's structural care: when the
image is resized, the coordinates must be multiplied by the same scale, and at
submission time predictions must be scaled BACK to original resolution.

``save_as_csv`` at the bottom is the worksheet's own export function, copied
verbatim (both asserts included) because the brief requires predictions to be
written by it. It expects points at ORIGINAL 256x256 resolution, so
``to_original_resolution`` must be applied first, and it preserves row order --
the test set is never permuted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


@dataclass
class PreprocConfig:
    out_size: int = 64
    grayscale: bool = True
    equalise: bool = False        # CLAHE; ablate on/off, default off (noise risk)
    to_float: bool = True


def load_raw(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load images and (if present) landmarks from an .npz/.npy file.

    Expected keys: 'images' and optionally 'points'/'landmarks'. Adapt here if
    your assignment loader differs -- this is the single integration point.
    """
    data = np.load(path, allow_pickle=True)
    if hasattr(data, "files"):                       # npz
        imgs = data["images"]
        pts = None
        for k in ("points", "landmarks", "pts"):
            if k in data.files:
                pts = data[k]; break
        return imgs, pts
    return data, None                                # bare array of images


def preprocess(img: np.ndarray, pts: Optional[np.ndarray], cfg: PreprocConfig):
    """Resize+grayscale+normalise one image and (if given) its landmarks.

    Returns (proc_img, proc_pts, scale) where scale=(sx, sy) maps ORIGINAL ->
    resized; invert it to put predictions back in original resolution.
    """
    h0, w0 = img.shape[:2]
    g = img
    if cfg.grayscale and img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (cfg.out_size, cfg.out_size), interpolation=cv2.INTER_AREA)
    if cfg.equalise:
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g.astype(np.uint8))
    if cfg.to_float:
        g = g.astype(np.float32) / 255.0
    sx, sy = cfg.out_size / w0, cfg.out_size / h0
    proc_pts = None
    if pts is not None:
        proc_pts = pts.astype(np.float32).copy()
        proc_pts[:, 0] *= sx
        proc_pts[:, 1] *= sy
    return g, proc_pts, (sx, sy)


def to_original_resolution(pts_resized: np.ndarray, scale: Tuple[float, float]) -> np.ndarray:
    """Invert the resize scaling for submission (W: 4.9 tripwire)."""
    sx, sy = scale
    out = pts_resized.astype(np.float32).copy()
    out[..., 0] /= sx
    out[..., 1] /= sy
    return out


def save_as_csv(points, location='.'):
    """
    Save the points out as a .csv file
    :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert points.shape[0] == 554, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[1:]) == 5*2, 'wrong number of points provided. There should be 5 points with 2 values (x,y) per point'
    np.savetxt(location + '/results_task2.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')
