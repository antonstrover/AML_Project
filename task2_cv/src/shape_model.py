"""PCA shape model + HOG -> shape-parameter regression.

The classical comparator regresses in **shape-parameter space**, the option
W09_L18 names alongside direct coordinate regression and motivates with "face
shape is low-dimensional and can be approximated with PCA."

Pipeline:
  1. Procrustes-free PCA on the training landmark vectors (10-D) -> a mean
     shape plus a few principal shape modes b. Typically 4-6 modes explain
     >95% of shape variance for 5 points.
  2. Extract a global HOG descriptor from the resized grey image (W09_L17:
     HOG is approximately invariant to brightness and small rotations -- the
     robustness story).
  3. Ridge-regress HOG -> shape parameters b (a handful of targets instead of
     10 raw coords). Reconstruct landmarks as  mean + V @ b.

Regressing the low-dimensional, decorrelated b instead of raw coordinates
constrains predictions to the manifold of plausible face shapes -- the model
literally cannot output a geometrically impossible face. That inductive bias
is the point of the comparison against the CNN.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def hog_descriptor(img_gray: np.ndarray, pixels_per_cell=(6, 6),
                   cells_per_block=(3, 3), orientations=9) -> np.ndarray:
    """Global HOG vector, matching the W09_L17 parameterisation (6x6 cells,
    3x3 block normalisation, 9 orientation bins over 0-180 deg)."""
    return hog(img_gray, orientations=orientations,
               pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
               block_norm="L2-Hys", feature_vector=True)


@dataclass
class ShapeModelRegressor:
    n_modes: int = 6
    alpha: float = 1.0
    pca: Optional[PCA] = field(default=None, repr=False)
    ridge: Optional[Ridge] = field(default=None, repr=False)
    mean_shape_: Optional[np.ndarray] = field(default=None, repr=False)

    def fit(self, imgs_gray, shapes_flat) -> "ShapeModelRegressor":
        """imgs_gray: list of HxW arrays; shapes_flat: (N, 10) landmark coords."""
        self.pca = PCA(n_components=self.n_modes, random_state=0).fit(shapes_flat)
        self.mean_shape_ = self.pca.mean_
        B = self.pca.transform(shapes_flat)                # (N, n_modes)
        X = np.vstack([hog_descriptor(im) for im in imgs_gray])
        self.ridge = Ridge(alpha=self.alpha).fit(X, B)
        return self

    def predict(self, imgs_gray) -> np.ndarray:
        X = np.vstack([hog_descriptor(im) for im in imgs_gray])
        B = self.ridge.predict(X)
        flat = self.pca.inverse_transform(B)               # (N, 10)
        return flat.reshape(flat.shape[0], -1, 2)

    def explained_variance(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_ if self.pca else np.array([])


def mean_face_baseline(shapes_flat) -> np.ndarray:
    """Floor model: predict the mean training shape for every image (W09_L18)."""
    return shapes_flat.mean(axis=0).reshape(-1, 2)
