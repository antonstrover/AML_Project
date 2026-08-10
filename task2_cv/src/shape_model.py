"""A PCA shape model with a regression from the HOG to the shape parameters.

This classical model does the regression in the space of the shape
parameters. W09_L18 gives this option and also direct coordinate regression.
The lecture gives this reason: the shape of a face has few dimensions, and a
PCA can approximate it.

The sequence is:

  1. Calculate a PCA of the landmark vectors of the training set. Each vector
     has 10 dimensions. The PCA gives a mean shape and some principal shape
     modes b. This code does not do a Procrustes alignment first. Usually 4 to
     6 modes contain more than 95% of the shape variance of 5 landmarks.
  2. Calculate a global HOG descriptor of the resized grey image. W09_L17
     shows that the HOG is approximately invariant to the brightness and to a
     small rotation. Thus the HOG makes the model more robust.
  3. Do a ridge regression from the HOG to the shape parameters b. The model
     has few targets and does not use the 10 raw coordinates. Then calculate
     the landmarks with the equation mean + V @ b.

The parameters b have few dimensions and are not correlated. A regression to
b keeps each prediction on the manifold of the possible face shapes. Thus the
model cannot give a geometrically impossible face. This constraint is the
reason for the comparison with the CNN.
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
    """Calculate the global HOG vector of one image.

    The parameters are the parameters in W09_L17. The cells are 6x6 pixels.
    The blocks for the normalisation are 3x3 cells. There are 9 orientation
    bins between 0 and 180 degrees.
    """
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
        """Fit the PCA and the ridge regression to the training data.

        The argument imgs_gray is a list of HxW arrays. The argument
        shapes_flat is an (N, 10) array of landmark coordinates.
        """
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
    """Give the mean shape of the training set for each image.

    This model is the minimum reference in W09_L18. Each other model must be
    better than this model.
    """
    return shapes_flat.mean(axis=0).reshape(-1, 2)
