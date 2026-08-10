"""Data load and preprocess functions for face alignment (Task 2).

The assignment gives the images and the landmarks in an array from Colab. The
container is different in each year. Usually it is an .npz file or a pickled
dict with the keys images and points. This module uses the standard shapes.
Thus it operates correctly when load_raw reads your file:

    images : (N, H, W, 3) uint8   or  (N, H, W) grey
    points : (N, 5, 2) float      landmark coordinates in ORIGINAL pixels

The preprocess sequence is from W06_L12. The report gives the reason for each
step:

    1. Change to grayscale. The 3 channels become 1 channel. The data that
       shows a landmark is structural, not chromatic.
    2. Resize to 64x64. The brief says that the full resolution is not
       necessary. IMPORTANT: multiply the landmarks by the same factors
       (sx, sy). If you do not, the landmarks move away from the face.
    3. Change the intensity range to [0,1]. Float values make the
       optimisation stable.
    4. Optional: apply CLAHE or histogram equalisation. This step increases
       the global contrast. W06_L12 warns that this step can increase the
       noise. Thus this step is a switch for an ablation, and the default is
       off.

The most important correctness point is the same as the structural care in
Task 1. When the code resizes the image, it must multiply the coordinates by
the same scale. At submission time the code must change the predictions BACK
to the original resolution.

The function save_as_csv at the end of this module is the export function from
the worksheet. This module keeps it without a change, and it keeps the two
asserts, because the brief tells you to write the predictions with it. The
function expects landmarks at the ORIGINAL resolution of 256x256. Thus apply
to_original_resolution first. The function keeps the order of the rows,
because the test set stays in its initial order.
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
    # CLAHE. The ablation sets this on and off. The default is off, because
    # CLAHE can increase the noise.
    equalise: bool = False
    to_float: bool = True


def load_raw(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read the images and the landmarks from an .npz file or an .npy file.

    The landmarks are optional.

    The function reads the key 'images'. It also reads the key 'points' or
    'landmarks' if the file contains one of them. Change this function if your
    assignment gives the data in a different form. This function is the only
    connection to the data file.
    """
    data = np.load(path, allow_pickle=True)
    if hasattr(data, "files"):                       # an npz file
        imgs = data["images"]
        pts = None
        for k in ("points", "landmarks", "pts"):
            if k in data.files:
                pts = data[k]; break
        return imgs, pts
    return data, None                                # an array of images only


def preprocess(img: np.ndarray, pts: Optional[np.ndarray], cfg: PreprocConfig):
    """Resize, change to grayscale and normalise one image.

    The function also changes the landmarks of the image if you give them.

    The function returns (proc_img, proc_pts, scale). The value scale is
    (sx, sy). It changes ORIGINAL coordinates into resized coordinates. Invert
    the scale to change the predictions back to the original resolution.
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
    """Invert the resize scale before the submission.

    W: 4.9 gives a warning about this step.
    """
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
