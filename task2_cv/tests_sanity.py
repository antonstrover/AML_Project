"""The checks of the Task 2 code.

These checks operate without the image data and without a GPU.

These checks do NOT measure the accuracy of the face alignment, because this
archive contains no face data. The checks make sure that the mechanisms of the
deep model and of the classical model are correct. Almost all the errors in a
face-alignment program are in these mechanisms:

  1. The horizontal flip changes the indices of the eye landmarks and of the
     mouth landmarks. Two flips give the initial data again.
  2. The soft-argmax finds a known peak of a heatmap to less than one pixel.
  3. The PCA shape model calculates the shapes again. The regression from the
     HOG to the shape is better than the mean face on artificial data with a
     known structure.
  4. The normalised error and the CED operate as their definitions say.
  5. The change to the resized space and back to the original space is exact.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from augment import hflip, FLIP_PERM, affine
from heatmap import make_heatmaps, soft_argmax
from shape_model import ShapeModelRegressor, mean_face_baseline
from evaluate import (auc_ced, ced, euclid_dist, normalised_errors, pixel_errors,
                      pose_proxies, threshold_rates)
from dataset import preprocess, to_original_resolution, PreprocConfig, save_as_csv

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond):
    results.append((name, PASS if cond else FAIL))
    print(f"[{PASS if cond else FAIL}] {name}")


# 1. the flip changes the indices, and two flips give the initial data ------ #
img = np.random.rand(64, 64).astype(np.float32)
pts = np.array([[10, 20], [50, 22], [30, 35], [18, 50], [44, 51]], float)  # the
# left eye, the right eye, the nose, the left mouth corner, the right corner
fimg, fpts = hflip(img, pts)
# A correct flip keeps this rule: the landmark with the index 0 is on the left
# side of the image. Thus the new index 0 holds the mirror image of the initial
# right eye, which is in the left half. The new index 1 holds the mirror image
# of the initial left eye, which is in the right half.
check("flip preserves left-eye-on-left convention",
      fpts[0, 0] < 32 and fpts[1, 0] > 32)
# Two flips give the initial image and the initial landmarks exactly.
ffimg, ffpts = hflip(fimg, fpts)
check("hflip round-trips image", np.allclose(ffimg, img))
check("hflip round-trips points", np.allclose(ffpts, pts))
check("flip perm swaps eyes (0<->1) and mouth (3<->4), nose fixed",
      FLIP_PERM.tolist() == [1, 0, 2, 4, 3])

# 2. the soft-argmax finds a known peak ------------------------------------- #
# The decode is accurate to less than one pixel for a peak in the image.
peak = np.array([[40.0, 12.0], [8.0, 55.0], [32.0, 32.0], [20.0, 44.0], [50.0, 30.0]])
hm = make_heatmaps(peak, (64, 64), sigma=1.5)
dec = soft_argmax(hm)  # a positive Gaussian heatmap gives the expected position
err = np.linalg.norm(dec - peak, axis=1).max()
check(f"soft-argmax sub-pixel on interior peaks (max err={err:.3f}px < 0.5)", err < 0.5)
# This check records the known error at the edge of the image. The image
# contains only one part of the Gaussian there. Thus the expected position
# moves to the centre of the image.
edge = make_heatmaps(np.array([[1.0, 1.0]]), (64, 64), sigma=1.5)
edge_err = np.linalg.norm(soft_argmax(edge)[0] - [1.0, 1.0])
check(f"boundary peak bias is small and inward ({edge_err:.2f}px)", 0 < edge_err < 2.0)

# 3. the PCA shape model and the HOG regression on artificial data ---------- #
# The code makes artificial faces. Each face is a base shape with 5 landmarks.
# Two latent factors change the shape. The code then draws each landmark as a
# dot and applies a blur. Thus the HOG shows data that changes with the shape.
rng = np.random.default_rng(0)
base = np.array([[20, 25], [44, 25], [32, 38], [24, 50], [40, 50]], float)
modes = np.array([[[3, 0]] * 5, [[0, 4]] * 5], float)  # the 2 shape modes
def render(shape):
    im = np.zeros((64, 64), np.float32)
    for (x, y) in shape:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < 64 and 0 <= yi < 64:
            im[max(0, yi-2):yi+3, max(0, xi-2):xi+3] = 1.0
    import cv2
    return cv2.GaussianBlur(im, (5, 5), 1.5)
N = 300
coeffs = rng.normal(0, 1, (N, 2))
shapes = np.array([base + c[0]*modes[0] + c[1]*modes[1] + rng.normal(0, 0.3, base.shape)
                   for c in coeffs])
imgs = [render(s) for s in shapes]
shapes_flat = shapes.reshape(N, -1)
tr, te = slice(0, 240), slice(240, N)
sm = ShapeModelRegressor(n_modes=4, alpha=1.0).fit(imgs[tr], shapes_flat[tr])
pred = sm.predict(imgs[te])
gt = shapes_flat[te].reshape(-1, 5, 2)
mean_pred = np.repeat(mean_face_baseline(shapes_flat[tr])[None], gt.shape[0], axis=0)
e_model = normalised_errors(pred, gt).mean()
e_mean = normalised_errors(mean_pred, gt).mean()
check(f"shape model beats mean-face floor ({e_model:.3f} < {e_mean:.3f})", e_model < e_mean)
check(f"PCA captures >90% shape variance ({sm.explained_variance()[:2].sum():.2f})",
      sm.explained_variance()[:2].sum() > 0.90)

# 4. the behaviour of the metric and of the CED ------------------------------ #
perfect = normalised_errors(gt, gt)
check("zero error when pred==gt", np.allclose(perfect, 0))
ts = np.linspace(0, 0.2, 50)
c = ced(normalised_errors(pred, gt), ts)
check("CED is monotonic non-decreasing", np.all(np.diff(c) >= -1e-9))
check("CED ends at 1.0", abs(c[-1] - 1.0) < 1e-6)
check(f"AUC-CED in (0,1) ({auc_ced(normalised_errors(pred, gt)):.3f})",
      0 < auc_ced(normalised_errors(pred, gt)) < 1)

# 5. the change to the resized space and back to the original space --------- #
big = (rng.random((100, 80, 3)) * 255).astype(np.uint8)
big_pts = np.array([[20, 30], [60, 32], [40, 50], [25, 70], [55, 72]], float)
proc_img, proc_pts, scale = preprocess(big, big_pts, PreprocConfig(out_size=64))
restored = to_original_resolution(proc_pts, scale)
check(f"coord round-trip exact (max err={np.abs(restored-big_pts).max():.4f})",
      np.allclose(restored, big_pts, atol=1e-3))
check("preprocessed image is 64x64 float in [0,1]",
      proc_img.shape == (64, 64) and proc_img.dtype == np.float32 and proc_img.max() <= 1.0)

# 6. the PyTorch targets are the same as the NumPy targets ------------------ #
# The training loop makes the targets on the device, because this method is
# faster. If these targets became different from the NumPy targets, the model
# would train against a different objective from the objective in the report.
# The program would give no error message.
try:
    import torch
    from model import HeatmapNet, gaussian_heatmaps, soft_argmax2d

    # The output heatmap must have the SAME resolution as the input image,
    # because the landmark coordinates in the loss are in input pixels. An
    # encoder-decoder with one pool operation and two upsample operations makes
    # a 128x128 heatmap. Then each Gaussian is at one half of its correct
    # relative position. The network then trains against an incorrect target
    # and gives a prediction near the centre of each image.
    net = HeatmapNet(n_landmarks=5)
    out = net(torch.zeros(2, 1, 64, 64))
    check(f"HeatmapNet preserves input resolution (got {tuple(out.shape)})",
          out.shape == (2, 5, 64, 64))
    check("soft-argmax decodes network output to (B,K,2)",
          soft_argmax2d(out).shape == (2, 5, 2))

    pts_t = torch.tensor(peak[None], dtype=torch.float32)      # (1,5,2)
    torch_hm = gaussian_heatmaps(pts_t, (64, 64), sigma=1.5)[0].numpy()
    numpy_hm = make_heatmaps(peak, (64, 64), sigma=1.5)
    check(f"torch heatmap targets match the NumPy reference "
          f"(max abs diff={np.abs(torch_hm-numpy_hm).max():.2e})",
          np.allclose(torch_hm, numpy_hm, atol=1e-5))
    check("training targets peak at 1.0, not at the 1/(2*pi*sigma^2) pmf value",
          abs(torch_hm.max() - 1.0) < 1e-3)
except ImportError:
    print("[SKIP] torch heatmap parity (PyTorch not installed)")

# 7. the metric of the marker: raw pixels at the original resolution --------- #
a = np.array([[[0.0, 0.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
b = np.zeros_like(a)
check("euclid_dist is the plain Euclidean distance (3,4 -> 5)",
      np.allclose(euclid_dist(a, b).reshape(1, 5), [[0, 5, 0, 0, 0]]))
check("pixel_errors keeps the (N, K) layout", pixel_errors(a, b).shape == (1, 5))
rates = threshold_rates(np.array([1.0, 4.0, 9.0, 16.0]), [2.0, 10.0, 100.0])
check("threshold_rates counts images at or below each threshold",
      rates == {"2": 0.25, "10": 0.75, "100": 1.0})
check("threshold_rates is monotonic in the threshold",
      list(rates.values()) == sorted(rates.values()))

# The pose values of a frontal face with level eyes are roll 0 and yaw 0.
frontal = np.array([[[20., 30.], [44., 30.], [32., 40.], [24., 50.], [40., 50.]]])
roll, yaw = pose_proxies(frontal)
check("level eyes give zero roll and a centred nose gives zero yaw",
      abs(roll[0]) < 1e-6 and abs(yaw[0]) < 1e-6)
rolled = frontal.copy(); rolled[0, 1, 1] += 24.0        # move the right eye down
check("tilting the eye line is detected as roll", abs(pose_proxies(rolled)[0][0]) > 40)
turned = frontal.copy(); turned[0, 2, 0] += 12.0        # move the nose sideways
check("an off-centre nose is detected as yaw", pose_proxies(turned)[1][0] > 0.4)

# 8. the format of the submission file --------------------------------------- #
import tempfile
with tempfile.TemporaryDirectory() as d:
    pts554 = rng.uniform(0, 256, (554, 5, 2))
    save_as_csv(pts554, d)
    lines = open(os.path.join(d, "results_task2.csv")).read().splitlines()
    check("save_as_csv writes 554 headerless rows", len(lines) == 554)
    check("each row is 10 comma-separated values",
          all(len(l.split(",")) == 10 for l in lines[:5]))
    check("values round-trip within float precision",
          np.allclose(np.loadtxt(os.path.join(d, "results_task2.csv"), delimiter=","),
                      pts554.reshape(554, 10)))
    for bad, why in ((np.zeros((100, 5, 2)), "wrong image count"),
                     (np.zeros((554, 7, 2)), "wrong point count")):
        try:
            save_as_csv(bad, d); ok = False
        except AssertionError:
            ok = True
        check(f"save_as_csv rejects the {why}", ok)

# the summary ---------------------------------------------------------------- #
n_pass = sum(s == PASS for _, s in results)
print(f"\n{n_pass}/{len(results)} checks passed")
sys.exit(0 if n_pass == len(results) else 1)
