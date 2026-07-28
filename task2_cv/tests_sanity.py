"""Sanity checks for Task 2 logic (runs without the image data or a GPU).

These do NOT claim face-alignment accuracy -- there is no face data bundled.
They verify that the *mechanics* the deep/classical pipelines depend on are
correct, which is where face-alignment bugs actually hide:

  1. horizontal flip swaps the eye/mouth landmark indices and round-trips
  2. soft-argmax recovers a known heatmap peak to sub-pixel accuracy
  3. the PCA shape model reconstructs shapes and the HOG->shape regressor
     beats the mean-face floor on a synthetic-but-structured task
  4. inter-ocular normalised error and CED behave as defined
  5. resize <-> original-resolution coordinate round-trip is exact
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


# 1. flip index swap + round-trip ------------------------------------------- #
img = np.random.rand(64, 64).astype(np.float32)
pts = np.array([[10, 20], [50, 22], [30, 35], [18, 50], [44, 51]], float)  # L/R eye, nose, L/R mouth
fimg, fpts = hflip(img, pts)
# left eye (idx0) should now sit on the right side of the image
# A correct flip keeps the convention "left-eye index sits on image-left":
# new idx0 (left eye) = mirrored original right eye -> left half;
# new idx1 (right eye) = mirrored original left eye -> right half.
check("flip preserves left-eye-on-left convention",
      fpts[0, 0] < 32 and fpts[1, 0] > 32)
# double flip restores original image and points exactly
ffimg, ffpts = hflip(fimg, fpts)
check("hflip round-trips image", np.allclose(ffimg, img))
check("hflip round-trips points", np.allclose(ffpts, pts))
check("flip perm swaps eyes (0<->1) and mouth (3<->4), nose fixed",
      FLIP_PERM.tolist() == [1, 0, 2, 4, 3])

# 2. soft-argmax recovers a known peak -------------------------------------- #
# interior peaks: expectation decode is sub-pixel accurate
peak = np.array([[40.0, 12.0], [8.0, 55.0], [32.0, 32.0], [20.0, 44.0], [50.0, 30.0]])
hm = make_heatmaps(peak, (64, 64), sigma=1.5)
dec = soft_argmax(hm)  # positive Gaussian heatmap -> direct spatial expectation
err = np.linalg.norm(dec - peak, axis=1).max()
check(f"soft-argmax sub-pixel on interior peaks (max err={err:.3f}px < 0.5)", err < 0.5)
# document the known boundary bias (expectation pulls inward when the Gaussian is clipped)
edge = make_heatmaps(np.array([[1.0, 1.0]]), (64, 64), sigma=1.5)
edge_err = np.linalg.norm(soft_argmax(edge)[0] - [1.0, 1.0])
check(f"boundary peak bias is small and inward ({edge_err:.2f}px)", 0 < edge_err < 2.0)

# 3. PCA shape model + HOG regression on a structured synthetic task -------- #
# Build synthetic "faces": a base 5-point shape morphed by 2 latent factors,
# rendered as blurred dots so HOG sees something correlated with the shape.
rng = np.random.default_rng(0)
base = np.array([[20, 25], [44, 25], [32, 38], [24, 50], [40, 50]], float)
modes = np.array([[[3, 0]] * 5, [[0, 4]] * 5], float)  # 2 shape modes
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

# 4. metric + CED behaviour -------------------------------------------------- #
perfect = normalised_errors(gt, gt)
check("zero error when pred==gt", np.allclose(perfect, 0))
ts = np.linspace(0, 0.2, 50)
c = ced(normalised_errors(pred, gt), ts)
check("CED is monotonic non-decreasing", np.all(np.diff(c) >= -1e-9))
check("CED ends at 1.0", abs(c[-1] - 1.0) < 1e-6)
check(f"AUC-CED in (0,1) ({auc_ced(normalised_errors(pred, gt)):.3f})",
      0 < auc_ced(normalised_errors(pred, gt)) < 1)

# 5. resize <-> original coordinate round-trip ------------------------------ #
big = (rng.random((100, 80, 3)) * 255).astype(np.uint8)
big_pts = np.array([[20, 30], [60, 32], [40, 50], [25, 70], [55, 72]], float)
proc_img, proc_pts, scale = preprocess(big, big_pts, PreprocConfig(out_size=64))
restored = to_original_resolution(proc_pts, scale)
check(f"coord round-trip exact (max err={np.abs(restored-big_pts).max():.4f})",
      np.allclose(restored, big_pts, atol=1e-3))
check("preprocessed image is 64x64 float in [0,1]",
      proc_img.shape == (64, 64) and proc_img.dtype == np.float32 and proc_img.max() <= 1.0)

# 6. torch heatmap targets agree with the NumPy reference ------------------- #
# The training loop builds targets on-device for speed; if they drifted from the
# unit-tested NumPy version the model would train against a different objective
# from the one the report describes, silently.
try:
    import torch
    from model import HeatmapNet, gaussian_heatmaps, soft_argmax2d

    # The output map must be the SAME resolution as the input, because the
    # landmark coordinates the loss compares against live in input pixels. An
    # encoder/decoder that pools once and upsamples twice silently emits a
    # 128x128 map, placing every Gaussian at half its correct relative position
    # -- the network then trains happily against a wrong target and predicts
    # near the image centre for everything.
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

# 7. the graded metric: raw pixels at original resolution -------------------- #
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

# pose proxies: a level frontal face reads as roll 0, yaw 0
frontal = np.array([[[20., 30.], [44., 30.], [32., 40.], [24., 50.], [40., 50.]]])
roll, yaw = pose_proxies(frontal)
check("level eyes give zero roll and a centred nose gives zero yaw",
      abs(roll[0]) < 1e-6 and abs(yaw[0]) < 1e-6)
rolled = frontal.copy(); rolled[0, 1, 1] += 24.0        # drop the right eye
check("tilting the eye line is detected as roll", abs(pose_proxies(rolled)[0][0]) > 40)
turned = frontal.copy(); turned[0, 2, 0] += 12.0        # push the nose sideways
check("an off-centre nose is detected as yaw", pose_proxies(turned)[1][0] > 0.4)

# 8. submission format ------------------------------------------------------- #
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

# summary -------------------------------------------------------------------- #
n_pass = sum(s == PASS for _, s in results)
print(f"\n{n_pass}/{len(results)} checks passed")
sys.exit(0 if n_pass == len(results) else 1)
