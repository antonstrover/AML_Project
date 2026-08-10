"""Evaluation metrics and plots for face alignment (Task 2).

W09_L18 gives the metric. The report gives it as a numbered equation:

    Let p_ik be the prediction for the image i and the landmark k. Let g_ik be
    the applicable ground truth. Then the normalised point-to-point error is

        e_ik = || p_ik - g_ik || / d_i      where d_i = || g_i,0 - g_i,1 ||

    Thus the metric is the Euclidean error divided by the inter-ocular
    distance. The inter-ocular distance is the distance between the two eye
    landmarks, which have the indices 0 and 1. This division removes the
    effect of the image resolution and the effect of the face size. The
    lecture makes this division necessary.

This module calculates the mean error and the median error for each landmark.
It also calculates the Cumulative Error Distribution (CED). The CED is the
fraction of the landmarks with a normalised error less than a threshold. The
primary figure shows one CED line for each approach on the same axes.

This module gives two metrics together, because they give different data:

  * euclid_dist is the Euclidean distance for each landmark from the
    worksheet. Its unit is RAW pixels at the original resolution of 256x256.
    The markers measure this quantity. Thus the code selects the model with
    this metric.
  * normalised_errors is the same distance divided by the inter-ocular
    distance. This metric does not change with the scale. Thus it compares
    models correctly across faces of different sizes.

The brief defines the accuracy as the percentage of the images with an error
less than a threshold. This value is a point on the CED, not a mean. The
function threshold_rates calculates it, and the code selects the model with
it. Thus a model with a low mean but many large errors gets a low score.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

LEFT_EYE, RIGHT_EYE = 0, 1

LANDMARK_NAMES = ["L eye", "R eye", "nose", "L mouth", "R mouth"]


def euclid_dist(pred_pts, gt_pts):
    """
    Calculate the euclidean distance between pairs of points
    :param pred_pts: The predicted points
    :param gt_pts: The ground truth points
    :return: An array of shape (no_points,) containing the distance of each predicted point from the ground truth
    """
    import numpy as np
    pred_pts = np.reshape(pred_pts, (-1, 2))
    gt_pts = np.reshape(gt_pts, (-1, 2))
    return np.sqrt(np.sum(np.square(pred_pts - gt_pts), axis=-1))


def pixel_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Calculate the raw Euclidean errors as an (N, K) array.

    The function uses euclid_dist from the worksheet.

    Give the landmarks at the ORIGINAL resolution of 256x256. The assignment
    defines its error in this space.
    """
    n, k, _ = pred.shape
    return euclid_dist(pred, gt).reshape(n, k)


def threshold_rates(per_image_err: np.ndarray, thresholds: Sequence[float]) -> Dict[str, float]:
    """Calculate the fraction of the IMAGES with an error less than a threshold.

    The markers use this rule.

    The argument per_image_err holds one value for each image. This code uses
    the mean of the errors of the five landmarks. Thus each result gives the
    percentage of the faces with all landmarks in a distance of t pixels.
    """
    return {f"{float(t):g}": float((per_image_err <= t).mean()) for t in thresholds}


def normalised_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Calculate the error of each landmark as an (N, K) array.

    The function divides each error by the inter-ocular distance.
    """
    d = np.linalg.norm(gt[:, LEFT_EYE] - gt[:, RIGHT_EYE], axis=1)      # (N,)
    d = np.maximum(d, 1e-6)
    dist = np.linalg.norm(pred - gt, axis=2)                            # (N, K)
    return dist / d[:, None]


def ced(errors_flat: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Calculate the Cumulative Error Distribution.

    The function calculates P(error <= t) for each threshold t.
    """
    e = errors_flat.ravel()
    return np.array([(e <= t).mean() for t in thresholds])


def auc_ced(errors_flat: np.ndarray, max_thr: float = 0.10) -> float:
    """Calculate the area below the CED between 0 and max_thr.

    The function divides the area by max_thr. Thus the result is in the range
    [0,1]. A large value shows a good model.
    """
    ts = np.linspace(0, max_thr, 100)
    c = ced(errors_flat, ts)
    return float(np.trapezoid(c, ts) / max_thr)


def per_landmark_summary(errors: np.ndarray) -> Dict[str, List[float]]:
    """Calculate the mean and the median normalised error of each landmark."""
    return {"mean": errors.mean(axis=0).tolist(),
            "median": np.median(errors, axis=0).tolist()}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_ced(curves: Dict[str, np.ndarray], thresholds: np.ndarray, path: str,
             title: str = "Cumulative Error Distribution",
             xlabel: str = "inter-ocular normalised error threshold",
             markers: Sequence[float] = (), show_auc: bool = True):
    """Plot the CED of each approach on the same axes.

    The argument markers gives the thresholds of the accuracy score. The
    function draws a vertical line at each one. Thus a marker can read the
    score from the figure and does not calculate it from a mean.
    """
    plt = _plt()
    plt.figure(figsize=(6.0, 4.2))
    for name, errs in curves.items():
        lab = f"{name} (AUC={auc_ced(errs):.3f})" if show_auc else name
        plt.plot(thresholds, ced(errs, thresholds), lw=2, label=lab)
    for m in markers:
        plt.axvline(m, ls=":", c="0.4", lw=1)
        plt.text(m, 0.02, f" {m:g}", fontsize=8, color="0.3", rotation=90,
                 va="bottom", ha="left")
    plt.xlabel(xlabel); plt.ylabel("fraction below threshold")
    plt.title(title, fontsize=11); plt.legend(fontsize=8, loc="lower right")
    plt.grid(alpha=.3); plt.ylim(0, 1); plt.xlim(thresholds[0], thresholds[-1])
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_boxplots(curves: Dict[str, np.ndarray], path: str,
                  title: str = "Normalised error by approach",
                  ylabel: str = "inter-ocular normalised error"):
    plt = _plt()
    plt.figure(figsize=(6.0, 4.2))
    plt.boxplot([e.ravel() for e in curves.values()], tick_labels=list(curves.keys()),
                showfliers=False)
    plt.ylabel(ylabel); plt.title(title, fontsize=11)
    plt.xticks(rotation=15); plt.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_landmark_boxplots(errors_by_model: Dict[str, np.ndarray], path: str,
                           title: str = "Error by landmark",
                           ylabel: str = "pixel error at 256x256"):
    """Plot a group of boxplots for each landmark.

    Each group contains one box for each model.

    This figure shows the landmark that is most difficult for each model.
    Usually the corners of the mouth give the largest error. Usually the nose
    gives the smallest error.
    """
    plt = _plt()
    names = list(errors_by_model)
    k = len(LANDMARK_NAMES)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.8 / len(names)
    colours = plt.cm.tab10.colors
    for m, name in enumerate(names):
        e = errors_by_model[name]
        pos = np.arange(k) + (m - (len(names) - 1) / 2) * width
        bp = ax.boxplot([e[:, j] for j in range(k)], positions=pos, widths=width * 0.85,
                        showfliers=False, patch_artist=True)
        for box in bp["boxes"]:
            box.set(facecolor=colours[m % 10], alpha=.65)
        for med in bp["medians"]:
            med.set(color="black", lw=1.2)
        ax.plot([], [], color=colours[m % 10], lw=6, alpha=.65, label=name)
    ax.set_xticks(np.arange(k)); ax.set_xticklabels(LANDMARK_NAMES)
    ax.set_xlabel("landmark"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def pose_proxies(gt: np.ndarray):
    """Calculate two simple values that show the pose of the head.

    The function uses only the ground-truth landmarks.

    roll : the angle of the line between the eyes, in degrees. A value of 0
           shows that the two eyes are at the same height.
    yaw  : the horizontal distance from the centre point between the eyes to
           the nose, divided by the inter-ocular distance. The value is
           approximately 0 for a frontal face. The value increases when the
           head turns, because the nose moves towards the nearer eye. This
           value is an indication, not a calibrated angle. The argument about
           the systematic error does not need more accuracy.
    """
    d = gt[:, RIGHT_EYE] - gt[:, LEFT_EYE]
    roll = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    iod = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
    mid_x = 0.5 * (gt[:, LEFT_EYE, 0] + gt[:, RIGHT_EYE, 0])
    yaw = (gt[:, 2, 0] - mid_x) / iod
    return roll, yaw


def plot_error_vs_pose(per_image_err, roll, yaw, path,
                       title: str = "Error against head pose"):
    """Plot the error of each image against the roll and against the yaw.

    The figure shows a scatter plot. It also shows the mean error in each
    group of values.
    """
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for ax, v, lab in ((axes[0], np.abs(roll), "|head roll| (degrees)"),
                       (axes[1], np.abs(yaw), "|yaw proxy| (nose offset / IOD)")):
        ax.scatter(v, per_image_err, s=12, alpha=.45, color="#2b6cb0")
        edges = np.quantile(v, np.linspace(0, 1, 6))
        edges[-1] += 1e-9
        centres, means = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (v >= a) & (v < b)
            if m.any():
                centres.append(v[m].mean()); means.append(per_image_err[m].mean())
        ax.plot(centres, means, "-o", color="crimson", lw=2, label="binned mean")
        r = np.corrcoef(v, per_image_err)[0, 1]
        ax.set_xlabel(lab); ax.set_ylabel("mean pixel error (256x256)")
        ax.set_title(f"r = {r:+.2f}", fontsize=10)
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140); plt.close(fig)


def plot_landmark_grid(images, pred, gt, idx, path, title, ncols: int = 4,
                       errs=None):
    """Plot a grid of faces with their landmarks.

    A red + shows a prediction. A green x shows a ground-truth landmark, if
    you give the ground truth.

    The argument images holds the ORIGINAL images of 256x256 pixels. The
    arguments pred and gt hold landmarks in the same space. Thus the figure
    shows the same data as the submission file.
    """
    plt = _plt()
    idx = list(idx)
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.7 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for a, i in enumerate(idx):
        ax = axes[a // ncols][a % ncols]
        ax.imshow(images[i])
        if gt is not None:
            ax.plot(gt[i][:, 0], gt[i][:, 1], "xg", ms=7, mew=1.6, label="ground truth")
        ax.plot(pred[i][:, 0], pred[i][:, 1], "+r", ms=8, mew=1.6, label="predicted")
        sub = f"#{i}" if errs is None else f"#{i}  {errs[a]:.1f}px"
        ax.set_title(sub, fontsize=8)
        ax.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9,
                   frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.05 if handles else 0, 1, 0.96))
    fig.savefig(path, dpi=140); plt.close(fig)
