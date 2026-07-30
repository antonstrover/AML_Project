"""Evaluation metrics and plots for face alignment (Task 2).

Metric (W09_L18, stated as a numbered equation in the report):

    For image i, landmark k, with prediction p_ik and ground truth g_ik,
    the normalised point-to-point error is

        e_ik = || p_ik - g_ik || / d_i      where d_i = || g_i,0 - g_i,1 ||

    i.e. the Euclidean error normalised by the inter-ocular distance (between
    the two eye landmarks, indices 0 and 1). Normalising removes the effect of
    image resolution and face size, exactly as the lecture requires.

We report the mean/median per landmark and the Cumulative Error Distribution
(CED): the fraction of points with normalised error below a sweeping
threshold. One CED line per approach on shared axes is the headline figure.

Two metrics are reported side by side, because they answer different questions:

  * ``euclid_dist`` -- the worksheet's own per-point Euclidean distance, in RAW
    pixels at the original 256x256 resolution. This is the quantity the graders
    measure, so it is the one the deployed model is selected on.
  * ``normalised_errors`` -- the same distance divided by the inter-ocular
    distance, which is scale-invariant and therefore the fair way to compare
    models across faces of different size.

The brief marks accuracy as "% of images with error below a certain threshold",
i.e. a *point on the CED* rather than a mean. ``threshold_rates`` reports
exactly that, and model selection uses it: a model with a lower mean but a
fatter tail scores worse under this rule.
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
    """(N, K) raw Euclidean errors, via the worksheet's ``euclid_dist``.

    Inputs must already be at ORIGINAL 256x256 resolution -- that is the space
    the assignment's error is defined in.
    """
    n, k, _ = pred.shape
    return euclid_dist(pred, gt).reshape(n, k)


def threshold_rates(per_image_err: np.ndarray, thresholds: Sequence[float]) -> Dict[str, float]:
    """Fraction of IMAGES whose error is below each threshold (the graded rule).

    ``per_image_err`` is one number per image (we use the mean over its five
    landmarks), so the result reads directly as "x% of faces are localised to
    within t pixels".
    """
    return {f"{float(t):g}": float((per_image_err <= t).mean()) for t in thresholds}


def normalised_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return (N, K) inter-ocular-normalised per-landmark errors."""
    d = np.linalg.norm(gt[:, LEFT_EYE] - gt[:, RIGHT_EYE], axis=1)      # (N,)
    d = np.maximum(d, 1e-6)
    dist = np.linalg.norm(pred - gt, axis=2)                            # (N, K)
    return dist / d[:, None]


def ced(errors_flat: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Cumulative Error Distribution: P(error <= t) for each t."""
    e = errors_flat.ravel()
    return np.array([(e <= t).mean() for t in thresholds])


def auc_ced(errors_flat: np.ndarray, max_thr: float = 0.10) -> float:
    """Area under the CED up to max_thr, normalised to [0,1] (higher = better)."""
    ts = np.linspace(0, max_thr, 100)
    c = ced(errors_flat, ts)
    return float(np.trapezoid(c, ts) / max_thr)


def per_landmark_summary(errors: np.ndarray) -> Dict[str, List[float]]:
    """Per-landmark mean and median normalised error."""
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
    """CED for each approach on shared axes.

    ``markers`` draws vertical lines at the operating thresholds the accuracy
    component is graded on, so the marker can read the graded quantity straight
    off the figure instead of inferring it from a mean.
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
    """Grouped boxplots: one group per landmark, one box per model within it.

    This is the figure that exposes *which* landmark each model struggles with
    (mouth corners are the usual answer, the nose the usual best).
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
    """Cheap head-pose proxies read straight off the ground-truth points.

    roll : angle of the eye line, in degrees (0 = level eyes).
    yaw  : the nose's horizontal offset from the eye midpoint, divided by the
           inter-ocular distance. Near 0 for a frontal face; it grows in
           magnitude as the head turns, because the nose projects toward the
           near eye. It is a proxy, not a calibrated angle, which is all the
           systematic-bias argument needs.
    """
    d = gt[:, RIGHT_EYE] - gt[:, LEFT_EYE]
    roll = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    iod = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
    mid_x = 0.5 * (gt[:, LEFT_EYE, 0] + gt[:, RIGHT_EYE, 0])
    yaw = (gt[:, 2, 0] - mid_x) / iod
    return roll, yaw


def plot_error_vs_pose(per_image_err, roll, yaw, path,
                       title: str = "Error against head pose"):
    """Scatter + binned trend of per-image error against roll and the yaw proxy."""
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
    """Grid of faces with predictions (red +) and, if given, ground truth (green x).

    ``images`` are the ORIGINAL 256x256 frames and ``pred``/``gt`` the points in
    that same space, so what is drawn is exactly what gets submitted.
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
