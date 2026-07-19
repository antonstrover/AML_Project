"""Task 2 orchestrator: 5-point face-landmark alignment.

RUNNING THIS
------------
This script is complete and runnable. It needs the assignment's image+landmark
arrays, which are NOT redistributed here. Point DATA_TRAIN / DATA_TEST below at
your .npz files (see `load_raw` in src/dataset.py -- the single integration
point) and, for the CNN path, `pip install torch`. The classical shape-model
path runs with NumPy/sklearn/OpenCV alone.

If the data files are absent the script prints exactly what to provide and
exits cleanly (return code 0) rather than failing.
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import dataset as D
import augment as A
import shape_model as SM
import evaluate as E
import robustness as R

# --------------------------------------------------------------------------
# DATA INTEGRATION POINT -- edit these two lines (or set the env vars).
# Expected npz contents (see src/dataset.py):
#     train: key 'images' (N,H,W,3|N,H,W)  + 'points'/'landmarks' (N,5,2) or (N,10)
#     test : key 'images' only
# --------------------------------------------------------------------------
DATA_TRAIN = os.environ.get("AML_T2_TRAIN", os.path.join(HERE, "data", "train.npz"))
DATA_VAL = os.environ.get("AML_T2_VAL", os.path.join(HERE, "data", "val.npz"))
DATA_TEST = os.environ.get("AML_T2_TEST", os.path.join(HERE, "data", "test.npz"))

OUT_FIG = os.path.join(HERE, "figures")
OUT_RES = os.path.join(HERE, "results")
OUT_SUB = os.path.join(HERE, "submission")
for d in (OUT_FIG, OUT_RES, OUT_SUB):
    os.makedirs(d, exist_ok=True)

RNG = np.random.default_rng(0)
CFG = D.PreprocConfig(out_size=64, grayscale=True, equalise=False, to_float=True)
AUG = A.AugmentConfig()


def _missing_data_message():
    print("=" * 70)
    print("Task 2 data not found.")
    print(f"  expected train: {DATA_TRAIN}")
    print(f"  expected test : {DATA_TEST}")
    print()
    print("Provide the assignment's npz arrays (or set AML_T2_TRAIN / AML_T2_TEST)")
    print("with keys: 'images' (N,H,W,3) and 'points' (N,5,2) for train,")
    print("'images' only for test. See src/dataset.py::load_raw -- the single")
    print("integration point. Then re-run:  python3 run_task2.py")
    print()
    print("For the CNN path also:  pip install torch")
    print("The classical PCA-shape-model path needs only numpy/sklearn/opencv.")
    print("=" * 70)


def _preprocess_set(imgs, pts):
    """Vectorised preprocess over a stack; returns (G, P, scales)."""
    G, P, S = [], [], []
    for i in range(len(imgs)):
        p = None if pts is None else pts[i]
        g, pp, sc = D.preprocess(imgs[i], p, CFG)
        G.append(g); S.append(sc)
        if pp is not None:
            P.append(pp)
    G = np.stack(G).astype(np.float32)
    P = np.stack(P).astype(np.float32) if P else None
    return G, P, S


def _build_augmented(G, P, factor=4):
    """Offline-expand the training set; the CNN also augments on the fly."""
    augG, augP = [G], [P]
    for _ in range(factor - 1):
        gg, pp = [], []
        for i in range(len(G)):
            a_img, a_pts = A.augment_pair(G[i], P[i], AUG, rng=RNG)
            gg.append(a_img); pp.append(a_pts)
        augG.append(np.stack(gg)); augP.append(np.stack(pp))
    return np.concatenate(augG), np.concatenate(augP)


def run_shape_model(Gtr, Ptr, Gva, Pva):
    """Classical comparator: PCA shape model + HOG->ridge. No torch needed."""
    print("\n[shape-model] fitting PCA shape model + HOG ridge ...")
    flat_tr = Ptr.reshape(len(Ptr), -1)
    reg = SM.ShapeModelRegressor().fit(Gtr, flat_tr)
    pred_va = reg.predict(Gva).reshape(len(Gva), 5, 2)

    mean_pred = SM.mean_face_baseline(flat_tr).reshape(1, 5, 2)
    mean_pred = np.repeat(mean_pred, len(Gva), axis=0)

    err_shape = E.normalised_errors(pred_va, Pva)
    err_mean = E.normalised_errors(mean_pred, Pva)
    print(f"[shape-model] inter-ocular mean err: shape={err_shape.mean():.4f} "
          f"mean-face={err_mean.mean():.4f}")
    return {
        "shape_model": err_shape,
        "mean_face": err_mean,
    }, reg


def run_cnn(Gtr, Ptr, Gva, Pva, augmented: bool, epochs=40):
    """Heatmap CNN + soft-argmax. Trains only if torch is importable."""
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        import model as M
        import heatmap as H
    except Exception as e:                       # torch absent -> skip cleanly
        print(f"[cnn:{'aug' if augmented else 'noaug'}] torch unavailable "
              f"({e.__class__.__name__}); skipping deep model.")
        return None, None

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = "aug" if augmented else "noaug"
    print(f"[cnn:{tag}] training on {dev} ...")

    if augmented:
        Gx, Px = _build_augmented(Gtr, Ptr, factor=4)
    else:
        Gx, Px = Gtr, Ptr

    # Gaussian target heatmaps at 64x64.
    HM = np.stack([H.make_heatmaps(Px[i], (64, 64), sigma=1.5) for i in range(len(Px))])
    X = torch.tensor(Gx[:, None, :, :], dtype=torch.float32)
    Y_hm = torch.tensor(HM, dtype=torch.float32)
    Y_xy = torch.tensor(Px, dtype=torch.float32)
    dl = DataLoader(TensorDataset(X, Y_hm, Y_xy), batch_size=32, shuffle=True)

    net = M.HeatmapNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(epochs):
        net.train(); tot = 0.0
        for xb, yhm, yxy in dl:
            xb, yhm, yxy = xb.to(dev), yhm.to(dev), yxy.to(dev)
            opt.zero_grad()
            phm = net(xb)
            pxy = M.soft_argmax2d(phm)
            loss = M.combined_loss(phm, yhm, pxy, yxy)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        if (ep + 1) % 10 == 0:
            print(f"[cnn:{tag}] epoch {ep+1}/{epochs} loss={tot/len(X):.5f}")

    # Validation prediction, with test-time augmentation (hflip average).
    net.eval()
    with torch.no_grad():
        xv = torch.tensor(Gva[:, None], dtype=torch.float32, device=dev)
        p1 = M.soft_argmax2d(net(xv)).cpu().numpy()
        # TTA: horizontal flip, predict, unflip + landmark un-swap, average.
        xvf = torch.flip(xv, dims=[-1])
        p2 = M.soft_argmax2d(net(xvf)).cpu().numpy()
        p2[..., 0] = (Gva.shape[-1] - 1) - p2[..., 0]
        p2 = p2[:, A.FLIP_PERM]
        pred_va = 0.5 * (p1 + p2)

    err = E.normalised_errors(pred_va, Pva)
    print(f"[cnn:{tag}] inter-ocular mean err: {err.mean():.4f}")
    return err, net


def main():
    if not (os.path.exists(DATA_TRAIN) and os.path.exists(DATA_TEST)):
        _missing_data_message()
        return 0

    print("Loading data ...")
    tr_imgs, tr_pts = D.load_raw(DATA_TRAIN)
    va_imgs, va_pts = D.load_raw(DATA_VAL)
    te_imgs, _ = D.load_raw(DATA_TEST)
    for p in (tr_pts, va_pts):
        if p is not None and p.ndim == 2 and p.shape[1] == 10:
            p.shape = (-1, 5, 2)                      # (N,10) -> (N,5,2)
    print(f"  train images {tr_imgs.shape}, points {None if tr_pts is None else tr_pts.shape}")
    print(f"  val   images {va_imgs.shape}, points {None if va_pts is None else va_pts.shape}")
    print(f"  test  images {te_imgs.shape}")

    # Preprocess (coords scaled with the image). The brief supplies a real
    # held-out validation set, so use it rather than splitting train.
    Gtr, Ptr, _ = _preprocess_set(tr_imgs, tr_pts)
    Gva, Pva, _ = _preprocess_set(va_imgs, va_pts)
    print(f"  split: train={len(Gtr)} val={len(Gva)}")

    curves = {}

    # 1) classical PCA shape model (+ mean-face floor)
    shape_errs, shape_reg = run_shape_model(Gtr, Ptr, Gva, Pva)
    curves.update(shape_errs)

    # 2) heatmap CNN, no augmentation
    err_noaug, net_noaug = run_cnn(Gtr, Ptr, Gva, Pva, augmented=False)
    if err_noaug is not None:
        curves["cnn_noaug"] = err_noaug

    # 3) heatmap CNN, with augmentation
    err_aug, net_aug = run_cnn(Gtr, Ptr, Gva, Pva, augmented=True)
    if err_aug is not None:
        curves["cnn_aug"] = err_aug

    # ---- metrics & figures -------------------------------------------------
    thr = np.linspace(0, 0.10, 100)
    ced_curves = {k: E.ced(v, thr) for k, v in curves.items()}
    E.plot_ced(ced_curves, thr, os.path.join(OUT_FIG, "ced_curve.png"))
    E.plot_boxplots(curves, os.path.join(OUT_FIG, "error_boxplots.png"))

    summary = {}
    for k, v in curves.items():
        summary[k] = {
            "mean_ioe": float(v.mean()),
            "median_ioe": float(np.median(v)),
            "auc_ced": float(E.auc_ced(v)),
            "per_landmark": E.per_landmark_summary(v.reshape(-1, 5)),
        }
    with open(os.path.join(OUT_RES, "task2_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nMetrics written to results/task2_metrics.json")

    # ---- robustness: aug vs no-aug under controlled perturbation ----------
    if net_aug is not None and net_noaug is not None:
        import torch, model as M

        def _mk_predict(net):
            def predict_fn(imgs):
                with torch.no_grad():
                    x = torch.tensor(imgs[:, None], dtype=torch.float32)
                    return M.soft_argmax2d(net(x)).cpu().numpy()
            return predict_fn

        rob = {}
        for name, net in (("cnn_aug", net_aug), ("cnn_noaug", net_noaug)):
            rob[name] = R.robustness_curve(_mk_predict(net), Gva, Pva,
                                           kind="rotation", rng=RNG)
        R.plot_robustness(rob, "rotation",
                          os.path.join(OUT_FIG, "robustness_rotation.png"))
        print("Robustness curve written to figures/robustness_rotation.png")

    # ---- test submission at ORIGINAL resolution ---------------------------
    Gte, _, scales = _preprocess_set(te_imgs, None)
    if net_aug is not None:
        import torch, model as M
        with torch.no_grad():
            xt = torch.tensor(Gte[:, None], dtype=torch.float32)
            pred_te = M.soft_argmax2d(net_aug(xt)).cpu().numpy()
        chosen = "cnn_aug"
    else:
        pred_te = shape_reg.predict(Gte).reshape(len(Gte), 5, 2)
        chosen = "shape_model"

    pred_orig = np.stack([
        D.to_original_resolution(pred_te[i], scales[i]) for i in range(len(pred_te))
    ])
    D.save_as_csv(pred_orig, OUT_SUB)
    print(f"Submission ({chosen}) written: submission/results_task2.csv "
          f"shape={pred_orig.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
