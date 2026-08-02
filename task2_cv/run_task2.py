"""Task 2 orchestrator: 5-point face-landmark alignment.

WHAT THIS RUN PRODUCES
----------------------
    figures/   CED curves (normalised and raw-pixel), per-landmark boxplots,
               best/failure landmark grids, error-vs-pose trend, robustness
               sweeps, and a visual check of the test predictions
    results/   task2_metrics.json (all metrics, timings and the device used)
    submission/results_task2.csv  (554 rows, original 256x256 resolution)

RUNNING THIS
------------
    python run_task2.py

It needs the assignment's image+landmark arrays, which are not redistributed
here; point DATA_TRAIN / DATA_VAL / DATA_TEST below at your .npz files (or set
the matching env vars). If they are absent the script prints exactly what to
provide and exits cleanly. The CNN path needs PyTorch; the classical
shape-model path runs on NumPy/sklearn/OpenCV alone.
"""
from __future__ import annotations

import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import dataset as D
import augment as A
import shape_model as SM
import evaluate as E
import robustness as R

# --------------------------------------------------------------------------
# DATA INTEGRATION POINT -- edit these lines (or set the env vars).
# Expected npz contents (see src/dataset.py):
#     train/val: key 'images' (N,256,256,3) + 'points' (N,5,2)
#     test     : key 'images' only
# --------------------------------------------------------------------------
DATA_TRAIN = os.environ.get("AML_T2_TRAIN", os.path.join(HERE, "data", "train.npz"))
DATA_VAL = os.environ.get("AML_T2_VAL", os.path.join(HERE, "data", "val.npz"))
DATA_TEST = os.environ.get("AML_T2_TEST", os.path.join(HERE, "data", "test.npz"))

OUT_FIG = os.path.join(HERE, "figures")
OUT_RES = os.path.join(HERE, "results")
OUT_SUB = os.path.join(HERE, "submission")
for d in (OUT_FIG, OUT_RES, OUT_SUB):
    os.makedirs(d, exist_ok=True)

SEED = 0
RNG = np.random.default_rng(SEED)
CFG = D.PreprocConfig(out_size=64, grayscale=True, equalise=False, to_float=True)
AUG = A.AugmentConfig()

# Hyperparameters, each justified in the report:
#   sigma      Gaussian target width on the 64x64 map. 1.5px keeps the blob ~7px
#              wide: broad enough to give gradient everywhere near the landmark,
#              tight enough that the soft-argmax expectation is not dragged by
#              the tail of a neighbouring landmark.
#   lr/opt     Adam 1e-3, cosine-annealed. Heatmap MSE against a normalised pmf
#              produces very small gradients, which Adam's per-parameter scaling
#              handles without hand-tuning; cosine removes the final-epochs
#              plateau without another knob to justify.
#   w_coord    0.1 on the coordinate term. The heatmap MSE is the primary loss
#              (it supervises every pixel); the coordinate term only aligns the
#              decode that is actually used at test time, so it is weighted down
#              to avoid it dominating early training.
#   batch      32 -- largest that keeps BatchNorm statistics stable at this
#              dataset size while still fitting comfortably in MPS memory.
#   aug factor 4x offline expansion, on top of which the model sees the same
#              image under different jitter each epoch.
EPOCHS = 150
PATIENCE = 25
BATCH = 32
LR = 1e-3
SIGMA = 1.5
AUG_FACTOR = 4

# Operating thresholds for the graded accuracy rule ("% of images with error
# below a certain threshold"), in pixels at the original 256x256 resolution.
THRESH_PX = (3.0, 5.0, 8.0, 12.0)
SELECT_PX = 5.0                       # the threshold the deployed model is chosen on


def _missing_data_message():
    print("=" * 70)
    print("Task 2 data not found.")
    for name, p in (("train", DATA_TRAIN), ("val", DATA_VAL), ("test", DATA_TEST)):
        print(f"  expected {name:5s}: {p}")
    print()
    print("Provide the assignment's npz arrays (or set AML_T2_TRAIN / AML_T2_VAL /")
    print("AML_T2_TEST) with keys 'images' (N,256,256,3) and 'points' (N,5,2) for")
    print("train and val, 'images' only for test. See src/dataset.py::load_raw --")
    print("the single integration point. Then re-run:  python run_task2.py")
    print("=" * 70)


def _device():
    """Apple M4 -> MPS; otherwise CUDA if present, else CPU."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def _to_original(pts_resized, scales):
    """Map a whole stack of (N,5,2) predictions back to 256x256 pixel space."""
    return np.stack([D.to_original_resolution(pts_resized[i], scales[i])
                     for i in range(len(pts_resized))])


def _build_augmented(G, P, factor=AUG_FACTOR):
    """Offline-expand the training set; the CNN also re-jitters on the fly."""
    augG, augP = [G], [P]
    for _ in range(factor - 1):
        gg, pp = [], []
        for i in range(len(G)):
            a_img, a_pts = A.augment_pair(G[i], P[i], AUG, rng=RNG)
            gg.append(a_img); pp.append(a_pts)
        augG.append(np.stack(gg)); augP.append(np.stack(pp))
    return np.concatenate(augG), np.concatenate(augP)


# --------------------------------------------------------------------------- #
# Models                                                                       #
# --------------------------------------------------------------------------- #
def run_shape_model(Gtr, Ptr, Gva):
    """Classical comparator: PCA shape model + HOG->ridge, plus the mean floor."""
    print("\n[shape-model] fitting PCA shape model + HOG ridge ...")
    flat_tr = Ptr.reshape(len(Ptr), -1)

    t0 = time.perf_counter()
    reg = SM.ShapeModelRegressor().fit(Gtr, flat_tr)
    fit_s = time.perf_counter() - t0
    pred_shape = reg.predict(Gva).reshape(len(Gva), 5, 2)

    mean_shape = SM.mean_face_baseline(flat_tr).reshape(1, 5, 2)
    pred_mean = np.repeat(mean_shape, len(Gva), axis=0)

    var = reg.explained_variance()
    print(f"[shape-model] fitted in {fit_s:.1f}s; {len(var)} modes explain "
          f"{var.sum():.1%} of shape variance")
    preds = {"mean_face": pred_mean, "shape_model": pred_shape}
    timing = {"mean_face": {"seconds": 0.0, "device": "cpu"},
              "shape_model": {"seconds": fit_s, "device": "cpu",
                              "explained_variance": float(var.sum())}}
    return preds, timing, reg


def _cnn_predict(net, imgs, dev, tta=True):
    """Predict (N,5,2) in 64x64 space, optionally averaging a flipped pass."""
    import torch
    import model as M
    net.eval()
    with torch.no_grad():
        x = torch.as_tensor(imgs[:, None], dtype=torch.float32, device=dev)
        p = M.soft_argmax2d(net(x)).cpu().numpy()
        if tta:
            # Flip, predict, un-mirror x and undo the landmark swap, average.
            p2 = M.soft_argmax2d(net(torch.flip(x, dims=[-1]))).cpu().numpy()
            p2[..., 0] = (imgs.shape[-1] - 1) - p2[..., 0]
            p2 = p2[:, A.FLIP_PERM]
            p = 0.5 * (p + p2)
    return p


def run_cnn(Gtr, Ptr, Gva, Pva, augmented: bool, epochs=EPOCHS, patience=PATIENCE):
    """Heatmap CNN + soft-argmax, early-stopped on validation AUC-CED.

    Selection uses AUC-CED rather than the training loss because the loss is
    dominated by the heatmap MSE, which keeps improving after the decoded
    coordinates have stopped getting better.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        import model as M
    except Exception as e:                       # torch absent -> skip cleanly
        print(f"[cnn:{'aug' if augmented else 'noaug'}] torch unavailable "
              f"({e.__class__.__name__}); skipping deep model.")
        return None, None, None

    dev = _device()
    tag = "aug" if augmented else "noaug"
    Gx, Px = _build_augmented(Gtr, Ptr) if augmented else (Gtr, Ptr)
    print(f"[cnn:{tag}] training on {dev}: {len(Gx)} samples, "
          f"<= {epochs} epochs, patience {patience}")

    X = torch.tensor(Gx[:, None, :, :], dtype=torch.float32)
    Y = torch.tensor(Px, dtype=torch.float32)
    dl = DataLoader(TensorDataset(X, Y), batch_size=BATCH, shuffle=True)

    torch.manual_seed(SEED)
    net = M.HeatmapNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_auc, best_state, best_ep, since = -1.0, None, 0, 0
    history = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        net.train(); tot = 0.0
        for xb, yxy in dl:
            xb, yxy = xb.to(dev), yxy.to(dev)
            opt.zero_grad()
            phm = net(xb)
            # Targets are built on-device per batch: cheaper than materialising
            # ~1 GB of float32 heatmaps for the augmented training set.
            yhm = M.gaussian_heatmaps(yxy, phm.shape[-2:], sigma=SIGMA)
            loss = M.combined_loss(phm, yhm, M.soft_argmax2d(phm), yxy)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        sched.step()

        # Early-stopping signal: validation AUC-CED (no TTA -- it is only a
        # selection signal, and skipping the extra pass halves the cost).
        auc = E.auc_ced(E.normalised_errors(_cnn_predict(net, Gva, dev, tta=False), Pva))
        history.append({"epoch": ep, "loss": tot / len(X), "val_auc_ced": auc})
        if auc > best_auc:
            best_auc, best_ep, since = auc, ep, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            since += 1
        if ep % 10 == 0 or ep == 1:
            print(f"[cnn:{tag}] epoch {ep:3d}/{epochs} loss={tot/len(X):.6f} "
                  f"val AUC-CED={auc:.4f} (best {best_auc:.4f} @ {best_ep})")
        if since >= patience:
            print(f"[cnn:{tag}] early stop at epoch {ep}; best was epoch {best_ep}")
            break
    train_s = time.perf_counter() - t0

    net.load_state_dict(best_state)
    pred_va = _cnn_predict(net, Gva, dev, tta=True)
    print(f"[cnn:{tag}] trained in {train_s/60:.1f} min on {dev}; "
          f"best epoch {best_ep}, val AUC-CED {best_auc:.4f}")
    timing = {"seconds": train_s, "device": dev, "epochs_run": len(history),
              "best_epoch": best_ep, "n_train_samples": int(len(Gx)),
              "n_parameters": int(sum(p.numel() for p in net.parameters()))}
    return pred_va, {"net": net, "device": dev, "history": history}, timing


# --------------------------------------------------------------------------- #
# Analysis                                                                     #
# --------------------------------------------------------------------------- #
def summarise(preds_orig, gt_orig):
    """Both metrics, per model: raw pixels at 256x256 and inter-ocular normalised."""
    out = {}
    for name, p in preds_orig.items():
        px = E.pixel_errors(p, gt_orig)                  # (N,5) graded quantity
        ioe = E.normalised_errors(p, gt_orig)            # (N,5) scale-invariant
        per_image = px.mean(axis=1)
        out[name] = {
            "mean_px": float(px.mean()),
            "median_px": float(np.median(px)),
            "mean_ioe": float(ioe.mean()),
            "median_ioe": float(np.median(ioe)),
            "auc_ced": float(E.auc_ced(ioe)),
            "pct_images_below_px": E.threshold_rates(per_image, THRESH_PX),
            "per_landmark_px": {"mean": px.mean(axis=0).tolist(),
                                "median": np.median(px, axis=0).tolist()},
            "per_landmark_ioe": E.per_landmark_summary(ioe),
        }
    return out


def qualitative(chosen, preds_orig, gt_orig, va_imgs):
    """Best/worst landmark grids and the error-vs-pose trend for the deployed model."""
    px = E.pixel_errors(preds_orig[chosen], gt_orig)
    per_image = px.mean(axis=1)
    order = np.argsort(per_image)
    best, worst = order[:4], order[-4:][::-1]

    E.plot_landmark_grid(va_imgs, preds_orig[chosen], gt_orig, best,
                         os.path.join(OUT_FIG, "qualitative_best.png"),
                         f"Best validation cases ({chosen}): predicted (+) vs ground truth (x)",
                         errs=per_image[best])
    E.plot_landmark_grid(va_imgs, preds_orig[chosen], gt_orig, worst,
                         os.path.join(OUT_FIG, "qualitative_worst.png"),
                         f"Worst validation cases ({chosen}): predicted (+) vs ground truth (x)",
                         errs=per_image[worst])

    roll, yaw = E.pose_proxies(gt_orig)
    E.plot_error_vs_pose(per_image, roll, yaw,
                         os.path.join(OUT_FIG, "error_vs_pose.png"),
                         title=f"Error against head pose ({chosen})")
    return {
        "best_idx": best.tolist(), "best_px": per_image[best].tolist(),
        "worst_idx": worst.tolist(), "worst_px": per_image[worst].tolist(),
        "corr_err_abs_roll": float(np.corrcoef(np.abs(roll), per_image)[0, 1]),
        "corr_err_abs_yaw": float(np.corrcoef(np.abs(yaw), per_image)[0, 1]),
    }


def robustness(cnns, shape_reg, Gva, Pva):
    """Aug vs no-aug (and the classical model) under increasing perturbation."""
    sweeps = {"noise": [0.0, 0.02, 0.05, 0.10, 0.15, 0.20],
              "rotation": [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]}
    predictors = {name: (lambda imgs, c=c: _cnn_predict(c["net"], imgs, c["device"], tta=False))
                  for name, c in cnns.items()}
    predictors["shape_model"] = lambda imgs: shape_reg.predict(imgs).reshape(len(imgs), 5, 2)

    out = {}
    for kind, levels in sweeps.items():
        curves = {name: R.robustness_curve(fn, Gva, Pva, kind, levels, seed=SEED)
                  for name, fn in predictors.items()}
        R.plot_robustness(curves, kind,
                          os.path.join(OUT_FIG, f"robustness_{kind}.png"))
        out[kind] = curves
        print(f"[robustness] {kind}: " + ", ".join(
            f"{n} {c['auc_ced'][0]:.3f}->{c['auc_ced'][-1]:.3f}" for n, c in curves.items()))
    return out


# --------------------------------------------------------------------------- #
def main():
    if not all(os.path.exists(p) for p in (DATA_TRAIN, DATA_VAL, DATA_TEST)):
        _missing_data_message()
        return 0

    print("Loading data ...")
    tr_imgs, tr_pts = D.load_raw(DATA_TRAIN)
    va_imgs, va_pts = D.load_raw(DATA_VAL)
    te_imgs, _ = D.load_raw(DATA_TEST)
    for p in (tr_pts, va_pts):
        if p is not None and p.ndim == 2 and p.shape[1] == 10:
            p.shape = (-1, 5, 2)                      # (N,10) -> (N,5,2)
    print(f"  train images {tr_imgs.shape}, points {tr_pts.shape}")
    print(f"  val   images {va_imgs.shape}, points {va_pts.shape}")
    print(f"  test  images {te_imgs.shape}")

    # Preprocess (coords scaled with the image). The brief supplies a real
    # held-out validation set, so use it rather than splitting train.
    t0 = time.perf_counter()
    Gtr, Ptr, _ = _preprocess_set(tr_imgs, tr_pts)
    Gva, Pva, va_scales = _preprocess_set(va_imgs, va_pts)
    Gte, _, te_scales = _preprocess_set(te_imgs, None)
    print(f"  preprocessed to {CFG.out_size}x{CFG.out_size} grey in "
          f"{time.perf_counter()-t0:.1f}s")

    log = {"config": {"out_size": CFG.out_size, "sigma": SIGMA, "lr": LR,
                      "batch": BATCH, "max_epochs": EPOCHS, "patience": PATIENCE,
                      "aug_factor": AUG_FACTOR, "seed": SEED,
                      "n_train": int(len(Gtr)), "n_val": int(len(Gva)),
                      "n_test": int(len(Gte))}}

    # ---- 1..4: the four approaches ----------------------------------------
    preds, timing, shape_reg = run_shape_model(Gtr, Ptr, Gva)

    cnns = {}
    for augmented in (False, True):
        name = "cnn_aug" if augmented else "cnn_noaug"
        pred, ctx, t = run_cnn(Gtr, Ptr, Gva, Pva, augmented=augmented)
        if pred is not None:
            preds[name] = pred
            cnns[name] = ctx
            timing[name] = t
    log["training"] = timing

    # ---- metrics, in ORIGINAL 256x256 pixel space --------------------------
    gt_orig = _to_original(Pva, va_scales)
    preds_orig = {k: _to_original(v, va_scales) for k, v in preds.items()}
    log["metrics"] = summarise(preds_orig, gt_orig)

    print("\n%-14s %8s %8s %8s %8s" % ("model", "mean px", "med px", "AUC-CED",
                                       f"<={SELECT_PX:g}px"))
    for name, m in log["metrics"].items():
        print("%-14s %8.2f %8.2f %8.3f %7.1f%%" % (
            name, m["mean_px"], m["median_px"], m["auc_ced"],
            100 * m["pct_images_below_px"][f"{SELECT_PX:g}"]))

    # ---- figures -----------------------------------------------------------
    ioe = {k: E.normalised_errors(v, gt_orig) for k, v in preds_orig.items()}
    pxe = {k: E.pixel_errors(v, gt_orig) for k, v in preds_orig.items()}

    E.plot_ced(ioe, np.linspace(0, 0.25, 200), os.path.join(OUT_FIG, "ced_curve.png"),
               title="CED, inter-ocular normalised error (validation, n=%d)" % len(Gva))
    E.plot_ced({k: v.mean(axis=1) for k, v in pxe.items()},
               np.linspace(0, 25, 200), os.path.join(OUT_FIG, "ced_pixels.png"),
               title="CED, mean per-image error in original 256x256 pixels",
               xlabel="pixel error threshold (256x256)", markers=THRESH_PX,
               show_auc=False)
    E.plot_boxplots(pxe, os.path.join(OUT_FIG, "error_boxplots.png"),
                    title="Per-point error by approach (validation)",
                    ylabel="pixel error at 256x256")
    E.plot_landmark_boxplots(pxe, os.path.join(OUT_FIG, "landmark_boxplots.png"),
                             title="Per-landmark error by approach (validation)")

    # ---- deploy the model that wins on the GRADED rule ---------------------
    key = f"{SELECT_PX:g}"
    chosen = max(log["metrics"],
                 key=lambda k: (log["metrics"][k]["pct_images_below_px"][key],
                                log["metrics"][k]["auc_ced"]))
    log["chosen_model"] = chosen
    log["selection_rule"] = f"highest %% of validation images with mean error <= {SELECT_PX:g}px"
    print(f"\n[select] deploying '{chosen}' on the graded threshold rule")

    log["qualitative"] = qualitative(chosen, preds_orig, gt_orig, va_imgs)
    print(f"[qualitative] err vs |roll| r={log['qualitative']['corr_err_abs_roll']:+.2f}, "
          f"err vs |yaw proxy| r={log['qualitative']['corr_err_abs_yaw']:+.2f}")

    # ---- robustness: aug vs no-aug under controlled perturbation ----------
    if len(cnns) == 2:
        log["robustness"] = robustness(cnns, shape_reg, Gva, Pva)

    # ---- test submission at ORIGINAL resolution ---------------------------
    if chosen in cnns:
        pred_te = _cnn_predict(cnns[chosen]["net"], Gte, cnns[chosen]["device"], tta=True)
    elif chosen == "shape_model":
        pred_te = shape_reg.predict(Gte).reshape(len(Gte), 5, 2)
    else:
        pred_te = np.repeat(Ptr.mean(axis=0)[None], len(Gte), axis=0)
    pred_te_orig = _to_original(pred_te, te_scales)

    # Visual check before shipping: the points must land on eyes/nose/mouth.
    E.plot_landmark_grid(te_imgs, pred_te_orig, None, [0, 1, 2, 3],
                         os.path.join(OUT_FIG, "test_predictions.png"),
                         f"Test-set predictions ({chosen}), original resolution")

    D.save_as_csv(pred_te_orig, OUT_SUB)
    lo, hi = float(pred_te_orig.min()), float(pred_te_orig.max())
    assert 0 <= lo and hi <= 256, f"predictions outside the image: [{lo}, {hi}]"
    print(f"Submission ({chosen}) written: submission/results_task2.csv "
          f"shape={pred_te_orig.shape} range=[{lo:.1f}, {hi:.1f}]")
    log["test_prediction_range"] = [lo, hi]

    with open(os.path.join(OUT_RES, "task2_metrics.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("Metrics written to results/task2_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
