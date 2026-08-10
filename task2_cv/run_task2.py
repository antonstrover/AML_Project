"""The full sequence of Task 2: the alignment of 5 face landmarks.

THE OUTPUT OF THIS SCRIPT
-------------------------
    figures/   The CED curves, in normalised units and in raw pixels. A
               boxplot for each landmark. A grid of the best results and a
               grid of the failures. The trend of the error against the pose.
               The robustness sweeps. A visual check of the test predictions.
    results/   task2_metrics.json, which contains each metric, each time and
               the device.
    submission/results_task2.csv, which has 554 rows at the original
               resolution of 256x256.

TO START THE SCRIPT
-------------------
    python run_task2.py

The script needs the image array and the landmark array of the assignment.
This archive does not contain them. Put the paths of your .npz files in
DATA_TRAIN, DATA_VAL and DATA_TEST below, or set the applicable environment
variables. If a file does not exist, the script tells you which file to supply
and then stops correctly.

Install PyTorch before you use the CNN. The classical shape model needs only
NumPy, scikit-learn and OpenCV.
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
# THE CONNECTION TO THE DATA. Change these lines, or set the environment
# variables. The file src/dataset.py gives more data about the npz files:
#     train and val: the key 'images' (N,256,256,3) and 'points' (N,5,2)
#     test         : the key 'images' only
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

# The hyperparameters. The report gives the reason for each one:
#   sigma      The width of the Gaussian target on the 64x64 heatmap. A value
#              of 1.5 pixels makes the blob approximately 7 pixels wide. The
#              blob is thus sufficiently wide to give a gradient near the
#              landmark. It is also sufficiently narrow: the tail of the
#              adjacent landmark does not move the result of the soft-argmax.
#   targets    Each Gaussian target has a peak value of 1.0. The code does NOT
#              normalise the sum to one. A sum-to-one target has a peak value
#              of approximately 0.07 and decreases the heatmap MSE to
#              approximately 1e-5. This value is smaller than the coordinate
#              term by six orders of magnitude. Then the model becomes the
#              direct coordinate regressor that this approach must replace.
#              The function model.combined_loss gives more data.
#   lr/opt     Adam with a learning rate of 1e-3 and a cosine schedule. The
#              background pixels control the heatmap MSE, and thus the
#              gradients are small. Adam scales each parameter and does not
#              need a manual adjustment. The cosine schedule removes the
#              constant loss in the last epochs and adds no other parameter.
#   w_coord    The weight of the coordinate term is 0.1 in grid-relative
#              units. The heatmap MSE is the primary loss, because it
#              supervises each pixel. The coordinate term only corrects the
#              decode that the code uses at test time. Thus the weight is
#              small and the coordinate term does not control the early
#              training.
#   batch      32. This is the largest batch that keeps the statistics of the
#              BatchNorm layers stable for this quantity of data. It is also
#              sufficiently small for the memory of the MPS device.
#   aug factor The code makes 4 copies of the training set before the
#              training. The model also gets a different random change of each
#              image in each epoch.
EPOCHS = 150
PATIENCE = 25
BATCH = 32
LR = 1e-3
SIGMA = 1.5
AUG_FACTOR = 4

# The thresholds of the accuracy rule of the marker. The rule gives the
# percentage of the images with an error less than a threshold. The unit is
# pixels at the original resolution of 256x256.
THRESH_PX = (3.0, 5.0, 8.0, 12.0)
SELECT_PX = 5.0                       # the threshold that selects the model


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
    """Select the device for the training.

    The function selects MPS on an Apple M4. If MPS is not available, it
    selects CUDA. If CUDA is not available, it selects the CPU.
    """
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _preprocess_set(imgs, pts):
    """Preprocess each image of a set.

    The function returns (G, P, scales).
    """
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
    """Change an (N,5,2) array of predictions back to the 256x256 space."""
    return np.stack([D.to_original_resolution(pts_resized[i], scales[i])
                     for i in range(len(pts_resized))])


def _build_augmented(G, P, factor=AUG_FACTOR):
    """Make more training data before the training starts.

    The CNN also applies a different random change in each epoch.
    """
    augG, augP = [G], [P]
    for _ in range(factor - 1):
        gg, pp = [], []
        for i in range(len(G)):
            a_img, a_pts = A.augment_pair(G[i], P[i], AUG, rng=RNG)
            gg.append(a_img); pp.append(a_pts)
        augG.append(np.stack(gg)); augP.append(np.stack(pp))
    return np.concatenate(augG), np.concatenate(augP)


# --------------------------------------------------------------------------- #
# The models                                                                    #
# --------------------------------------------------------------------------- #
def run_shape_model(Gtr, Ptr, Gva):
    """Fit and apply the classical models.

    The first model is the PCA shape model with a ridge regression from the
    HOG. The second model is the mean face, which is the minimum reference.
    """
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
    """Predict an (N,5,2) array in the 64x64 space.

    Set tta to True to also predict the flipped image. The function then
    calculates the mean of the two predictions.
    """
    import torch
    import model as M
    net.eval()
    with torch.no_grad():
        x = torch.as_tensor(imgs[:, None], dtype=torch.float32, device=dev)
        p = M.soft_argmax2d(net(x)).cpu().numpy()
        if tta:
            # Flip the image and predict. Then make the mirror image of x
            # again and change the landmark indices back. Then calculate the
            # mean of the two predictions.
            p2 = M.soft_argmax2d(net(torch.flip(x, dims=[-1]))).cpu().numpy()
            p2[..., 0] = (imgs.shape[-1] - 1) - p2[..., 0]
            p2 = p2[:, A.FLIP_PERM]
            p = 0.5 * (p + p2)
    return p


def run_cnn(Gtr, Ptr, Gva, Pva, augmented: bool, epochs=EPOCHS, patience=PATIENCE):
    """Train the heatmap CNN and decode its output with the soft-argmax.

    The training stops early when the AUC-CED of the validation set stops to
    increase.

    The code uses the AUC-CED and not the training loss, because the heatmap
    MSE controls the loss. The heatmap MSE continues to decrease after the
    accuracy of the decoded coordinates stops to increase.
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
            # The code makes the targets on the device for each batch. The
            # alternative is a store of approximately 1 GB of float32 heatmaps
            # for the augmented training set.
            yhm = M.gaussian_heatmaps(yxy, phm.shape[-2:], sigma=SIGMA)
            loss = M.combined_loss(phm, yhm, M.soft_argmax2d(phm), yxy)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        sched.step()

        # The AUC-CED of the validation set controls the early stop. The code
        # does not use the flipped image here, because this value only selects
        # the epoch. Thus the code needs one half of the time.
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
# The analysis                                                                  #
# --------------------------------------------------------------------------- #
def summarise(preds_orig, gt_orig):
    """Calculate the two metrics for each model.

    The first metric is in raw pixels at the resolution of 256x256. The second
    metric is the error divided by the inter-ocular distance.
    """
    out = {}
    for name, p in preds_orig.items():
        px = E.pixel_errors(p, gt_orig)                  # (N,5) the marker uses
                                                         # this quantity
        ioe = E.normalised_errors(p, gt_orig)            # (N,5) does not change
                                                         # with the scale
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
    """Make the qualitative figures of the selected model.

    The function makes a grid of the best results and a grid of the worst
    results. It also plots the trend of the error against the pose.
    """
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
    """Compare the models when the change of the images becomes larger.

    The function compares the CNN with augmentation, the CNN without
    augmentation and the classical model.
    """
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
            p.shape = (-1, 5, 2)                      # change (N,10) to (N,5,2)
    print(f"  train images {tr_imgs.shape}, points {tr_pts.shape}")
    print(f"  val   images {va_imgs.shape}, points {va_pts.shape}")
    print(f"  test  images {te_imgs.shape}")

    # Preprocess each set. The code scales the coordinates with the image. The
    # brief supplies a separate validation set. Thus the code uses that set and
    # does not divide the training set.
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

    # ---- 1 to 4: the four approaches --------------------------------------
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

    # ---- the metrics, in the ORIGINAL 256x256 pixel space ------------------
    gt_orig = _to_original(Pva, va_scales)
    preds_orig = {k: _to_original(v, va_scales) for k, v in preds.items()}
    log["metrics"] = summarise(preds_orig, gt_orig)

    print("\n%-14s %8s %8s %8s %8s" % ("model", "mean px", "med px", "AUC-CED",
                                       f"<={SELECT_PX:g}px"))
    for name, m in log["metrics"].items():
        print("%-14s %8.2f %8.2f %8.3f %7.1f%%" % (
            name, m["mean_px"], m["median_px"], m["auc_ced"],
            100 * m["pct_images_below_px"][f"{SELECT_PX:g}"]))

    # ---- the figures -------------------------------------------------------
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

    # ---- select the best model with the rule of the MARKER -----------------
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

    # ---- the robustness of the CNN with and without the augmentation ------
    if len(cnns) == 2:
        log["robustness"] = robustness(cnns, shape_reg, Gva, Pva)

    # ---- the submission for the test set, at the ORIGINAL resolution ------
    if chosen in cnns:
        pred_te = _cnn_predict(cnns[chosen]["net"], Gte, cnns[chosen]["device"], tta=True)
    elif chosen == "shape_model":
        pred_te = shape_reg.predict(Gte).reshape(len(Gte), 5, 2)
    else:
        pred_te = np.repeat(Ptr.mean(axis=0)[None], len(Gte), axis=0)
    pred_te_orig = _to_original(pred_te, te_scales)

    # Examine the figure before you send the submission. Each landmark must be
    # on an eye, on the nose or on a corner of the mouth.
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
