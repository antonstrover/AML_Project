"""Generate the Task 2 *method* diagrams: the preprocessing and augmentation
flowcharts and the heatmap-CNN architecture.

Nothing here plots results. Every CED curve, boxplot, landmark example and
robustness sweep is produced by run_task2.py from the real validation set --
this file used to emit synthetic demo versions of them as placeholders, and
those are exactly what the previous submission was marked down for. If a plot
shows numbers, it came from the real data.
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)


def flow(boxes, title, path, color="#cfe8ff"):
    fig, ax = plt.subplots(figsize=(3.2, 0.9 * len(boxes) + 0.6))
    ax.set_xlim(0, 4); ax.set_ylim(0, len(boxes)); ax.axis("off")
    ax.set_title(title, fontsize=11, weight="bold")
    for i, txt in enumerate(boxes):
        y = len(boxes) - i - 1
        ax.add_patch(FancyBboxPatch((0.4, y + 0.12), 3.2, 0.66,
                     boxstyle="round,pad=0.04", fc=color, ec="#3b7dd8"))
        ax.text(2.0, y + 0.45, txt, ha="center", va="center", fontsize=9)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((2.0, y + 0.12), (2.0, y - 0.12),
                         arrowstyle="-|>", mutation_scale=12, color="#333"))
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# 1. preprocessing flowchart
flow(["RGB image (orig HxW)", "grayscale (3->1 ch)",
      "resize 64x64  +  scale landmarks", "intensity -> [0,1] float",
      "(opt) CLAHE equalise [ablated]"],
     "Pre-processing (W06_L12)", os.path.join(FIG, "preprocess_flowchart.png"))

# 2. augmentation flowchart
flow(["(image, 5 landmarks)", "h-flip  +  SWAP indices 0<->1, 3<->4",
      "rotate / scale / translate (both)", "brightness / contrast / gamma",
      "Gaussian noise", "(augmented image, landmarks)"],
     "Augmentation (W09_L18)", os.path.join(FIG, "augment_flowchart.png"),
     color="#ffe2c2")

# 3. heatmap CNN architecture
fig, ax = plt.subplots(figsize=(8.5, 3.0)); ax.axis("off")
ax.set_xlim(0, 11); ax.set_ylim(0, 3)
stages = [("input\n1x64x64", "#eeeeee"), ("conv32\n64x64", "#cfe8ff"),
          ("conv64\n32x32", "#add4ff"), ("bottleneck\n64x16x16", "#7fb8f5"),
          ("up+conv32\n32x32", "#add4ff"), ("up+conv16\n64x64", "#cfe8ff"),
          ("heatmaps\n5x64x64", "#ffd9b3"), ("soft-argmax\n-> 5x2", "#ffc2a3")]
for i, (txt, c) in enumerate(stages):
    x = 0.3 + i * 1.32
    ax.add_patch(FancyBboxPatch((x, 1.0), 1.1, 1.0, boxstyle="round,pad=0.04",
                 fc=c, ec="#2b6cb0"))
    ax.text(x + 0.55, 1.5, txt, ha="center", va="center", fontsize=8)
    if i < len(stages) - 1:
        ax.add_patch(FancyArrowPatch((x + 1.1, 1.5), (x + 1.32, 1.5),
                     arrowstyle="-|>", mutation_scale=11, color="#333"))
ax.text(5.5, 2.6, "Heatmap-regression CNN (encoder-decoder, soft-argmax head)",
        ha="center", fontsize=11, weight="bold")
ax.text(5.5, 0.55, "loss = MSE(pred heatmap, Gaussian target) + 0.1 * MSE(decoded xy, gt xy)",
        ha="center", fontsize=8.5, style="italic")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "cnn_heatmap_arch.png"), dpi=140); plt.close(fig)

print("Task 2 method diagrams written. Result figures come from run_task2.py.")
