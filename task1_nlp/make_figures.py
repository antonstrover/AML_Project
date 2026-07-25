"""Generate the Task 1 method diagrams.

pipeline_flowchart.png   raw text -> features -> gate -> classifier -> label
model_architectures.png  the sparse TF-IDF head and the BiLSTM side by side

The drawing helpers are the same ones used for the Task 2 diagrams
(task2_cv/make_figures.py), so both tasks' figures share a visual language.

Run:  python make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GREEN, GREY = "#cfe8ff", "#ffe2c2", "#d7f0d0", "#eeeeee"
EDGE = "#2b6cb0"


def box(ax, x, y, w, h, text, fc=BLUE, fontsize=8.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                fc=fc, ec=EDGE, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight)


def arrow(ax, xy_from, xy_to, label=None, style="-|>"):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style,
                                 mutation_scale=12, color="#333", lw=1.1))
    if label:
        ax.text((xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2 + 0.08,
                label, ha="center", va="bottom", fontsize=7.5, style="italic",
                color="#444")


# --------------------------------------------------------------------------- #
# (a) Preprocessing / representation pipeline.                                 #
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
ax.set_title("Task 1 pipeline: raw text to a three-way label", fontsize=12,
             weight="bold")

box(ax, 0.15, 2.1, 1.5, 0.9, "raw document\n(review or\nEnron email)", fc=GREY)
box(ax, 2.05, 3.3, 1.9, 0.9, "structural features\n10 surface statistics\n(length, headers, digits)")
box(ax, 2.05, 0.9, 1.9, 0.9, "light clean\nlowercase +\nwhitespace collapse")
box(ax, 4.35, 3.3, 1.7, 0.9, "GMM spam gate\n2 components,\nunsupervised", fc=ORANGE)
box(ax, 4.35, 0.9, 1.7, 0.9, "TF-IDF union\nword (1,2)\nchar_wb (3,5)")
box(ax, 6.45, 0.9, 1.7, 0.9, "sentiment models\nNB / SVM / BiLSTM\n-> soft vote", fc=GREEN)
box(ax, 8.5, 2.1, 1.35, 0.9, "label\n0 / 1 / -1", fc=GREY, weight="bold")

arrow(ax, (0.9, 3.0), (0.9, 3.75)); arrow(ax, (0.9, 3.75), (2.05, 3.75))
arrow(ax, (0.9, 2.1), (0.9, 1.35)); arrow(ax, (0.9, 1.35), (2.05, 1.35))
arrow(ax, (3.95, 3.75), (4.35, 3.75))
arrow(ax, (3.95, 1.35), (4.35, 1.35))
arrow(ax, (6.05, 1.35), (6.45, 1.35))
arrow(ax, (6.05, 3.75), (9.17, 3.75), style="-")
arrow(ax, (9.17, 3.75), (9.17, 3.0), label=None)
ax.text(7.6, 3.85, "P(spam) > 0.5  ->  dummy label -1", ha="center", fontsize=8,
        style="italic", color="#a15c00")
arrow(ax, (8.15, 1.35), (9.17, 1.35), style="-")
arrow(ax, (9.17, 1.35), (9.17, 2.1))
ax.text(8.6, 1.05, "else 0 / 1", ha="center", fontsize=8, style="italic",
        color="#2f6b2a")
ax.text(5.0, 2.55, "the two branches are decoupled: the gate never sees the\n"
                   "sentiment vocabulary, the classifier never sees spam",
        ha="center", fontsize=8, style="italic", color="#555")

fig.tight_layout()
fig.savefig(os.path.join(FIG, "pipeline_flowchart.png"), dpi=140)
plt.close(fig)

# --------------------------------------------------------------------------- #
# (b) The two model architectures, side by side.                               #
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))

ax = axes[0]
ax.set_xlim(0, 4); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("Sparse head: calibrated Linear SVM", fontsize=11, weight="bold")
stack = [("tokens + character n-grams", BLUE),
         ("TF-IDF, sublinear tf\nword(1,2) union char_wb(3,5)", BLUE),
         ("~200k sparse dimensions", BLUE),
         ("LinearSVC, hinge loss\nclass_weight=balanced", ORANGE),
         ("Platt scaling (sigmoid, cv=3)\n-> calibrated P(pos)", GREEN)]
for i, (txt, c) in enumerate(stack):
    y = 5.0 - i * 1.05
    box(ax, 0.25, y, 3.5, 0.8, txt, fc=c, fontsize=8)
    if i < len(stack) - 1:
        arrow(ax, (2.0, y), (2.0, y - 0.25))
ax.text(2.0, 0.35, "no word order: a bag of substrings", ha="center",
        fontsize=8, style="italic", color="#555")

ax = axes[1]
ax.set_xlim(0, 4); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("Sequence head: BiLSTM over GloVe", fontsize=11, weight="bold")
stack = [("token ids, padded to 60", BLUE),
         ("GloVe-100d embedding (frozen)\n~96% vocabulary coverage", BLUE),
         ("BiLSTM, 128 units per direction", ORANGE),
         ("max-over-time pooling (256-d)\ndropout 0.4", ORANGE),
         ("linear -> softmax P(pos)", GREEN)]
for i, (txt, c) in enumerate(stack):
    y = 5.0 - i * 1.05
    box(ax, 0.25, y, 3.5, 0.8, txt, fc=c, fontsize=8)
    if i < len(stack) - 1:
        arrow(ax, (2.0, y), (2.0, y - 0.25))
ax.text(2.0, 0.35, "reads order in both directions:\nnegation scope, mixed-polarity clauses",
        ha="center", fontsize=8, style="italic", color="#555")

fig.suptitle("The two prediction methods; their probabilities are averaged in the "
             "deployed soft-vote ensemble", fontsize=10, y=0.02)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(FIG, "model_architectures.png"), dpi=140)
plt.close(fig)

print("Task 1 figures written:", sorted(os.listdir(FIG)))
