"""Figures for DLCV 06B · Video Understanding and Action Recognition
(deep-learning/dl-06b-video-understanding).

Re-runnable: writes PNGs only to src/assets/deep-learning/dl-06b-video-understanding/.
All data are synthetic (a small square moving on a noisy background, random
segments on a timeline). The scene-bias toy and the temporal-AP toy reproduce the
code blocks of the tutorial exactly, so the plotted numbers match the printed ones.
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.linear_model import LogisticRegression

OUT = Path(__file__).resolve().parents[2] / "src/assets/deep-learning/dl-06b-video-understanding"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, MUTED = "#8a8a85", "#222222", "#555555"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- toy data (as in the text)
def make_clips(n, T=8, S=20, noise=0.5, scene_bias=0.5, seed=0):
    """A 3x3 square moves 1 px/frame left (y=0) or right (y=1) on a noisy background.
    scene_bias = P(background brightness agrees with the label); 0.5 = no bias."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = noise * rng.standard_normal((n, T, S, S))
    for i in range(n):
        step = 1 if y[i] == 1 else -1
        c_mid = rng.integers(T // 2, S - 3 - T // 2 + 1)
        r = rng.integers(0, S - 3 + 1)
        agree = rng.random() < scene_bias
        X[i] += 0.3 * (y[i] if agree else 1 - y[i])
        for t in range(T):
            c = c_mid + step * (t - T // 2)
            X[i, t, r:r + 3, c:c + 3] += 1.0
    return X, y


def frame_features(X):
    return X[:, X.shape[1] // 2].reshape(len(X), -1)


def motion_features(X):
    dt = X[:, 1:] - X[:, :-1]
    dx = np.gradient(X[:, :-1], axis=3)
    return -(dt * dx).sum(axis=(2, 3)) / ((dx ** 2).sum(axis=(2, 3)) + 1e-6)


def box(ax, x, y, w, h, text, fc, ec=INK, fs=10, color=INK, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color)


def arrow(ax, x0, y0, x1, y1, color=INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 lw=1.2, color=color))


# ---------------------------------------------------------------- Figure 1: tasks
def fig_tasks():
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4))
    rng = np.random.default_rng(3)
    # (a) clip classification: five frames of one clip
    ax = axes[0, 0]
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 4); ax.axis("off")
    for k in range(5):
        x0 = 0.2 + 2.05 * k
        ax.add_patch(Rectangle((x0, 1.2), 1.8, 1.8, fc="#eeeeec", ec=INK, lw=1))
        ax.add_patch(Rectangle((x0 + 0.15 + 0.28 * k, 1.85), 0.45, 0.45, fc=BLUE, ec="none"))
        ax.text(x0 + 0.9, 0.9, f"t = {k}", ha="center", fontsize=10, color=MUTED)
    box(ax, 2.6, 3.25, 5.2, 0.6, 'one label per clip: "moving right"', "#dbe8f8", fs=10)
    ax.set_title("(a) Clip classification (trimmed clip)", loc="left")

    # (b) temporal action localization: segments on a long timeline
    ax = axes[0, 1]
    ax.set_xlim(0, 100); ax.set_ylim(0, 4); ax.axis("off")
    ax.add_patch(Rectangle((0, 1.6), 100, 0.8, fc="#eeeeec", ec=INK, lw=1))
    for s, e, lab in [(12, 27, "jump"), (48, 56, "throw"), (71, 90, "jump")]:
        ax.add_patch(Rectangle((s, 1.6), e - s, 0.8, fc=ORANGE, ec="white", lw=2))
        ax.text((s + e) / 2, 2.65, lab, ha="center", fontsize=10)
        ax.text(s, 1.15, f"{s}s", ha="center", fontsize=10, color=MUTED)
        ax.text(e, 1.15, f"{e}s", ha="center", fontsize=10, color=MUTED)
    ax.text(50, 0.35, "untrimmed video: find (start, end, class) of every action",
            ha="center", fontsize=10, color=MUTED)
    ax.set_title("(b) Temporal action localization", loc="left")

    # (c) spatio-temporal action detection: boxes on keyframes
    ax = axes[1, 0]
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 4); ax.axis("off")
    for k in range(3):
        x0 = 0.3 + 3.45 * k
        ax.add_patch(Rectangle((x0, 0.7), 3.0, 2.3, fc="#eeeeec", ec=INK, lw=1))
        # two "people": stick-like rectangles
        p1 = (x0 + 0.4 + 0.25 * k, 1.0)
        p2 = (x0 + 2.0, 1.0)
        ax.add_patch(Rectangle(p1, 0.35, 1.3, fc=GRAY, ec="none"))
        ax.add_patch(Rectangle(p2, 0.35, 1.1 if k != 1 else 0.7, fc=GRAY, ec="none"))
        ax.add_patch(Rectangle((p1[0] - 0.12, 0.9), 0.6, 1.55, fc="none", ec=BLUE, lw=2))
        ax.add_patch(Rectangle((p2[0] - 0.12, 0.9), 0.6, 1.35 if k != 1 else 0.95,
                               fc="none", ec=AQUA, lw=2))
        ax.text(p1[0] + 0.18, 2.55, "walk", ha="center", fontsize=10, color=INK)
        ax.text(p2[0] + 0.18, 2.55 if k != 1 else 2.05, "sit" if k == 1 else "stand",
                ha="center", fontsize=10, color=INK)
        ax.text(x0 + 1.5, 0.3, f"keyframe {k + 1}", ha="center", fontsize=10, color=MUTED)
    ax.set_title("(c) Spatio-temporal action detection", loc="left")

    # (d) video-text retrieval and question answering
    ax = axes[1, 1]
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 4); ax.axis("off")
    box(ax, 0.2, 2.6, 4.6, 0.9, 'query: "a dog catches\na frisbee"', "#dbe8f8", fs=10)
    box(ax, 0.2, 0.5, 4.6, 0.9, 'question: "what does the\nperson pick up first?"', "#fde3d7", fs=10)
    sims = np.sort(rng.uniform(0.2, 0.9, 3))[::-1]
    for k, s in enumerate(sims):
        y0 = 3.25 - 0.8 * k
        ax.add_patch(Rectangle((6.2, y0), 1.6, 0.62, fc="#eeeeec", ec=INK, lw=1))
        ax.text(8.0, y0 + 0.31, f"rank {k + 1}  (sim {s:.2f})", va="center", fontsize=10)
    arrow(ax, 4.9, 3.05, 6.1, 3.25)
    arrow(ax, 4.9, 0.8, 6.1, 0.8)
    ax.text(6.2, 0.8, 'answer: "a cup"', va="center", fontsize=10, color=MUTED)
    ax.set_title("(d) Retrieval and video question answering", loc="left")
    fig.tight_layout()
    save(fig, "tasks.png")


# ---------------------------------------------------------------- Figure 2: families
def fig_families():
    fig, ax = plt.subplots(figsize=(10.4, 8.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 17.6); ax.axis("off")
    rows = [16.0, 12.4, 9.0, 5.6, 2.2]
    titles = ["(a) Per-frame 2D CNN + pooling (TSN, TSM)", "(b) Two-stream (RGB + optical flow)",
              "(c) 3D CNN (C3D, I3D, R(2+1)D)", "(d) SlowFast (two frame rates)",
              "(e) Video transformer (TimeSformer, ViViT, MViT, Video Swin)"]
    for y, t in zip(rows, titles):
        ax.text(0.0, y + 0.95, t, fontsize=11, fontweight="bold", color=INK)
    lab = "#eeeeec"
    # (a)
    y = rows[0]
    for k in range(3):
        box(ax, 0.2 + 1.25 * k, y - 0.55, 1.0, 0.7, f"frame {k + 1}", lab, fs=10)
        box(ax, 0.2 + 1.25 * k, y - 1.55, 1.0, 0.6, "2D CNN", "#dbe8f8", fs=10)
        arrow(ax, 0.7 + 1.25 * k, y - 0.6, 0.7 + 1.25 * k, y - 0.9)
        ax.plot([0.7 + 1.25 * k] * 2, [y - 1.6, y - 1.95], color=INK, lw=1.2)
    ax.plot([0.7, 5.9], [y - 1.95, y - 1.95], color=INK, lw=1.2)
    arrow(ax, 5.9, y - 1.95, 5.9, y - 1.0)
    box(ax, 4.7, y - 0.95, 2.4, 0.7, "average scores", "#e3f4ec", fs=10)
    arrow(ax, 7.15, y - 0.6, 8.0, y - 0.6)
    box(ax, 8.05, y - 0.95, 1.2, 0.7, "label", lab, fs=10)
    ax.text(9.5, y - 1.05, "weights shared;\nTSM adds channel\nshifts between frames",
            fontsize=10, color=MUTED)
    # (b)
    y = rows[1]
    box(ax, 0.2, y - 0.45, 1.9, 0.65, "one RGB frame", lab, fs=10)
    box(ax, 0.2, y - 1.55, 1.9, 0.65, "stacked flow ×L", lab, fs=10)
    box(ax, 2.6, y - 0.45, 2.0, 0.65, "spatial CNN", "#dbe8f8", fs=10)
    box(ax, 2.6, y - 1.55, 2.0, 0.65, "temporal CNN", "#fde3d7", fs=10)
    arrow(ax, 2.15, y - 0.12, 2.55, y - 0.12); arrow(ax, 2.15, y - 1.22, 2.55, y - 1.22)
    box(ax, 5.4, y - 1.0, 1.7, 0.7, "late fusion", "#e3f4ec", fs=10)
    arrow(ax, 4.65, y - 0.12, 5.35, y - 0.55); arrow(ax, 4.65, y - 1.22, 5.35, y - 0.8)
    arrow(ax, 7.15, y - 0.65, 8.0, y - 0.65)
    box(ax, 8.05, y - 1.0, 1.2, 0.7, "label", lab, fs=10)
    ax.text(9.5, y - 1.0, "flow computed\nbeforehand", fontsize=10, color=MUTED)
    # (c)
    y = rows[2]
    for k in range(4):
        ax.add_patch(Rectangle((0.3 + 0.18 * k, y - 1.4 + 0.15 * k), 1.2, 1.0, fc=lab, ec=INK, lw=1))
    ax.text(1.2, y - 1.75, "clip T×H×W", ha="center", fontsize=10, color=MUTED)
    arrow(ax, 2.2, y - 0.7, 2.75, y - 0.7)
    box(ax, 2.8, y - 1.05, 3.6, 0.7, "stack of t×d×d convolutions", "#dbe8f8", fs=10)
    arrow(ax, 6.45, y - 0.7, 8.0, y - 0.7)
    box(ax, 8.05, y - 1.05, 1.2, 0.7, "label", lab, fs=10)
    ax.text(9.5, y - 1.05, "I3D: inflate a\npretrained 2D net", fontsize=10, color=MUTED)
    # (d)
    y = rows[3]
    box(ax, 0.2, y - 0.45, 1.9, 0.65, "few frames", lab, fs=10)
    box(ax, 0.2, y - 1.55, 1.9, 0.65, "many frames (α×)", lab, fs=10)
    box(ax, 2.6, y - 0.45, 2.6, 0.65, "Slow: many channels", "#dbe8f8", fs=10)
    box(ax, 2.6, y - 1.55, 2.6, 0.65, "Fast: β× channels", "#fde3d7", fs=10)
    arrow(ax, 2.15, y - 0.12, 2.55, y - 0.12); arrow(ax, 2.15, y - 1.22, 2.55, y - 1.22)
    arrow(ax, 3.9, y - 0.88, 3.9, y - 0.52, color=ORANGE)
    ax.text(4.0, y - 0.78, "lateral", fontsize=10, color=ORANGE)
    box(ax, 5.8, y - 1.0, 1.3, 0.7, "concat", "#e3f4ec", fs=10)
    arrow(ax, 5.25, y - 0.12, 5.75, y - 0.55); arrow(ax, 5.25, y - 1.22, 5.75, y - 0.8)
    arrow(ax, 7.15, y - 0.65, 8.0, y - 0.65)
    box(ax, 8.05, y - 1.0, 1.2, 0.7, "label", lab, fs=10)
    ax.text(9.5, y - 1.0, "α = 8, β = 1/8\nis typical", fontsize=10, color=MUTED)
    # (e)
    y = rows[4]
    for k in range(4):
        ax.add_patch(Rectangle((0.3 + 0.18 * k, y - 1.4 + 0.15 * k), 1.2, 1.0, fc=lab, ec=INK, lw=1))
    arrow(ax, 2.2, y - 0.7, 2.6, y - 0.7)
    box(ax, 2.65, y - 1.05, 1.9, 0.7, "tubelet tokens", "#dbe8f8", fs=10)
    arrow(ax, 4.6, y - 0.7, 4.95, y - 0.7)
    box(ax, 5.0, y - 1.05, 2.4, 0.7, "space–time attention", "#fde3d7", fs=10)
    arrow(ax, 7.45, y - 0.7, 8.0, y - 0.7)
    box(ax, 8.05, y - 1.05, 1.2, 0.7, "label", lab, fs=10)
    ax.text(9.5, y - 1.15, "joint, divided,\nwindowed or\npooled attention", fontsize=10, color=MUTED)
    save(fig, "families.png")


# ---------------------------------------------------------------- Figure 3: scene bias toy
def fig_scene_bias():
    data = {"train": make_clips(2000, scene_bias=0.9, seed=1),
            "same": make_clips(1000, scene_bias=0.9, seed=2),
            "no cue": make_clips(1000, scene_bias=0.5, seed=3),
            "shift": make_clips(1000, scene_bias=0.1, seed=4)}
    feats = [("one frame", frame_features), ("motion", motion_features),
             ("frame+motion", lambda X: np.c_[frame_features(X), motion_features(X)])]
    accs = {}
    for name, f in feats:
        clf = LogisticRegression(C=0.1, max_iter=5000).fit(f(data["train"][0]), data["train"][1])
        accs[name] = [clf.score(f(X), y) for k, (X, y) in data.items() if k != "train"]
        print(name, np.round(accs[name], 3))

    fig = plt.figure(figsize=(10.4, 4.3))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.55, 4.2], hspace=0.35, wspace=0.08)
    # example clips drawn with less noise (0.1 instead of 0.5) so the square is visible
    X, y = make_clips(40, noise=0.1, scene_bias=1.0, seed=11)
    i_r = int(np.where(y == 1)[0][0])
    Xs, ys = make_clips(40, noise=0.1, scene_bias=0.0, seed=12)
    i_s = int(np.where(ys == 1)[0][0])
    for row, (clip, lab) in enumerate([(X[i_r], "train: right,\nbright scene"),
                                        (Xs[i_s], "shifted test: right,\ndark scene")]):
        for col, t in enumerate([1, 4, 7]):
            a = fig.add_subplot(gs[row, col])
            a.imshow(clip[t], cmap="gray", vmin=-0.3, vmax=1.5)
            a.set_xticks([]); a.set_yticks([])
            if row == 0:
                a.set_title(f"t = {t}", fontsize=10)
            if col == 0:
                a.set_ylabel(lab, fontsize=10)
    ax = fig.add_subplot(gs[:, 4])
    groups = ["same distribution", "no scene cue", "scene misleads"]
    w = 0.26
    for k, (name, col) in enumerate(zip(accs, [BLUE, ORANGE, AQUA])):
        xs = np.arange(3) + (k - 1) * (w + 0.02)
        ax.bar(xs, accs[name], width=w, color=col, label=name, edgecolor="white", linewidth=2)
        for xx, v in zip(xs, accs[name]):
            ax.text(xx, v + 0.015, f"{v:.2f}", ha="center", fontsize=10, color=INK)
    ax.axhline(0.5, color=GRAY, lw=1, ls="--")
    ax.text(2.55, 0.51, "chance", fontsize=10, color=MUTED, ha="right", va="bottom")
    ax.set_xticks(range(3)); ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.08); ax.set_ylabel("test accuracy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=10)
    ax.set_title("trained where the scene agrees with the label 90% of the time", fontsize=11)
    save(fig, "scene-bias.png")


# ---------------------------------------------------------------- Figure 4: (2+1)D factorization
def draw_kernel(ax, x0, y0, kt, kd, cell=0.35, fc="#dbe8f8"):
    """Draw a kt x kd x kd kernel as kt stacked kd x kd slices offset in depth."""
    for t in reversed(range(kt)):
        ox, oy = x0 + 0.28 * t, y0 + 0.22 * t
        for i in range(kd):
            for j in range(kd):
                ax.add_patch(Rectangle((ox + j * cell, oy + i * cell), cell, cell, fc=fc, ec=INK, lw=0.8))


def fig_factorization():
    c, t, d, T, H, W = 64, 3, 3, 16, 56, 56
    M = math.floor(t * d * d * c * c / (d * d * c + t * c))
    p3 = c * c * t * d * d
    p21 = c * M * d * d + M * c * t
    p2 = c * c * d * d
    fig = plt.figure(figsize=(10.4, 4.0))
    ax = fig.add_axes([0.0, 0.0, 0.66, 1.0])
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(0.2, 4.55, "3D convolution", fontsize=12, fontweight="bold")
    draw_kernel(ax, 0.6, 1.6, 3, 3)
    ax.text(1.4, 1.0, "t × d × d = 3 × 3 × 3\nspace and time at once", ha="center", fontsize=10)
    ax.text(3.5, 2.4, "≈", fontsize=22, ha="center")
    ax.text(4.3, 4.55, "(2+1)D: R(2+1)D block", fontsize=12, fontweight="bold")
    draw_kernel(ax, 4.5, 1.9, 1, 3, fc="#dbe8f8")
    ax.text(5.0, 1.1, "1 × d × d\nspatial", ha="center", fontsize=10)
    arrow(ax, 5.7, 2.4, 6.4, 2.4)
    box(ax, 6.45, 2.1, 0.9, 0.6, "ReLU", "#e3f4ec", fs=10)
    arrow(ax, 7.4, 2.4, 8.0, 2.4)
    for k in range(3):
        ax.add_patch(Rectangle((8.2 + 0.28 * k, 2.05 + 0.22 * k), 0.35, 0.35, fc="#fde3d7", ec=INK, lw=0.8))
    ax.text(8.6, 1.1, "t × 1 × 1\ntemporal", ha="center", fontsize=10)
    ax.text(6.6, 3.65, f"M = {M} intermediate channels", ha="center", fontsize=10, color=MUTED)

    ax2 = fig.add_axes([0.72, 0.17, 0.27, 0.68])
    names = ["2D", "3D", "(2+1)D"]
    vals = [p2 / 1e3, p3 / 1e3, p21 / 1e3]
    ax2.barh(names, vals, color=[GRAY, BLUE, ORANGE], height=0.55, edgecolor="white")
    for k, v in enumerate(vals):
        ax2.text(v + 3, k, f"{v:.1f}k", va="center", fontsize=10)
    ax2.set_xlim(0, 140); ax2.invert_yaxis()
    ax2.set_xlabel("parameters (thousands)")
    ax2.set_title("one layer, 64 → 64 channels", fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)
    save(fig, "factorization.png")


# ---------------------------------------------------------------- Figure 5: sampling
def fig_sampling():
    n = 300
    rng = np.random.default_rng(0)
    edges = np.linspace(0, n, 9)
    train = np.floor(edges[:-1] + rng.random(8) * np.diff(edges)).astype(int)
    test = np.floor((edges[:-1] + edges[1:]) / 2).astype(int)
    dense = 100 + 8 * np.arange(8)
    starts = np.linspace(0, n - 64, 10).astype(int)
    fig, ax = plt.subplots(figsize=(10.4, 3.6))
    rows = {"TSN, training": 3, "TSN, testing": 2, "dense clip (8 × stride 8)": 1, "10 test clips": 0}
    for lab, yy in rows.items():
        ax.hlines(yy, 0, n, color="#cccccc", lw=6)
    for e in edges:
        ax.vlines(e, 1.6, 3.4, color=GRAY, lw=0.8, ls=":")
    ax.plot(train, np.full(8, 3), "o", color=BLUE, ms=9)
    ax.plot(test, np.full(8, 2), "o", color=ORANGE, ms=9)
    ax.plot(dense, np.full(8, 1), "o", color=AQUA, ms=8)
    for k, s in enumerate(starts):        # one thin bar per clip, staggered so overlaps stay visible
        ax.add_patch(Rectangle((s, -0.45 + 0.09 * k), 64, 0.07, fc=BLUE, ec="none", zorder=3))
    ax.set_yticks(list(rows.values())); ax.set_yticklabels(list(rows.keys()))
    ax.set_xlim(-5, n + 5); ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel("frame index (a 10-second video at 30 fps); dotted lines = 8 TSN segments")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "sampling.png")


# ---------------------------------------------------------------- Figure 6: tIoU and AP
def tiou(seg, segs):
    inter = np.clip(np.minimum(seg[1], segs[:, 1]) - np.maximum(seg[0], segs[:, 0]), 0, None)
    union = (seg[1] - seg[0]) + (segs[:, 1] - segs[:, 0]) - inter
    return inter / union


def average_precision(preds, gts, thr, return_hits=False):
    n_gt = sum(len(g) for g in gts.values())
    used = {v: np.zeros(len(g), bool) for v, g in gts.items()}
    tp, order = [], sorted(preds, key=lambda p: -p[3])
    for v, s, e, _ in order:
        ious = tiou(np.array([s, e]), gts[v])
        ious[used[v]] = -1
        j = int(np.argmax(ious))
        hit = ious[j] >= thr
        if hit:
            used[v][j] = True
        tp.append(hit)
    tp = np.array(tp, float)
    recall = np.cumsum(tp) / n_gt
    precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    ap = np.sum(np.diff(np.r_[0, recall]) * precision)
    return (ap, order, tp) if return_hits else ap


def toy_localization():
    rng = np.random.default_rng(0)
    gts, preds = {}, []
    for v in range(20):
        starts = np.sort(rng.uniform(0, 100, rng.integers(1, 4)))
        g = np.c_[starts, starts + rng.uniform(4, 15, len(starts))]
        gts[v] = g
        for s, e in g:
            if rng.random() < 0.85:
                j = rng.normal(0, 0.2 * (e - s), 2)
                preds.append((v, s + j[0], e + j[1], rng.uniform(0.5, 1.0)))
        for _ in range(rng.integers(0, 3)):
            s = rng.uniform(0, 100)
            preds.append((v, s, s + rng.uniform(3, 12), rng.uniform(0.0, 0.7)))
    return gts, preds


def fig_tiou():
    gts, preds = toy_localization()
    thrs = [0.3, 0.4, 0.5, 0.6, 0.7]
    aps = [average_precision(preds, gts, t) for t in thrs]
    print("AP", np.round(aps, 3), "mean", round(float(np.mean(aps)), 3))
    _, order, tp = average_precision(preds, gts, 0.5, return_hits=True)
    hit = {(p[0], p[1]): h for p, h in zip(order, tp)}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw={"width_ratios": [2.3, 1]})
    vids = [0, 1, 2, 3, 4, 5]
    for r, v in enumerate(vids):
        yy = len(vids) - 1 - r
        ax.hlines(yy, 0, 115, color="#e4e4e2", lw=14)
        for s, e in gts[v]:
            ax.add_patch(Rectangle((s, yy + 0.02), e - s, 0.3, fc=GRAY, ec="white", lw=1, zorder=3))
        for p in preds:
            if p[0] != v:
                continue
            ok = hit[(p[0], p[1])]
            ax.add_patch(Rectangle((p[1], yy - 0.32), p[2] - p[1], 0.3,
                                   fc=AQUA if ok else ORANGE, ec="white", lw=1,
                                   hatch=None if ok else "///", zorder=3))
    ax.set_yticks(range(len(vids))); ax.set_yticklabels([f"video {v}" for v in vids[::-1]])
    ax.set_xlim(0, 115); ax.set_ylim(-0.7, len(vids) - 0.4)
    ax.set_xlabel("time (s)")
    ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=GRAY, label="ground truth (upper)"),
                       Patch(fc=AQUA, label="prediction: true positive at tIoU 0.5"),
                       Patch(fc=ORANGE, hatch="///", ec="white", label="prediction: false positive")],
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=10)
    ax.set_title("6 of the 20 synthetic videos", fontsize=11)
    ax2.plot(thrs, aps, "-o", color=BLUE, lw=2, ms=8)
    for t, a in zip(thrs, aps):
        ax2.text(t + 0.012, a + 0.04, f"{a:.2f}", ha="left", fontsize=10)
    ax2.set_xticks(thrs); ax2.set_xlim(0.27, 0.76); ax2.set_ylim(0, 1); ax2.set_xlabel("tIoU threshold")
    ax2.set_ylabel("AP"); ax2.set_title(f"average over 0.3–0.7: {np.mean(aps):.3f}", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "tiou-ap.png")


if __name__ == "__main__":
    fig_tasks()
    fig_families()
    fig_scene_bias()
    fig_factorization()
    fig_sampling()
    fig_tiou()
