"""Figures for DLCV 01C - Model Compression and Efficient Inference.

Run:  .venv/bin/python scripts/figures/dl_01c_model_compression.py
Writes only to src/assets/deep-learning/dl-01c-model-compression/.
Data: scikit-learn's digits dataset (bundled, nothing is downloaded) and a made-up accelerator
for the roofline sketch. All results are toy results for teaching.
"""
import copy
import warnings
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-01c-model-compression"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    kb = path.stat().st_size / 1024
    w = plt.imread(path).shape[1]
    print(f"{name}: {kb:.0f} KB, {w} px wide")
    assert kb < 1024 and w <= 1600


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- Figure 1: roofline sketch
def roofline():
    peak, bw = 4e12, 100e9  # made-up accelerator: 4 TMAC/s, 100 GB/s
    ridge = peak / bw

    def layer(kind, C=64, H=56, W=56, batch=1):
        if kind == "conv":
            weights, macs = 9 * C * C, 9 * C * C * H * W
        elif kind == "dw":
            weights, macs = 9 * C, 9 * C * H * W
        elif kind == "pw":
            weights, macs = C * C, C * C * H * W
        else:  # linear 2048 x 2048
            weights, macs, C, H, W = 2048 * 2048, 2048 * 2048, 2048, 1, 1
        return batch * macs, weights + batch * 2 * C * H * W

    pts = [("3x3 conv, fp32", "conv", 1, 4), ("depthwise 3x3, fp32", "dw", 1, 4),
           ("1x1 conv, fp32", "pw", 1, 4), ("depthwise 3x3, int8", "dw", 1, 1),
           ("linear, batch 1, fp32", "fc", 1, 4), ("linear, batch 64, fp32", "fc", 64, 4)]
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    xs = np.logspace(-1, 3, 400)
    ax.plot(xs, np.minimum(peak, bw * xs) / 1e12, color=INK, lw=2.2)
    ax.axvline(ridge, color=MUTED, ls="--", lw=1)
    ax.text(ridge * 1.08, 0.012, f"ridge point\n{ridge:.0f} MACs/byte", color=MUTED, fontsize=10)
    ax.text(0.12, 0.35, "memory-bound:\nspeed = bandwidth x intensity", fontsize=10, color=MUTED, rotation=0)
    ax.text(130, 4.6, "compute-bound: speed = peak", fontsize=10, color=MUTED)
    offsets = {0: (8, -14), 1: (8, -4), 2: (8, -12), 3: (8, -4), 4: (8, -4), 5: (8, -14)}
    for i, (label, kind, batch, nb) in enumerate(pts):
        macs, vals = layer(kind, batch=batch)
        inten = macs / (vals * nb)
        perf = min(peak, bw * inten) / 1e12
        ax.scatter([inten], [perf], s=70, color=C[i % len(C)], zorder=3, edgecolor="white", lw=1)
        ax.annotate(label, (inten, perf), textcoords="offset points", xytext=offsets[i], fontsize=10, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0.008, 9)
    ax.set_xlabel("arithmetic intensity (MACs per byte moved)")
    ax.set_ylabel("attainable speed (TMAC/s)")
    ax.set_title("Roofline of a made-up accelerator (4 TMAC/s, 100 GB/s)", fontsize=12)
    style(ax)
    save(fig, "roofline.png")


# ---------------------------------------------------------------- shared toy model
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X / 16.0, y, test_size=0.3, random_state=0, stratify=y)
BASE = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=0).fit(Xtr, ytr)


def forward(Ws, bs, Xq):
    h = Xq
    for i, (w, b) in enumerate(zip(Ws, bs)):
        h = h @ w + b
        if i < len(Ws) - 1:
            h = np.maximum(h, 0)
    return h


def acc_of(Ws, bs=None):
    bs = BASE.intercepts_ if bs is None else bs
    return (forward(Ws, bs, Xte).argmax(1) == yte).mean()


def q_weight(w, bits, per_channel):
    qmax = 2 ** (bits - 1) - 1
    m = np.abs(w).max(axis=0, keepdims=True) if per_channel else np.abs(w).max()
    return np.clip(np.round(w * qmax / m), -qmax, qmax) * m / qmax


# ---------------------------------------------------------------- Figure 2: what pruning and quantization do
def weight_views():
    w = BASE.coefs_[1].ravel()
    thr = np.quantile(np.abs(w), 0.8)
    bits = 3
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(w).max() / qmax
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    ax = axes[0]
    bins = np.linspace(w.min(), w.max(), 81)
    ax.hist(w[np.abs(w) <= thr], bins=bins, color=C[1], alpha=0.85, label="removed")
    ax.hist(w[np.abs(w) > thr], bins=bins, color=C[0], alpha=0.85, label="kept")
    ax.axvline(thr, color=INK, ls="--", lw=1)
    ax.axvline(-thr, color=INK, ls="--", lw=1)
    ax.set_title("(a) magnitude pruning: remove the 80% smallest |w|", fontsize=12)
    ax.set_xlabel("weight value")
    ax.set_ylabel("count")
    ax.legend(fontsize=10, frameon=False, loc="center left")
    style(ax)

    ax = axes[1]
    ax.hist(w, bins=bins, color=MUTED, alpha=0.35, label="original weights")
    levels = np.arange(-qmax, qmax + 1) * s
    q = np.clip(np.round(w / s), -qmax, qmax) * s
    vals, counts = np.unique(np.round(q, 8), return_counts=True)
    ax2 = ax.twinx()
    ax2.vlines(vals, 0, counts, color=C[2], lw=4, label="3-bit values")
    ax2.set_ylabel("count after rounding", color=C[2])
    ax2.set_ylim(0, None)
    for lv in levels:
        ax.axvline(lv, color=C[2], lw=0.6, alpha=0.6)
    ax.set_title("(b) 3-bit per-tensor grid: 7 levels", fontsize=12)
    ax.set_xlabel("weight value")
    ax.set_ylabel("count")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=10, frameon=False, loc="upper left")
    style(ax)
    ax2.spines["top"].set_visible(False)
    fig.suptitle("Second-layer weights (256 x 128) of the toy MLP trained on digits", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "weight-views.png")


# ---------------------------------------------------------------- Figure 3: toy trade-off curves
def fine_tune(net, masks, epochs=10):
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        order = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), 128):
            net.partial_fit(Xtr[order[i:i + 128]], ytr[order[i:i + 128]])
            for w, m in zip(net.coefs_, masks):
                w *= m


def unstructured_masks(net, sparsity):
    return [np.abs(w) > np.quantile(np.abs(w), sparsity) for w in net.coefs_]


def structured_masks(net, keep):
    masks = [np.ones_like(w) for w in net.coefs_]
    for l in range(len(net.coefs_) - 1):
        norms = np.abs(net.coefs_[l]).sum(axis=0)
        dead = np.argsort(norms)[: int(round((1 - keep) * norms.size))]
        masks[l][:, dead] = 0
        masks[l + 1][dead, :] = 0
    return masks


def tradeoffs():
    total = sum(w.size for w in BASE.coefs_)
    dense_acc = BASE.score(Xte, yte)
    curves = {"weights, no fine-tuning": [], "weights + fine-tuning": [], "units + fine-tuning": []}
    for sp in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]:
        net = copy.deepcopy(BASE)
        masks = unstructured_masks(net, sp)
        for w, m in zip(net.coefs_, masks):
            w *= m
        kept = sum(m.sum() for m in masks) / total
        curves["weights, no fine-tuning"].append((kept, net.score(Xte, yte)))
        fine_tune(net, masks)
        curves["weights + fine-tuning"].append((kept, net.score(Xte, yte)))
    for keep in [0.5, 0.35, 0.25, 0.18, 0.125, 0.0625]:
        net = copy.deepcopy(BASE)
        masks = structured_masks(net, keep)
        for w, m in zip(net.coefs_, masks):
            w *= m
        fine_tune(net, masks)
        curves["units + fine-tuning"].append((sum(m.sum() for m in masks) / total, net.score(Xte, yte)))

    fig, axes = plt.subplots(1, 3, figsize=(10.3, 4.3))
    ax = axes[0]
    for i, (name, pts) in enumerate(curves.items()):
        pts = np.array(pts)
        ax.plot(pts[:, 0] * 100, pts[:, 1], "o-", color=C[i], label=name, lw=2, ms=5)
    ax.axhline(dense_acc, color=MUTED, ls=":", lw=1.2, label="dense model")
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.set_xticklabels(["2", "5", "10", "20", "50"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("weights kept (% of 50,432)")
    ax.set_ylabel("test accuracy")
    ax.set_title("(a) pruning", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")
    style(ax)

    ax = axes[1]
    bit_list = [2, 3, 4, 5, 6, 8]
    for i, (pc, name) in enumerate([(False, "per-tensor scale"), (True, "per-channel scales")]):
        accs = [acc_of([q_weight(w, b, pc) for w in BASE.coefs_]) for b in bit_list]
        ax.plot(bit_list, accs, "o-", color=C[i + 3], label=name, lw=2, ms=5)
    ax.axhline(dense_acc, color=MUTED, ls=":", lw=1.2, label="float32 model")
    ax.set_xlabel("weight bits (symmetric, round to nearest)")
    ax.set_ylabel("test accuracy")
    ax.set_title("(b) weight quantization", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")
    style(ax)

    ax = axes[2]
    W1, W2, W3 = BASE.coefs_
    U, S, Vt = np.linalg.svd(W2, full_matrices=False)
    ranks = [1, 2, 4, 6, 8, 12, 16, 32, 64]
    accs = [acc_of([W1, (U[:, :r] * S[:r]) @ Vt[:r], W3]) for r in ranks]
    energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    ax.plot(ranks, accs, "o-", color=C[6], lw=2, ms=5, label="test accuracy")
    ax.plot(ranks, energy[np.array(ranks) - 1], "s--", color=C[4], lw=1.6, ms=4,
            label="share of squared singular values")
    ax.axhline(dense_acc, color=MUTED, ls=":", lw=1.2, label="full-rank model")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("rank r of the second layer (256 x 128)")
    ax.set_title("(c) low-rank factorization", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9.5, frameon=False, loc="center right")
    style(ax)
    fig.suptitle("Toy results: a 64-256-128-10 MLP on scikit-learn digits (one seed)", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "toy-tradeoffs.png")
    for name, pts in curves.items():
        print(name, [(round(k * 100, 1), round(a, 3)) for k, a in pts])


if __name__ == "__main__":
    roofline()
    weight_views()
    tradeoffs()
