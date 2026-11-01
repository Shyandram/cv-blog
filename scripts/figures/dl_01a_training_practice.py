"""Figures for DLCV 01A · Training Deep Vision Models in Practice.

Run:  .venv/bin/python scripts/figures/dl_01a_training_practice.py
Writes PNGs to src/assets/deep-learning/dl-01a-training-practice/ and prints the
numbers quoted in the tutorial text. NumPy + scikit-learn only, CPU, a few seconds.

The training code below is the same code as the snippets in the tutorial: a small
NumPy MLP (64-32-10, ReLU) trained with SGD + momentum on scikit-learn's 8x8 digits.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from scipy import ndimage
from skimage import data, filters, measure, segmentation
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src/assets/deep-learning/dl-01a-training-practice"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.facecolor": "white"})

BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = "#3b6fb6", "#e08a2c", "#3a9a5b", "#c8453c", "#7b5ea7", "#666666"


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p.name}  {p.stat().st_size / 1024:.0f} KB")


# ------------------------------------------------------------ schedules (snippet 1)
def constant(t, T, peak):
    return peak


def step_decay(t, T, peak, milestones=(0.5, 0.75), factor=0.1):
    return peak * factor ** sum(t >= m * T for m in milestones)


def warmup_cosine(t, T, peak, warmup=0.05):
    W = max(1, int(warmup * T))
    if t < W:
        return peak * (t + 1) / W
    q = (t - W) / max(1, T - W)
    return 0.5 * peak * (1 + np.cos(np.pi * q))


def one_cycle(t, T, peak, pct=0.3, div=25, final_div=1e4):
    up, lo = max(1, int(pct * T)), peak / div
    if t < up:
        return lo + (peak - lo) * 0.5 * (1 - np.cos(np.pi * t / up))
    q = (t - up) / max(1, T - up)
    end = lo / final_div
    return end + (peak - end) * 0.5 * (1 + np.cos(np.pi * q))


SCHEDULES = [(constant, "constant", GRAY), (step_decay, "step decay", ORANGE),
             (warmup_cosine, "warmup + cosine", BLUE), (one_cycle, "one-cycle", GREEN)]

# ------------------------------------------------------------ data and model (snippet 2)
X, y = load_digits(return_X_y=True)
X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=0)
mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-8
X_tr, X_va, X_te = [(a - mu) / sd for a in (X_tr, X_va, X_te)]


def init(seed, sizes=(64, 32, 10)):
    rng = np.random.default_rng(seed)
    return [[rng.normal(0, np.sqrt(2 / m), (m, n)), np.zeros(n)]
            for m, n in zip(sizes[:-1], sizes[1:])]


def forward(P, X):
    h = [X]
    for i, (W, b) in enumerate(P):
        z = h[-1] @ W + b
        h.append(np.maximum(z, 0) if i < len(P) - 1 else z)
    return h


def loss_and_grads(P, X, y):
    h = forward(P, X)
    z = h[-1] - h[-1].max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    loss = -logp[np.arange(len(y)), y].mean()
    d = np.exp(logp)
    d[np.arange(len(y)), y] -= 1
    d /= len(y)
    grads = []
    for i in reversed(range(len(P))):
        grads.insert(0, [h[i].T @ d, d.sum(0)])
        if i:
            d = (d @ P[i][0].T) * (h[i] > 0)
    return loss, grads


def evaluate(P, X, y):
    z = forward(P, X)[-1]
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return -logp[np.arange(len(y)), y].mean(), (z.argmax(1) == y).mean()


def train(schedule, peak, epochs=15, bs=16, momentum=0.9, seed=0, ema_decay=0.99):
    P = init(seed)
    V = [[np.zeros_like(W), np.zeros_like(b)] for W, b in P]
    E = [[W.copy(), b.copy()] for W, b in P]
    rng = np.random.default_rng(seed + 1000)
    T, t, log = epochs * int(np.ceil(len(y_tr) / bs)), 0, []
    for _ in range(epochs):
        order = rng.permutation(len(y_tr))
        for i in range(0, len(y_tr), bs):
            idx = order[i:i + bs]
            _, G = loss_and_grads(P, X_tr[idx], y_tr[idx])
            lr = schedule(t, T, peak)
            t += 1
            for (W, b), (gW, gb), v, e in zip(P, G, V, E):
                v[0] = momentum * v[0] + gW
                v[1] = momentum * v[1] + gb
                W -= lr * v[0]
                b -= lr * v[1]
                e[0] += (1 - ema_decay) * (W - e[0])
                e[1] += (1 - ema_decay) * (b - e[1])
        log.append(evaluate(P, X_tr, y_tr) + evaluate(P, X_va, y_va))
    return P, np.array(log), E


# ------------------------------------------------------------ range test (snippet 3)
def lr_range_test(lo=1e-4, hi=10.0, steps=200, bs=16, momentum=0.9, seed=0):
    P = init(seed)
    V = [[np.zeros_like(W), np.zeros_like(b)] for W, b in P]
    rng = np.random.default_rng(seed)
    lrs, smooth, raw, avg = [], [], [], 0.0
    for t in range(steps):
        lr = lo * (hi / lo) ** (t / (steps - 1))
        idx = rng.choice(len(y_tr), bs, replace=False)
        loss, G = loss_and_grads(P, X_tr[idx], y_tr[idx])
        avg = 0.9 * avg + 0.1 * loss
        lrs.append(lr)
        raw.append(loss)
        smooth.append(avg / (1 - 0.9 ** (t + 1)))
        if not np.isfinite(loss) or smooth[-1] > 4 * min(smooth):
            break
        for (W, b), (gW, gb), v in zip(P, G, V):
            v[0] = momentum * v[0] + gW
            v[1] = momentum * v[1] + gb
            W -= lr * v[0]
            b -= lr * v[1]
    return np.array(lrs), np.array(smooth), np.array(raw)


# ------------------------------------------------------------ overfit one batch (snippet 4)
def overfit_one_batch(double_softmax=False, steps=300, lr=0.1, seed=0):
    P = init(seed)
    xb, yb = X_tr[:16], y_tr[:16]
    curve = []
    for step in range(steps):
        h = forward(P, xb)
        z = h[-1]
        if double_softmax:
            z = np.exp(z - z.max(1, keepdims=True))
            z = z / z.sum(1, keepdims=True)
        s = z - z.max(1, keepdims=True)
        p = np.exp(s) / np.exp(s).sum(1, keepdims=True)
        loss = -np.log(p[np.arange(16), yb]).mean()
        curve.append(loss)
        d = p.copy()
        d[np.arange(16), yb] -= 1
        d /= 16
        if double_softmax:
            d = z * (d - (d * z).sum(1, keepdims=True))
        for i in reversed(range(len(P))):
            gW, gb = h[i].T @ d, d.sum(0)
            if i:
                d = (d @ P[i][0].T) * (h[i] > 0)
            P[i][0] -= lr * gW
            P[i][1] -= lr * gb
    return np.array(curve)


# ================================================================ figures
def fig_schedules():
    T, peak = 15 * int(np.ceil(len(y_tr) / 16)), 0.1
    t = np.arange(T)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for f, name, col in SCHEDULES:
        lr = np.array([f(s, T, peak) for s in t])
        for ax in axes:
            ax.plot(t, lr, color=col, lw=2, label=name)
    axes[0].set_title("Learning rate (linear scale)")
    axes[1].set_title("Same schedules, log scale")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-7, 0.3)
    for ax in axes:
        ax.set_xlabel("training step")
        ax.set_ylabel("learning rate")
        ax.grid(alpha=0.3)
    axes[0].axvspan(0, int(0.05 * T), color=BLUE, alpha=0.12)
    axes[0].text(int(0.05 * T) + 10, 0.03, "warmup", color=BLUE, fontsize=10)
    axes[0].set_ylim(-0.005, 0.135)
    axes[0].legend(fontsize=9.5, loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, "schedules.png")
    print(f"schedule figure: T = {T} steps, peak = {peak}")


def fig_schedule_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ep = np.arange(1, 16)
    print("schedule runs (seed 0):")
    for f, name, col in SCHEDULES:
        _, log, _ = train(f, peak=0.1)
        axes[0].plot(ep, log[:, 0], "-o", ms=3, color=col, lw=1.8, label=f"{name}, peak 0.1")
        axes[1].plot(ep, log[:, 3], "-o", ms=3, color=col, lw=1.8, label=f"{name}, peak 0.1")
        print(f"  {name:16s} final train loss {log[-1, 0]:.3g}  val acc {log[-1, 3]:.3f}")
    _, log, _ = train(constant, peak=0.01)
    axes[0].plot(ep, log[:, 0], "--", color=PURPLE, lw=1.8, label="constant, 0.01")
    axes[1].plot(ep, log[:, 3], "--", color=PURPLE, lw=1.8, label="constant, 0.01")
    print(f"  constant 0.01     final train loss {log[-1, 0]:.3g}  val acc {log[-1, 3]:.3f}")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-4, 1e4)
    axes[0].set_title("Training loss (full training set)")
    axes[0].set_ylabel("cross-entropy (log scale)")
    axes[1].set_title("Validation accuracy")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.6, 1.0)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "schedule-curves.png")


def fig_range_test():
    lrs, smooth, raw = lr_range_test()
    best = lrs[smooth.argmin()]
    print(f"range test: lowest smoothed loss {smooth.min():.3f} at lr {best:.3f}; stop at {lrs[-1]:.3f}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lrs, raw, color=GRAY, lw=0.8, alpha=0.6, label="batch loss")
    ax.plot(lrs, smooth, color=BLUE, lw=2.2, label="smoothed loss")
    ax.axvline(best, color=GREEN, ls="--", lw=1.5)
    ax.axvline(best / 10, color=ORANGE, ls=":", lw=1.8)
    ax.text(best / 1.12, 2.3, f"lowest loss\nlr ≈ {best:.2f}\n(peak for a\nschedule)", color=GREEN, fontsize=10,
            ha="right")
    ax.text(best / 10 / 1.12, 2.3, f"≈ {best / 10:.3f}\n(safe constant\nrate)", color=ORANGE, fontsize=10,
            ha="right")
    ax.set_xscale("log")
    ax.set_ylim(0, 3.2)
    ax.set_xlabel("learning rate (increased every step, log scale)")
    ax.set_ylabel("training loss")
    ax.set_title("Learning-rate range test: one short run, 200 steps at most")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower left")
    fig.tight_layout()
    save(fig, "lr-range-test.png")


def fig_overfit():
    good, bug = overfit_one_batch(), overfit_one_batch(double_softmax=True)
    floor = -np.log(np.e / (np.e + 9))
    print(f"overfit one batch: correct {good[-1]:.4f}, double softmax {bug[-1]:.4f}, floor {floor:.4f}")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(good, color=GREEN, lw=2, label="correct model: loss goes to ~0")
    ax.plot(bug, color=RED, lw=2, label="bug (softmax applied twice): loss stalls")
    ax.axhline(floor, color=RED, ls=":", lw=1.4)
    ax.text(150, floor * 1.25, f"lowest loss the bug allows: {floor:.2f}", color=RED, fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel("gradient step on the same 16 images")
    ax.set_ylabel("training loss (log scale)")
    ax.set_title("Overfit one batch: a healthy pipeline memorizes 16 images")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="center right")
    fig.tight_layout()
    save(fig, "overfit-one-batch.png")


def fig_seeds():
    seeds = range(5)
    A = np.array([train(warmup_cosine, peak=0.1, seed=s)[1][-1, 3] for s in seeds])
    B = np.array([train(one_cycle, peak=0.1, seed=s)[1][-1, 3] for s in seeds])
    d = B - A
    print("seed study A:", np.round(A, 3), "B:", np.round(B, 3))
    print(f"  A {A.mean():.3f} +- {A.std(ddof=1):.3f}; B {B.mean():.3f} +- {B.std(ddof=1):.3f}; "
          f"diff mean {d.mean():+.3f} range {d.min():+.3f}..{d.max():+.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    for s in seeds:
        ax.plot([0, 1], [A[s], B[s]], color=GRAY, lw=1, alpha=0.6)
    ax.scatter(np.zeros(5), A, color=BLUE, s=50, zorder=3)
    ax.scatter(np.ones(5), B, color=GREEN, s=50, zorder=3)
    for x, v, col in ((0, A, BLUE), (1, B, GREEN)):
        ax.errorbar(x + (-0.18 if x == 0 else 0.18), v.mean(), yerr=v.std(ddof=1), fmt="s",
                    color=col, capsize=5, ms=7)
    ax.set_xticks([0, 1], ["A: warmup + cosine", "B: one-cycle"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("validation accuracy")
    ax.set_title("5 seeds each (lines join equal seeds);\nsquares: mean ± 1 std")
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    cols = [GREEN if v > 0 else RED for v in d]
    ax.bar(list(seeds), d * 100, color=cols)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(d.mean() * 100, color=PURPLE, ls="--", lw=1.5, label=f"mean {d.mean() * 100:+.1f} points")
    ax.set_xlabel("seed")
    ax.set_ylabel("B − A (accuracy points)")
    ax.set_title("Paired difference per seed")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save(fig, "seed-variance.png")


def fig_augment():
    img = data.coins()
    fg = segmentation.clear_border(img > filters.threshold_otsu(img))
    lab = measure.label(ndimage.binary_fill_holes(fg))
    coin = max(measure.regionprops(lab), key=lambda r: r.area)
    mask = lab == coin.label
    box = tuple(int(v) for v in coin.bbox)
    H, W = img.shape

    def hflip(img, mask, box):
        y0, x0, y1, x1 = box
        return img[:, ::-1], mask[:, ::-1], (y0, W - x1, y1, W - x0)

    def crop(img, mask, box, top, left, h, w):
        y0, x0, y1, x1 = box
        new = (max(y0 - top, 0), max(x0 - left, 0), min(y1 - top, h), min(x1 - left, w))
        return img[top:top + h, left:left + w], mask[top:top + h, left:left + w], new

    i2, m2, b2 = crop(*hflip(img, mask, box), top=90, left=0, h=180, w=240)
    # wrong: transform the image only, keep the old annotations (cropped to the new frame)
    _, mw, bw = crop(img, mask, box, top=90, left=0, h=180, w=240)

    def show(ax, im, m, b, title, col):
        ax.imshow(im, cmap="gray")
        ov = np.zeros(m.shape + (4,))
        ov[m] = matplotlib.colors.to_rgba(col, 0.45)
        ax.imshow(ov)
        y0, x0, y1, x1 = b
        if y1 > y0 and x1 > x0:
            ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec=col, lw=2.2))
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [384, 240, 240]})
    show(axes[0], img, mask, box, "original: mask + box", BLUE)
    show(axes[1], i2, m2, b2, "flip + crop applied to\nimage, mask and box", GREEN)
    show(axes[2], i2, mw, bw, "WRONG: image changed,\nlabels not", RED)
    fig.tight_layout()
    save(fig, "augment-geometry.png")
    print(f"augment: box {box} -> {b2}")


def fig_flowchart():
    fig, ax = plt.subplots(figsize=(10, 8.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=10.5):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, label=None, lx=0.0, ly=0.0, col="black"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=col, lw=1.4))
        if label:
            ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label, fontsize=10, color=col, fontweight="bold")

    qs = [
        (9.4, "1. Is the loss finite\n(no NaN or inf)?",
         "No: find the first non-finite value; check inputs,\nlog/exp/division; lower the learning rate;\nreproduce in float32"),
        (7.6, "2. Can it overfit ONE\nsmall batch (loss ≈ 0)?",
         "No: a bug in model, loss or update; check the output\nlayer and loss pairing, run a gradient check,\nprint shapes"),
        (5.8, "3. Does the training loss\nfall on the full data?",
         "No: run a learning-rate range test; check init and\nnormalization; add warmup; add capacity"),
        (4.0, "4. Does validation follow\ntraining?",
         "Gap grows: augment, weight decay, early stop, more data.\nVal bad from step 0: check split, leakage,\npreprocessing mismatch"),
        (2.2, "5. Better than the baseline\nover several seeds?",
         "No: improve data or recipe; compare with the\nsame budget and the same seeds"),
    ]
    qx, ax_x = 2.05, 7.2
    for i, (yy, q, a) in enumerate(qs):
        box(qx, yy, 3.5, 1.2, q, "#e8eef8", BLUE)
        box(ax_x, yy, 5.5, 1.2, a, "#fbeaea", RED, fs=10)
        arrow(qx + 1.78, yy, ax_x - 2.78, yy, "no", lx=-0.12, ly=0.12, col=RED)
        if i < len(qs) - 1:
            arrow(qx, yy - 0.6, qx, qs[i + 1][0] + 0.6, "yes", lx=0.1, ly=-0.05, col=GREEN)
    box(qx, 0.55, 3.5, 0.85, "Done: evaluate the test set once;\nreport mean ± std over seeds", "#e9f5ec", GREEN)
    arrow(qx, 2.2 - 0.6, qx, 0.55 + 0.43, "yes", lx=0.1, ly=-0.05, col=GREEN)
    ax.text(5, 10.15, "Debugging a training run: ask the questions in order", ha="center", fontsize=12.5,
            fontweight="bold")
    save(fig, "debug-flowchart.png")


if __name__ == "__main__":
    fig_schedules()
    fig_schedule_curves()
    fig_range_test()
    fig_overfit()
    fig_seeds()
    fig_augment()
    fig_flowchart()
