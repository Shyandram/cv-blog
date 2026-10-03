"""Figures for DIP Chapter 13 - Image Pattern Classification.

Run:  .venv/bin/python scripts/figures/dip_ch13.py
Writes only to src/assets/dip/ch13/. All data are synthetic or from skimage.data.
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage import data  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "dip" / "ch13"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})
COLORS = ["#d62728", "#2ca02c", "#ff7f0e"]


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{name}: {path.stat().st_size / 1024:.0f} KB")


# ---------------------------------------------------------------- 1. minimum distance
def fig_min_distance():
    rng = np.random.default_rng(0)
    names = ["apple", "lime", "orange"]
    means = np.array([[0.95, 7.5], [0.35, 5.0], [0.12, 8.0]])  # (hue, diameter cm)
    stds = np.array([[0.05, 0.7], [0.05, 0.5], [0.04, 0.7]])
    X = np.concatenate([m + s * rng.standard_normal((60, 2)) for m, s in zip(means, stds)])
    y = np.repeat(np.arange(3), 60)
    # standardize features so both axes count equally
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    M = np.array([Z[y == k].mean(0) for k in range(3)])
    gx, gy = np.meshgrid(np.linspace(-0.05, 1.1, 400), np.linspace(3, 10.5, 400))
    G = (np.c_[gx.ravel(), gy.ravel()] - mu) / sd
    d = ((G[:, None, :] - M[None]) ** 2).sum(-1)
    lab = d.argmin(1).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    from matplotlib.colors import ListedColormap

    ax.contourf(gx, gy, lab, levels=[-0.5, 0.5, 1.5, 2.5], cmap=ListedColormap(["#f6c9c9", "#cdeccd", "#ffe0bf"]))
    ax.contour(gx, gy, lab, levels=[0.5, 1.5], colors="k", linewidths=1)
    for k in range(3):
        ax.scatter(*X[y == k].T, s=16, c=COLORS[k], label=names[k], edgecolors="none")
        m = M[k] * sd + mu
        ax.scatter(*m, marker="*", s=320, c=COLORS[k], edgecolors="k", linewidths=1.2)
    ax.set_xlabel("feature 1: hue (0-1)")
    ax.set_ylabel("feature 2: diameter (cm)")
    ax.set_title("Minimum-distance classifier: stars = class means")
    ax.legend(loc="upper center")
    save(fig, "min-distance.png")


# ---------------------------------------------------------------- 2. template matching
def fig_template():
    img = data.coins().astype(np.float32)
    tpl = img[170:225, 180:235].copy()  # one coin
    R = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    th = 0.6
    peaks = (R == ndi.maximum_filter(R, size=25)) & (R > th)
    ys, xs = np.nonzero(peaks)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [0.5, 1.6, 1.6]})
    axs[0].imshow(tpl, cmap="gray")
    axs[0].set_title("template w")
    axs[1].imshow(R, cmap="viridis", vmin=-0.5, vmax=1)
    axs[1].set_title("normalized correlation")
    axs[2].imshow(img, cmap="gray")
    h, w = tpl.shape
    for x0, y0 in zip(xs, ys):
        axs[2].add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec="#ff3b3b", lw=1.5))
    axs[2].set_title(f"peaks > {th}: {len(xs)} matches")
    for a in axs:
        a.axis("off")
    save(fig, "template-matching.png")


# ---------------------------------------------------------------- 3. Bayes
def gauss_logpdf(X, m, C):
    Ci = np.linalg.inv(C)
    d = X - m
    return -0.5 * np.einsum("ni,ij,nj->n", d, Ci, d) - 0.5 * np.log(np.linalg.det(C))


def fig_bayes():
    rng = np.random.default_rng(1)
    gx, gy = np.meshgrid(np.linspace(-4, 6, 400), np.linspace(-4, 6, 400))
    G = np.c_[gx.ravel(), gy.ravel()]
    cases = [
        ("equal covariances: linear boundary", [np.eye(2) * 1.0, np.eye(2) * 1.0]),
        ("unequal covariances: quadratic boundary", [np.array([[0.5, 0.0], [0.0, 0.5]]), np.array([[2.5, 1.2], [1.2, 1.5]])]),
    ]
    m1, m2 = np.array([0.0, 0.0]), np.array([2.5, 2.0])
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, (title, (C1, C2)) in zip(axs, cases):
        X1 = rng.multivariate_normal(m1, C1, 150)
        X2 = rng.multivariate_normal(m2, C2, 150)
        dlt = gauss_logpdf(G, m1, C1) - gauss_logpdf(G, m2, C2)
        ax.contour(gx, gy, dlt.reshape(gx.shape), levels=[0], colors="k", linewidths=2)
        for m, C, c in [(m1, C1, COLORS[0]), (m2, C2, COLORS[1])]:
            p = np.exp(gauss_logpdf(G, m, C)).reshape(gx.shape)
            ax.contour(gx, gy, p, levels=3, colors=c, linewidths=0.8, alpha=0.8)
        ax.scatter(*X1.T, s=8, c=COLORS[0], label="class 1")
        ax.scatter(*X2.T, s=8, c=COLORS[1], label="class 2")
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")
        ax.legend(loc="lower right", fontsize=9)
    save(fig, "bayes-boundaries.png")


def fig_multispectral():
    """Synthetic 4-band 'satellite' scene: water, vegetation, built-up."""
    rng = np.random.default_rng(2)
    H = W = 128
    yy, xx = np.mgrid[:H, :W]
    truth = np.full((H, W), 2)  # built-up
    truth[(xx - 30) ** 2 / 900 + (yy - 90) ** 2 / 500 < 1] = 0  # lake
    truth[ndi.gaussian_filter(rng.standard_normal((H, W)), 8) > 0.02] = 1  # vegetation blobs
    truth[(xx - 30) ** 2 / 900 + (yy - 90) ** 2 / 500 < 1] = 0
    # mean reflectance per class in bands (blue, green, red, near-IR)
    means = np.array([[0.10, 0.08, 0.05, 0.03], [0.05, 0.12, 0.06, 0.45], [0.18, 0.18, 0.20, 0.25]])
    noise = 0.06
    X = means[truth] + noise * rng.standard_normal((H, W, 4))
    # train on 50 random pixels per class
    feats = X.reshape(-1, 4)
    lab = truth.ravel()
    ms, Cs = [], []
    for k in range(3):
        idx = rng.choice(np.flatnonzero(lab == k), 50, replace=False)
        ms.append(feats[idx].mean(0))
        Cs.append(np.cov(feats[idx].T))
    priors = np.bincount(lab, minlength=3) / lab.size
    score = np.stack([gauss_logpdf(feats, ms[k], Cs[k]) + np.log(priors[k]) for k in range(3)], 1)
    pred = score.argmax(1).reshape(H, W)
    acc = (pred == truth).mean()
    # single-band threshold baseline on red
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#1f77b4", "#2ca02c", "#9a9a9a"])
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    axs[0].imshow(truth, cmap=cmap, vmin=0, vmax=2)
    axs[0].set_title("ground truth")
    axs[1].imshow(X[..., 2], cmap="gray")
    axs[1].set_title("band 3 (red)")
    axs[2].imshow(X[..., 3], cmap="gray")
    axs[2].set_title("band 4 (near-IR)")
    axs[3].imshow(pred, cmap=cmap, vmin=0, vmax=2)
    axs[3].set_title(f"Bayes result, {acc * 100:.1f}% correct")
    for a in axs:
        a.axis("off")
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(cmap.colors, ["water", "vegetation", "built-up"])],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.04))
    save(fig, "multispectral-bayes.png")
    return acc


# ---------------------------------------------------------------- 4. perceptron
def fig_perceptron():
    rng = np.random.default_rng(3)
    X1 = rng.normal([0.7, 0.8], 0.5, (25, 2))
    X2 = rng.normal([-0.8, -0.6], 0.5, (25, 2))
    X = np.r_[X1, X2]
    y = np.r_[np.ones(25), -np.ones(25)]
    Xa = np.c_[X, np.ones(len(X))]  # augmented with bias input 1
    w = np.array([1.0, -1.0, 0.0])
    alpha = 0.5
    snaps = [(0, w.copy(), None)]
    updates = 0
    order = rng.permutation(len(X))
    for epoch in range(1, 20):
        errs = 0
        for i in order:
            if y[i] * (w @ Xa[i]) <= 0:
                w = w + alpha * y[i] * Xa[i]
                updates += 1
                errs += 1
                snaps.append((updates, w.copy(), i))
        if errs == 0:
            break
    final = (updates, w.copy(), None)
    snaps = [snaps[0], snaps[1], snaps[max(2, len(snaps) // 2)], final]
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.6), sharey=True)
    xs = np.linspace(-2.5, 2.5, 10)
    for ax, (u, wv, i) in zip(axs, snaps):
        ax.scatter(*X1.T, s=14, c=COLORS[0], label="class +1")
        ax.scatter(*X2.T, s=14, c=COLORS[1], label="class -1")
        ax.plot(xs, -(wv[0] * xs + wv[2]) / wv[1], "k-", lw=2)
        if i is not None:
            ax.scatter(*X[i], s=160, facecolors="none", edgecolors="k", lw=1.5)
        mis = (np.sign(Xa @ wv) != y).sum()
        title = "start" if u == 0 else f"after {u} update{'s' if u > 1 else ''}"
        if wv is final[1]:
            title = f"converged ({u} updates)"
        ax.set_title(f"{title}\n{mis} misclassified")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x1")
    axs[0].set_ylabel("x2")
    axs[0].legend(loc="lower right", fontsize=8)
    save(fig, "perceptron.png")


# ---------------------------------------------------------------- 5. CNN diagram
def fig_cnn_arch():
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def stack(x, n, size, color, label, sub):
        for k in range(n):
            ax.add_patch(FancyBboxPatch((x + 0.12 * k, 1.5 + 0.12 * k), size, size, boxstyle="square,pad=0",
                                        fc=color, ec="k", lw=0.8))
        ax.text(x + size / 2 + 0.06 * n, 0.9, label, ha="center", fontsize=10)
        ax.text(x + size / 2 + 0.06 * n, 0.4, sub, ha="center", fontsize=9, color="#444")

    stack(0.3, 1, 3.0, "#dddddd", "input", "20x20x1")
    stack(4.6, 8, 2.6, "#9ecae1", "conv 5x5 + ReLU", "16x16x8")
    stack(9.0, 8, 1.4, "#6baed6", "max-pool 2x2", "8x8x8")
    for k in range(12):
        ax.add_patch(plt.Rectangle((13.0, 0.95 + k * 0.38), 0.35, 0.33, fc="#fdae6b", ec="k", lw=0.6))
    ax.text(13.2, 0.4, "flatten (512) -> FC", ha="center", fontsize=9, color="#444")
    for k in range(10):
        ax.add_patch(plt.Circle((16.0, 1.25 + k * 0.42), 0.16, fc="#74c476", ec="k", lw=0.6))
        ax.text(16.35, 1.25 + k * 0.42, str(k), va="center", fontsize=8)
    ax.text(16.0, 0.4, "10 scores", ha="center", fontsize=9, color="#444")
    ax.text(18.9, 3.1, "softmax\n-> class\nprobabilities", ha="center", va="center", fontsize=10)
    for x0, x1 in [(3.5, 4.5), (8.1, 8.9), (11.4, 12.9), (13.5, 15.7), (16.7, 17.8)]:
        ax.annotate("", xy=(x1, 3.1), xytext=(x0, 3.1), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(7.0, 5.6, "feature extractor (learned)", ha="center", fontsize=11, weight="bold")
    ax.plot([4.4, 11.6], [5.3, 5.3], "k-", lw=1)
    ax.text(16.0, 5.6, "classifier (learned)", ha="center", fontsize=11, weight="bold")
    ax.plot([12.8, 19.4], [5.3, 5.3], "k-", lw=1)
    save(fig, "cnn-architecture.png")


# ---------------------------------------------------------------- 6. tiny NumPy CNN on synthetic digits
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
         cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX]


def make_digits(n, rng, S=20):
    X = np.zeros((n, S, S), np.float32)
    y = rng.integers(0, 10, n)
    for i in range(n):
        canvas = np.zeros((40, 40), np.uint8)
        font = FONTS[rng.integers(len(FONTS))]
        scale = rng.uniform(0.9, 1.25) * (2.0 if font == cv2.FONT_HERSHEY_PLAIN else 1.0)
        th = int(rng.integers(2, 4))
        (tw, tht), _ = cv2.getTextSize(str(y[i]), font, scale, th)
        org = (int(20 - tw / 2 + rng.integers(-2, 3)), int(20 + tht / 2 + rng.integers(-2, 3)))
        cv2.putText(canvas, str(y[i]), org, font, scale, 255, th, cv2.LINE_AA)
        Mrot = cv2.getRotationMatrix2D((20, 20), rng.uniform(-15, 15), 1.0)
        canvas = cv2.warpAffine(canvas, Mrot, (40, 40))
        img = cv2.resize(canvas, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        X[i] = np.clip(img + 0.08 * rng.standard_normal((S, S)), 0, 1)
    return X, y


def im2col(X, k):
    n, H, W = X.shape
    Ho, Wo = H - k + 1, W - k + 1
    s = X.strides
    cols = np.lib.stride_tricks.as_strided(X, (n, Ho, Wo, k, k), (s[0], s[1], s[2], s[1], s[2]))
    return cols.reshape(n, Ho * Wo, k * k)


class TinyCNN:
    """conv(8 filters 5x5) -> ReLU -> maxpool 2x2 -> FC(512->10) -> softmax."""

    def __init__(self, rng, F=8, k=5, S=20):
        self.k, self.F = k, F
        self.Ho = S - k + 1
        self.Wc = rng.standard_normal((k * k, F)) * np.sqrt(2 / (k * k))
        self.bc = np.zeros(F)
        D = (self.Ho // 2) ** 2 * F
        self.Wf = rng.standard_normal((D, 10)) * np.sqrt(1 / D)
        self.bf = np.zeros(10)

    def forward(self, X):
        n = len(X)
        cols = im2col(X, self.k)  # n, P, k*k
        z = cols @ self.Wc + self.bc  # n, P, F
        a = np.maximum(z, 0).reshape(n, self.Ho, self.Ho, self.F)
        h = self.Ho // 2
        blocks = a.reshape(n, h, 2, h, 2, self.F).transpose(0, 1, 3, 5, 2, 4).reshape(n, h, h, self.F, 4)
        p = blocks.max(-1)
        arg = blocks.argmax(-1)
        flat = p.reshape(n, -1)
        s = flat @ self.Wf + self.bf
        s -= s.max(1, keepdims=True)
        prob = np.exp(s)
        prob /= prob.sum(1, keepdims=True)
        self.cache = (cols, z, arg, flat)
        return prob, a

    def step(self, X, y, lr, wd=1e-4):
        n = len(X)
        prob, _ = self.forward(X)
        loss = -np.log(prob[np.arange(n), y] + 1e-12).mean()
        cols, z, arg, flat = self.cache
        ds = prob.copy()
        ds[np.arange(n), y] -= 1
        ds /= n
        dWf = flat.T @ ds + wd * self.Wf
        dbf = ds.sum(0)
        dp = (ds @ self.Wf.T).reshape(n, self.Ho // 2, self.Ho // 2, self.F)
        h = self.Ho // 2
        dblocks = np.zeros((n, h, h, self.F, 4))
        np.put_along_axis(dblocks, arg[..., None], dp[..., None], axis=-1)
        da = dblocks.reshape(n, h, h, self.F, 2, 2).transpose(0, 1, 4, 2, 5, 3).reshape(n, self.Ho, self.Ho, self.F)
        dz = da.reshape(n, -1, self.F) * (z > 0)
        dWc = np.einsum("npk,npf->kf", cols, dz) + wd * self.Wc
        dbc = dz.sum((0, 1))
        for P, dP in [(self.Wf, dWf), (self.bf, dbf), (self.Wc, dWc), (self.bc, dbc)]:
            P -= lr * dP
        return loss


def augment(X, rng):
    out = np.empty_like(X)
    for i, img in enumerate(X):
        M = cv2.getRotationMatrix2D((10, 10), rng.uniform(-12, 12), rng.uniform(0.9, 1.1))
        M[:, 2] += rng.integers(-2, 3, 2)
        out[i] = cv2.warpAffine(img, M, (20, 20), borderMode=cv2.BORDER_CONSTANT)
    return out


def train(Xtr, ytr, Xva, yva, epochs, rng, aug=False, bs=32, lr=0.05):
    net = TinyCNN(rng)
    hist = []
    for ep in range(epochs):
        idx = rng.permutation(len(Xtr))
        for b in range(0, len(idx), bs):
            j = idx[b:b + bs]
            xb = augment(Xtr[j], rng) if aug else Xtr[j]
            net.step(xb, ytr[j], lr)
        ptr, _ = net.forward(Xtr)
        pva, _ = net.forward(Xva)
        hist.append(((ptr.argmax(1) == ytr).mean(), (pva.argmax(1) == yva).mean()))
    return net, np.array(hist)


def fig_cnn():
    rng = np.random.default_rng(4)
    Xtr, ytr = make_digits(3000, rng)
    Xva, yva = make_digits(1000, rng)
    net, hist = train(Xtr, ytr, Xva, yva, epochs=12, rng=rng)
    pva, _ = net.forward(Xva)
    pred = pva.argmax(1)
    acc = (pred == yva).mean()
    print(f"CNN val accuracy (3000 train): {acc:.3f}")

    # samples
    fig, axs = plt.subplots(2, 10, figsize=(10, 2.3))
    for i, a in enumerate(axs.ravel()):
        j = np.flatnonzero(ytr == i % 10)[i // 10]
        a.imshow(Xtr[j], cmap="gray")
        a.axis("off")
    fig.suptitle("synthetic 20x20 digits (random font, size, shift, rotation, noise)", fontsize=11)
    save(fig, "synthetic-digits.png")

    # filters, feature maps, confusion
    j = np.flatnonzero(yva == 3)[0]
    _, amap = net.forward(Xva[j:j + 1])
    fig = plt.figure(figsize=(10.5, 3.0))
    gs = fig.add_gridspec(2, 10, hspace=0.35, wspace=0.15)
    ax = fig.add_subplot(gs[0:2, 0:2])
    ax.imshow(Xva[j], cmap="gray")
    ax.set_title("input")
    ax.axis("off")
    W = net.Wc.reshape(5, 5, 8)
    for f in range(8):
        a = fig.add_subplot(gs[0, 2 + f])
        a.imshow(W[..., f], cmap="RdBu_r", vmin=-np.abs(W).max(), vmax=np.abs(W).max())
        a.set_title(f"filter {f + 1}", fontsize=9)
        a.axis("off")
        a = fig.add_subplot(gs[1, 2 + f])
        a.imshow(amap[0, ..., f], cmap="magma")
        a.set_title(f"map {f + 1}", fontsize=9)
        a.axis("off")
    fig.suptitle(f"learned 5x5 filters (top) and their ReLU feature maps (bottom); "
                 f"validation accuracy {acc * 100:.1f}%", fontsize=11)
    save(fig, "cnn-features.png")
    return acc, hist


def fig_overfitting():
    rng = np.random.default_rng(5)
    Xtr, ytr = make_digits(150, rng)
    Xva, yva = make_digits(1000, rng)
    _, h0 = train(Xtr, ytr, Xva, yva, epochs=60, rng=np.random.default_rng(6), aug=False, bs=16)
    _, h1 = train(Xtr, ytr, Xva, yva, epochs=60, rng=np.random.default_rng(6), aug=True, bs=16)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    ep = np.arange(1, len(h0) + 1)
    for ax, h, t in [(axs[0], h0, "no augmentation"), (axs[1], h1, "with random shift/rotate/scale")]:
        ax.plot(ep, h[:, 0], label="training accuracy", c="#1f77b4", lw=2)
        ax.plot(ep, h[:, 1], label="validation accuracy", c="#d62728", lw=2)
        ax.set_title(f"150 training images, {t}")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("accuracy")
    axs[0].legend(loc="lower right")
    save(fig, "overfitting.png")
    print(f"overfit final: no-aug train {h0[-1, 0]:.3f} val {h0[-1, 1]:.3f}; aug train {h1[-1, 0]:.3f} val {h1[-1, 1]:.3f}")


if __name__ == "__main__":
    import sys
    if "--quick" in sys.argv:
        fig_perceptron()
        fig_cnn()
        raise SystemExit
    fig_min_distance()
    fig_template()
    fig_bayes()
    print(f"multispectral accuracy: {fig_multispectral():.3f}")
    fig_perceptron()
    fig_cnn_arch()
    fig_cnn()
    fig_overfitting()
