"""Extra figures for ML 03 - Convolution, Image Filtering and CNNs.

Run:  .venv/bin/python scripts/figures/ml_03_convolutional_networks_extra.py
Writes PNGs only to src/assets/ml/ml-03-convolutional-networks/.
Uses skimage.data test images, scikit-learn digits and synthetic arrays; CPU only.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2
import torch
from torch import nn
from skimage import data
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parents[2] / "src/assets/ml/ml-03-convolutional-networks"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, GRAY = "#2a6fdb", "#e4572e", "#17a398", "#555555"
torch.set_num_threads(1)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name, f"{(OUT / name).stat().st_size / 1024:.0f} KB")


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------- 1. sliding window
def sliding_window():
    img = np.array([[1, 2, 0, 1, 3],
                    [0, 1, 3, 2, 1],
                    [2, 0, 1, 1, 0],
                    [1, 3, 2, 0, 2],
                    [0, 1, 1, 2, 1]], float)
    K = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float)
    pad = np.pad(img, 1)
    out = np.zeros_like(img)
    for y in range(5):
        for x in range(5):
            out[y, x] = (K * pad[y:y + 3, x:x + 3]).sum()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8),
                             gridspec_kw={"width_ratios": [7, 3, 5]})
    ax = axes[0]
    ax.imshow(pad, cmap="Greys", vmin=-1, vmax=6)
    for (r, c), v in np.ndenumerate(pad):
        is_pad = r in (0, 6) or c in (0, 6)
        ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                color=GRAY if is_pad else "black", fontsize=11)
    ax.add_patch(Rectangle((0.5, 0.5), 3, 3, fill=False, ec=ORANGE, lw=3))
    ax.add_patch(Rectangle((-0.5, -0.5), 7, 7, fill=False, ec=GRAY, lw=1, ls="--"))
    ax.set_title("Input with zero padding p = 1")
    off(ax)
    ax = axes[1]
    ax.imshow(K, cmap="coolwarm", vmin=-3, vmax=3)
    for (r, c), v in np.ndenumerate(K):
        ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=12)
    ax.set_title("Kernel K (3 × 3)")
    off(ax)
    ax = axes[2]
    ax.imshow(out, cmap="coolwarm", vmin=-8, vmax=8)
    for (r, c), v in np.ndenumerate(out):
        ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=11)
    ax.add_patch(Rectangle((0.5, 0.5), 1, 1, fill=False, ec=ORANGE, lw=3))
    ax.set_title("Output G (5 × 5)")
    off(ax)
    window = pad[1:4, 1:4]
    fig.suptitle(f"Orange window: sum of K ⊙ window = {(K * window).sum():.0f} → G[1, 1]",
                 y=0.04, fontsize=11)
    save(fig, "sliding-window.png")


# ---------------------------------------------------------------- 2. correlation vs convolution
def corr_vs_conv():
    img = np.zeros((9, 9), np.float32)
    img[4, 4] = 1.0
    K = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.float32)
    corr = cv2.filter2D(img, -1, K, borderType=cv2.BORDER_CONSTANT)
    conv = cv2.filter2D(img, -1, cv2.flip(K, -1), borderType=cv2.BORDER_CONSTANT)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    for ax, a, t in zip(axes, [K, img, corr, conv],
                        ["Kernel K", "Input: one bright pixel",
                         "Correlation (filter2D)", "Convolution (flipped K)"]):
        ax.imshow(a, cmap="Blues", vmin=0, vmax=9 if a.max() > 1 else 1)
        if a.shape[0] == 3:
            for (r, c), v in np.ndenumerate(a):
                ax.text(c, r, f"{v:.0f}", ha="center", va="center")
        else:
            for (r, c), v in np.ndenumerate(a[3:6, 3:6]):
                if v:
                    ax.text(c + 3, r + 3, f"{v:.0f}", ha="center", va="center", fontsize=9)
        ax.set_title(t, fontsize=11)
        off(ax)
    save(fig, "corr-vs-conv.png")


# ---------------------------------------------------------------- 3. filter gallery
def filter_gallery():
    img = data.camera().astype(np.float32) / 255.0
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    box = cv2.blur(img, (7, 7), borderType=cv2.BORDER_REFLECT_101)
    gauss = cv2.GaussianBlur(img, (0, 0), 2.0)
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gx = cv2.filter2D(img, -1, sx)
    gy = cv2.filter2D(img, -1, sx.T)
    mag = np.hypot(gx, gy)
    sharp = np.clip(img + 1.5 * (img - gauss), 0, 1)
    panels = [(img, "Input", "gray", 0, 1), (box, "Box 7 × 7", "gray", 0, 1),
              (gauss, "Gaussian σ = 2", "gray", 0, 1),
              (gx, "Sobel x (signed)", "coolwarm", -1, 1),
              (mag, "Gradient magnitude", "gray", 0, 2),
              (sharp, "Unsharp mask λ = 1.5", "gray", 0, 1)]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6))
    for ax, (a, t, cm, lo, hi) in zip(axes.ravel(), panels):
        ax.imshow(a, cmap=cm, vmin=lo, vmax=hi)
        ax.set_title(t)
        off(ax)
    save(fig, "filter-gallery.png")


# ---------------------------------------------------------------- 4. border modes
def border_modes():
    img = data.camera().astype(np.float32) / 255.0
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    modes = [("Zero (CONSTANT)", cv2.BORDER_CONSTANT),
             ("Replicate", cv2.BORDER_REPLICATE),
             ("Reflect-101 (OpenCV default)", cv2.BORDER_REFLECT_101),
             ("Wrap (circular)", cv2.BORDER_WRAP)]
    k = np.ones((21, 21), np.float32) / 441
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 6.0))
    for col, (name, m) in enumerate(modes):
        padded = cv2.copyMakeBorder(img, 24, 24, 24, 24, m, value=0)
        axes[0, col].imshow(padded, cmap="gray", vmin=0, vmax=1)
        axes[0, col].add_patch(Rectangle((23.5, 23.5), 128, 128, fill=False, ec=ORANGE, lw=1.5))
        axes[0, col].set_title(name, fontsize=10)
        out = cv2.filter2D(img, -1, k, borderType=m)
        axes[1, col].imshow(out, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title("21 × 21 box blur", fontsize=10)
        off(axes[0, col]); off(axes[1, col])
    fig.suptitle("Top: what the filter sees beyond the image (orange = real image). "
                 "Bottom: the blurred result", fontsize=11, y=0.02)
    save(fig, "border-modes.png")


# ---------------------------------------------------------------- 5. padding / stride / dilation
def geometry():
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    cases = [("k=3, p=0, s=1 (valid)", 0, 1, 1),
             ("k=3, p=1, s=1 (same)", 1, 1, 1),
             ("k=3, p=1, s=2", 1, 2, 1),
             ("k=3, p=2, s=1, d=2", 2, 1, 2)]
    H = 7
    for ax, (title, p, s, d) in zip(axes, cases):
        n = H + 2 * p
        ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(n - 0.4, -0.6)
        for r in range(n):
            for c in range(n):
                inside = p <= r < p + H and p <= c < p + H
                ax.add_patch(Rectangle((c - .5, r - .5), 1, 1,
                                       fc="#dfe8f7" if inside else "white",
                                       ec="#9aa7bd", lw=.6, ls="-" if inside else ":"))
        # first two output positions' taps
        for idx, col in enumerate([BLUE, ORANGE]):
            oy, ox = 0, idx * s
            for i in range(3):
                for j in range(3):
                    ax.add_patch(Rectangle((ox + j * d - .45, oy + i * d - .45), .9, .9,
                                           fc=col, alpha=.55, ec=None))
        Hout = (H + 2 * p - d * 2 - 1) // s + 1
        ax.set_title(f"{title}\noutput {Hout} × {Hout}", fontsize=10)
        ax.set_aspect("equal"); off(ax)
    fig.suptitle("7 × 7 input (shaded), padding (dotted), taps of two neighbouring outputs "
                 "(blue, orange)", fontsize=11, y=0.03)
    save(fig, "padding-stride-dilation.png")


# ---------------------------------------------------------------- 6. receptive field: theory vs effective
def receptive_field():
    torch.manual_seed(0)
    size = 65
    configs = [("3 layers, stride 1", [1, 1, 1]),
               ("6 layers, stride 1", [1] * 6),
               ("6 layers, stride 1,2,1,2,1,1", [1, 2, 1, 2, 1, 1])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, (title, strides) in zip(axes, configs):
        layers = []
        for s in strides:
            conv = nn.Conv2d(1, 1, 3, stride=s, padding=1, bias=False)
            nn.init.constant_(conv.weight, 1 / 9)
            layers += [conv]
        net = nn.Sequential(*layers)
        x = torch.zeros(1, 1, size, size, requires_grad=True)
        y = net(x)
        cy, cx = y.shape[2] // 2, y.shape[3] // 2
        y[0, 0, cy, cx].backward()
        g = x.grad[0, 0].abs().numpy()
        r, j = 1, 1
        for s in strides:
            r += 2 * j
            j *= s
        c0 = size // 2 - 17
        g = g[c0:c0 + 35, c0:c0 + 35]
        ax.imshow(g / g.max(), cmap="magma", vmin=0, vmax=1)
        ys, xs = np.nonzero(g > 0)
        ax.add_patch(Rectangle((xs.min() - .5, ys.min() - .5), np.ptp(xs) + 1, np.ptp(ys) + 1,
                               fill=False, ec="#7fd1ff", lw=1.5, ls="--"))
        ax.set_title(f"{title}\ntheoretical RF {r} × {r}", fontsize=10)
        off(ax)
    fig.suptitle("Gradient of one centre output w.r.t. the input (averaging kernels): "
                 "influence is concentrated in the middle of the theoretical field",
                 fontsize=10, y=0.04)
    save(fig, "receptive-field.png")


# ---------------------------------------------------------------- 7. pooling
def pooling():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 10, (4, 4)).astype(float)
    mx = x.reshape(2, 2, 2, 2).max(axis=(1, 3))
    av = x.reshape(2, 2, 2, 2).mean(axis=(1, 3))
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0), gridspec_kw={"width_ratios": [4, 2, 2]})
    cols = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]])
    pal = ["#dfe8f7", "#fde2d8", "#d6f2ee", "#f6ead0"]
    for ax, a, t, c in zip(axes, [x, mx, av], ["Input 4 × 4", "Max pool 2 × 2, s = 2",
                                            "Average pool 2 × 2, s = 2"],
                           [cols, np.array([[0, 1], [2, 3]]), np.array([[0, 1], [2, 3]])]):
        n = a.shape[0]
        for r in range(n):
            for q in range(n):
                ax.add_patch(Rectangle((q - .5, r - .5), 1, 1, fc=pal[c[r, q]], ec="k", lw=.8))
                ax.text(q, r, f"{a[r, q]:g}", ha="center", va="center", fontsize=12)
        ax.set_xlim(-.5, n - .5); ax.set_ylim(n - .5, -.5); ax.set_aspect("equal")
        ax.set_title(t, fontsize=10); off(ax)
    save(fig, "pooling.png")


# ---------------------------------------------------------------- 8. learned filters on digits
def learned_filters():
    torch.manual_seed(0)
    d = load_digits()
    X = torch.tensor(d.images[:, None] / 16.0, dtype=torch.float32)
    y = torch.tensor(d.target)
    tr, te = train_test_split(np.arange(len(y)), test_size=.2, stratify=d.target, random_state=0)
    model = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                          nn.Flatten(), nn.Linear(16 * 16, 10))
    opt = torch.optim.AdamW(model.parameters(), 3e-3, weight_decay=1e-4)
    for _ in range(15):
        perm = torch.tensor(tr)[torch.randperm(len(tr))]
        for idx in perm.split(64):
            loss = nn.functional.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    w = model[0].weight.detach()[:, 0].numpy()
    sample = X[te[0]:te[0] + 1]
    with torch.no_grad():
        fmap = torch.relu(model[0](sample))[0].numpy()
    fig, axes = plt.subplots(2, 9, figsize=(11, 3.0))
    axes[0, 0].imshow(sample[0, 0], cmap="gray"); axes[0, 0].set_title("input", fontsize=9)
    axes[1, 0].axis("off")
    for k in range(8):
        m = np.abs(w[k]).max()
        axes[0, k + 1].imshow(w[k], cmap="coolwarm", vmin=-m, vmax=m)
        axes[0, k + 1].set_title(f"filter {k}", fontsize=9)
        axes[1, k + 1].imshow(fmap[k], cmap="magma")
    axes[1, 1].set_ylabel("ReLU map", fontsize=9)
    for ax in axes.ravel():
        off(ax)
    fig.subplots_adjust(hspace=.3)
    fig.text(0.5, 0.0, "Top: the 8 learned 3 × 3 first-layer filters (red +, blue −). "
             "Bottom: each filter's ReLU response to the input digit.",
             ha="center", fontsize=10)
    save(fig, "learned-filters.png")


if __name__ == "__main__":
    sliding_window()
    corr_vs_conv()
    filter_gallery()
    border_modes()
    geometry()
    receptive_field()
    pooling()
    learned_filters()
