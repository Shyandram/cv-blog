"""Extra figures for ML 04 - Attention and Vision Transformers.

Run:  .venv/bin/python scripts/figures/ml_04_vision_transformers_extra.py
Writes PNGs only to src/assets/ml/ml-04-vision-transformers/.
Uses skimage.data.astronaut and synthetic arrays; no model weights, CPU only.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import cv2
from skimage import data

OUT = Path(__file__).resolve().parents[2] / "src/assets/ml/ml-04-vision-transformers"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, GOLD, PURPLE, GRAY = ("#2a6fdb", "#e4572e", "#17a398",
                                           "#f3a712", "#8e5ad6", "#555555")


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name, f"{(OUT / name).stat().st_size / 1024:.0f} KB")


def off(ax):
    ax.set_xticks([]); ax.set_yticks([])


def box(ax, xy, w, h, text, fc, fs=11):
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec="k", lw=1.2))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, color="k", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=color,
                                 lw=1.4, connectionstyle=f"arc3,rad={rad}"))


# --------------------------------------------------------------- 1. patchify
def patchify():
    img = cv2.resize(data.astronaut(), (224, 224), interpolation=cv2.INTER_AREA)
    P = 32                                   # 7 x 7 grid keeps the figure readable
    n = 224 // P
    fig = plt.figure(figsize=(12, 4.2))
    ax = fig.add_axes([0.0, 0.05, 0.3, 0.9])
    ax.imshow(img)
    for k in range(1, n):
        ax.axhline(k * P - .5, color="white", lw=1.2)
        ax.axvline(k * P - .5, color="white", lw=1.2)
    ax.set_title(f"224 × 224 image, {P} × {P} patches → {n * n} tokens", fontsize=10)
    off(ax)
    # first 12 patches in raster order
    for i in range(12):
        r, c = divmod(i, n)
        a = fig.add_axes([0.33 + i * 0.043, 0.55, 0.038, 0.11])
        a.imshow(img[r * P:(r + 1) * P, c * P:(c + 1) * P])
        a.set_title(str(i), fontsize=8); off(a)
    fig.text(0.33 + 12 * 0.043, 0.6, "…  (49)", fontsize=11, va="center")
    fig.text(0.33, 0.73, "1. cut into patches, read in raster order", fontsize=10)
    fig.text(0.33, 0.42, "2. flatten each patch (P·P·C numbers), multiply by E  →  D-dim token",
             fontsize=10)
    fig.text(0.33, 0.33, "3. add a position embedding $p_i$; optionally prepend a [CLS] token",
             fontsize=10)
    fig.text(0.33, 0.24, "4. the sequence of N tokens (N × D) enters the Transformer encoder",
             fontsize=10)
    fig.text(0.33, 0.13, "ViT-Base uses P = 16 on 224 × 224 → 196 tokens; P = 32 is drawn here "
             "for clarity.", fontsize=9, color=GRAY)
    save(fig, "patchify.png")


# --------------------------------------------------------------- 2. positional encodings
def sinusoid(n_pos, d):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d // 2)[None]
    ang = pos / 10000 ** (2 * i / d)
    pe = np.zeros((n_pos, d))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return pe


def positional():
    pe = sinusoid(64, 128)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8),
                             gridspec_kw={"width_ratios": [1.3, 1, 1], "wspace": 0.45})
    im = axes[0].imshow(pe.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xlabel("position $pos$"); axes[0].set_ylabel("dimension")
    axes[0].set_title("Sinusoidal encoding, $D = 128$")
    fig.colorbar(im, ax=axes[0], fraction=.04)
    sim = pe @ pe.T / (128 / 2)
    axes[1].plot(sim[20], color=BLUE, lw=2)
    axes[1].axvline(20, color=GRAY, ls="--")
    axes[1].set_xlabel("position $m$"); axes[1].set_ylabel(r"$p_{20}\cdot p_m$ (normalised)")
    axes[1].set_title("Similarity depends on offset")
    axes[1].grid(alpha=.3)
    # 2-D: concatenate row and column encodings for a 14 x 14 patch grid
    g = 14
    row, col = sinusoid(g, 64), sinusoid(g, 64)
    pe2 = np.concatenate([np.repeat(row, g, 0), np.tile(col, (g, 1))], 1)   # 196 x 128
    q = 4 * g + 9
    s2 = (pe2 @ pe2[q]).reshape(g, g)
    axes[2].imshow(s2, cmap="viridis")
    axes[2].plot(9, 4, "r*", ms=12)
    axes[2].set_title("2-D (row ‖ column) encoding:\nsimilarity to the starred patch", fontsize=10)
    off(axes[2])
    save(fig, "positional-encoding.png")


# --------------------------------------------------------------- 3. transformer block
def block():
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis("off")
    y = 1.2
    items = [(0.7, "$X$\n(N × D)", "#f3f3f3", 1.0), (2.3, "LayerNorm", "#efe3fb", 1.3),
             (4.1, "Multi-head\nself-attention", "#dfe8f7", 1.7), (5.8, "+", "white", 0.5),
             (7.2, "LayerNorm", "#efe3fb", 1.3), (9.0, "MLP\nD→4D→D, GELU", "#d6f2ee", 1.7),
             (10.7, "+", "white", 0.5)]
    for x, t, fc, w in items:
        box(ax, (x, y), w, 0.9 if t != "+" else 0.5, t, fc, fs=10 if t != "+" else 14)
    for (x0, _, _, w0), (x1, _, _, w1) in zip(items[:-1], items[1:]):
        arrow(ax, (x0 + w0 / 2, y), (x1 - w1 / 2, y))
    arrow(ax, (10.95, y), (11.7, y)); ax.text(11.75, y, "$Y$", va="center", fontsize=12)
    # residual paths
    ax.plot([1.45], [y], "o", color=ORANGE, ms=5)
    ax.add_patch(FancyArrowPatch((1.45, y), (5.8, y + 0.25), arrowstyle="-|>",
                                 mutation_scale=13, color=ORANGE, lw=1.6,
                                 connectionstyle="arc3,rad=-0.35"))
    ax.add_patch(FancyArrowPatch((6.05, y + 0.25), (10.7, y + 0.25), arrowstyle="-|>",
                                 mutation_scale=13, color=ORANGE, lw=1.6,
                                 connectionstyle="arc3,rad=-0.35"))
    ax.text(5.9, 2.75, "residual (identity) paths", ha="center", color=ORANGE, fontsize=10)
    ax.text(4.1, 0.25, "mixes information ACROSS tokens", ha="center", fontsize=9, color=GRAY)
    ax.text(9.0, 0.25, "transforms EACH token separately", ha="center", fontsize=9, color=GRAY)
    save(fig, "transformer-block.png")


# --------------------------------------------------------------- 4. Swin windows
def swin():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    n, M = 8, 4
    cmap = plt.get_cmap("tab10")
    for ax, shift, title in [(axes[0], 0, "Layer ℓ: regular 4 × 4 windows"),
                             (axes[1], M // 2, "Layer ℓ+1: windows shifted by 2")]:
        ids = np.zeros((n, n), int)
        for r in range(n):
            for c in range(n):
                rr, cc = (r - shift) % n if shift else r, (c - shift) % n if shift else c
                if shift:
                    rr = (r + shift) // M if r + shift < n else 2 + (r + shift - n) // M
                    cc = (c + shift) // M if c + shift < n else 2 + (c + shift - n) // M
                    ids[r, c] = min(rr, 2) * 3 + min(cc, 2)
                else:
                    ids[r, c] = (r // M) * 2 + c // M
        for r in range(n):
            for c in range(n):
                ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, fc=cmap(ids[r, c] % 10), alpha=.35,
                                       ec="white", lw=1))
        # window borders
        cuts = [0, M, n] if not shift else [0, M - shift, n - shift, n]
        for cpos in cuts:
            ax.axvline(cpos, color="k", lw=2); ax.axhline(n - cpos, color="k", lw=2)
        ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_aspect("equal"); off(ax)
        ax.set_title(title, fontsize=10)
    # receptive coverage of one token after the two layers
    ax = axes[2]
    reach = np.zeros((n, n))
    q = (1, 3)                                    # token near a window border
    w1 = [(r, c) for r in range(0, M) for c in range(0, M)]             # its first window
    reach1 = np.zeros((n, n)); [reach1.__setitem__(p, 1) for p in w1]
    # second layer: union of shifted windows that contain any reached token
    shifted_id = lambda r, c: (min((r + 2) // M if r + 2 < n else 2, 2),
                               min((c + 2) // M if c + 2 < n else 2, 2))
    wins = {shifted_id(r, c) for r, c in w1}
    for r in range(n):
        for c in range(n):
            if shifted_id(r, c) in wins:
                reach[r, c] = 1
    ax.imshow(reach + reach1, cmap="Blues", vmin=0, vmax=2.4)
    ax.plot(q[1], q[0], "r*", ms=14)
    ax.set_title("Tokens influenced by ★ after 2 layers\n(dark: layer ℓ, light: added by the shift)",
                 fontsize=10)
    for k in range(n + 1):
        ax.axhline(k - .5, color="white", lw=.8); ax.axvline(k - .5, color="white", lw=.8)
    off(ax)
    save(fig, "swin-windows.png")


if __name__ == "__main__":
    patchify()
    positional()
    block()
    swin()
