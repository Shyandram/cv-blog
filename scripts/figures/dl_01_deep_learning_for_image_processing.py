"""Figures for DL 01 - Deep Learning for Image Processing and Computer Vision.

Run from the repo root:
    .venv/bin/python scripts/figures/dl_01_deep_learning_for_image_processing.py
Writes PNGs to src/assets/deep-learning/dl-01-deep-learning-for-image-processing/.
All inputs are skimage.data images, scikit-learn's load_digits, or synthetic arrays.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from skimage import data, img_as_float, transform
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
OUT = (Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning"
       / "dl-01-deep-learning-for-image-processing")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

BLUE, ORANGE, GREEN, GRAY, PURPLE = "#4C72B0", "#DD8452", "#55A868", "#8C8C8C", "#8172B3"


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(path).convert("RGB")
    if im.width > 1600:
        im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
    im.quantize(colors=128, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)


def box(ax, x, y, w, h, text, color, fontsize=11, alpha=0.18, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=color, alpha=alpha, lw=0))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="none", ec=color, lw=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color="black", style="-|>", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, linestyle=ls))


# ---------------------------------------------------------------- 1. three levels
def fig_three_levels():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.text(2.0, 5.35, "Classical (textbook) view", ha="center", fontsize=12, fontweight="bold")
    ax.text(8.0, 5.35, "Deep-learning view", ha="center", fontsize=12, fontweight="bold")
    box(ax, 0.2, 3.7, 3.6, 1.2, "Computer vision (high level)\nimage -> meaning\nclassify, recognize", ORANGE)
    box(ax, 0.2, 2.1, 3.6, 1.2, "Image analysis (mid level)\nimage -> attributes / regions\nsegment, extract features", PURPLE)
    box(ax, 0.2, 0.5, 3.6, 1.2, "Image processing (low level)\nimage -> image\nfilter, equalize, transform", BLUE)
    box(ax, 6.2, 3.2, 3.6, 1.6, "High-level vision\noutput = semantics\nlabels, boxes, per-pixel classes\nloss: classification", ORANGE)
    box(ax, 6.2, 0.6, 3.6, 1.6, "Low-level vision\noutput = an image\nrestore, enhance, translate\nloss: pixel fidelity", BLUE)
    arrow(ax, 3.9, 4.3, 6.1, 4.1, ORANGE, lw=2)
    arrow(ax, 3.9, 1.1, 6.1, 1.3, BLUE, lw=2)
    arrow(ax, 3.9, 2.85, 6.1, 3.5, PURPLE, ls="--")
    arrow(ax, 3.9, 2.55, 6.1, 1.9, PURPLE, ls="--")
    ax.text(5.0, 3.55, "semantic maps\n(segmentation)", ha="center", fontsize=9.5, color=PURPLE)
    ax.text(5.0, 1.95, "cleaned signals\n(enhancement)", ha="center", fontsize=9.5, color=PURPLE)
    save(fig, "three-levels.png")


# ---------------------------------------------------------------- 2. pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    ax.text(0.1, 4.55, "Classical pattern recognition", fontsize=12, fontweight="bold")
    xs = [0.1, 2.3, 4.5, 6.7, 8.9]
    labels = ["Sensor", "Feature\ngeneration\n(hand-designed)", "Feature\nselection",
              "Classifier\n(learned)", "System\nevaluation"]
    colors = [GRAY, BLUE, BLUE, ORANGE, GRAY]
    for x, t, c in zip(xs, labels, colors):
        box(ax, x, 2.75, 1.9, 1.35, t, c, fontsize=10.5)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + 1.95, 3.42, b - 0.05, 3.42)
    arrow(ax, 9.85, 2.55, 3.3, 2.55, GRAY, ls="--", style="-|>")
    ax.text(6.6, 2.22, "redesign features by hand if the error is too high", ha="center",
            fontsize=9.5, color=GRAY)

    ax.text(0.1, 1.95, "End-to-end deep learning", fontsize=12, fontweight="bold")
    box(ax, 0.1, 0.2, 1.9, 1.35, "Sensor\n(raw pixels)", GRAY, fontsize=10.5)
    box(ax, 2.3, 0.2, 6.3, 1.35, "One deep network: features + classifier\nall weights learned jointly",
        GREEN, fontsize=11)
    for k in range(5):
        ax.add_patch(Rectangle((2.6 + k * 1.15, 0.32), 0.25 + 0.08 * k, 0.28, fc=GREEN, alpha=0.5))
    box(ax, 8.9, 0.2, 1.9, 1.35, "Loss +\nevaluation", GRAY, fontsize=10.5)
    arrow(ax, 2.05, 0.87, 2.25, 0.87)
    arrow(ax, 8.65, 0.87, 8.85, 0.87)
    ax.annotate("", xy=(5.4, 1.62), xytext=(9.8, 1.62),
                arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.6,
                                connectionstyle="arc3,rad=0.25"))
    ax.text(10.9, 1.75, "gradients\n(backprop)", color="crimson", fontsize=9.5, ha="center")
    save(fig, "pipeline-vs-end-to-end.png")


# ---------------------------------------------------------------- 3. feature maps
def fig_feature_maps():
    img = img_as_float(data.camera())
    img = transform.resize(img, (256, 256), anti_aliasing=True)
    x = torch.tensor(img, dtype=torch.float32)[None, None]
    torch.manual_seed(1)
    rand = nn.Conv2d(1, 3, 3, padding=1, bias=False)
    sobel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
    hand = nn.Conv2d(1, 3, 3, padding=1, bias=False)
    with torch.no_grad():
        hand.weight[0, 0] = sobel
        hand.weight[1, 0] = sobel.T
        hand.weight[2, 0] = -lap
        yr = F.relu(rand(x))[0].numpy()
        yh = F.relu(hand(x))[0].numpy()
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.9))
    rows = [("random", yr, rand.weight.detach()[:, 0].numpy(),
             ["random #1", "random #2", "random #3"]),
            ("hand-set", yh, hand.weight.detach()[:, 0].numpy(),
             ["Sobel x (vertical edges)", "Sobel y (horizontal edges)", "negative Laplacian"])]
    for r, (_, maps, ws, names) in enumerate(rows):
        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 0].set_title("input")
        for c in range(3):
            axes[r, c + 1].imshow(maps[c], cmap="gray")
            axes[r, c + 1].set_title(names[c], fontsize=10)
            ins = axes[r, c + 1].inset_axes([0.0, 0.0, 0.28, 0.28])
            ins.imshow(ws[c], cmap="coolwarm", vmin=-np.abs(ws[c]).max(), vmax=np.abs(ws[c]).max())
            ins.set_xticks([]); ins.set_yticks([])
    axes[0, 0].set_ylabel("random filters", fontsize=11)
    axes[1, 0].set_ylabel("hand-set filters", fontsize=11)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    save(fig, "feature-maps.png")


# ---------------------------------------------------------------- 4. learned filters
def fig_learned_filters():
    torch.manual_seed(0)
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X / 16.0, dtype=torch.float32).reshape(-1, 1, 8, 8)
    Xtr, Xte, ytr, yte = train_test_split(X, torch.tensor(y), test_size=0.25, random_state=0)
    net = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 2 * 2, 10),
    )
    before = net[0].weight.detach()[:, 0].clone().numpy()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(30):
        for i in range(0, len(Xtr), 64):
            loss = F.cross_entropy(net(Xtr[i:i + 64]), ytr[i:i + 64])
            opt.zero_grad(); loss.backward(); opt.step()
    acc = (net(Xte).argmax(1) == yte).float().mean().item()
    after = net[0].weight.detach()[:, 0].numpy()
    fig, axes = plt.subplots(2, 8, figsize=(11, 3.3))
    for r, (ws, lab) in enumerate([(before, "before training"),
                                   (after, f"after training (test acc. {acc:.2f})")]):
        for c in range(8):
            w = ws[c]
            axes[r, c].imshow(w, cmap="gray", vmin=-np.abs(w).max(), vmax=np.abs(w).max())
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 3].set_title(lab, fontsize=11, x=1.1)
    fig.tight_layout()
    save(fig, "learned-filters.png")


# ---------------------------------------------------------------- 5. receptive field
def rf_curve(layers):
    r, j, out = 1, 1, []
    for k, s in layers:
        r, j = r + (k - 1) * j, j * s
        out.append(r)
    return out


def fig_receptive_field():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    a.set_xlim(-0.5, 12.5)
    a.set_ylim(-0.6, 3.8)
    a.axis("off")
    n = 13
    centre = 6
    for lvl in range(4):
        lo, hi = centre - lvl, centre + lvl          # units below that feed the top unit
        for i in range(n):
            active = (3 - lvl) <= 3 and abs(i - centre) <= (3 - lvl)
            col = BLUE if active else "#DDDDDD"
            a.add_patch(Rectangle((i - 0.4, lvl - 0.25), 0.8, 0.5, fc=col,
                                  alpha=0.85 if active else 1, ec="white"))
        del lo, hi
    for lvl in range(3, 0, -1):
        half = 3 - lvl
        for i in range(centre - half, centre + half + 1):
            for d in (-1, 0, 1):
                a.plot([i, i + d], [lvl - 0.25, lvl - 0.75], color=GRAY, lw=0.7)
    labels = ["input pixels", "layer 1 (3-tap)", "layer 2 (3-tap)", "layer 3 (3-tap)"]
    for lvl, t in enumerate(labels):
        a.text(-0.6, lvl, t, ha="right", va="center", fontsize=10)
    a.annotate("", xy=(centre - 3 - 0.4, -0.45), xytext=(centre + 3 + 0.4, -0.45),
               arrowprops=dict(arrowstyle="<->", color="crimson"))
    a.text(centre, -0.62, "receptive field = 7 pixels", ha="center", va="top", color="crimson")
    a.set_title("One output unit after three 3x3 layers (1-D slice)")
    a.set_xlim(-4.5, 12.5)

    depth = np.arange(1, 13)
    plain = rf_curve([(3, 1)] * 12)
    down = []
    layers = []
    for d in depth:
        layers.append((3, 1))
        if d % 2 == 0:
            layers.append((2, 2))
        down.append(rf_curve(layers)[-1])
    b.plot(depth, plain, "o-", color=BLUE, label="3x3, stride 1 everywhere")
    b.plot(depth, down, "s-", color=ORANGE, label="3x3 + 2x2 pool after every 2nd layer")
    b.set_xlabel("number of 3x3 conv layers")
    b.set_ylabel("theoretical receptive field (pixels)")
    b.set_yscale("log", base=2)
    b.grid(alpha=0.3)
    b.legend(fontsize=9.5, loc="upper left")
    b.set_title("Receptive field versus depth")
    fig.tight_layout()
    save(fig, "receptive-field.png")


# ---------------------------------------------------------------- 6. timeline
def fig_timeline():
    lanes = {
        "CNN backbones": (2.0, BLUE, [
            (2012, "AlexNet"), (2014, "VGG / GoogLeNet"), (2015, "ResNet"),
            (2022, "ConvNeXt"), (2023, "ConvNeXt V2")]),
        "Transformers &\nself-supervision": (1.0, PURPLE, [
            (2020, "ViT"), (2021, "Swin / DINO / MAE"), (2023, "DINOv2"),
            (2024, "Vim / VMamba"), (2025, "DINOv3")]),
        "Task & foundation\nmodels": (0.0, ORANGE, [
            (2014, "FCN"), (2015, "U-Net /\nFaster R-CNN"), (2020, "DETR / DDPM"),
            (2021, "CLIP / latent\ndiffusion"), (2022, "DiT /\nInstructPix2Pix"),
            (2023, "SAM / LLaVA"), (2024, "SAM 2"), (2025, "SAM 3")]),
    }
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    for name, (yv, col, items) in lanes.items():
        ax.axhline(yv, color=col, lw=2, alpha=0.35)
        ax.text(2011.2, yv, name, ha="right", va="center", fontsize=10.5, color=col, fontweight="bold")
        for k, (yr, lab) in enumerate(items):
            ax.plot(yr, yv, "o", color=col, ms=8)
            off = 0.22 if k % 2 == 0 else -0.22
            ax.text(yr, yv + off, lab, ha="center", va="bottom" if off > 0 else "top", fontsize=9.5)
    ax.set_xlim(2011.5, 2025.7)
    ax.set_ylim(-0.75, 2.6)
    ax.set_yticks([])
    ax.set_xticks(range(2012, 2026))
    ax.tick_params(axis="x", labelsize=10)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.set_title("Architectures in this tutorial, by arXiv year")
    save(fig, "timeline.png")


# ---------------------------------------------------------------- 7. residual block
def fig_residual():
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for ax, title, layers in [
        (axes[0], "Basic block (ResNet-18/34)",
         ["3x3 conv, C", "BatchNorm", "ReLU", "3x3 conv, C", "BatchNorm"]),
        (axes[1], "Bottleneck block (ResNet-50/101/152)",
         ["1x1 conv, C/4", "BN + ReLU", "3x3 conv, C/4", "BN + ReLU", "1x1 conv, C + BN"]),
    ]:
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(title)
        ax.text(3.6, 9.55, "x", ha="center", fontsize=13, fontweight="bold")
        arrow(ax, 3.6, 9.3, 3.6, 8.75)
        ys = [8.0, 6.9, 5.8, 4.7, 3.6]
        for yv, t in zip(ys, layers):
            col = GREEN if "conv" in t else GRAY
            box(ax, 2.6, yv, 2.0, 0.7, t, col, fontsize=10)
        for y0, y1 in zip(ys[:-1], ys[1:]):
            arrow(ax, 3.6, y0 - 0.02, 3.6, y1 + 0.72)
        ax.text(4.9, 5.9, "F(x)", fontsize=13, color=GREEN, fontweight="bold")
        ax.plot([3.6, 1.0, 1.0], [9.0, 9.0, 2.6], color=BLUE, lw=2)
        arrow(ax, 1.0, 2.6, 3.35, 2.6, BLUE, lw=2)
        ax.text(0.85, 5.8, "identity\nshortcut", ha="right", va="center", color=BLUE, fontsize=10.5,
                rotation=90)
        arrow(ax, 3.6, 3.58, 3.6, 2.85)
        ax.add_patch(plt.Circle((3.6, 2.6), 0.25, fc="white", ec="black", lw=1.5))
        ax.text(3.6, 2.6, "+", ha="center", va="center", fontsize=14)
        arrow(ax, 3.6, 2.35, 3.6, 1.85)
        box(ax, 2.6, 1.15, 2.0, 0.65, "ReLU", GRAY, fontsize=10)
        arrow(ax, 3.6, 1.12, 3.6, 0.6)
        ax.text(3.6, 0.35, "y = ReLU(x + F(x))", ha="center", fontsize=12)
    fig.tight_layout()
    save(fig, "residual-block.png")


# ---------------------------------------------------------------- 8. ViT patchify
def fig_patchify():
    img = transform.resize(img_as_float(data.astronaut()), (224, 224), anti_aliasing=True)
    g = 4
    p = 224 // g
    fig = plt.figure(figsize=(12, 4.8))
    a = fig.add_axes([0.0, 0.08, 0.3, 0.84])
    a.imshow(img)
    for k in range(1, g):
        a.axhline(k * p - 0.5, color="white", lw=2.5)
        a.axvline(k * p - 0.5, color="white", lw=2.5)
    a.set_xticks([]); a.set_yticks([])
    a.set_title(f"image split into {g}x{g} patches")
    b = fig.add_axes([0.32, 0.0, 0.68, 1.0])
    b.set_xlim(0, 20)
    b.set_ylim(0, 10)
    b.axis("off")
    # tokens: cls + first 8 patches + dots
    tx = 0.4
    b.add_patch(Rectangle((tx, 6.6), 0.9, 0.8, fc=ORANGE, ec="black"))
    b.text(tx + 0.45, 7.0, "cls", ha="center", va="center", fontsize=9.5, rotation=90)
    b.text(tx + 0.45, 6.2, "0", ha="center", fontsize=9.5)
    for k in range(8):
        r, c = divmod(k, g)
        x0 = tx + 1.15 * (k + 1)
        patch = img[r * p:(r + 1) * p, c * p:(c + 1) * p]
        b.imshow(patch, extent=(x0, x0 + 0.9, 6.6, 7.4))
        b.add_patch(Rectangle((x0, 6.6), 0.9, 0.8, fc="none", ec="black"))
        b.text(x0 + 0.45, 6.2, str(k + 1), ha="center", fontsize=9.5)
    b.text(tx + 1.15 * 9 + 0.5, 7.0, "...", fontsize=14, va="center")
    b.text(tx + 1.15 * 10 + 0.3, 6.2, f"{g * g}", ha="center", fontsize=9.5)
    b.add_patch(Rectangle((tx + 1.15 * 10 - 0.15, 6.6), 0.9, 0.8, fc="#DDDDDD", ec="black"))
    b.text(6.6, 8.6, "flatten each patch -> linear embedding E;  + position embedding (index below)",
           ha="center", fontsize=10.5)
    b.text(6.6, 5.6, "sequence of N + 1 tokens, each a D-dim vector", ha="center", fontsize=10.5)
    arrow(b, 6.6, 5.3, 6.6, 4.5)
    box(b, 2.2, 2.6, 8.8, 1.8, "Transformer encoder  x L\n[LN -> multi-head self-attention -> +]\n"
        "[LN -> MLP -> +]", PURPLE, fontsize=10.5)
    arrow(b, 11.1, 3.5, 13.0, 3.5)
    b.text(12.05, 3.75, "cls token", ha="center", fontsize=9.5)
    box(b, 13.1, 2.9, 2.6, 1.2, "MLP head", ORANGE, fontsize=11)
    arrow(b, 15.8, 3.5, 17.2, 3.5)
    b.text(17.4, 3.5, "class", va="center", fontsize=11)
    save(fig, "vit-patchify.png")


if __name__ == "__main__":
    fig_three_levels()
    fig_pipeline()
    fig_feature_maps()
    fig_learned_filters()
    fig_receptive_field()
    fig_timeline()
    fig_residual()
    fig_patchify()
    for f in sorted(OUT.glob("*.png")):
        im = Image.open(f)
        print(f"{f.name:28s} {im.size}  {f.stat().st_size / 1024:.0f} KB")
