"""Figures for DLCV 10 - Image Generation.

Run:  .venv/bin/python scripts/figures/dl_10_image_generation.py
Writes PNGs to src/assets/deep-learning/dl-10-image-generation/ only.
Data: synthetic arrays only (random features, drawn disks and squares);
nothing is downloaded and no network weights are used. Runs in a few seconds on a CPU.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402
from skimage.draw import disk, rectangle  # noqa: E402
from skimage.measure import label, regionprops  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-10-image-generation"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.facecolor": "white"})

BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"


def save(fig, name, quantize=True):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if quantize:
        im = Image.open(path).convert("RGB")
        im = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        im.save(path, optimize=True)
    w, h = Image.open(path).size
    assert w <= 1600, f"{name} is {w} px wide"
    print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB, {w}x{h} px)")


def box(ax, x, y, w, h, text, fc, ec="#333333", fs=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight,
            wrap=True)


def arrow(ax, x0, y0, x1, y1, color="#333333", style="-|>", ls="-", lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, linestyle=ls))


# ------------------------------------------------------------------ Figure 1: family map
def family_map():
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.text(0.1, 5.3, "Generator: how new content is produced", fontsize=11, weight="bold")
    ax.text(0.1, 2.05, "Representation: where the generator works", fontsize=11, weight="bold")
    gens = [
        (0.1, "GAN\none generator pass\n(StyleGAN, BigGAN)", "#fde0dd"),
        (2.15, "Diffusion\nmany denoising steps\n(DDPM, LDM, DiT)", "#deebf7"),
        (4.2, "Flow matching /\nrectified flow\nfollow a velocity field\n(SD3)", "#e5f5e0"),
        (6.25, "Autoregressive\none token, scale or\ntoken group per step\n(Parti, VAR, MAR)", "#fff2cc"),
        (8.3, "Unified multimodal\none transformer for\ntext and images\n(Chameleon, Janus)", "#efedf5"),
    ]
    for x, t, c in gens:
        box(ax, x, 3.2, 1.95, 1.85, t, c)
    reps = [
        (0.6, "Pixels\n(H x W x 3 values)", "#f0f0f0"),
        (3.95, "Continuous latent\n(VAE / autoencoder grid)", "#f0f0f0"),
        (7.3, "Discrete tokens\n(VQ codebook indices)", "#f0f0f0"),
    ]
    for x, t, c in reps:
        box(ax, x, 0.2, 2.6, 1.5, t, c)
    # which generator usually works in which representation
    links = [(1.07, 1.9, BLUE), (1.07, 5.25, BLUE),           # GAN -> pixels / latent (GAN priors)
             (3.12, 1.9, BLUE), (3.12, 5.25, BLUE),           # diffusion -> pixels / latent
             (5.17, 5.25, GREEN), (5.17, 1.9, GREEN),         # flow -> latent / pixels
             (7.22, 8.6, ORANGE), (7.22, 5.25, ORANGE),       # AR -> tokens / continuous (MAR)
             (9.27, 8.6, PURPLE), (9.27, 5.25, PURPLE)]       # unified -> tokens / continuous
    for x0, x1, c in links:
        arrow(ax, x0, 3.18, x1, 1.72, color=c, lw=1.2)
    save(fig, "family-map.png")


# ------------------------------------------------------------------ Figure 2: T2I pipeline
def pipeline():
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    box(ax, 0.1, 3.5, 1.8, 0.9, '"a red ball\non the grass"', "#ffffff")
    box(ax, 2.3, 3.5, 2.0, 0.9, "Text encoder\n(CLIP and/or T5,\nfrozen)", "#fff2cc")
    box(ax, 0.1, 1.6, 1.8, 0.9, "Gaussian noise\n(latent grid)", "#ffffff")
    box(ax, 2.3, 1.2, 3.4, 1.7, "Denoiser / velocity network\n(U-Net or DiT)\nimage tokens attend to\ntext tokens (cross-attention)",
        "#deebf7")
    box(ax, 0.1, 0.1, 1.8, 0.9, "Optional control\n(edges, depth, pose)", "#e5f5e0")
    box(ax, 6.2, 1.6, 1.8, 0.9, "Clean latent", "#ffffff")
    box(ax, 8.4, 1.6, 1.9, 0.9, "VAE decoder\n-> image", "#fde0dd")
    arrow(ax, 1.9, 3.95, 2.3, 3.95)
    arrow(ax, 3.3, 3.5, 3.3, 2.9)
    ax.text(3.4, 3.15, "token embeddings", fontsize=10)
    arrow(ax, 1.9, 2.05, 2.3, 2.05)
    arrow(ax, 1.9, 0.55, 2.45, 1.2, color=GREEN)
    ax.text(2.3, 0.75, "ControlNet", fontsize=10, color=GREEN)
    arrow(ax, 5.7, 2.05, 6.2, 2.05)
    arrow(ax, 8.0, 2.05, 8.4, 2.05)
    # the sampling loop
    ax.add_patch(FancyArrowPatch((5.5, 1.2), (3.6, 1.2), connectionstyle="arc3,rad=-0.6",
                                 arrowstyle="-|>", mutation_scale=14, color=RED, lw=1.4))
    ax.text(5.9, 0.35, "repeat N steps; with guidance, each step\nruns the network with and without the prompt",
            ha="left", fontsize=10, color=RED)
    ax.text(6.25, 3.6, "Training: encode real images to latents,\nadd noise, learn to undo it given the caption.\n"
            "Sampling: start from noise, apply\nthe network N times, decode once.", fontsize=10, va="bottom")
    save(fig, "t2i-pipeline.png")


# ------------------------------------------------------------------ Figure 3: cross-attention toy
def cross_attention():
    rng = np.random.default_rng(0)
    d = 32
    words = ["a", "red", "ball", "on", "grass"]
    E = rng.normal(size=(len(words), d))
    yy, xx = np.mgrid[:16, :16]
    ball = ((yy - 6) ** 2 + (xx - 9) ** 2 <= 9).ravel()
    X = 1.0 * rng.normal(size=(256, d))
    X[ball] += 0.5 * (E[1] + E[2])
    X[~ball] += 0.5 * E[4]
    scores = X @ E.T / np.sqrt(d)
    A = np.exp(scores - scores.max(axis=1, keepdims=True))
    A /= A.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 6, figsize=(10.4, 2.3))
    axes[0].imshow(ball.reshape(16, 16), cmap="Greens_r", vmin=-0.5, vmax=1.3)
    axes[0].set_title("layout (ball region)")
    for j, w in enumerate(words):
        im = axes[j + 1].imshow(A[:, j].reshape(16, 16), cmap="magma", vmin=0, vmax=1)
        axes[j + 1].set_title(f'"{w}"')
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="attention weight")
    save(fig, "cross-attention.png")


# ------------------------------------------------------------------ Figure 4: compositional check toy
COLORS = {"red": (0.85, 0.1, 0.1), "blue": (0.1, 0.2, 0.85)}


def render(objects, size=96):
    img = np.full((size, size, 3), 0.8)
    for k, (col, shape) in enumerate(objects):
        cy, cx = 24 + 48 * (k // 2), 24 + 48 * (k % 2)
        if shape == "circle":
            rr, cc = disk((cy, cx), 14)
        else:
            rr, cc = rectangle((cy - 12, cx - 12), extent=(24, 24))
        img[rr, cc] = COLORS[col]
    return img


def detect(img):
    found = []
    for region in regionprops(label(np.abs(img - 0.8).sum(-1) > 0.3)):
        r, g, b = img[tuple(region.coords.T)].mean(0)
        col = "red" if r > b else "blue"
        shape = "square" if region.extent > 0.9 else "circle"
        found.append((col, shape))
    return found


def presence_score(prompt, img):
    words = {w for obj in prompt for w in obj}
    seen = {w for obj in detect(img) for w in obj}
    return len(words & seen) / len(words)


def exact_check(prompt, img):
    return sorted(detect(img)) == sorted(prompt)


def compositional():
    prompt = [("red", "circle"), ("red", "circle"), ("blue", "square")]
    candidates = {
        "correct": prompt,
        "wrong count": prompt + [("red", "circle")],
        "colors swapped": [("blue", "circle"), ("blue", "circle"), ("red", "square")],
        "object missing": prompt[:2],
    }
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 3.2))
    for a, (name, objs) in zip(axes, candidates.items()):
        img = render(objs)
        a.imshow(img, interpolation="nearest")
        ok = exact_check(prompt, img)
        a.set_title(f"{name}\npresence {presence_score(prompt, img):.2f} | exact: {'pass' if ok else 'fail'}",
                    fontsize=10, color=GREEN if ok else RED)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle('prompt: "two red circles and a blue square"', fontsize=11, y=1.02)
    save(fig, "compositional-check.png")


if __name__ == "__main__":
    family_map()
    pipeline()
    cross_attention()
    compositional()
