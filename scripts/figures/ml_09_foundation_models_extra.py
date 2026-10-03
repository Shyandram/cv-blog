"""Extra teaching figures for ML 09 (foundation models); synthetic data, CPU, no downloads.

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/ml_09_foundation_models_extra.py
Writes only into src/assets/ml/ml-09-foundation-models/. Every curve is synthetic or
computed from a toy experiment; nothing here is a benchmark result of a real model.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

OUT = Path(__file__).resolve().parents[2] / "src/assets/ml/ml-09-foundation-models"
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, PURPLE, GRAY = "#2a6fdb", "#e4572e", "#17a398", "#7b4fbf", "#555555"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name)


def box(ax, x, y, w, h, text, color, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="black", lw=1, alpha=.18))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="none", ec=color, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=GRAY, lw=1.4))


def scaling_law():
    rng = np.random.default_rng(0)
    E, A, alpha = 1.7, 40.0, 0.3
    c = np.logspace(0, 6, 13)
    obs = (E + A * c ** -alpha) * np.exp(.02 * rng.normal(size=c.size))
    grid = np.logspace(0, 6, 200)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.loglog(grid, E + A * grid ** -alpha, color=BLUE, label=r"$E + A\,C^{-\alpha}$")
    ax.loglog(grid, A * grid ** -alpha, color=BLUE, ls=":", label=r"pure power law $A\,C^{-\alpha}$")
    ax.axhline(E, color=GRAY, ls="--", lw=1, label="irreducible floor E")
    ax.scatter(c, obs, color=ORANGE, zorder=3, label="synthetic runs")
    ax.set(xlabel="Training compute C (arbitrary units)", ylabel="Held-out loss",
           title="(a) Power law with a floor (synthetic)")
    ax.legend(fontsize=9)
    # (b) Parametric L(N, D) = E + A/N^a + B/D^b with C = 6 N D (all constants synthetic).
    Ep, Ap, Bp, a, b = 1.7, 400.0, 400.0, .34, .28
    ax = axes[1]
    n = np.logspace(5, 10, 300)
    for budget, col in zip([1e17, 1e19, 1e21], [GREEN, BLUE, PURPLE]):
        d = budget / (6 * n)
        loss = Ep + Ap / n ** a + Bp / d ** b
        i = loss.argmin()
        ax.semilogx(n, loss, color=col, label=f"C = {budget:.0e}")
        ax.scatter(n[i], loss[i], color=col, edgecolor="black", zorder=3)
    ax.set(ylim=(1.6, 4.5), xlabel="Model parameters N (tokens D = C / 6N)",
           ylabel="Predicted loss", title="(b) Fixed compute: an optimal model size")
    ax.legend(fontsize=9, title="iso-compute curve", title_fontsize=9)
    fig.tight_layout()
    save(fig, "scaling-law.png")


def objectives():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.6); ax.axis("off")
    box(ax, 4.0, 4.6, 3.0, .8, "Pre-training objective:\nwhere does the target come from?", GRAY, 11)
    items = [
        ("Supervised\nclassification", "human labels\nfor fixed classes", BLUE),
        ("Contrastive\nimage-text", "paired caption /\nalt text", ORANGE),
        ("Masked image\nmodeling", "the hidden\npatches themselves", GREEN),
        ("Self-distillation\n(DINO family)", "a teacher network's\noutput on another view", PURPLE),
        ("Generative /\nnext-token", "the next token\nor the clean signal", "#b8860b"),
    ]
    for k, (name, target, col) in enumerate(items):
        x = .2 + k * 2.18
        box(ax, x, 2.55, 1.95, 1.05, name, col, 10)
        box(ax, x, .55, 1.95, 1.2, "target:\n" + target, col, 9)
        arrow(ax, (5.5, 4.55), (x + .97, 3.65))
        arrow(ax, (x + .97, 2.5), (x + .97, 1.8))
    ax.text(5.5, .1, "Examples: ViT on labels | CLIP, SigLIP | MAE, BEiT | DINO, DINOv2, DINOv3 | LLMs, diffusion",
            ha="center", fontsize=9, color=GRAY)
    save(fig, "pretraining-objectives.png")


def linear_probe():
    rng = np.random.default_rng(0)
    D, k_sem, k_nui = 300, 8, 8
    sem_map = rng.normal(size=(k_sem, D))
    nui_map = rng.normal(size=(k_nui, D))

    def sample(centers, n):
        y = rng.integers(len(centers), size=n)
        z = centers[y] + .7 * rng.normal(size=(n, k_sem))
        u = 4.0 * rng.normal(size=(n, k_nui))
        return z @ sem_map + u @ nui_map + rng.normal(size=(n, D)), y

    up_centers = rng.normal(size=(50, k_sem))
    encoder = LinearDiscriminantAnalysis(n_components=k_sem).fit(*sample(up_centers, 10_000))
    new_centers = rng.normal(size=(4, k_sem))
    x_te, y_te = sample(new_centers, 2_000)
    f_te = encoder.transform(x_te)
    shots = [1, 2, 5, 10, 20, 50, 100]
    stats = {"raw": [], "feat": []}
    for s in shots:
        raw, feat = [], []
        for _ in range(10):
            x_tr, y_tr = sample(new_centers, 4 * s)
            raw.append(LogisticRegression(max_iter=3000).fit(x_tr, y_tr).score(x_te, y_te))
            feat.append(LogisticRegression(max_iter=3000)
                        .fit(encoder.transform(x_tr), y_tr).score(f_te, y_te))
        stats["raw"].append((np.mean(raw), np.std(raw)))
        stats["feat"].append((np.mean(feat), np.std(feat)))
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, col, lab in [("raw", ORANGE, "linear probe on raw inputs (300-D)"),
                          ("feat", BLUE, "linear probe on frozen pretrained features (8-D)")]:
        m, sd = np.array(stats[key]).T
        ax.semilogx(shots, m, "o-", color=col, label=lab)
        ax.fill_between(shots, m - sd, m + sd, color=col, alpha=.15)
    ax.axhline(.25, color=GRAY, ls=":", lw=1, label="chance (4 classes)")
    ax.set(xlabel="Labeled examples per new class", ylabel="Test accuracy", ylim=(0, 1.02),
           title="Toy transfer: reusable features matter most with few labels")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "linear-probe.png")


def mllm_pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.6); ax.axis("off")
    box(ax, .1, 1.3, 1.5, 1.0, "Image /\nvideo frames", GRAY)
    box(ax, 2.1, 1.3, 1.8, 1.0, "Vision encoder\n(e.g. CLIP ViT)\noften frozen", BLUE)
    box(ax, 4.4, 1.3, 1.9, 1.0, "Connector\nlinear / MLP /\nQ-Former / resampler", ORANGE, 9)
    box(ax, 4.4, .05, 1.9, .8, "Text prompt tokens", GRAY)
    box(ax, 6.8, .9, 2.0, 1.8, "Language model\n(decoder-only)\nreads visual +\ntext tokens", PURPLE)
    box(ax, 9.3, 1.3, 1.6, 1.0, "Answer tokens\n(loss here)", GREEN)
    arrow(ax, (1.65, 1.8), (2.05, 1.8))
    arrow(ax, (3.95, 1.8), (4.35, 1.8))
    arrow(ax, (6.35, 1.8), (6.75, 1.8))
    arrow(ax, (6.35, .45), (6.75, 1.1))
    arrow(ax, (8.85, 1.8), (9.25, 1.8))
    ax.text(5.5, 3.2, "Visual tokens are projected into the language model's embedding space",
            ha="center", fontsize=10, color=GRAY)
    save(fig, "mllm-pipeline.png")


def similarity():
    image = np.array([[1, .1, 0], [.1, 1, 0], [.5, .5, .2]])
    image /= np.linalg.norm(image, axis=1, keepdims=True)
    scores = image @ np.eye(3)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    im = ax.imshow(scores, cmap="Blues", vmin=0, vmax=1)
    for (i, j), s in np.ndenumerate(scores):
        ax.text(j, i, f"{s:.2f}", ha="center", va="center",
                color="white" if s > .6 else "black", fontsize=11)
    ax.set_xticks(range(3), ["Text A", "Text B", "Text C"])
    ax.set_yticks(range(3), ["Image A", "Image B", "Ambiguous image"])
    ax.set_title("Synthetic unit embeddings: cosine similarity")
    fig.colorbar(im, ax=ax)
    save(fig, "similarity-matrix.png")


if __name__ == "__main__":
    scaling_law()
    objectives()
    linear_probe()
    mllm_pipeline()
    similarity()
