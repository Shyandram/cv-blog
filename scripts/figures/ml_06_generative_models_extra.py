"""Extra teaching figures for ML 06 (generative models).

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/ml_06_generative_models_extra.py
Synthetic data or scikit-learn's bundled digits only; CPU; no downloads.
Writes only to src/assets/ml/ml-06-generative-models/.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parents[2] / "src/assets/ml/ml-06-generative-models"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, GREY = "#2a6fdb", "#e4572e", "#17a398", "#6b7280"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name)


# ---------------------------------------------------------------- taxonomy
def taxonomy():
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.4); ax.axis("off")

    def box(x, y, w, h, text, color, fs=10.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    fc=color, ec="#333333", lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=GREY))

    box(3.6, 3.55, 3.3, 0.65, "Deep generative models", "#e5e7eb", 12)
    box(0.2, 2.35, 7.3, 0.7, "Likelihood-based: write down (or bound) log p(x)", "#dbeafe", 11)
    box(7.9, 2.35, 2.4, 0.7, "Implicit: sampler only", "#fde2d8", 11)
    arrow(4.6, 3.55, 3.85, 3.05); arrow(5.9, 3.55, 9.1, 3.05)
    leaves = [(0.2, "Exact likelihood", "Autoregressive,\nnormalizing flows", "#eff6ff"),
              (2.7, "Lower bound (ELBO)", "VAE; diffusion\n(variational view)", "#eff6ff"),
              (5.2, "Unnormalized density", "Energy-based models\n(score, MCMC)", "#eff6ff"),
              (7.9, "Adversarial", "GAN, WGAN,\nStyleGAN", "#fff1eb")]
    for x, head, body, color in leaves:
        w = 2.3 if x < 7 else 2.4
        box(x, 1.3, w, 0.6, head, color, 10.5)
        box(x, 0.2, w, 0.8, body, "white", 10)
        arrow(x + w / 2, 2.35, x + w / 2, 1.9)
    save(fig, "taxonomy.png")


# ---------------------------------------------------------------- VAE on digits
def vae_digits():
    torch.manual_seed(0); torch.set_num_threads(1)
    d = load_digits()
    train, test, ytr, yte = train_test_split(d.data, d.target, test_size=.2,
                                             random_state=0, stratify=d.target)
    X = torch.tensor((train > 8).astype("float32"))
    Xt = torch.tensor((test > 8).astype("float32"))
    enc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 4))
    dec = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 64))
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=2e-3)
    for step in range(4000):
        x = X[torch.randint(len(X), (128,))]
        mu, logvar = enc(x).chunk(2, -1)
        z = mu + (.5 * logvar).exp() * torch.randn_like(mu)
        rec = F.binary_cross_entropy_with_logits(dec(z), x, reduction="none").sum(1)
        kl = .5 * (mu.square() + logvar.exp() - logvar - 1).sum(1)
        loss = (rec + kl).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mu, _ = enc(Xt).chunk(2, -1)
        grid = torch.linspace(-2, 2, 7)
        zz = torch.stack(torch.meshgrid(grid, -grid, indexing="xy"), -1).reshape(-1, 2)
        imgs = dec(zz).sigmoid().reshape(7, 7, 8, 8).numpy()
    mosaic = np.ones((7 * 10, 7 * 10))          # 2-pixel white gutter
    for r in range(7):
        for c in range(7):
            mosaic[r * 10 + 1:r * 10 + 9, c * 10 + 1:c * 10 + 9] = 1 - imgs[r, c]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    sc = axes[0].scatter(mu[:, 0], mu[:, 1], c=yte, cmap="tab10", s=10)
    axes[0].set_title("Posterior means of test digits (2-D latent)")
    axes[0].set_xlabel("z1"); axes[0].set_ylabel("z2")
    axes[0].set_xlim(-3.5, 3.5); axes[0].set_ylim(-3.5, 3.5)
    axes[0].grid(alpha=.25)
    cb = fig.colorbar(sc, ax=axes[0], ticks=range(10)); cb.set_label("digit label (not used in training)")
    axes[1].imshow(mosaic, cmap="gray", vmin=0, vmax=1,
                   extent=(-2.33, 2.33, -2.33, 2.33))
    axes[1].set_title("Decoder output on a grid of z")
    axes[1].set_xlabel("z1"); axes[1].set_ylabel("z2")
    save(fig, "vae-digits.png")


# ---------------------------------------------------------------- GAN ring
def gan_ring():
    torch.manual_seed(0); torch.set_num_threads(1)
    angles = torch.arange(8) * (2 * math.pi / 8)
    centers = 2.0 * torch.stack([angles.cos(), angles.sin()], 1)

    def sample_data(n):
        return centers[torch.randint(8, (n,))] + .05 * torch.randn(n, 2)

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, 128), nn.ReLU(),
                             nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, o))

    G, D = mlp(8, 2), mlp(2, 1)
    og = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(.5, .999))
    od = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(.5, .999))
    snaps, z_fixed = {}, torch.randn(2000, 8, generator=torch.Generator().manual_seed(1))
    for step in range(3000):
        real = sample_data(256)
        fake = G(torch.randn(256, 8))
        dl = (F.binary_cross_entropy_with_logits(D(real), torch.ones(256, 1))
              + F.binary_cross_entropy_with_logits(D(fake.detach()), torch.zeros(256, 1)))
        od.zero_grad(); dl.backward(); od.step()
        gl = F.binary_cross_entropy_with_logits(D(fake), torch.ones(256, 1))
        og.zero_grad(); gl.backward(); og.step()
        if step + 1 in (200, 1000, 3000):
            with torch.no_grad():
                snaps[step + 1] = G(z_fixed).numpy()
    real = sample_data(2000).numpy()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    axes[0].scatter(real[:, 0], real[:, 1], s=3, c=BLUE, alpha=.5)
    axes[0].set_title("Data: 8 Gaussians")
    for ax, (k, v) in zip(axes[1:], snaps.items()):
        ax.scatter(v[:, 0], v[:, 1], s=3, c=ORANGE, alpha=.5)
        ax.scatter(centers[:, 0], centers[:, 1], marker="+", s=80, c="k")
        ax.set_title(f"Generator, step {k}")
    for ax in axes:
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
        ax.grid(alpha=.25)
    save(fig, "gan-ring.png")


# ---------------------------------------------------------------- precision / recall
def precision_recall():
    rng = np.random.default_rng(0)
    angles = np.arange(8) * 2 * np.pi / 8
    centers = 2 * np.stack([np.cos(angles), np.sin(angles)], 1)

    def ring(n, modes):
        return centers[rng.choice(modes, n)] + .05 * rng.standard_normal((n, 2))

    def coverage(query, ref, k=3):
        dr = np.linalg.norm(ref[:, None] - ref[None], axis=-1)
        r = np.sort(dr, 1)[:, k]
        d = np.linalg.norm(query[:, None] - ref[None], axis=-1)
        return np.mean((d <= r[None]).any(1))

    real = ring(1000, range(8))
    fakes = [("Good: all 8 modes", ring(1000, range(8))),
             ("Mode collapse: 2 modes", ring(1000, [0, 4])),
             ("Blurry: 8 modes + noise", ring(1000, range(8)) + .4 * rng.standard_normal((1000, 2)))]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, (name, fake) in zip(axes, fakes):
        p, r = coverage(fake, real), coverage(real, fake)
        ax.scatter(real[:, 0], real[:, 1], s=3, c=BLUE, alpha=.4, label="real")
        ax.scatter(fake[:, 0], fake[:, 1], s=3, c=ORANGE, alpha=.4, label="generated")
        ax.set_title(f"{name}\nprecision {p:.2f}, recall {r:.2f}", fontsize=11)
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal")
        ax.grid(alpha=.25)
    axes[0].legend(loc="center", markerscale=4, fontsize=9)
    save(fig, "precision-recall.png")


if __name__ == "__main__":
    taxonomy()
    vae_digits()
    precision_recall()
    gan_ring()
