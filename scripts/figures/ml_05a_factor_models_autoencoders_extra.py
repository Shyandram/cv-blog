"""Figures for ML 05A (linear factor models and autoencoders).

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/ml_05a_factor_models_autoencoders_extra.py
Uses only scikit-learn's bundled digits and synthetic arrays; CPU only, < 1 minute.
Writes only to src/assets/ml/ml-05a-factor-models-autoencoders/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.decomposition import (PCA, NMF, FactorAnalysis, FastICA,
                                   MiniBatchDictionaryLearning)
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parents[2] / "src/assets/ml/ml-05a-factor-models-autoencoders"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, PURPLE = "#2a6fdb", "#e4572e", "#17a398", "#7b4fb3"


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(path)


def digits():
    X = load_digits().data / 16.0
    return train_test_split(X, test_size=0.2, random_state=0)


def components_grid():
    train, _ = digits()
    k = 8
    models = [
        ("PCA", PCA(n_components=k, random_state=0)),
        ("Factor analysis", FactorAnalysis(n_components=k, random_state=0)),
        ("NMF", NMF(n_components=k, init="nndsvda", max_iter=1000, random_state=0)),
        ("ICA", FastICA(n_components=k, whiten="unit-variance", max_iter=2000,
                        random_state=0)),
        ("Sparse coding", MiniBatchDictionaryLearning(n_components=k, alpha=0.5,
                                                      batch_size=64, max_iter=50,
                                                      random_state=0)),
    ]
    fig, axes = plt.subplots(len(models), k, figsize=(10, 6.6))
    for r, (name, model) in enumerate(models):
        model.fit(train)
        comps = model.components_
        for c in range(k):
            ax = axes[r, c]
            img = comps[c].reshape(8, 8)
            lim = np.abs(img).max() + 1e-12
            if name == "NMF":
                ax.imshow(img, cmap="Greys", vmin=0, vmax=lim)
            else:
                ax.imshow(img, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center")
    fig.suptitle("Eight learned components per model (8 x 8 digits; red = +, blue = -)",
                 y=0.995)
    save(fig, "components.png")


def ica_demo():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 8, 2000)
    s1 = np.sin(2 * t)
    s2 = np.sign(np.sin(3 * t))
    s3 = 2 * (t % 1) - 1              # sawtooth
    S = np.c_[s1, s2, s3]
    S += 0.05 * rng.normal(size=S.shape)
    A = np.array([[1.0, 1.0, 1.0], [0.5, 2.0, 1.0], [1.5, 1.0, 2.0]])
    X = S @ A.T
    S_ica = FastICA(n_components=3, whiten="unit-variance", random_state=0).fit_transform(X)
    S_pca = PCA(n_components=3).fit_transform(X)
    cols = [BLUE, ORANGE, GREEN]
    titles = ["True sources", "Observed mixtures", "PCA components", "ICA estimates"]
    fig, axes = plt.subplots(4, 1, figsize=(9, 7.5), sharex=True)
    for ax, M, title in zip(axes, [S, X, S_pca, S_ica], titles):
        M = (M - M.mean(0)) / M.std(0)
        for j in range(3):
            ax.plot(t, M[:, j] + 3.2 * j, color=cols[j], lw=1.2)
        ax.set_title(title, loc="left")
        ax.set_yticks([])
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    save(fig, "ica-separation.png")


def pca_variance():
    train, test = digits()
    pca = PCA().fit(train)
    fig = plt.figure(figsize=(11, 3.8))
    ax = fig.add_subplot(1, 2, 1)
    cum = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(np.arange(1, 65), cum, color=BLUE, lw=2)
    for k in (2, 10, 20):
        ax.axvline(k, color="grey", ls=":", lw=1)
        ax.text(k + 0.8, 0.12, f"k={k}\n{cum[k-1]:.2f}", fontsize=10)
    ax.set_xlabel("number of components k")
    ax.set_ylabel("cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.set_title("PCA on digits (training split)")
    ks = [2, 5, 10, 20, 64]
    x = test[:1]
    gs = fig.add_gridspec(1, 2)[0, 1].subgridspec(1, len(ks) + 1, wspace=0.1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(x.reshape(8, 8), cmap="Greys"); ax0.set_title("input")
    ax0.set_xticks([]); ax0.set_yticks([])
    for i, k in enumerate(ks):
        p = PCA(n_components=k).fit(train)
        r = p.inverse_transform(p.transform(x))
        a = fig.add_subplot(gs[0, i + 1])
        a.imshow(r.reshape(8, 8), cmap="Greys", vmin=0, vmax=1)
        a.set_title(f"k={k}"); a.set_xticks([]); a.set_yticks([])
    save(fig, "pca-variance.png")


def dae_demo():
    torch.manual_seed(0); torch.set_num_threads(1)
    train, test = digits()
    X = torch.tensor(train, dtype=torch.float32)
    Xt = torch.tensor(test, dtype=torch.float32)
    enc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 12))
    dec = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 64), nn.Sigmoid())
    model = nn.Sequential(enc, dec)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(2000):
        clean = X[torch.randint(len(X), (64,))]
        noisy = (clean + 0.2 * torch.randn_like(clean)).clamp(0, 1)
        loss = (model(noisy) - clean).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        g = torch.Generator().manual_seed(1)
        x = Xt[:8]
        noisy = (x + 0.2 * torch.randn(x.shape, generator=g)).clamp(0, 1)
        out = model(noisy)
    fig, axes = plt.subplots(3, 8, figsize=(10, 4.1))
    for r, (M, name) in enumerate([(x, "clean"), (noisy, "noisy input"), (out, "DAE output")]):
        for c in range(8):
            ax = axes[r, c]
            ax.imshow(M[c].reshape(8, 8).numpy(), cmap="Greys", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center")
    fig.suptitle("Denoising autoencoder on held-out digits (Gaussian noise, sigma = 0.2)")
    save(fig, "dae-reconstructions.png")


def swiss_roll():
    X, t = make_swiss_roll(n_samples=1500, noise=0.05, random_state=0)
    Z = PCA(n_components=2).fit_transform(X)
    fig = plt.figure(figsize=(10.5, 3.6))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap="viridis", s=4)
    ax.set_title("3-D data (colour = position)")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(10, -70)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(Z[:, 0], Z[:, 1], c=t, cmap="viridis", s=4)
    ax2.set_title("Linear 2-D projection (PCA)")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(t, X[:, 1], c=t, cmap="viridis", s=4)
    ax3.set_title("True 2-D coordinates (unrolled)")
    ax3.set_xlabel("position along roll"); ax3.set_ylabel("height")
    ax3.set_xticks([]); ax3.set_yticks([])
    fig.tight_layout()
    save(fig, "swiss-roll.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            globals()[name]()
        raise SystemExit
    components_grid()
    ica_demo()
    pca_variance()
    dae_demo()
    swiss_roll()
