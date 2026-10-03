"""Extra figures for PR 05 (feature selection and dimensionality reduction).

Writes only into src/assets/pattern-recognition/pr-05-features-dimensionality/.
Run: .venv/bin/python scripts/figures/pr_05_features_dimensionality_extra.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, make_circles, make_swiss_roll
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[2] / "src/assets/pattern-recognition/pr-05-features-dimensionality"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white"})
C0, C1 = "#2b6cb0", "#c53030"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def variance_vs_class():
    rng = np.random.default_rng(4)
    n = 200
    y = np.repeat([0, 1], n)
    nuisance = rng.normal(0, 3.0, 2 * n)
    signal = np.where(y == 0, -0.6, 0.6) + rng.normal(0, 0.2, 2 * n)
    X = np.c_[nuisance, signal]
    pc1 = PCA(n_components=1).fit(X)
    z = pc1.transform(X)[:, 0]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    ax = axes[0]
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=8, c=C0, alpha=0.6, label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=8, c=C1, alpha=0.6, label="class 1")
    d = pc1.components_[0] * 8
    ax.annotate("", xy=d, xytext=-d, arrowprops=dict(arrowstyle="<->", color="k"))
    ax.text(d[0] * 0.55, d[1] + 0.25, "PC 1", fontsize=10)
    ax.set_xlabel("feature 1 (large nuisance variance)")
    ax.set_ylabel("feature 2 (small class signal)")
    ax.set_title("Synthetic measurements")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_ylim(-2.2, 1.6)
    bins = np.linspace(z.min(), z.max(), 30)
    axes[1].hist(z[y == 0], bins=bins, alpha=0.6, color=C0, density=True)
    axes[1].hist(z[y == 1], bins=bins, alpha=0.6, color=C1, density=True)
    axes[1].set_title("Projection on PC 1\n(%.0f%% of variance)" % (100 * pc1.explained_variance_ratio_[0]))
    axes[1].set_xlabel("PC 1 score")
    lda = LinearDiscriminantAnalysis(n_components=1).fit(X, y)
    w = lda.transform(X)[:, 0]
    bins = np.linspace(w.min(), w.max(), 30)
    axes[2].hist(w[y == 0], bins=bins, alpha=0.6, color=C0, density=True)
    axes[2].hist(w[y == 1], bins=bins, alpha=0.6, color=C1, density=True)
    axes[2].set_title("Projection on Fisher LDA direction")
    axes[2].set_xlabel("LDA score")
    fig.tight_layout()
    save(fig, "variance-vs-class-v2.png")


def pca_digits():
    X, _ = load_digits(return_X_y=True)
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    fig = plt.figure(figsize=(10.5, 3.6))
    ax = fig.add_axes([0.06, 0.15, 0.32, 0.72])
    ax.plot(np.arange(1, 65), cum, color="#2f855a")
    for r in [2, 10, 21]:
        ax.axvline(r, ls=":", color="gray")
        ax.text(r + 0.8, 0.12, f"r={r}", fontsize=9)
    ax.set_xlabel("number of components r")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("PCA on 8x8 digits")
    idx = [0, 7, 21]
    ranks = [2, 10, 21, 64]
    for i, k in enumerate(idx):
        for j, r in enumerate(ranks):
            p = PCA(n_components=r).fit(X)
            rec = p.inverse_transform(p.transform(X[k:k + 1]))[0]
            a = fig.add_axes([0.45 + j * 0.135, 0.62 - i * 0.29, 0.12, 0.25])
            a.imshow(rec.reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
            a.set_xticks([])
            a.set_yticks([])
            if i == 0:
                a.set_title("original" if r == 64 else f"r={r}", fontsize=10)
    save(fig, "pca-digits.png")


def pca_vs_lda():
    X, y = load_digits(return_X_y=True)
    Xs = StandardScaler().fit_transform(X)
    Zp = PCA(n_components=2).fit_transform(Xs)
    Zl = LinearDiscriminantAnalysis(n_components=2).fit_transform(Xs, y)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    cmap = plt.get_cmap("tab10")
    for ax, Z, title in [(axes[0], Zp, "PCA (unsupervised): top-2 variance"),
                         (axes[1], Zl, "Fisher LDA (supervised): top-2 discriminants")]:
        for c in range(10):
            ax.scatter(Z[y == c, 0], Z[y == c, 1], s=5, color=cmap(c), alpha=0.6)
            m = Z[y == c].mean(0)
            ax.text(m[0], m[1], str(c), fontsize=13, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=cmap(c), alpha=0.85))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save(fig, "pca-vs-lda-digits.png")


def kernel_pca():
    X, y = make_circles(n_samples=400, factor=0.4, noise=0.07, random_state=0)
    Z = KernelPCA(n_components=3, kernel="rbf", gamma=2).fit_transform(X)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    axes[0].scatter(X[y == 0, 0], X[y == 0, 1], s=6, c=C0)
    axes[0].scatter(X[y == 1, 0], X[y == 1, 1], s=6, c=C1)
    axes[0].set_aspect("equal")
    axes[0].set_title("Input data")
    axes[1].scatter(Z[y == 0, 0], Z[y == 0, 1], s=6, c=C0)
    axes[1].scatter(Z[y == 1, 0], Z[y == 1, 1], s=6, c=C1)
    axes[1].set_title("Kernel PCA: components 1 and 2")
    axes[1].set_xlabel("kPC 1")
    axes[1].set_ylabel("kPC 2")
    axes[2].scatter(Z[y == 0, 0], Z[y == 0, 2], s=6, c=C0)
    axes[2].scatter(Z[y == 1, 0], Z[y == 1, 2], s=6, c=C1)
    axes[2].set_title("Kernel PCA: components 1 and 3")
    axes[2].set_xlabel("kPC 1")
    axes[2].set_ylabel("kPC 3")
    fig.tight_layout()
    save(fig, "kernel-pca-circles.png")


def ica():
    t = np.linspace(0, 8, 2000)
    S = np.c_[np.sin(3 * t), np.sign(np.sin(5 * t)), ((2 * t) % 2) - 1]
    A = np.array([[1.0, 0.5, 0.3], [0.6, 1.0, 0.4], [0.4, 0.3, 1.0]])
    X = S @ A.T
    S_ica = FastICA(n_components=3, whiten="unit-variance", random_state=0).fit_transform(X)
    S_pca = PCA(n_components=3).fit_transform(X)
    fig, axes = plt.subplots(3, 4, figsize=(10.5, 4.6), sharex=True)
    cols = [S, X, S_pca, S_ica]
    titles = ["true sources", "observed mixtures", "PCA components", "ICA components"]
    colors = ["#2b6cb0", "#6b46c1", "#dd6b20", "#2f855a"]
    for j, (M, ti, co) in enumerate(zip(cols, titles, colors)):
        for i in range(3):
            ax = axes[i, j]
            ax.plot(t[:800], M[:800, i], color=co, lw=1)
            ax.set_yticks([])
            if i == 0:
                ax.set_title(ti)
    fig.tight_layout()
    save(fig, "ica-unmixing.png")


def manifold():
    X, t = make_swiss_roll(n_samples=1000, noise=0.05, random_state=0)
    methods = [("PCA", PCA(n_components=2)),
               ("Isomap (k=10)", Isomap(n_neighbors=10, n_components=2)),
               ("LLE (k=12)", LocallyLinearEmbedding(n_neighbors=12, n_components=2, random_state=0)),
               ("t-SNE (perplexity 30)", TSNE(n_components=2, perplexity=30, init="pca", random_state=0))]
    fig = plt.figure(figsize=(10.5, 2.9))
    ax = fig.add_subplot(1, 5, 1, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap="viridis", s=3)
    ax.set_title("Swiss roll (3-D)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(10, -70)
    for i, (name, m) in enumerate(methods):
        Y = m.fit_transform(X)
        a = fig.add_subplot(1, 5, i + 2)
        a.scatter(Y[:, 0], Y[:, 1], c=t, cmap="viridis", s=3)
        a.set_title(name)
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    save(fig, "manifold-swiss-roll.png")


if __name__ == "__main__":
    variance_vs_class()
    pca_digits()
    pca_vs_lda()
    kernel_pca()
    ica()
    manifold()
    for f in sorted(OUT.glob("*.png")):
        print(f.name, f.stat().st_size // 1024, "KB")
