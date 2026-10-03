"""Extra original figures for PR 07 (clustering, mixtures and EM).

Synthetic data only; re-runnable; writes into
src/assets/pattern-recognition/pr-07-clustering-mixtures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

OUT = (Path(__file__).resolve().parents[2]
       / "src/assets/pattern-recognition/pr-07-clustering-mixtures")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})
PALETTE = ["#2563eb", "#db2777", "#16a34a", "#ea580c", "#7c3aed", "#0891b2",
           "#ca8a04", "#64748b", "#be123c", "#4d7c0f", "#1d4ed8", "#c026d3",
           "#0f766e", "#b45309", "#334155"]


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def kmeans_pp_init(X, k, rng):
    centers = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(centers)


def lloyd(X, centers, max_iter=100):
    hist = []
    for _ in range(max_iter):
        d2 = ((X[:, None] - centers[None]) ** 2).sum(-1)
        z = d2.argmin(1)
        hist.append(d2[np.arange(len(X)), z].sum())
        new = np.array([X[z == k].mean(0) if np.any(z == k) else centers[k]
                        for k in range(len(centers))])
        if np.allclose(new, centers):
            break
        centers = new
    return centers, z, hist


def fig_kmeans_seeding():
    X, _ = make_blobs(n_samples=1500, centers=15, cluster_std=0.6, random_state=2)
    runs = {"random": [], "k-means++": []}
    for s in range(20):
        r = np.random.default_rng(s)
        runs["random"].append(lloyd(X, X[r.choice(len(X), 15, replace=False)]))
        runs["k-means++"].append(lloyd(X, kmeans_pp_init(X, 15, np.random.default_rng(s))))
    worst_random = max(runs["random"], key=lambda t: t[2][-1])
    worst_pp = max(runs["k-means++"], key=lambda t: t[2][-1])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, (c, z, h), title in [(axes[0], worst_random, "Worst of 20 random-seed runs"),
                                 (axes[1], worst_pp, "Worst of 20 k-means++ runs")]:
        ax.scatter(X[:, 0], X[:, 1], c=[PALETTE[i % 15] for i in z], s=6, alpha=0.6)
        ax.scatter(c[:, 0], c[:, 1], marker="X", s=90, c="black", edgecolor="white")
        ax.set(title=f"{title}\nJ = {h[-1]:.0f}", xticks=[], yticks=[])
    finals = [[t[2][-1] for t in runs[k]] for k in ["random", "k-means++"]]
    axes[2].boxplot(finals, tick_labels=["random", "k-means++"], widths=0.5)
    for i, f in enumerate(finals):
        axes[2].scatter(np.full(len(f), i + 1) + np.random.default_rng(i).uniform(-.12, .12, len(f)),
                        f, s=14, color=PALETTE[i], alpha=0.7, zorder=3)
    axes[2].set(title="Final objective J, 20 single runs", ylabel="J (lower is better)")
    fig.tight_layout()
    save(fig, "kmeans-seeding.png")


def fig_linkage():
    rng = np.random.default_rng(4)
    a = rng.normal([0, 0], 0.35, (70, 2))
    b = rng.normal([3.2, 0], 0.35, (70, 2))
    bridge = np.column_stack([np.linspace(0.5, 2.7, 14), rng.normal(0, 0.04, 14)])
    c = rng.normal([1.6, 2.4], [0.9, 0.18], (60, 2))
    X = np.vstack([a, b, bridge, c])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
    axes = axes.ravel()
    for ax, link in zip(axes[:3], ["single", "complete", "ward"]):
        z = AgglomerativeClustering(n_clusters=3, linkage=link).fit_predict(X)
        ax.scatter(X[:, 0], X[:, 1], c=[PALETTE[i] for i in z], s=12, alpha=0.75)
        ax.set(title=f"{link.capitalize()} linkage, 3 clusters", xticks=[], yticks=[])
    Z = linkage(X, method="ward")
    dendrogram(Z, ax=axes[3], truncate_mode="lastp", p=20, no_labels=True,
               color_threshold=Z[-3, 2] + 1e-9, above_threshold_color="#64748b")
    axes[3].axhline((Z[-3, 2] + Z[-2, 2]) / 2, color="black", ls="--", lw=1)
    axes[3].set(title="Ward dendrogram (cut gives 3)", ylabel="Merge height")
    fig.tight_layout()
    save(fig, "linkage-comparison.png")


def fig_dbscan():
    X, _ = make_moons(n_samples=320, noise=0.06, random_state=5)
    X = np.vstack([X, np.random.default_rng(1).uniform([-1.3, -0.8], [2.3, 1.3], (25, 2))])
    eps, mpts = 0.18, 5
    dist, _ = NearestNeighbors(n_neighbors=mpts).fit(X).kneighbors(X)
    kd = np.sort(dist[:, -1])[::-1]
    db = DBSCAN(eps=eps, min_samples=mpts).fit(X)
    core = np.zeros(len(X), bool)
    core[db.core_sample_indices_] = True
    noise = db.labels_ == -1
    border = ~core & ~noise
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].plot(np.arange(len(kd)), kd, color=PALETTE[0])
    axes[0].axhline(eps, color=PALETTE[1], ls="--", label=f"eps = {eps}")
    axes[0].set(title=f"Sorted distance to {mpts}th neighbour (self included)",
                xlabel="Points sorted by k-distance (descending)", ylabel="k-distance")
    axes[0].legend(frameon=False)
    cols = np.array([PALETTE[l % 15] if l >= 0 else "#94a3b8" for l in db.labels_])
    axes[1].scatter(X[core, 0], X[core, 1], c=cols[core], s=16, label="core")
    axes[1].scatter(X[border, 0], X[border, 1], facecolors="white", edgecolors=cols[border],
                    s=30, linewidths=1.4, label="border")
    axes[1].scatter(X[noise, 0], X[noise, 1], marker="x", c="#475569", s=26, label="noise")
    axes[1].set(title="DBSCAN: core, border and noise points", xticks=[], yticks=[])
    axes[1].legend(frameon=False, loc="lower left", fontsize=10)
    fig.tight_layout()
    save(fig, "dbscan-kdistance.png")


def ellipse(ax, mean, cov, color):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    for s in [1, 2]:
        ax.add_patch(Ellipse(mean, 2 * s * np.sqrt(vals[1]), 2 * s * np.sqrt(vals[0]),
                             angle=angle, fill=False, color=color, lw=1.6, alpha=0.9 / s))


def fig_em():
    rng = np.random.default_rng(3)
    means = np.array([[0.0, 0.0], [4.0, 1.0], [1.0, 4.0]])
    covs = [np.array([[1.0, 0.6], [0.6, 0.8]]), np.array([[0.5, -0.2], [-0.2, 1.2]]),
            np.array([[0.7, 0.0], [0.0, 0.3]])]
    X = np.vstack([rng.multivariate_normal(m, S, n) for m, S, n in zip(means, covs, [250, 150, 100])])
    N, D, K = len(X), 2, 3
    mu = X[rng.choice(N, K, replace=False)]
    Sigma = np.array([np.cov(X.T) for _ in range(K)])
    pi = np.full(K, 1 / K)
    snaps, ll = {}, []
    for it in range(200):
        lj = np.column_stack([np.log(pi[k]) + multivariate_normal(mu[k], Sigma[k]).logpdf(X)
                              for k in range(K)])
        ln = logsumexp(lj, 1)
        R = np.exp(lj - ln[:, None])
        ll.append(ln.mean())
        if it in (0, 2, 6, 12):
            snaps[it] = (mu.copy(), Sigma.copy(), R.copy())
        Nk = R.sum(0)
        pi, mu = Nk / N, (R.T @ X) / Nk[:, None]
        for k in range(K):
            d = X - mu[k]
            Sigma[k] = (R[:, k, None] * d).T @ d / Nk[k] + 1e-6 * np.eye(D)
        if it > 0 and ll[-1] - ll[-2] < 1e-10:
            break
    snaps["final"] = (mu, Sigma, R)
    rgb = np.array([[0.15, 0.39, 0.92], [0.86, 0.15, 0.47], [0.09, 0.64, 0.29]])
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    axes = axes.ravel()
    for ax, key in zip(axes[:5], [0, 2, 6, 12, "final"]):
        m, S, R = snaps[key]
        ax.scatter(X[:, 0], X[:, 1], c=np.clip(R @ rgb, 0, 1), s=7, alpha=0.7)
        for k in range(K):
            ellipse(ax, m[k], S[k], rgb[k])
        ax.set(title=f"EM iteration {key}" if key != "final" else f"Converged ({len(ll)} it.)",
               xticks=[], yticks=[], xlim=(-3.5, 6.5), ylim=(-3, 6.5))
    axes[5].plot(np.arange(len(ll)), ll, color=PALETTE[0], marker=".", ms=4)
    axes[5].set(title="Mean log-likelihood", xlabel="Iteration")
    fig.tight_layout()
    save(fig, "gmm-em-progress.png")


def fig_selection():
    X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.0, random_state=7)
    Xs = X @ np.array([[1.0, 0.0], [0.6, 0.5]])
    Ks = np.arange(1, 9)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4))
    for cov, col in zip(["spherical", "diag", "full"], PALETTE):
        bic = [GaussianMixture(k, covariance_type=cov, n_init=3, random_state=0).fit(Xs).bic(Xs)
               for k in Ks]
        axes[0].plot(Ks, bic, marker="o", color=col, label=cov)
    axes[0].set(title="BIC of Gaussian mixtures", xlabel="K", ylabel="BIC (lower is better)")
    axes[0].legend(frameon=False)
    K2 = np.arange(2, 9)
    sil, dbi, stab = [], [], []
    rng = np.random.default_rng(1)
    for k in K2:
        base = KMeans(k, n_init=10, random_state=0).fit(X)
        sil.append(silhouette_score(X, base.labels_))
        dbi.append(davies_bouldin_score(X, base.labels_))
        s = []
        for b in range(10):
            idx = rng.integers(0, len(X), len(X))
            s.append(adjusted_rand_score(base.predict(X),
                                         KMeans(k, n_init=10, random_state=b).fit(X[idx]).predict(X)))
        stab.append(np.mean(s))
    axes[1].plot(K2, sil, marker="o", color=PALETTE[0], label="silhouette (higher better)")
    axes[1].plot(K2, dbi, marker="s", color=PALETTE[1], label="Davies-Bouldin (lower better)")
    axes[1].set(title="Internal indices, k-means", xlabel="K")
    axes[1].legend(frameon=False, fontsize=10)
    axes[2].plot(K2, stab, marker="o", color=PALETTE[2])
    axes[2].set(title="Bootstrap stability (mean ARI)", xlabel="K", ylim=(0.4, 1.05))
    fig.tight_layout()
    save(fig, "model-selection.png")


def fig_moons():
    """Width-limited redraw of the two-moons comparison (generating labels, k-means, DBSCAN)."""
    X, truth = make_moons(n_samples=320, noise=0.06, random_state=5)
    km = KMeans(n_clusters=2, n_init=10, random_state=5).fit_predict(X)
    density = DBSCAN(eps=0.18, min_samples=5).fit_predict(X)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
    for ax, labels, title in zip(axes, [truth, km, density],
                                 ["Generating groups", "K-means: compact groups",
                                  "DBSCAN: density connectivity"]):
        for label in sorted(set(labels)):
            color = "#64748b" if label == -1 else PALETTE[label % 2]
            pts = X[labels == label]
            ax.scatter(pts[:, 0], pts[:, 1], color=color, s=12, alpha=0.65)
        ax.set(title=title, xlabel="Coordinate 1", ylabel="Coordinate 2")
    fig.tight_layout()
    save(fig, "moons-comparison.png")


if __name__ == "__main__":
    fig_moons()
    fig_kmeans_seeding()
    fig_linkage()
    fig_dbscan()
    fig_em()
    fig_selection()
