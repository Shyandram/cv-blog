"""Original synthetic figures for the Pattern Recognition course; no downloads."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons

ROOT = Path(__file__).resolve().parents[2] / "src/assets/pattern-recognition"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})


def save(fig, chapter, name):
    folder = ROOT / chapter
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def features():
    rng = np.random.default_rng(5)
    labels = np.repeat([0, 1], 160)
    x = rng.normal(0, 2.8, len(labels))
    y = (2 * labels - 1) * 0.65 + rng.normal(0, 0.18, len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    colors = ["#2563eb", "#db2777"]
    for k in [0, 1]:
        chosen = labels == k
        axes[0].scatter(x[chosen], y[chosen], s=12, alpha=0.45,
                        color=colors[k], label=f"Class {k}")
        axes[1].hist(x[chosen], bins=np.linspace(-9, 9, 22), alpha=0.5,
                     color=colors[k], density=True)
        axes[2].hist(y[chosen], bins=np.linspace(-1.3, 1.3, 25), alpha=0.5,
                     color=colors[k], density=True)
    axes[0].set(title="Synthetic measurements", xlabel="High-variance nuisance",
                ylabel="Low-variance class signal")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].set(title="Projection onto nuisance axis", xlabel="Nuisance value",
                ylabel="Density")
    axes[2].set(title="Projection onto class axis", xlabel="Class signal",
                ylabel="Density")
    fig.tight_layout()
    save(fig, "pr-05-features-dimensionality", "variance-vs-class.png")


def clustering():
    X, truth = make_moons(n_samples=320, noise=0.06, random_state=5)
    km = KMeans(n_clusters=2, n_init=10, random_state=5).fit_predict(X)
    density = DBSCAN(eps=0.18, min_samples=5).fit_predict(X)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, labels, title in zip(axes, [truth, km, density],
                                 ["Generating groups", "K-means: compact groups",
                                  "DBSCAN: density connectivity"]):
        for label in sorted(set(labels)):
            color = "#64748b" if label == -1 else ["#2563eb", "#db2777"][label % 2]
            points = X[labels == label]
            ax.scatter(points[:, 0], points[:, 1], color=color, s=13, alpha=0.65)
        ax.set(title=title, xlabel="Coordinate 1", ylabel="Coordinate 2")
        ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "pr-07-clustering-mixtures", "clustering-geometry.png")


def posterior():
    rho = 0.85
    cov = np.array([[1, rho], [rho, 1]])
    precision = np.linalg.inv(cov)
    axis = np.linspace(-3, 3, 250)
    xx, yy = np.meshgrid(axis, axis)
    points = np.stack([xx, yy], axis=-1)
    target = np.exp(-0.5 * np.einsum("...i,ij,...j->...", points, precision, points))
    # Optimal factorized Gaussian under KL(q || p) for this Gaussian target.
    variance = 1 - rho**2
    mean_field = np.exp(-0.5 * (xx**2 + yy**2) / variance)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.5))
    for ax, values, title in zip(axes, [target, mean_field],
                                 ["Correlated Gaussian target",
                                  "Reverse-KL mean-field optimum"]):
        ax.contour(xx, yy, values, levels=[0.05, 0.15, 0.35, 0.65, 0.9],
                   colors="#2563eb")
        ax.set(title=title, xlabel="Parameter 1", ylabel="Parameter 2",
               xlim=(-3, 3), ylim=(-3, 3))
        ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "pr-08-approximate-inference", "mean-field-correlation.png")


if __name__ == "__main__":
    features()
    clustering()
    posterior()
