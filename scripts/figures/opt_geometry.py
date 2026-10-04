"""Original convex-geometry teaching figures, using only synthetic data."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Wedge

ROOT = Path(__file__).resolve().parents[2] / "src/assets/optimization/opt-01-modeling-convex-sets"
plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sets():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].add_patch(Circle((0, 0), 1, color="#93c5fd", alpha=.6))
    axes[0].plot([-.8, .6], [.2, -.6], color="#db2777", marker="o")
    axes[0].set_title("Convex: disk")
    axes[1].add_patch(Wedge((0, 0), 1, 0, 360, width=.5,
                            color="#93c5fd", alpha=.6))
    axes[1].plot([-1, 1], [0, 0], color="#db2777", marker="o")
    axes[1].set_title("Nonconvex: annulus")
    triangle = np.array([[-1, -.7], [1, -.7], [0, 1]])
    axes[2].add_patch(Polygon(triangle, color="#93c5fd", alpha=.6))
    average = np.array([.2, .3, .5]) @ triangle
    axes[2].scatter(*average, color="#db2777", s=45)
    axes[2].set_title("Convex hull: triangle")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set(xlim=(-1.25, 1.25), ylim=(-1.15, 1.2), xlabel="x1", ylabel="x2")
    fig.tight_layout()
    save(fig, "sets.png")


def projection():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.add_patch(Circle((0, 0), 1, color="#93c5fd", alpha=.6))
    y = np.array([2., 1.])
    z = y / np.linalg.norm(y)
    tangent = np.array([-z[1], z[0]])
    endpoints = z[None, :] + np.array([-1.5, 1.5])[:, None] * tangent
    ax.plot(endpoints[:, 0], endpoints[:, 1], "--", color="#64748b",
            label="Separating hyperplane")
    ax.plot([z[0], y[0]], [z[1], y[1]], color="#db2777", marker="o")
    ax.annotate("Query y", y, xytext=(y[0] + .08, y[1] + .08))
    ax.annotate("Projection z", z, xytext=(z[0] - .7, z[1] - .3))
    ax.set_aspect("equal")
    ax.set(xlim=(-1.3, 2.7), ylim=(-1.2, 1.65), xlabel="x1", ylabel="x2",
           title="Projection onto a closed convex set")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, "projection.png")


if __name__ == "__main__":
    sets()
    projection()
