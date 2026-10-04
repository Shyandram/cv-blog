"""Original LP/QP and logarithmic-coordinate examples."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[2] / "src/assets/optimization/opt-03-convex-programs"
plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def programs():
    vertices = np.array([[0, 0], [2.5, 0], [1, 3], [0, 4]])
    grid = np.linspace(-.3, 4.3, 200)
    xx, yy = np.meshgrid(grid, grid)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7))
    values = [-3 * xx - 2 * yy, .5 * ((xx - 3)**2 + (yy - 2)**2)]
    optima = [[1, 3], [1.8, 1.4]]
    titles = ["Linear objective", "Quadratic objective"]
    for ax, value, optimum, title in zip(axes, values, optima, titles):
        ax.add_patch(Polygon(vertices, color="#93c5fd", alpha=.6))
        ax.contour(xx, yy, value, levels=10, colors="#64748b", linewidths=.7)
        ax.scatter(*optimum, color="#db2777", s=45, zorder=5, label="Optimum")
        ax.set(xlim=(-.3, 4.3), ylim=(-.3, 4.3), xlabel="x1", ylabel="x2", title=title)
        ax.set_aspect("equal")
        ax.legend(frameon=False)
    axes[1].scatter(3, 2, marker="x", color="#0f172a", label="Target")
    fig.tight_layout()
    save(fig, "lp-qp.png")


def geometry():
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    vertices = np.array([[0, 0], [4, 0], [0, 4]])
    axes[0].add_patch(Polygon(vertices, color="#93c5fd", alpha=.6))
    axes[0].set(xlim=(0, 4.2), ylim=(0, 4.2), xlabel="x", ylabel="y",
                title="Positive coordinates: x + y <= 4")
    z = np.linspace(-2, np.log(4) - .02, 500)
    boundary = np.log(4 - np.exp(z))
    axes[1].fill_between(z, -2, boundary, where=boundary >= -2, color="#93c5fd", alpha=.6)
    axes[1].plot(z, boundary, color="#2563eb")
    axes[1].set(xlim=(-2, 1.5), ylim=(-2, 1.5), xlabel="log x", ylabel="log y",
                title="Log coordinates: exp(u) + exp(v) <= 4")
    axes[0].scatter(2, 2, color="#db2777")
    axes[1].scatter(np.log(2), np.log(2), color="#db2777")
    fig.tight_layout()
    save(fig, "geometric-transform.png")


if __name__ == "__main__":
    programs()
    geometry()
