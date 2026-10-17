"""Original synthetic figures for the Pattern Recognition course; no downloads."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "src/assets/pattern-recognition"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})


def save(fig, chapter, name):
    folder = ROOT / chapter
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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
    posterior()
