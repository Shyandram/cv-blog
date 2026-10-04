"""Original analytical figures for convex functions; no external data."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2] / "src/assets/optimization/opt-02-convex-functions"
plt.rcParams.update({"font.size": 10, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bounds():
    x = np.linspace(-2, 2, 400)
    f = lambda x: .5 * x**2
    anchor, left, right = -.5, -1.5, 1.5
    tangent = f(anchor) + anchor * (x - anchor)
    chord_x = np.linspace(left, right, 100)
    chord = f(left) + (f(right) - f(left)) * (chord_x - left) / (right - left)
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.plot(x, f(x), lw=2, label="Convex function")
    ax.plot(x, tangent, "--", label="Supporting tangent")
    ax.plot(chord_x, chord, color="#db2777", label="Endpoint chord")
    ax.scatter([left, right], [f(left), f(right)], color="#db2777")
    ax.scatter([anchor], [f(anchor)], color="#64748b")
    ax.set(xlabel="x", ylabel="Value", title="Convexity bounds for f(x) = x^2 / 2")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "chord-tangent.png")


def smooth_max():
    x = np.linspace(-2, 2, 400)
    scores = np.stack([x, -x], axis=-1)
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.plot(x, scores.max(axis=1), color="#0f172a", lw=2, label="max(x, -x)")
    for temperature in [.8, .3, .08]:
        value = temperature * logsumexp(scores / temperature, axis=1)
        ax.plot(x, value, label=f"Temperature {temperature}")
    ax.set(xlabel="x", ylabel="Value", title="Smooth convex approximation of a maximum")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "log-sum-exp.png")


if __name__ == "__main__":
    bounds()
    smooth_max()
