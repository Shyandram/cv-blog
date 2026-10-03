"""Figures for PR 03 (linear models for classification).

Run from the repository root:
    MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/pr_03_linear_models_extra.py

Writes only to src/assets/pattern-recognition/pr-03-linear-models/.
All data are synthetic (fixed seeds); nothing is downloaded.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC

OUT = Path(__file__).resolve().parents[2] / "src/assets/pattern-recognition/pr-03-linear-models"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
C0, C1, C2 = "#1f77b4", "#d62728", "#2ca02c"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def losses():
    m = np.linspace(-2.5, 3, 600)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(m, (m <= 0).astype(float), color="black", lw=2, label="zero-one")
    ax.plot(m, np.maximum(0, -m), color="#9467bd", label="perceptron  max(0, -m)")
    ax.plot(m, np.maximum(0, 1 - m), color=C1, label="hinge  max(0, 1 - m)")
    ax.plot(m, np.log1p(np.exp(-m)) / np.log(2), color=C0,
            label="logistic  log2(1 + exp(-m))")
    ax.plot(m, (1 - m) ** 2, color=C2, label="squared  (1 - m)^2")
    ax.set_ylim(-0.1, 4)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel("margin  m = y f(x)")
    ax.set_ylabel("loss")
    ax.set_title("Surrogate losses for y in {-1, +1}")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    save(fig, "surrogate-losses.png")


def fisher():
    rng = np.random.default_rng(2)
    cov = np.array([[3.0, 2.4], [2.4, 2.4]])
    a = rng.multivariate_normal([0, 0], cov, 200)
    b = rng.multivariate_normal([2.0, 0.0], cov, 200)
    m0, m1 = a.mean(0), b.mean(0)
    sw = np.cov(a.T) + np.cov(b.T)
    w_mean = (m1 - m0) / np.linalg.norm(m1 - m0)
    w_f = np.linalg.solve(sw, m1 - m0)
    w_f /= np.linalg.norm(w_f)
    fig = plt.figure(figsize=(10.5, 3.8))
    ax = fig.add_subplot(1, 3, 1)
    ax.scatter(*a.T, s=6, color=C0)
    ax.scatter(*b.T, s=6, color=C1)
    c = (m0 + m1) / 2
    for w, col, lab in [(w_mean, "gray", "mean difference"), (w_f, "black", "Fisher")]:
        ax.plot([c[0] - 3 * w[0], c[0] + 3 * w[0]], [c[1] - 3 * w[1], c[1] + 3 * w[1]],
                color=col, lw=2, label=lab)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Data and two directions")

    def j(w):
        return (w @ (m1 - m0)) ** 2 / (w @ sw @ w)
    for k, (w, title) in enumerate([(w_mean, "Projected on mean difference"),
                                     (w_f, "Projected on Fisher direction")]):
        ax = fig.add_subplot(1, 3, k + 2)
        pa, pb = a @ w, b @ w
        bins = np.linspace(min(pa.min(), pb.min()), max(pa.max(), pb.max()), 40)
        ax.hist(pa, bins=bins, color=C0, alpha=0.6)
        ax.hist(pb, bins=bins, color=C1, alpha=0.6)
        ax.set_title(f"{title}\nJ(w) = {j(w):.2f}", fontsize=10)
        ax.set_yticks([])
    fig.tight_layout()
    save(fig, "fisher-projection.png")
    return j(w_mean), j(w_f)


def ls_vs_logistic():
    """Least-squares classification is pulled by far-away correct points."""
    rng = np.random.default_rng(5)
    a = rng.normal([-1, -1], 0.5, (50, 2))
    b = rng.normal([1, 1], 0.5, (50, 2))
    extra = rng.normal([6, 2], 0.4, (20, 2))   # correct-side class-1 points far away
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    xs = np.linspace(-3, 8, 2)
    for ax, use_extra in zip(axes, [False, True]):
        X = np.vstack([a, b] + ([extra] if use_extra else []))
        y = np.r_[-np.ones(50), np.ones(50 + (20 if use_extra else 0))]
        Phi = np.c_[np.ones(len(X)), X]
        w = np.linalg.lstsq(Phi, y, rcond=None)[0]
        lr = LogisticRegression(C=100).fit(X, y)
        ax.scatter(*X[y < 0].T, s=10, color=C0)
        ax.scatter(*X[y > 0].T, s=10, color=C1)
        ax.plot(xs, -(w[0] + w[1] * xs) / w[2], color=C2, lw=2, label="least squares")
        wl, bl = lr.coef_[0], lr.intercept_[0]
        ax.plot(xs, -(bl + wl[0] * xs) / wl[1], "k--", lw=2, label="logistic regression")
        ax.set_xlim(-3, 8)
        ax.set_ylim(-3.5, 4)
        ax.set_title("added far, correctly placed red points" if use_extra else "two compact classes")
        ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "least-squares-vs-logistic.png")


def margins():
    rng = np.random.default_rng(7)
    a = rng.normal([-1.5, -1.0], 0.6, (30, 2))
    b = rng.normal([1.5, 1.0], 0.6, (30, 2))
    X = np.vstack([a, b])
    y = np.r_[-np.ones(30), np.ones(30)]
    xs = np.linspace(-4, 4, 2)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.scatter(*a.T, s=12, color=C0)
    ax.scatter(*b.T, s=12, color=C1)
    for seed in range(6):
        p = Perceptron(random_state=seed, shuffle=True, max_iter=1000, tol=None).fit(X, y)
        w, b0 = p.coef_[0], p.intercept_[0]
        ax.plot(xs, -(b0 + w[0] * xs) / w[1], color="gray", lw=1)
    ax.set_title("Perceptron: any separator (6 random orders)")
    ax = axes[1]
    svm = SVC(kernel="linear", C=1e6).fit(X, y)
    w, b0 = svm.coef_[0], svm.intercept_[0]
    ax.scatter(*a.T, s=12, color=C0)
    ax.scatter(*b.T, s=12, color=C1)
    for off, ls in [(0, "-"), (1, "--"), (-1, "--")]:
        ax.plot(xs, -(b0 - off + w[0] * xs) / w[1], color="black", ls=ls, lw=1.5)
    sv = svm.support_vectors_
    ax.scatter(*sv.T, s=110, facecolors="none", edgecolors="black", linewidths=1.5,
               label=f"{len(sv)} support vectors")
    ax.set_title(f"Hard-margin SVM: margin 2/||w|| = {2 / np.linalg.norm(w):.2f}")
    ax.legend(fontsize=9, loc="lower right")
    for ax in axes:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "perceptron-vs-svm.png")


if __name__ == "__main__":
    losses()
    print("J(mean diff), J(Fisher):", fisher())
    ls_vs_logistic()
    margins()
