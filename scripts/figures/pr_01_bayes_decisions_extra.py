"""Extra figures for PR 01 (Bayes decision theory).

Run from the repository root:
    MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/pr_01_bayes_decisions_extra.py

Writes only to src/assets/pattern-recognition/pr-01-bayes-decisions/.
All data are synthetic (fixed seeds); nothing is downloaded.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, multivariate_normal

OUT = Path(__file__).resolve().parents[2] / "src/assets/pattern-recognition/pr-01-bayes-decisions"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
C0, C1, C2 = "#1f77b4", "#d62728", "#2ca02c"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def posteriors_1d():
    """Likelihoods, prior-weighted likelihoods and posteriors for two 1-D Gaussians."""
    x = np.linspace(-4, 6, 1001)
    l0, l1 = norm.pdf(x, 0, 1), norm.pdf(x, 2, 1)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    axes[0].plot(x, l0, color=C0, label=r"$p(x\mid c_0)$")
    axes[0].plot(x, l1, color=C1, label=r"$p(x\mid c_1)$")
    axes[0].set_title("Class likelihoods")
    for prior1, ls in [(0.5, "-"), (0.1, "--")]:
        j0, j1 = (1 - prior1) * l0, prior1 * l1
        post1 = j1 / (j0 + j1)
        axes[1].plot(x, j0, color=C0, ls=ls)
        axes[1].plot(x, j1, color=C1, ls=ls)
        axes[2].plot(x, post1, color=C1, ls=ls,
                     label=f"P(c1) = {prior1}")
        # boundary where the posterior is 0.5
        xb = 1 + np.log((1 - prior1) / prior1) / 2
        axes[2].axvline(xb, color="gray", ls=ls, lw=1)
    axes[1].set_title("Likelihood x prior (solid 0.5, dashed 0.1)")
    axes[2].axhline(0.5, color="black", lw=0.8)
    axes[2].set_title(r"Posterior $P(c_1\mid x)$ and boundary")
    for ax in axes:
        ax.set_xlabel("measurement x")
    axes[0].legend(fontsize=9)
    axes[2].legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "posterior-prior-shift.png")


def gaussian_cases():
    """Bayes boundaries for three covariance cases (equal priors)."""
    rng = np.random.default_rng(0)
    mu0, mu1 = np.array([0.0, 0.0]), np.array([2.5, 1.0])
    cases = [
        ("Case 1: S0 = S1 = s^2 I\n(linear, perpendicular to means)",
         np.eye(2), np.eye(2)),
        ("Case 2: S0 = S1 (shared)\n(linear, tilted)",
         np.array([[1.0, 0.7], [0.7, 1.2]]), np.array([[1.0, 0.7], [0.7, 1.2]])),
        ("Case 3: S0 \u2260 S1\n(quadratic)",
         np.array([[0.4, 0.0], [0.0, 0.4]]), np.array([[2.0, -0.6], [-0.6, 1.0]])),
    ]
    xx, yy = np.meshgrid(np.linspace(-3, 6, 400), np.linspace(-3.5, 5, 400))
    grid = np.dstack([xx, yy])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for ax, (title, s0, s1) in zip(axes, cases):
        g = (multivariate_normal(mu1, s1).logpdf(grid)
             - multivariate_normal(mu0, s0).logpdf(grid))
        ax.contourf(xx, yy, g > 0, levels=[-0.5, 0.5, 1.5],
                    colors=["#dbe8f5", "#f7dcdc"], alpha=0.8)
        ax.contour(xx, yy, g, levels=[0], colors="black", linewidths=1.6)
        a = rng.multivariate_normal(mu0, s0, 80)
        b = rng.multivariate_normal(mu1, s1, 80)
        ax.scatter(*a.T, s=10, color=C0)
        ax.scatter(*b.T, s=10, color=C1)
        ax.plot(*np.c_[mu0, mu1], "k:", lw=1)
        ax.set_title(title.replace("S0", r"$\Sigma_0$").replace("S1", r"$\Sigma_1$")
                     .replace("s^2 I", r"$\sigma^2 I$"), fontsize=10)
        ax.set_xlim(-3, 6)
        ax.set_ylim(-3.5, 5)
        ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "gaussian-boundaries.png")


def roc():
    """ROC curves for equal-variance Gaussian scores with different separations d'."""
    t = np.linspace(-6, 8, 2001)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for d, col in [(0.5, "#9467bd"), (1.0, C0), (2.0, C2), (3.0, C1)]:
        fpr = norm.sf(t, 0, 1)
        tpr = norm.sf(t, d, 1)
        auc = norm.cdf(d / np.sqrt(2))
        axes[0].plot(fpr, tpr, color=col, label=f"d' = {d}: AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    # operating points on d'=2 for cost ratios (equal priors)
    d = 2.0
    for ratio, mk in [(1, "o"), (9, "s"), (1 / 9, "^")]:
        # Predict 1 when LR > C_FP/C_FN * P0/P1; ratio = C_FN/C_FP
        thr = (np.log(1 / ratio) + d ** 2 / 2) / d
        axes[0].plot(norm.sf(thr), norm.sf(thr, d), mk, color="black", ms=7,
                     label=f"d'=2, C_FN/C_FP = {ratio:.3g}")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves (equal priors)")
    axes[0].legend(fontsize=8, loc="lower right")
    # right: score distributions with a sliding threshold for d'=2
    x = np.linspace(-4, 6, 800)
    axes[1].plot(x, norm.pdf(x), color=C0, label="negatives")
    axes[1].plot(x, norm.pdf(x, d), color=C1, label="positives")
    thr = 1.4
    axes[1].fill_between(x, norm.pdf(x), where=x > thr, color=C0, alpha=0.3,
                         label=f"FPR = {norm.sf(thr):.2f}")
    axes[1].fill_between(x, norm.pdf(x, d), where=x > thr, color=C1, alpha=0.2,
                         label=f"TPR = {norm.sf(thr, d):.2f}")
    axes[1].axvline(thr, color="black", ls="--", lw=1)
    axes[1].set_title(f"One threshold (t = {thr}) on d' = 2 scores")
    axes[1].set_xlabel("score")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    save(fig, "roc-curves.png")


def reject_curve():
    """Error-reject trade-off (Chow) for 3 overlapping 2-D Gaussian classes."""
    rng = np.random.default_rng(1)
    means = np.array([[0, 0], [2, 0], [1, 1.7]])
    n = 20000
    y = rng.integers(0, 3, n)
    x = means[y] + rng.normal(0, 1, (n, 2))
    logp = np.stack([multivariate_normal(m, np.eye(2)).logpdf(x) for m in means], 1)
    post = np.exp(logp - logp.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    pmax = post.max(1)
    pred = post.argmax(1)
    costs = np.linspace(0.0, 0.7, 141)
    rej, err = [], []
    for c in costs:
        accept = pmax >= 1 - c
        rej.append(1 - accept.mean())
        err.append(((pred != y) & accept).mean())
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.plot(rej, err, color=C0, lw=2)
    for c in [0.1, 0.2, 0.3, 0.5]:
        i = np.argmin(abs(costs - c))
        ax.plot(rej[i], err[i], "o", color=C1)
        ax.annotate(f"C_R = {c}", (rej[i], err[i]), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Reject rate")
    ax.set_ylabel("Error rate (accepted and wrong)")
    ax.set_title("Error-reject trade-off, 3 Gaussian classes")
    fig.tight_layout()
    save(fig, "error-reject.png")


if __name__ == "__main__":
    posteriors_1d()
    gaussian_cases()
    roc()
    reject_curve()
