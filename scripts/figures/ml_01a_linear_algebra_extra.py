"""Figures for ML 01A - Linear Algebra for Learning.

Writes only to src/assets/ml/ml-01a-linear-algebra/.
Run: .venv/bin/python scripts/figures/ml_01a_linear_algebra_extra.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from skimage import data, img_as_float

OUT = Path(__file__).resolve().parents[2] / "src/assets/ml/ml-01a-linear-algebra"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})
DPI = 150


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path, path.stat().st_size // 1024, "KB")


# 1. SVD as rotate - stretch - rotate --------------------------------------
def fig_svd_circle():
    A = np.array([[1.6, 0.9], [0.3, 1.1]])
    U, s, Vt = np.linalg.svd(A)
    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])
    e = np.eye(2)
    stages = [
        (np.eye(2), r"input: unit circle"),
        (Vt, r"after $V^\top$ (rotate/reflect)"),
        (np.diag(s) @ Vt, r"after $\Sigma V^\top$ (stretch axes)"),
        (U @ np.diag(s) @ Vt, r"after $U\Sigma V^\top = A$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    v_cols = Vt.T  # right singular vectors as columns
    for ax, (M, title) in zip(axes, stages):
        pts = M @ circle
        ax.plot(pts[0], pts[1], color="#3b6ea5", lw=2)
        ax.fill(pts[0], pts[1], color="#3b6ea5", alpha=0.08)
        for j, c in enumerate(["#d1495b", "#edae49"]):
            v = M @ v_cols[:, j]
            ax.annotate("", xy=v, xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=c, lw=2.2))
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.axvline(0, color="0.8", lw=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].text(-2.1, -2.05, r"red/yellow: $v_1, v_2$", fontsize=10)
    axes[3].text(-2.1, -2.05, rf"$\sigma_1$={s[0]:.2f}, $\sigma_2$={s[1]:.2f}", fontsize=10)
    save(fig, "svd-unit-circle.png")


# 2. Rank-k approximation of an image --------------------------------------
def fig_low_rank():
    img = img_as_float(data.camera())[::2, ::2]  # 256 x 256
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    ks = [5, 20, 50]
    fig, axes = plt.subplots(1, 5, figsize=(13, 3.1),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25], "wspace": 0.35})
    for ax, k in zip(axes[:3], ks):
        Ak = (U[:, :k] * s[:k]) @ Vt[:k]
        rel = np.linalg.norm(img - Ak) / np.linalg.norm(img)
        ax.imshow(np.clip(Ak, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"rank {k}\nrel. error {rel:.3f}", fontsize=11)
        ax.axis("off")
    axes[3].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"original\nrank {np.linalg.matrix_rank(img)}", fontsize=11)
    axes[3].axis("off")
    ax = axes[4]
    ax.semilogy(np.arange(1, len(s) + 1), s, color="#3b6ea5", lw=2)
    for k in ks:
        ax.axvline(k, color="#d1495b", ls="--", lw=1)
    ax.set_xlabel("index j")
    ax.set_ylabel(r"$\sigma_j$")
    ax.set_title("singular value spectrum", fontsize=11)
    ax.grid(alpha=0.3)
    save(fig, "low-rank-image.png")


# 3. PCA on correlated 2-D data -------------------------------------------
def fig_pca():
    rng = np.random.default_rng(0)
    C = np.array([[3.0, 1.6], [1.6, 1.4]])
    X = rng.multivariate_normal([2.0, 1.0], C, size=300)
    mu = X.mean(0)
    Xc = X - mu
    S = Xc.T @ Xc / (len(X) - 1)
    lam, W = np.linalg.eigh(S)
    order = np.argsort(lam)[::-1]
    lam, W = lam[order], W[:, order]
    W = W * np.sign(W[0])  # make both axes point to the right (sign is arbitrary)
    d1 = W[:, 0]
    z = Xc @ d1
    recon = mu + np.outer(z, d1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5, color="#3b6ea5")
    for j, c in enumerate(["#d1495b", "#edae49"]):
        v = W[:, j] * 2 * np.sqrt(lam[j])
        ax.annotate("", xy=mu + v, xytext=mu,
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.5))
        ax.text(*(mu + v * 1.12 + np.array([0.15, 0.0])), f"PC{j+1}, $\\lambda$={lam[j]:.2f}",
                color=c, fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    ax.set_aspect("equal")
    ax.set_title("data and principal axes (length 2$\\sqrt{\\lambda}$)", fontsize=11)
    ax.grid(alpha=0.3)

    ax = axes[1]
    idx = rng.choice(len(X), 40, replace=False)
    ax.scatter(X[idx, 0], X[idx, 1], s=16, color="#3b6ea5", label="data $x$")
    ax.scatter(recon[idx, 0], recon[idx, 1], s=16, color="#d1495b", label="decode(encode($x$))")
    for i in idx:
        ax.plot([X[i, 0], recon[i, 0]], [X[i, 1], recon[i, 1]], color="0.6", lw=0.8)
    tline = np.linspace(z.min(), z.max(), 2)
    line = mu + np.outer(tline, d1)
    ax.plot(line[:, 0], line[:, 1], color="#d1495b", lw=1.5)
    ax.set_aspect("equal")
    ax.set_title("1-D PCA: residuals are perpendicular", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    save(fig, "pca-2d.png")


# 4. Determinant as signed area --------------------------------------------
def fig_determinant():
    mats = [
        (np.array([[1.0, 0.0], [0.0, 1.0]]), "identity"),
        (np.array([[2.0, 0.6], [0.0, 1.0]]), "stretch + shear"),
        (np.array([[0.0, 1.0], [1.0, 0.0]]), "reflection"),
        (np.array([[1.0, 2.0], [0.5, 1.0]]), "singular"),
    ]
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (M, name) in zip(axes, mats):
        P = M @ square
        ax.add_patch(Polygon(square.T, closed=True, fill=False, ls="--", ec="0.6"))
        ax.add_patch(Polygon(P.T, closed=True, alpha=0.3, color="#3b6ea5"))
        ax.annotate("", xy=M[:, 0], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#d1495b", lw=2))
        ax.annotate("", xy=M[:, 1], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#edae49", lw=2))
        ax.set_title(f"{name}\ndet = {np.linalg.det(M):.2f}", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-0.3, 3.0)
        ax.set_ylim(-0.3, 2.0)
        ax.grid(alpha=0.3)
    axes[0].text(-0.25, 1.8, r"red: $Ae_1$, yellow: $Ae_2$", fontsize=10)
    save(fig, "determinant-area.png")


if __name__ == "__main__":
    fig_svd_circle()
    fig_low_rank()
    fig_pca()
    fig_determinant()
