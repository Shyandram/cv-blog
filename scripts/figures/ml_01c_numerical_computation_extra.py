"""Figures for ML 01C · Numerical Computation.

Writes only into src/assets/ml/ml-01c-numerical-computation/.
Run: .venv/bin/python scripts/figures/ml_01c_numerical_computation_extra.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src" / "assets" / "ml" / "ml-01c-numerical-computation"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, RED, GRAY = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# 1. Finite-difference error versus step size --------------------------------
def fd_error():
    x0 = 1.0
    true = np.cos(x0)
    eps = np.logspace(-14, -0.5, 120)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for dtype, color, label in ((np.float32, ORANGE, "float32"), (np.float64, BLUE, "float64")):
        x = dtype(x0)
        fwd, cen = [], []
        for e in eps:
            h = dtype(e)
            hx = (x + h) - x  # the step actually represented
            if hx == 0:
                fwd.append(np.nan)
                cen.append(np.nan)
                continue
            fwd.append(abs(float((np.sin(x + h) - np.sin(x)) / hx) - true))
            cen.append(abs(float((np.sin(x + h) - np.sin(x - h)) / (2 * hx)) - true))
        fwd = np.maximum(np.array(fwd), 1e-17)
        cen = np.maximum(np.array(cen), 1e-17)
        ax.loglog(eps, fwd, color=color, ls="--", lw=1.6, label=f"forward, {label}")
        ax.loglog(eps, cen, color=color, lw=2.0, label=f"central, {label}")
    ax.set_xlabel(r"step size $\epsilon$")
    ax.set_ylabel("absolute error of derivative")
    ax.set_title(r"Estimating $d\,\sin(x)/dx$ at $x=1$")
    ax.text(4e-3, 1e-9, "truncation error\ndominates", color=GRAY, ha="center")
    ax.text(3e-12, 3e-2, "roundoff error\ndominates", color=GRAY, ha="center")
    ax.set_ylim(1e-15, 10)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    save(fig, "finite-difference-error.png")


# 2. Gradient descent zigzag vs Newton on an ill-conditioned quadratic -------
def gd_vs_newton():
    H = np.array([[1.0, 0.0], [0.0, 20.0]])
    f = lambda W1, W2: 0.5 * (H[0, 0] * W1**2 + H[1, 1] * W2**2)
    w0 = np.array([-9.0, 1.6])

    def path(step_rule, n=25):
        w, pts = w0.copy(), [w0.copy()]
        for _ in range(n):
            g = H @ w
            w = w - step_rule(g) * g
            pts.append(w.copy())
        return np.array(pts)

    fixed = path(lambda g: 0.09)  # close to 2/lambda_max = 0.1
    exact = path(lambda g: (g @ g) / (g @ H @ g))
    newton = np.array([w0, w0 - np.linalg.solve(H, H @ w0)])

    x = np.linspace(-10, 10, 300)
    y = np.linspace(-2.2, 2.2, 300)
    X, Y = np.meshgrid(x, y)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.6), sharex=True)
    for ax, pts, color, title in (
        (axes[0], fixed, ORANGE, "GD, fixed step 0.09\n(close to 2/λmax = 0.1)"),
        (axes[1], exact, BLUE, "GD with exact line search"),
        (axes[2], newton, GREEN, "Newton: one step"),
    ):
        ax.contour(X, Y, f(X, Y), levels=np.geomspace(0.05, 60, 14), colors=GRAY,
                   linewidths=0.7, alpha=0.7)
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=color, ms=3, lw=1.4)
        ax.plot(0, 0, "*", color=RED, ms=14)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(r"$w_2$")
        ax.set_aspect("equal")
    axes[-1].set_xlabel(r"$w_1$ (curvature 1);  $w_2$ has curvature 20")
    fig.suptitle("Hessian condition number 20: gradient steps zigzag, Newton rescales", y=1.0)
    save(fig, "gd-vs-newton.png")


# 3. Critical points: minimum, maximum, saddle -------------------------------
def critical_points():
    x = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(x, x)
    cases = (
        ("minimum: H positive definite", X**2 + Y**2, "eigenvalues +2, +2"),
        ("maximum: H negative definite", -(X**2) - Y**2, "eigenvalues -2, -2"),
        ("saddle: H indefinite", X**2 - Y**2, "eigenvalues +2, -2"),
    )
    fig = plt.figure(figsize=(13, 4.2))
    for i, (title, Z, note) in enumerate(cases, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
        ax.scatter([0], [0], [0], color=RED, s=40, depthshade=False)
        ax.set_title(f"{title}\n{note}", fontsize=10)
        ax.set_xlabel("$x_1$", labelpad=-6)
        ax.set_ylabel("$x_2$", labelpad=-6)
        ax.tick_params(labelsize=8, pad=-2)
        ax.view_init(elev=28, azim=-55)
    fig.suptitle(r"All three have $\nabla f(0)=0$; the Hessian tells them apart", y=1.0)
    save(fig, "critical-points.png")


# 4. Learning-rate stability on a quadratic ---------------------------------
def lr_stability():
    lam = np.array([1.0, 10.0])
    lim = 2 / lam.max()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for frac, color in ((0.2, BLUE), (0.5, GREEN), (0.95, ORANGE), (1.02, RED)):
        eta = frac * lim
        w = np.array([1.0, 1.0])
        losses = []
        for _ in range(60):
            losses.append(0.5 * np.sum(lam * w**2))
            w = w - eta * lam * w
        ax.semilogy(losses, color=color, lw=2, label=rf"$\eta = {frac:.2f}\times 2/\lambda_{{\max}}$")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"loss $\frac{1}{2}w^\top H w$")
    ax.set_title(r"Fixed-step gradient descent, $H=\mathrm{diag}(1,10)$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    save(fig, "learning-rate-stability.png")


# 5. KKT conditions and projected gradient ----------------------------------
def kkt_projection():
    A = np.array([[2.0, 0.6], [0.6, 1.0]])
    c = np.array([1.6, 1.1])  # unconstrained minimizer
    r = 1.0
    f = lambda W: 0.5 * np.einsum("...i,ij,...j->...", W - c, A, W - c)
    grad = lambda w: A @ (w - c)

    # projected gradient path
    w = np.array([-0.8, -0.2])
    pts = [w.copy()]
    for _ in range(40):
        w = w - 0.12 * grad(w)
        n = np.linalg.norm(w)
        if n > r:
            w = w * r / n
        pts.append(w.copy())
    pts = np.array(pts)
    w_star = pts[-1]

    g = np.linspace(-1.6, 2.4, 300)
    X, Y = np.meshgrid(g, g)
    Z = f(np.stack([X, Y], axis=-1))
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.contour(X, Y, Z, levels=np.geomspace(0.02, 6, 14), colors=GRAY, linewidths=0.7)
    t = np.linspace(0, 2 * np.pi, 300)
    ax.fill(r * np.cos(t), r * np.sin(t), color=BLUE, alpha=0.12)
    ax.plot(r * np.cos(t), r * np.sin(t), color=BLUE, lw=1.6, label=r"feasible set $\|w\|\leq 1$")
    ax.plot(*c, "*", color=RED, ms=14, label="unconstrained minimum")
    ax.plot(pts[:, 0], pts[:, 1], "-o", color=ORANGE, ms=3, lw=1.2, label="projected gradient path")
    ax.plot(*w_star, "o", color=GREEN, ms=9, label="constrained solution $w^*$")
    neg_g = -grad(w_star)
    neg_g = 0.75 * neg_g / np.linalg.norm(neg_g)
    normal = 0.4 * w_star / np.linalg.norm(w_star)
    ax.annotate("", xy=w_star + neg_g, xytext=w_star,
                arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax.annotate("", xy=w_star + normal, xytext=w_star,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2, ls="--"))
    ax.text(*(w_star + neg_g + np.array([0.0, -0.25])), r"$-\nabla f(w^*)$", color=RED)
    ax.text(*(w_star + np.array([-0.25, 0.22])), r"$\nabla g(w^*)$", color=GREEN)
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 2.4)
    ax.set_ylim(-1.6, 2.4)
    ax.set_xlabel("$w_1$")
    ax.set_ylabel("$w_2$")
    ax.set_title("KKT: at $w^*$, $-\\nabla f = \\lambda \\nabla g$ with $\\lambda \\geq 0$")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "kkt-projection.png")


if __name__ == "__main__":
    fd_error()
    gd_vs_newton()
    critical_points()
    lr_stability()
    kkt_projection()
