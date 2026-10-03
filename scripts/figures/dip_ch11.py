"""Figures for DIP Chapter 11 - Active contours: snakes and level sets.

Run:  .venv/bin/python scripts/figures/dip_ch11.py
Writes PNGs to src/assets/dip/ch11/. Re-runnable; uses only skimage.data and
synthetic arrays.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage import data, draw  # noqa: E402
from skimage.color import rgb2gray  # noqa: E402
from skimage.filters import gaussian  # noqa: E402
from skimage.segmentation import (  # noqa: E402
    active_contour,
    chan_vese,
    disk_level_set,
    inverse_gaussian_gradient,
    morphological_chan_vese,
    morphological_geodesic_active_contour,
)

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "dip" / "ch11"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})
DPI = 150


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


# --------------------------------------------------------------------------
# A small, self-contained snake solver (semi-implicit Kass-Witkin-Terzopoulos)
# --------------------------------------------------------------------------
def snake_matrix(n, alpha, beta, gamma):
    """Return (A + gamma I)^-1 for a closed snake with n points."""
    a = 2 * alpha + 6 * beta
    b = -alpha - 4 * beta
    c = beta
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = a
        A[i, (i - 1) % n] = A[i, (i + 1) % n] = b
        A[i, (i - 2) % n] = A[i, (i + 2) % n] = c
    return np.linalg.inv(A + gamma * np.eye(n))


def resample(x, y, spacing=1.0):
    """Redistribute the points of a closed curve evenly along its length."""
    xc, yc = closed(x, y)
    seg = np.hypot(np.diff(xc), np.diff(yc))
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(s[-1] / spacing), 12)
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.interp(t, s, xc), np.interp(t, s, yc)


def run_snake(fx, fy, x, y, alpha=0.1, beta=0.1, gamma=1.0, kappa=1.0,
              balloon=0.0, iters=300, resample_every=0, spacing=1.0):
    """Evolve a closed snake under the external force field (fx, fy).

    fx, fy : arrays with the x (column) and y (row) components of the force.
    balloon: magnitude of an outward normal pressure force (Cohen 1991).
    resample_every: if > 0, re-space the points evenly every that many steps.
    """
    x, y = x.astype(float).copy(), y.astype(float).copy()
    inv = snake_matrix(len(x), alpha, beta, gamma)
    for k in range(iters):
        if resample_every and k % resample_every == 0:
            x, y = resample(x, y, spacing)
            inv = snake_matrix(len(x), alpha, beta, gamma)
        coords = np.vstack([y, x])
        ex = kappa * ndi.map_coordinates(fx, coords, order=1, mode="nearest")
        ey = kappa * ndi.map_coordinates(fy, coords, order=1, mode="nearest")
        if balloon:
            tx = np.roll(x, -1) - np.roll(x, 1)
            ty = np.roll(y, -1) - np.roll(y, 1)
            norm = np.hypot(tx, ty) + 1e-9
            nx, ny = ty / norm, -tx / norm
            # flip so the normal points outward (away from the centroid)
            if np.mean((x - x.mean()) * nx + (y - y.mean()) * ny) < 0:
                nx, ny = -nx, -ny
            ex = ex + balloon * nx
            ey = ey + balloon * ny
        x = inv @ (gamma * x + ex)
        y = inv @ (gamma * y + ey)
        x = np.clip(x, 0, fx.shape[1] - 1)
        y = np.clip(y, 0, fx.shape[0] - 1)
    return x, y


def gvf(f, mu=0.2, iters=400):
    """Gradient vector flow (Xu & Prince 1998) of an edge map f in [0, 1]."""
    fy, fx = np.gradient(f)
    mag2 = fx ** 2 + fy ** 2
    u, v = fx.copy(), fy.copy()
    for _ in range(iters):
        u = u + mu * ndi.laplace(u, mode="nearest") - (u - fx) * mag2
        v = v + mu * ndi.laplace(v, mode="nearest") - (v - fy) * mag2
    return u, v


def circle(cx, cy, r, n=120):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return cx + r * np.cos(t), cy + r * np.sin(t)


def closed(x, y):
    return np.append(x, x[0]), np.append(y, y[0])


# --------------------------------------------------------------------------
# Figure 11.1 - snake on the astronaut's face + effect of the bending weight
# --------------------------------------------------------------------------
def fig_snake_astronaut():
    img = rgb2gray(data.astronaut())
    smooth = gaussian(img, sigma=3, preserve_range=False)
    s = np.linspace(0, 2 * np.pi, 400)
    init = np.array([100 + 100 * np.sin(s), 220 + 100 * np.cos(s)]).T
    stages = [15, 60, 2500]
    colors = ["#f4a261", "#2a9d8f", "#1d4ed8"]

    star = np.zeros((200, 200))
    rr, cc = draw.polygon([30, 170, 170, 100, 30], [30, 30, 170, 120, 170], star.shape)
    star[rr, cc] = 1.0
    star_s = gaussian(star, sigma=2)
    t = np.linspace(0, 2 * np.pi, 300)
    sinit = np.array([100 + 90 * np.sin(t), 100 + 90 * np.cos(t)]).T

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    ax = axes[0]
    ax.imshow(img, cmap="gray")
    ax.plot(init[:, 1], init[:, 0], "--r", lw=2, label="initial circle")
    for it, col in zip(stages, colors):
        snake = active_contour(smooth, init, alpha=0.015, beta=10, gamma=0.001,
                               max_num_iter=it)
        ax.plot(snake[:, 1], snake[:, 0], color=col, lw=2, label=f"after {it} iterations")
    ax.set_xlim(90, 350)
    ax.set_ylim(230, 0)
    ax.set_title("(a) Snake shrinking onto the face")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.axis("off")

    ax = axes[1]
    ax.imshow(star, cmap="gray", vmin=-0.3, vmax=1.3)
    ax.plot(sinit[:, 1], sinit[:, 0], "--r", lw=2, label="initial circle")
    for beta, col in [(0.001, "#e76f51"), (20.0, "#1d4ed8")]:
        snake = active_contour(star_s, sinit, alpha=0.01, beta=beta, gamma=0.01,
                               w_edge=1, max_num_iter=2500)
        ax.plot(snake[:, 1], snake[:, 0], color=col, lw=2, label=f"beta = {beta}")
    ax.set_title("(b) Bending weight beta on a notched shape")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.axis("off")
    save(fig, "snake-astronaut.png")


# --------------------------------------------------------------------------
# Figure 11.2 - gradient field vs GVF field on a U shape, and the two snakes
# --------------------------------------------------------------------------
def u_shape(n=64):
    img = np.zeros((n, n))
    img[16:48, 16:48] = 1.0
    img[16:40, 26:38] = 0.0  # the notch opening at the top
    return img


def fig_gvf():
    img = u_shape()
    edge = ndi.gaussian_gradient_magnitude(img, sigma=1.0)
    edge = edge / edge.max()
    gy, gx = np.gradient(edge)
    u, v = gvf(edge, mu=0.2, iters=800)

    x0, y0 = circle(32, 32, 26, n=100)
    opts = dict(alpha=0.05, beta=0.05, gamma=1.0, kappa=4.0, iters=1500,
                resample_every=25, spacing=1.0)
    sx_g, sy_g = run_snake(gx, gy, x0, y0, **opts)
    sx_v, sy_v = run_snake(u, v, x0, y0, **opts)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.5))
    step = 2
    Y, X = np.mgrid[0:64:step, 0:64:step]
    for ax, (fx, fy), title in [(axes[0, 0], (gx, gy), "(a) Gradient of the edge map"),
                                (axes[0, 1], (u, v), "(b) Gradient vector flow (GVF)")]:
        U = fx[::step, ::step]
        V = fy[::step, ::step]
        mag = np.hypot(U, V)
        nz = mag > 1e-4
        Un = np.where(nz, U / (mag + 1e-12), 0)
        Vn = np.where(nz, V / (mag + 1e-12), 0)
        ax.imshow(img, cmap="gray", vmin=-0.4, vmax=1.6)
        ax.quiver(X[nz], Y[nz], Un[nz], -Vn[nz], color="#1d4ed8", scale=40, width=0.004)
        ax.set_title(title + "\n(unit arrows; blank = zero force)", fontsize=11)
        ax.set_xlim(-0.5, 63.5)
        ax.set_ylim(63.5, -0.5)
        ax.axis("off")
    for ax, (sx, sy), title in [(axes[1, 0], (sx_g, sy_g), "(c) Snake with gradient force"),
                                (axes[1, 1], (sx_v, sy_v), "(d) Snake with GVF force")]:
        ax.imshow(img, cmap="gray", vmin=-0.4, vmax=1.6)
        ax.plot(*closed(x0, y0), "--r", lw=2, label="initial")
        ax.plot(*closed(sx, sy), color="#f4a261", lw=2.5, label="converged")
        ax.set_title(title)
        ax.legend(loc="lower center", fontsize=10, ncol=2, framealpha=0.9)
        ax.set_xlim(-0.5, 63.5)
        ax.set_ylim(63.5, -0.5)
        ax.axis("off")
    fig.tight_layout()
    save(fig, "gvf-vs-gradient.png")


# --------------------------------------------------------------------------
# Figure 11.3 - balloon force
# --------------------------------------------------------------------------
def fig_balloon():
    n = 160
    img = np.zeros((n, n))
    rr, cc = draw.ellipse(80, 80, 55, 65, img.shape)
    img[rr, cc] = 1.0
    rng = np.random.default_rng(0)
    noisy = img + 0.15 * rng.standard_normal(img.shape)
    edge = ndi.gaussian_gradient_magnitude(noisy, sigma=3.0)
    edge = np.clip(edge / np.percentile(edge, 99.5), 0, 1)
    gy, gx = np.gradient(edge)
    x0, y0 = circle(75, 85, 12, n=80)
    runs = [(0.0, "(a) No balloon: the snake collapses"),
            (0.3, "(b) Balloon pressure 0.3: it inflates to the edge")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (b, title) in zip(axes, runs):
        sx, sy = run_snake(gx, gy, x0, y0, alpha=0.05, beta=0.05, gamma=1.0,
                           kappa=10.0, balloon=b, iters=600, resample_every=25,
                           spacing=2.0)
        ax.imshow(noisy, cmap="gray")
        ax.plot(*closed(x0, y0), "--r", lw=2, label="initial")
        ax.plot(*closed(sx, sy), color="#f4a261", lw=2.5, label="after 600 iterations")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="lower center", fontsize=10, ncol=2, framealpha=0.9)
        ax.axis("off")
    save(fig, "balloon.png")


# --------------------------------------------------------------------------
# Figure 11.4 - the level-set function as a surface
# --------------------------------------------------------------------------
def fig_level_set_surface():
    n = 120
    Y, X = np.mgrid[0:n, 0:n]
    d1 = np.hypot(X - 42, Y - 60) - 22
    d2 = np.hypot(X - 82, Y - 60) - 22
    phi = np.minimum(d1, d2)  # signed distance (negative inside) to two discs
    fig = plt.figure(figsize=(11, 4.8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(X, Y, phi, cmap="viridis", rstride=3, cstride=3, linewidth=0,
                    antialiased=True, alpha=0.85)
    ax.contour(X, Y, phi, levels=[0], zdir="z", offset=0, colors="red", linewidths=2.5)
    xx, yy = np.meshgrid([0, n], [0, n])
    ax.plot_surface(xx, yy, np.zeros_like(xx, dtype=float), color="gray", alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("phi(x, y)")
    ax.set_title("(a) phi as a surface; red = zero level set")
    ax.view_init(elev=28, azim=-60)

    ax = fig.add_subplot(1, 2, 2)
    levels = [(0, "#d62828", "phi = 0"), (-12, "#f4a261", "phi = -12"),
              (6, "#2a9d8f", "phi = +6"), (14, "#1d4ed8", "phi = +14")]
    im = ax.imshow(phi, cmap="viridis", origin="upper")
    for lev, col, lab in levels:
        ax.contour(phi, levels=[lev], colors=col, linewidths=2.2, linestyles="solid")
        ax.plot([], [], color=col, lw=2.2, label=lab)
    ax.set_title("(b) Slicing at other heights changes topology")
    ax.legend(loc="lower center", fontsize=10, ncol=2, framealpha=0.9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="phi")
    save(fig, "level-set-surface.png")


# --------------------------------------------------------------------------
# Figure 11.5 - Chan-Vese on the camera image
# --------------------------------------------------------------------------
def fig_chan_vese():
    img = data.camera().astype(float) / 255.0
    snaps = {}

    def store(ls, it=[0]):
        it[0] += 1
        if it[0] in (5, 25, 100):
            snaps[it[0]] = ls.copy()

    init = disk_level_set(img.shape, center=(200, 260), radius=60)
    morphological_chan_vese(img, num_iter=100, init_level_set=init, smoothing=3,
                            iter_callback=store)
    cv = chan_vese(img, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_num_iter=200,
                   dt=0.5, init_level_set="checkerboard", extended_output=True)
    seg, phi = cv[0], cv[1]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    ax = axes[0]
    ax.imshow(img, cmap="gray")
    ax.contour(init, levels=[0.5], colors="red", linestyles="--", linewidths=1.8)
    ax.plot([], [], "--r", label="initial disk")
    for (k, ls), col in zip(sorted(snaps.items()), ["#f4a261", "#2a9d8f", "#1d4ed8"]):
        ax.contour(ls, levels=[0.5], colors=col, linewidths=1.6)
        ax.plot([], [], color=col, label=f"iteration {k}")
    ax.set_title("(a) Morphological Chan-Vese", fontsize=10)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.axis("off")
    axes[1].imshow(seg, cmap="gray")
    axes[1].set_title("(b) PDE Chan-Vese: final regions", fontsize=10)
    axes[1].axis("off")
    im = axes[2].imshow(phi, cmap="coolwarm")
    axes[2].set_title("(c) Final level-set function phi", fontsize=10)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.03)
    fig.tight_layout()
    save(fig, "chan-vese-camera.png")


# --------------------------------------------------------------------------
# Figure 11.6 - topology: one snake vs. level sets on two objects
# --------------------------------------------------------------------------
def fig_topology():
    n = 200
    img = np.zeros((n, n))
    for cy, cx, r in [(100, 60, 35), (100, 145, 30)]:
        rr, cc = draw.disk((cy, cx), r, shape=img.shape)
        img[rr, cc] = 1.0
    rng = np.random.default_rng(1)
    noisy = np.clip(img + 0.2 * rng.standard_normal(img.shape), -0.5, 1.5)

    t = np.linspace(0, 2 * np.pi, 300)
    init = np.array([100 + 92 * np.sin(t), 100 + 92 * np.cos(t)]).T
    snake = active_contour(gaussian(noisy, sigma=3), init, alpha=0.1, beta=1.0,
                           gamma=0.01, max_num_iter=5000)

    gimg = inverse_gaussian_gradient(noisy, alpha=100, sigma=2)
    ls0 = disk_level_set(img.shape, center=(100, 100), radius=92)
    gac = morphological_geodesic_active_contour(gimg, num_iter=300, init_level_set=ls0,
                                                smoothing=2, balloon=-1, threshold=0.5)
    mcv = morphological_chan_vese(noisy, num_iter=60, init_level_set="checkerboard",
                                  smoothing=2)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    for ax in axes:
        ax.imshow(noisy, cmap="gray")
        ax.axis("off")
    axes[0].plot(init[:, 1], init[:, 0], "--r", lw=2, label="initial")
    axes[0].plot(snake[:, 1], snake[:, 0], color="#f4a261", lw=2.5, label="final snake")
    axes[0].set_title("(a) Snake: one curve stays one curve", fontsize=10)
    axes[0].legend(loc="lower center", fontsize=10, ncol=2, framealpha=0.9)
    axes[1].contour(ls0, levels=[0.5], colors="red", linestyles="--", linewidths=2)
    axes[1].contour(gac, levels=[0.5], colors="#f4a261", linewidths=2.5)
    axes[1].set_title("(b) Geodesic AC (level set): splits", fontsize=10)
    axes[2].contour(mcv, levels=[0.5], colors="#f4a261", linewidths=2.5)
    axes[2].set_title("(c) Chan-Vese, checkerboard start", fontsize=10)
    fig.tight_layout()
    save(fig, "topology-comparison.png")


if __name__ == "__main__":
    fig_snake_astronaut()
    fig_gvf()
    fig_balloon()
    fig_level_set_surface()
    fig_chan_vese()
    fig_topology()
