"""Figures for DLCV 04C - Generative Image Restoration.

Run:  .venv/bin/python scripts/figures/dl_04c_generative_restoration.py
Writes PNGs to src/assets/deep-learning/dl-04c-generative-restoration/ only.
Everything is synthetic (1-D step edges, a 2-D Gaussian mixture); nothing is downloaded.
Uses NumPy/SciPy/Matplotlib only and runs in a few seconds on a CPU.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-04c-generative-restoration"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.facecolor": "white"})

C_TRUE, C_MEAN, C_SAMP, C_LS, C_OBS = "#222222", "#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(path).convert("RGB")
    im = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    im.save(path, optimize=True)
    w, h = Image.open(path).size
    print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB, {w}x{h} px)")


# ---------------------------------------------------------------- 1-D step-edge toy (snippet 1)
n, factor, sigma = 64, 8, 0.2
positions = np.arange(8, 56)
X = (np.arange(n)[None, :] >= positions[:, None]).astype(float)


def A(x):
    return gaussian_filter1d(x, 3.0, axis=-1, mode="nearest")[..., ::factor]


AX = A(X)


def posterior(y):
    logp = -np.sum((y - AX) ** 2, axis=1) / (2 * sigma**2)
    p = np.exp(logp - logp.max())
    return p / p.sum()


def fig_posterior_mean_vs_samples():
    rng = np.random.default_rng(11)
    x = X[20]
    y = A(x) + sigma * rng.standard_normal(n // factor)
    p = posterior(y)
    mean = p @ X
    samples = X[rng.choice(len(positions), size=3, p=p)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    pix = np.arange(n)
    ax1.step(pix, x, where="mid", color=C_TRUE, lw=2.4, label="clean signal x (unknown)")
    ax1.plot(pix[::factor], y, "s", color=C_OBS, ms=7, label="measurement y (8 noisy values)")
    for i, s in enumerate(samples):
        ax1.step(pix, s + 0.02 * (i - 1), where="mid", color=C_SAMP, lw=1.3, alpha=0.9,
                 label="posterior samples (3)" if i == 0 else None)
    ax1.plot(pix, mean, color=C_MEAN, lw=2.4, label="posterior mean (MMSE)")
    ax1.set_xlabel("pixel")
    ax1.set_ylabel("intensity")
    ax1.set_ylim(-0.6, 1.5)
    ax1.set_title("(a) The mean is a ramp; every sample is a sharp edge")
    ax1.legend(loc="upper left", fontsize=9, frameon=False)

    ax2.bar(positions, p, color=C_SAMP, alpha=0.8)
    ax2.axvline(np.argmax(x), color=C_TRUE, ls="--", lw=1.5, label="true edge position")
    ax2.set_xlim(12, 44)
    ax2.set_xlabel("edge position k")
    ax2.set_ylabel("p(k | y)")
    ax2.set_title("(b) Exact posterior over edge positions")
    ax2.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, "posterior-mean-vs-samples.png")


def fig_perception_distortion():
    rng = np.random.default_rng(0)
    truth, mean, post, prior = [], [], [], []
    for _ in range(2000):
        x = X[rng.integers(len(positions))]
        y = A(x) + sigma * rng.standard_normal(n // factor)
        p = posterior(y)
        truth.append(x)
        mean.append(p @ X)
        post.append(X[rng.choice(len(positions), p=p)])
        prior.append(X[rng.integers(len(positions))])
    truth, mean, post, prior = map(np.array, (truth, mean, post, prior))

    def point(est):
        return np.mean((est - truth) ** 2), wasserstein_distance(est.ravel(), truth.ravel())

    lams = np.linspace(0, 1, 11)
    blend = np.array([point((1 - lam) * mean + lam * post) for lam in lams])
    pm, ps, pp = point(mean), point(post), point(prior)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(blend[:, 0], blend[:, 1], "o-", color="#7f7f7f", ms=4, lw=1.2,
            label=r"blends $(1-\lambda)\,$mean $+\,\lambda\,$sample, $\lambda = 0, 0.1, \dots, 1$")
    ax.plot(*pm, "o", color=C_MEAN, ms=11, label="posterior mean (lowest MSE)")
    ax.plot(*ps, "D", color=C_SAMP, ms=10, label="posterior sample (about 2x the MSE)")
    ax.plot(*pp, "s", color=C_LS, ms=10, label="prior sample (ignores y)")
    ax.annotate("lowest distortion,\nbut not realistic", pm, xytext=(pm[0] + 0.012, pm[1] + 0.001),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.annotate("realistic and\nconsistent with y", ps, xytext=(ps[0] + 0.02, ps[1] + 0.006),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.set_xscale("log")
    ax.set_xlabel("distortion: MSE to the clean signal (log scale)")
    ax.set_ylabel("perception gap: W1 between pixel-value\ndistributions of outputs and real signals")
    ax.set_title("Toy perception-distortion plane (measured on 2000 test signals)")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_ylim(-0.001, 0.03)
    fig.tight_layout()
    save(fig, "perception-distortion-toy.png")
    print("  points (MSE, W1): mean", np.round(pm, 4), "sample", np.round(ps, 4), "prior", np.round(pp, 4))


# ---------------------------------------------------------------- 2-D mixture diffusion (snippets 2-3)
mu = np.array([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])
w = np.array([0.1, 0.2, 0.3, 0.4])
s2 = 0.3**2
T = 200
beta = np.linspace(1e-4, 0.05, T)
alpha = 1 - beta
abar = np.cumprod(alpha)


def score(x, ab):
    m, v = np.sqrt(ab) * mu, ab * s2 + (1 - ab)
    d2 = ((x[:, None, :] - m[None]) ** 2).sum(-1)
    logr = np.log(w) - d2 / (2 * v)
    r = np.exp(logr - logr.max(1, keepdims=True))
    r /= r.sum(1, keepdims=True)
    return (r[:, :, None] * (m[None] - x[:, None, :])).sum(1) / v, r


def x0_hat(x, ab):
    return (x + (1 - ab) * score(x, ab)[0]) / np.sqrt(ab)


def ddpm_step(x, t, x0, rng):
    ab_prev = abar[t - 1] if t > 0 else 1.0
    mean = (np.sqrt(ab_prev) * beta[t] * x0 + np.sqrt(alpha[t]) * (1 - ab_prev) * x) / (1 - abar[t])
    var = beta[t] * (1 - ab_prev) / (1 - abar[t])
    return mean + np.sqrt(var) * rng.standard_normal(x.shape)


def density(grid, ab):
    m, v = np.sqrt(ab) * mu, ab * s2 + (1 - ab)
    d2 = ((grid[:, None, :] - m[None]) ** 2).sum(-1)
    return (w * np.exp(-d2 / (2 * v)) / (2 * np.pi * v)).sum(1)


def fig_diffusion_trajectories():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1500, 2))
    keep = {T - 1: x.copy()}
    traj = [x[:25].copy()]
    for t in reversed(range(T)):
        x = ddpm_step(x, t, x0_hat(x, abar[t]), rng)
        traj.append(x[:25].copy())
        if t in (120, 40):
            keep[t] = x.copy()
    keep[-1] = x.copy()
    traj = np.array(traj)

    g = np.linspace(-4, 4, 161)
    G = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.1))
    panels = [(T - 1, "start: pure noise (t = T)"), (120, "t = 120"), (40, "t = 40"), (-1, "end: samples (t = 0)")]
    for ax, (t, title) in zip(axes, panels):
        ab = abar[t] if t >= 0 else 1.0
        ax.contour(g, g, density(G, ab).reshape(161, 161), levels=8, cmap="Greys", linewidths=0.8)
        ax.scatter(keep[t][:, 0], keep[t][:, 1], s=2, color=C_SAMP, alpha=0.35)
        ax.set_title(title)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([-3, 0, 3])
    for k in range(traj.shape[1]):
        axes[3].plot(traj[:, k, 0], traj[:, k, 1], color=C_MEAN, lw=0.6, alpha=0.6)
    axes[3].plot([], [], color=C_MEAN, lw=1, label="25 reverse paths")
    axes[3].legend(loc="lower left", fontsize=8, frameon=True)
    axes[0].set_ylabel("contours: noised density $p_t$")
    fig.suptitle("Reverse diffusion with the exact score of a four-blob mixture (weights 0.1, 0.2, 0.3, 0.4)",
                 y=1.02)
    fig.tight_layout()
    save(fig, "diffusion-trajectories.png")


def fig_dps_toy():
    rng = np.random.default_rng(3)
    Am = np.array([[1.0, 1.0]]) / np.sqrt(2)
    sig_y = 0.2
    x_true = np.array([1.8, -2.1])
    y = Am @ x_true + sig_y * rng.standard_normal(1)
    S = (Am @ Am.T)[0, 0] * s2 + sig_y**2
    logpost = np.log(w) - (y - mu @ Am.T)[:, 0] ** 2 / (2 * S)
    post_w = np.exp(logpost - logpost.max())
    post_w /= post_w.sum()
    gain = s2 * Am.T / S
    post_mu = mu + (y - mu @ Am.T) @ gain.T
    post_cov = s2 * (np.eye(2) - gain @ Am)
    k = rng.choice(4, size=2000, p=post_w)
    exact = post_mu[k] + rng.multivariate_normal(np.zeros(2), post_cov, size=2000)
    mmse = post_w @ post_mu
    least_sq = (np.linalg.pinv(Am) @ y).ravel()

    def data_grad(x, ab, h=1e-4):
        J = np.stack([(x0_hat(x + h * e, ab) - x0_hat(x - h * e, ab)) / (2 * h) for e in np.eye(2)], -1)
        resid = y - x0_hat(x, ab) @ Am.T
        return -2 * np.einsum("nij,ik,nk->nj", J, Am.T, resid), np.abs(resid)

    x = rng.standard_normal((2000, 2))
    for t in reversed(range(T)):
        x_prior = ddpm_step(x, t, x0_hat(x, abar[t]), rng)
        gr, res = data_grad(x, abar[t])
        x = x_prior - 0.1 / (res + 1e-8) * gr

    g = np.linspace(-4, 4, 161)
    G = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, pts, title, col in [(axes[0], exact, "(a) Exact posterior samples", "#9467bd"),
                                (axes[1], x, "(b) DPS-style samples (prior step + data step)", C_SAMP)]:
        ax.contour(g, g, density(G, 1.0).reshape(161, 161), levels=6, cmap="Greys", linewidths=0.8)
        xs = np.linspace(-4, 4, 2)
        c = y[0] * np.sqrt(2)
        ax.fill_between(xs, c - xs - 2 * sig_y * np.sqrt(2), c - xs + 2 * sig_y * np.sqrt(2),
                        color=C_OBS, alpha=0.15, label=r"$|y - Ax| \leq 2\sigma_y$")
        ax.plot(xs, c - xs, color=C_OBS, lw=1.2)
        ax.scatter(pts[:, 0], pts[:, 1], s=3, color=col, alpha=0.4, label="samples")
        ax.plot(*least_sq, "s", color=C_LS, ms=9, mec="k", label="least squares (min. norm)")
        ax.plot(*mmse, "o", color=C_MEAN, ms=10, mec="k", label="posterior mean (MMSE)")
        ax.plot(*x_true, "*", color="gold", ms=16, mec="k", label="truth")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    save(fig, "dps-toy.png")


def fig_guidance_loop():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(xc, yc, wd, ht, text, fc):
        ax.add_patch(FancyBboxPatch((xc - wd / 2, yc - ht / 2), wd, ht, boxstyle="round,pad=0.08",
                                    fc=fc, ec="#333333", lw=1.2))
        ax.text(xc, yc, text, ha="center", va="center", fontsize=10)

    def arrow(p, q, text=None, off=(0, 0.18), rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#333333",
                                     connectionstyle=f"arc3,rad={rad}"))
        if text:
            ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha="center", fontsize=9)

    box(0.9, 2.6, 1.4, 0.8, "noisy state\n$x_t$", "#f2f2f2")
    box(3.3, 2.6, 2.3, 1.1, "PRIOR STEP\ndenoise: $\\hat x_0 = E[x_0 \\mid x_t]$\nthen step to level $t-1$",
        "#dbe9f6")
    box(6.3, 2.6, 2.4, 1.1, "DATA-CONSISTENCY STEP\nnudge so that $A\\hat x_0 \\approx y$\n(gradient or projection)",
        "#e2f2dd")
    box(9.4, 2.6, 1.5, 0.8, "less noisy\n$x_{t-1}$", "#f2f2f2")
    box(6.3, 0.75, 2.4, 0.75, "measurement $y$ and\ndegradation $A$ (known)", "#fff2cc")
    arrow((1.65, 2.6), (2.1, 2.6))
    arrow((4.5, 2.6), (5.05, 2.6))
    arrow((7.55, 2.6), (8.6, 2.6))
    arrow((6.3, 1.15), (6.3, 2.0))
    ax.add_patch(FancyArrowPatch((9.4, 3.05), (0.9, 3.05), arrowstyle="-|>", mutation_scale=14, lw=1.2,
                                 color="#888888", connectionstyle="arc3,rad=0.18", ls="--"))
    ax.text(5.15, 4.15, "repeat for t = T, T-1, ..., 1   (each pass costs one or more network calls)",
            ha="center", fontsize=9.5, color="#555555")
    ax.text(3.3, 1.5, "learned once, from clean\nimages only (task-agnostic)", ha="center", fontsize=9,
            color="#1f4e79")
    ax.text(1.0, 0.75, "DPS: gradient through $\\hat x_0$\nDDNM: replace the range of $A$\n"
            "$\\Pi$GDM: pseudoinverse guidance", ha="left", va="center", fontsize=9, color="#2d572c")
    save(fig, "guidance-loop.png")


if __name__ == "__main__":
    fig_posterior_mean_vs_samples()
    fig_perception_distortion()
    fig_diffusion_trajectories()
    fig_dps_toy()
    fig_guidance_loop()
