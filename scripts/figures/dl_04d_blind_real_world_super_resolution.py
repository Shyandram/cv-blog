"""Figures for DLCV 04D (blind and real-world super-resolution).

Writes PNGs to src/assets/deep-learning/dl-04d-blind-real-world-super-resolution/.
Data: skimage.data images and synthetic signals only. Run from the repository root:
    .venv/bin/python scripts/figures/dl_04d_blind_real_world_super_resolution.py
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse.linalg import LinearOperator, cg
from scipy.special import j1
from skimage import data, img_as_float32, img_as_float64
from skimage.color import rgb2gray

OUT = Path(__file__).resolve().parents[2] / "src/assets/deep-learning/dl-04d-blind-real-world-super-resolution"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- 1. kernel mismatch (non-blind SR with a wrong kernel)
S = 2
X = img_as_float64(data.camera())[128:384, 128:384]
H, W = X.shape


def blur(z, sigma):
    return ndi.gaussian_filter(z, sigma, mode="wrap")


def A(z, sigma):
    return blur(z, sigma)[::S, ::S]


def At(r, sigma):
    up = np.zeros((H, W))
    up[::S, ::S] = r
    return blur(up, sigma)


def nonblind_sr(y, sigma, lam=2e-3):
    op = LinearOperator((H * W, H * W), dtype=float,
                        matvec=lambda v: (At(A(v.reshape(H, W), sigma), sigma)
                                          - lam * ndi.laplace(v.reshape(H, W), mode="wrap")).ravel())
    sol, _ = cg(op, At(y, sigma).ravel(), maxiter=200)
    return sol.reshape(H, W)


def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - np.clip(b, 0, 1)) ** 2))


def fig_kernel_mismatch():
    rng = np.random.default_rng(0)
    assumed = np.round(np.arange(0.6, 3.01, 0.2), 1)
    curves = {}
    outs = {}
    for true in [1.0, 1.6, 2.2]:
        y = A(X, true) + 0.003 * rng.standard_normal((H // S, W // S))
        curves[true] = []
        for a in assumed:
            sr = nonblind_sr(y, a)
            curves[true].append(psnr(X, sr))
            if true == 1.6 and a in (0.6, 1.6, 2.8):
                outs[a] = sr
        if true == 1.6:
            y16 = y
    fig = plt.figure(figsize=(10.6, 5.2))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.6], hspace=0.32, wspace=0.18)
    crop = (slice(40, 140), slice(70, 170))
    lr_up = np.kron(y16, np.ones((S, S)))
    panels = [("HR (truth)", X), ("LR input, true σ = 1.6", lr_up),
              ("assumed σ = 0.6\n(too narrow)", outs[0.6]), ("assumed σ = 1.6\n(correct)", outs[1.6]),
              ("assumed σ = 2.8\n(too wide)", outs[2.8])]
    pos = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    for (title, img), (r, c) in zip(panels, pos):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(np.clip(img[crop], 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        t = title if "assumed" not in title else title + f"\n{psnr(X, img):.2f} dB"
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.text(0.0, 0.5, "Non-blind ×2 SR\n(Tikhonov + CG)\nwith a Gaussian\nkernel of the\nassumed width", fontsize=10, va="center")
    ax = fig.add_subplot(gs[:, 3])
    ax.set_position(ax.get_position().translated(0.04, 0))
    for (true, ps), col in zip(curves.items(), ["#1f77b4", "#2ca02c", "#d62728"]):
        ax.plot(assumed, ps, "-o", ms=3, color=col, label=f"true σ = {true}")
        ax.axvline(true, color=col, ls=":", lw=1)
    ax.set_xlabel("assumed kernel width σ")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Each curve peaks at the true width")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    save(fig, "kernel-mismatch.png")


# ---------------------------------------------------------------- 2. randomized second-order degradation pipeline
def gaussian_kernel(rng, size, smax):
    s1, s2 = rng.uniform(0.2, smax, 2)
    if rng.random() < 0.5:
        s2 = s1
    th = rng.uniform(0, np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Sg = R @ np.diag([s1 ** 2, s2 ** 2]) @ R.T
    ax = np.arange(size) - size // 2
    C = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1)
    k = np.exp(-0.5 * np.einsum("...i,ij,...j", C, np.linalg.inv(Sg), C))
    return k / k.sum(), f"gauss({s1:.1f},{s2:.1f})"


def sinc_kernel(size, wc):
    ax = np.arange(size) - size // 2
    r = np.hypot(*np.meshgrid(ax, ax, indexing="ij"))
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(r == 0, wc ** 2 / (4 * np.pi), wc * j1(wc * r) / (2 * np.pi * r))
    return k / k.sum()


def jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", (np.clip(img, 0, 1) * 255).round().astype(np.uint8)[..., ::-1],
                           [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255


def one_round(img, rng, log, smax, nmax):
    size = int(rng.choice(np.arange(7, 22, 2)))
    if rng.random() < 0.1:
        k, name = sinc_kernel(size, rng.uniform(np.pi / 3, np.pi)), "sinc"
    else:
        k, name = gaussian_kernel(rng, size, smax)
    img = np.stack([ndi.convolve(img[..., c], k, mode="reflect") for c in range(3)], -1)
    f = rng.uniform(0.3, 1.2)
    interp = rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    img = cv2.resize(img, None, fx=f, fy=f, interpolation=interp)
    if rng.random() < 0.5:
        sig = rng.uniform(1, nmax) / 255
        shape = img.shape[:2] + ((1,) if rng.random() < 0.4 else (3,))
        img = img + sig * rng.standard_normal(shape)
        noise = f"gauss{sig * 255:.0f}"
    else:
        peak = 10 ** rng.uniform(1.5, 3)
        img = rng.poisson(np.clip(img, 0, 1) * peak) / peak
        noise = f"poisson{peak:.0f}"
    q = rng.uniform(30, 95)
    img = jpeg(img, q)
    log.append(f"{name} {noise} q{q:.0f}")
    return img


def degrade(hr, rng, scale=4):
    log = []
    img = one_round(hr, rng, log, smax=3.0, nmax=30)
    if rng.random() < 0.8:
        img = one_round(img, rng, log, smax=1.5, nmax=25)
    h, w = hr.shape[0] // scale, hr.shape[1] // scale
    def resize_sinc(z):
        z = cv2.resize(z, (w, h), interpolation=cv2.INTER_AREA)
        k = sinc_kernel(15, rng.uniform(np.pi / 3, np.pi))  # same kernel for all color channels
        return np.stack([ndi.convolve(z[..., c], k, mode="reflect") for c in range(3)], -1)
    final = [lambda z: jpeg(z, rng.uniform(30, 95)), resize_sinc]
    if rng.random() < 0.5:
        final = final[::-1]
    for op in final:
        img = op(img)
    return np.clip(img, 0, 1), log


def fig_degradation_samples():
    hr = img_as_float32(data.astronaut())
    crop = (slice(20, 52), slice(48, 80))       # LR-pixel crop of the face (x4 -> 128 x 128 HR pixels)
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.9))
    axes = axes.ravel()
    axes[0].imshow(hr[80:208, 192:320])
    axes[0].set_title("HR crop", fontsize=10)
    for i, seed in enumerate(range(7), start=1):
        lr, log = degrade(hr, np.random.default_rng(seed))
        axes[i].imshow(lr[crop], interpolation="nearest")
        rounds = "2 rounds" if len(log) == 2 else "1 round"
        first = log[0].split()[1]
        axes[i].set_title(f"seed {seed}: {rounds}\n1st noise: {first}", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Seven training inputs from one HR image: random blur, resize, noise, JPEG and sinc (×4, shown enlarged)", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.86, hspace=0.32, wspace=0.05)
    save(fig, "degradation-samples.png")


# ---------------------------------------------------------------- 3. sinc filter: kernel and ringing on an edge
def fig_sinc_ringing():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    for ax, wc in zip(axes[:2], [np.pi / 3, 2 * np.pi / 3]):
        k = sinc_kernel(21, wc)
        lim = np.abs(k).max()
        ax.imshow(k, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"2-D sinc kernel, ωc = {wc / np.pi:.2f}π", fontsize=10)
        ax.axis("off")
    edge = np.zeros((64, 64))
    edge[:, 32:] = 1.0
    xs = np.arange(16, 48)
    axes[2].plot(xs, edge[32, xs], "k-", lw=1.5, label="sharp edge")
    axes[2].plot(xs, ndi.gaussian_filter(edge, 1.5, mode="reflect")[32, xs], "-", color="#1f77b4", label="Gaussian blur σ = 1.5")
    sincblur = ndi.convolve(edge, sinc_kernel(21, np.pi / 3), mode="reflect")
    axes[2].plot(xs, sincblur[32, xs], "-", color="#d62728", label="sinc filter ωc = π/3")
    axes[2].axhline(1, color="gray", lw=0.6, ls=":")
    axes[2].axhline(0, color="gray", lw=0.6, ls=":")
    axes[2].set_title("Profile across a step edge", fontsize=10)
    axes[2].set_xlabel("pixel")
    axes[2].legend(fontsize=10, loc="center right")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "sinc-ringing.png")
    print("sinc overshoot above 1:", round(float(sincblur.max() - 1), 3), "undershoot below 0:", round(float(-sincblur.min()), 3))


# ---------------------------------------------------------------- 4. metric disagreement
def crispness(img):
    return np.mean(np.hypot(ndi.sobel(img, 0), ndi.sobel(img, 1)))


def fig_metric_disagreement():
    hr = img_as_float64(rgb2gray(data.astronaut()))
    rng = np.random.default_rng(0)
    smooth = ndi.gaussian_filter(hr, 1.5)
    cands = [("smooth", smooth),
             ("unsharp-masked ×2", smooth + 2.0 * (smooth - ndi.gaussian_filter(smooth, 2))),
             ("smooth + fake texture", smooth + 0.04 * rng.standard_normal(hr.shape))]
    crop = (slice(90, 220), slice(170, 300))
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.2))
    axes[0].imshow(hr[crop], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"ground truth\ncrispness {crispness(hr):.3f}", fontsize=10)
    for ax, (name, img) in zip(axes[1:], cands):
        img = np.clip(img, 0, 1)
        ax.imshow(img[crop], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{name}\nPSNR {psnr(hr, img):.2f} dB, crispness {crispness(img):.3f}", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    save(fig, "metric-disagreement.png")


if __name__ == "__main__":
    fig_kernel_mismatch()
    fig_degradation_samples()
    fig_sinc_ringing()
    fig_metric_disagreement()
