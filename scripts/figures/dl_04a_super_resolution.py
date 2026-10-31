"""Figures for DLCV 04A - Image Super-Resolution.

Run:  .venv/bin/python scripts/figures/dl_04a_super_resolution.py
Writes PNGs to src/assets/deep-learning/dl-04a-super-resolution/ only.
All inputs come from skimage.data or synthetic arrays; nothing is downloaded.
The linear regressor is fitted in closed form (ridge regression) in about a second on a CPU.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402
from skimage import data, img_as_float64  # noqa: E402
from skimage.color import rgb2gray  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-04a-super-resolution"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.titlesize": 10})
S = 4  # scale factor used throughout


def save(fig, name, quantize=True):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(path).convert("RGB")
    if quantize:
        im = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    im.save(path, optimize=True)
    w, h = Image.open(path).size
    print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB, {w}x{h})")


def save_array(arr, name, scale=3):
    """Save a [0,1] grayscale array as a PNG of exact pixel size (for ImageCompare)."""
    a = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
    a = np.kron(a, np.ones((scale, scale), dtype=np.uint8))  # nearest enlargement for display only
    path = OUT / name
    Image.fromarray(a, mode="L").save(path, optimize=True)
    print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB) {a.shape}")


def box(ax, xy, w, h, text, fc, fontsize=10, ec="#333333"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, p, q, color="#333333"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.2))


# ------------------------------------------------------------ shared SR helpers (same as the tutorial code)
def degrade_rgb(hr, s=S, blur=1.2, noise=0.01, rng=None):
    rng = rng or np.random.default_rng(0)
    x = ndi.gaussian_filter(hr, sigma=(blur, blur, 0))
    H, W, C = x.shape
    x = x.reshape(H // s, s, W // s, s, C).mean(axis=(1, 3))
    x = x + noise * rng.standard_normal(x.shape)
    return np.round(np.clip(x, 0, 1) * 255) / 255


def degrade_gray(x, blur, s=S):
    x = ndi.gaussian_filter(x, blur) if blur else x
    H, W = x.shape[0] // s * s, x.shape[1] // s * s
    return x[:H, :W].reshape(H // s, s, W // s, s).mean(axis=(1, 3))


def bicubic(lr, s=S):
    return cv2.resize(lr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


P = 7


def features(lr):
    f = sliding_window_view(np.pad(lr, P // 2, mode="reflect"), (P, P)).reshape(lr.size, P * P)
    return np.c_[f - f.mean(axis=1, keepdims=True), np.ones(lr.size)]


def to_blocks(r, s=S):
    h, w = r.shape[0] // s, r.shape[1] // s
    return r.reshape(h, s, w, s).transpose(0, 2, 1, 3).reshape(h * w, s * s)


def from_blocks(b, h, w, s=S):
    return b.reshape(h, w, s, s).transpose(0, 2, 1, 3).reshape(h * s, w * s)


def train(images, blur, lam=1e-3):
    A, B = [], []
    for im in images:
        im = im[: im.shape[0] // S * S, : im.shape[1] // S * S]
        lr = degrade_gray(im, blur)
        A.append(features(lr))
        B.append(to_blocks(im - bicubic(lr)))
    A, B = np.concatenate(A), np.concatenate(B)
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ B)


def super_resolve(lr, Wt):
    return bicubic(lr) + from_blocks(features(lr) @ Wt, *lr.shape)


def psnr(a, b, shave=S, peak=1.0):
    a, b = a[shave:-shave, shave:-shave], np.clip(b, 0, 1)[shave:-shave, shave:-shave]
    return 10 * np.log10(peak**2 / np.mean((a - b) ** 2))


TRAIN = [img_as_float64(im) for im in (data.camera(), data.moon(), data.coins(), data.page(),
                                        data.text(), data.brick(), data.grass(), data.gravel(),
                                        rgb2gray(data.coffee()), rgb2gray(data.chelsea()),
                                        rgb2gray(data.rocket()))]
TEST = img_as_float64(rgb2gray(data.astronaut()))


# ------------------------------------------------------------ 1. degradation pipeline
def degradation_pipeline():
    hr = img_as_float64(data.astronaut())[40:232, 160:352]  # 192 x 192, divisible by 4
    rng = np.random.default_rng(0)
    blurred = ndi.gaussian_filter(hr, sigma=(1.2, 1.2, 0))
    lr_clean = blurred.reshape(48, S, 48, S, 3).mean(axis=(1, 3))
    lr = np.round(np.clip(lr_clean + 0.03 * rng.standard_normal(lr_clean.shape), 0, 1) * 255) / 255
    u8 = (lr * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 30])
    lr_jpeg = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1] / 255.0
    near = lambda a: np.kron(a, np.ones((S, S, 1)))  # noqa: E731  (display only)
    panels = [(hr, "HR image x\n192 x 192"), (blurred, "1) blur  x * k\n(Gaussian, σ = 1.2 px)"),
              (near(lr_clean), "2) average 4 x 4, keep 1\n48 x 48 (shown enlarged)"),
              (near(lr), "3) noise + 8-bit\nσ = 0.03"), (near(lr_jpeg), "4) JPEG, quality 30\nthe LR input y")]
    fig, axes = plt.subplots(1, 5, figsize=(10.4, 2.75))
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    save(fig, "degradation-pipeline.png")


# ------------------------------------------------------------ 2. aliasing and spectra
def radial_spectrum(img):
    F = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean()))) ** 2
    h, w = img.shape
    yy, xx = np.indices(F.shape)
    r = np.hypot((yy - h // 2) / h, (xx - w // 2) / w)  # radius in cycles per pixel
    bins = np.linspace(0, 0.5, 65)
    idx = np.digitize(r.ravel(), bins)
    p = np.array([F.ravel()[idx == i].mean() for i in range(1, len(bins))])
    return 0.5 * (bins[1:] + bins[:-1]), p


def aliasing():
    n = 256
    yy, xx = np.indices((n, n))
    f0 = 0.30  # cycles per pixel, above the x4 LR Nyquist limit of 0.125
    grating = 0.5 + 0.5 * np.cos(2 * np.pi * f0 * (xx + yy) / np.sqrt(2))
    naive = grating[::S, ::S]
    pre = ndi.gaussian_filter(grating, 0.5 * S)[::S, ::S]
    blur_trained = train(TRAIN, 1.2)
    lr = degrade_gray(TEST, 1.2)
    f_hr, p_hr = radial_spectrum(TEST)
    _, p_bic = radial_spectrum(bicubic(lr))
    _, p_rr = radial_spectrum(np.clip(super_resolve(lr, blur_trained), 0, 1))

    fig = plt.figure(figsize=(10.4, 3.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.7])
    for k, (im, title) in enumerate([(grating, "HR grating\n0.30 cycles/pixel"),
                                     (naive, "keep every 4th pixel\n(false coarse stripes)"),
                                     (pre, "blur first, then keep\n(detail removed, no alias)")]):
        ax = fig.add_subplot(gs[k])
        ax.imshow(im[:64, :64] if k == 0 else im[:16, :16], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    ax = fig.add_subplot(gs[3])
    ax.semilogy(f_hr, p_hr, color="#222222", lw=2, label="HR image")
    ax.semilogy(f_hr, p_bic, color="#2c6fbb", lw=1.8, label="bicubic from LR")
    ax.semilogy(f_hr, p_rr, color="#c0504d", lw=1.8, ls="--", label="learned linear SR")
    ax.axvline(0.5 / S, color="#888888", ls=":", lw=1.5)
    ax.text(0.5 / S + 0.01, p_hr.max() * 0.05, "LR Nyquist\n1/(2s) = 0.125", fontsize=10, color="#555555")
    ax.set_xlabel("spatial frequency (cycles per HR pixel)")
    ax.set_ylabel("mean power")
    ax.set_title("Radial power spectrum, astronaut (gray), x4", fontsize=10)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "aliasing-spectrum.png")


# ------------------------------------------------------------ 3. interpolation kernels
def keys(x, a=-0.5):
    x = np.abs(x)
    return np.where(x <= 1, (a + 2) * x**3 - (a + 3) * x**2 + 1,
                    np.where(x < 2, a * x**3 - 5 * a * x**2 + 8 * a * x - 4 * a, 0.0))


def lanczos(x, a=3):
    return np.where(np.abs(x) < a, np.sinc(x) * np.sinc(x / a), 0.0)


def kernels():
    x = np.linspace(-3.5, 3.5, 1401)
    ks = {"nearest (box)": np.where(np.abs(x) < 0.5, 1.0, 0.0),
          "linear (tent)": np.clip(1 - np.abs(x), 0, None),
          "Keys cubic, a = -1/2": keys(x),
          "Lanczos-3": lanczos(x),
          "ideal sinc": np.sinc(x)}
    colors = ["#888888", "#2c6fbb", "#c0504d", "#7a5195", "#222222"]
    styles = ["-", "-", "-", "-", ":"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.6))
    dx = x[1] - x[0]
    nfft = 1 << 16
    freqs = np.fft.rfftfreq(nfft, d=dx)
    for (name, k), c, ls in zip(ks.items(), colors, styles):
        a1.plot(x, k, color=c, ls=ls, lw=1.8, label=name)
        if name == "ideal sinc":
            a2.plot([0, 0.5, 0.5, 1.5], [1, 1, 0, 0], color=c, ls=ls, lw=1.8, label=name)
        else:
            H = np.abs(np.fft.rfft(k, nfft)) * dx
            a2.plot(freqs, H, color=c, ls=ls, lw=1.8, label=name)
    a1.axhline(0, color="#bbbbbb", lw=0.8)
    a1.set_xlabel("distance from the output position (LR pixels)")
    a1.set_ylabel("weight")
    a1.set_title("Interpolation kernels", fontsize=10)
    a2.legend(fontsize=10, frameon=False, loc="upper right")
    a2.axvline(0.5, color="#bbbbbb", lw=0.8, ls="--")
    a2.set_xlim(0, 1.5)
    a2.set_ylim(-0.02, 1.15)
    a2.set_xlabel("frequency (cycles per LR pixel)")
    a2.set_ylabel("|frequency response|")
    a2.set_title("Frequency response (Nyquist at 0.5)", fontsize=10)
    fig.tight_layout()
    save(fig, "interpolation-kernels.png")


# ------------------------------------------------------------ 4. interpolation crops
def interp_crops():
    hr = img_as_float64(data.astronaut())
    lr = degrade_rgb(hr)
    y = lambda im: 16 + np.clip(im, 0, 1) @ np.array([65.481, 128.553, 24.966])  # noqa: E731
    r0, c0, sz = 60, 200, 128  # face and helmet region
    outs = [("HR (ground truth)", hr, None)]
    for name, flag in [("nearest", cv2.INTER_NEAREST), ("bilinear", cv2.INTER_LINEAR),
                       ("bicubic", cv2.INTER_CUBIC), ("Lanczos-4 (OpenCV)", cv2.INTER_LANCZOS4)]:
        sr = np.clip(cv2.resize(lr, (512, 512), interpolation=flag), 0, 1)
        a, b = y(hr)[S:-S, S:-S], y(sr)[S:-S, S:-S]
        outs.append((name, sr, 10 * np.log10(255.0**2 / np.mean((a - b) ** 2))))
    fig, axes = plt.subplots(1, 5, figsize=(10.4, 2.65))
    for ax, (name, im, p) in zip(axes, outs):
        ax.imshow(im[r0:r0 + sz, c0:c0 + sz], interpolation="nearest")
        ax.set_title(name if p is None else f"{name}\nPSNR-Y {p:.2f} dB", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    save(fig, "interpolation-crops.png")


# ------------------------------------------------------------ 5. pixel shuffle diagram
def pixel_shuffle_diagram():
    s, h, w = 2, 3, 3
    cols = ["#f4a6a6", "#a6c8f4", "#b5e3a1", "#f7d78b"]
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(-0.3, 5.0)
    ax.set_aspect("equal")
    ax.axis("off")
    cell = 0.5
    for k in range(s * s):
        ox, oy = 0.3 + (k % 2) * 2.2, 3.3 - (k // 2) * 2.45
        for r in range(h):
            for c in range(w):
                ax.add_patch(Rectangle((ox + c * cell, oy - r * cell), cell, cell, fc=cols[k], ec="#333333", lw=0.8))
        ax.add_patch(Rectangle((ox, oy), cell, cell, fill=False, ec="black", lw=2.2))
        ax.text(ox, oy + 0.6, f"channel {k}\n(i, j) = ({k // s}, {k % s})", fontsize=10)
    ax.text(0.3, 4.6, "LR features T: (C·s², H, W) = (4, 3, 3)", fontsize=10, weight="bold")
    arrow(ax, (4.9, 2.2), (6.6, 2.2))
    ax.text(5.75, 2.5, "pixel shuffle", ha="center", fontsize=10)
    ax.text(5.75, 1.75, "no weights,\njust re-indexing", ha="center", fontsize=10, va="top")
    ox, oy = 7.0, 3.5
    for R in range(h * s):
        for Cc in range(w * s):
            k = s * (R % s) + (Cc % s)
            ax.add_patch(Rectangle((ox + Cc * cell, oy - R * cell), cell, cell, fc=cols[k], ec="#333333", lw=0.8))
            ax.text(ox + Cc * cell + cell / 2, oy - R * cell + cell / 2, str(k), ha="center", va="center", fontsize=10)
    ax.add_patch(Rectangle((ox, oy - cell), 2 * cell, 2 * cell, fill=False, ec="black", lw=2.2))
    ax.text(ox, 4.6, "HR output: (C, sH, sW) = (1, 6, 6)", fontsize=10, weight="bold")
    ax.text(ox + 3.2, 2.6, "output pixel (s·h + i, s·w + j)\ncomes from channel s·i + j\nat LR position (h, w)",
            fontsize=10, va="center")
    ax.text(ox + 3.2, 1.2, "bold squares: LR position (0, 0)\nin every channel → one s x s block", fontsize=10, va="center")
    save(fig, "pixel-shuffle.png")


# ------------------------------------------------------------ 6. upsampling strategies
def upsampling_strategies():
    fig, ax = plt.subplots(figsize=(10.4, 3.6))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    rows = [(2.55, "Pre-upsampling\n(SRCNN, VDSR)",
             [("LR", "#eeeeee", 0.55), ("bicubic\nx s", "#fcf3cf", 0.9), ("conv layers\nat HR size", "#f6dbe0", 2.6),
              ("HR", "#eeeeee", 0.55)]),
            (0.75, "Post-upsampling\n(ESPCN, EDSR,\nRCAN, SwinIR)",
             [("LR", "#eeeeee", 0.55), ("conv / attention\nat LR size", "#dbe9f6", 2.6),
              ("conv to\nC·s² ch.", "#dbe9f6", 0.95), ("pixel\nshuffle", "#dff0d8", 0.85), ("HR", "#eeeeee", 0.55)])]
    for y0, label, blocks in rows:
        ax.text(0.05, y0 + 0.35, label, fontsize=10, va="center")
        x0 = 2.1
        for k, (txt, fc, wd) in enumerate(blocks):
            hgt = 0.7 if txt in ("LR", "HR") else 0.8
            box(ax, (x0, y0 + 0.35 - hgt / 2), wd, hgt, txt, fc, 10)
            if k < len(blocks) - 1:
                arrow(ax, (x0 + wd, y0 + 0.35), (x0 + wd + 0.3, y0 + 0.35))
            x0 += wd + 0.3
    ax.text(7.7, 2.9, "every layer costs\ns² times more", fontsize=10, color="#c0504d", va="center")
    ax.text(9.2, 1.1, "layers run at\nLR size; resize\nonly at the end", fontsize=10, color="#2c6fbb", va="center")
    save(fig, "upsampling-strategies.png")


# ------------------------------------------------------------ 7. learned linear regressor
def regressor_results():
    W_sharp, W_blur = train(TRAIN, 0.0), train(TRAIN, 1.2)
    r0, c0, sz = 40, 160, 160  # crop in HR pixels (face)
    lr_blur, lr_sharp = degrade_gray(TEST, 1.2), degrade_gray(TEST, 0.0)
    panels = [("bicubic", bicubic(lr_blur), TEST),
              ("linear SR, trained blurry\n(matched)", super_resolve(lr_blur, W_blur), TEST),
              ("HR (ground truth)", TEST, None),
              ("bicubic, sharp input", bicubic(lr_sharp), TEST),
              ("linear SR, trained blurry\n(mismatched: too sharp)", super_resolve(lr_sharp, W_blur), TEST),
              ("linear SR, trained sharp\n(matched)", super_resolve(lr_sharp, W_sharp), TEST)]
    fig = plt.figure(figsize=(10.4, 7.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.05])
    for k, (name, im, ref) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        ax.imshow(np.clip(im, 0, 1)[r0:r0 + sz, c0:c0 + sz], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name if ref is None else f"{name}\nPSNR {psnr(ref, im):.2f} dB", fontsize=10)
        ax.axis("off")
    # the 16 learned sub-pixel filters (blurry-trained model)
    ax = fig.add_subplot(gs[:, 3])
    filt = W_blur[:-1].T.reshape(S, S, P, P)
    pad = 1
    mosaic = np.full((S * (P + pad) - pad, S * (P + pad) - pad), np.nan)
    for i in range(S):
        for j in range(S):
            mosaic[i * (P + pad):i * (P + pad) + P, j * (P + pad):j * (P + pad) + P] = filt[i, j]
    v = np.nanmax(np.abs(mosaic))
    im = ax.imshow(mosaic, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    ax.set_title("16 learned 7 x 7 filters\n(one per sub-pixel offset (i, j))", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.5)
    fig.text(0.02, 0.02, "Top row: blurry test input (σ = 1.2). Bottom row: sharp test input (box average only). "
             "PSNR on the full 512 x 512 test image, 4-pixel border shaved.", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "linear-regressor.png", quantize=False)
    # same-size pair for the ImageCompare slider
    crop = (slice(48, 176), slice(176, 304))
    save_array(np.clip(bicubic(lr_blur), 0, 1)[crop], "compare-bicubic.png")
    save_array(np.clip(super_resolve(lr_blur, W_blur), 0, 1)[crop], "compare-linear-sr.png")


# ------------------------------------------------------------ 8. posterior mean vs sample toy
def posterior_toy():
    rng = np.random.default_rng(0)
    n, s, sigma = 64, 4, 0.2
    cand = np.arange(16, 48)
    X = (np.arange(n)[None, :] >= cand[:, None]).astype(float)
    DX = X.reshape(len(cand), n // s, s).mean(axis=2)
    mm, sp, tr = [], [], []
    ex = None
    for trial in range(2000):
        t = rng.integers(len(cand))
        y = DX[t] + sigma * rng.standard_normal(n // s)
        logp = -((y - DX) ** 2).sum(axis=1) / (2 * sigma**2)
        post = np.exp(logp - logp.max())
        post /= post.sum()
        x_mmse = post @ X
        x_samp = X[rng.choice(len(cand), p=post)]
        mm.append(x_mmse); sp.append(x_samp); tr.append(X[t])
        if ex is None and post.max() < 0.5:
            ex = (X[t], y, x_mmse, x_samp, post)
    mm, sp, tr = map(np.array, (mm, sp, tr))
    alphas = np.linspace(0, 1, 11)
    mse, dist = [], []
    for a in alphas:
        xa = mm + a * (sp - mm)
        mse.append(np.mean((xa - tr) ** 2))
        dist.append(wasserstein_distance(np.diff(xa, axis=1).max(axis=1), np.diff(tr, axis=1).max(axis=1)))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.7))
    x_true, y, x_mmse, x_samp, post = ex
    idx = np.arange(n)
    a1.step(idx, x_true, where="mid", color="#222222", lw=2, label="true HR edge")
    a1.step(idx, np.repeat(y, s), where="mid", color="#999999", lw=1.2, label="noisy LR (shown at HR size)")
    a1.plot(idx, x_mmse, color="#2c6fbb", lw=2, label="posterior mean (MMSE)")
    a1.step(idx, x_samp, where="mid", color="#c0504d", lw=1.6, ls="--", label="one posterior sample")
    a1.set_xlim(8, 56)
    a1.set_xlabel("pixel index")
    a1.set_ylabel("intensity")
    a1.set_title("One ambiguous example", fontsize=10)
    a1.legend(fontsize=10, frameon=False, loc="upper left")
    a2.plot(mse, dist, "o-", color="#7a5195", lw=1.8)
    a2.annotate("posterior mean (α = 0)", (mse[0], dist[0]), xytext=(14, -6), textcoords="offset points", fontsize=10)
    a2.annotate("posterior sample (α = 1)", (mse[-1], dist[-1]), xytext=(-165, 6), textcoords="offset points",
                fontsize=10)
    a2.set_xlabel("distortion: MSE to the true edge")
    a2.set_ylabel("realism gap: W1 distance of\nedge-height distributions")
    a2.set_title("Blends x_α = mean + α (sample − mean)", fontsize=10)
    fig.tight_layout()
    save(fig, "perception-distortion-toy.png")
    print(f"toy: MSE mean {mse[0]:.4f}, sample {mse[-1]:.4f}, W1 mean {dist[0]:.3f}, sample {dist[-1]:.3f}")


if __name__ == "__main__":
    degradation_pipeline()
    aliasing()
    kernels()
    interp_crops()
    pixel_shuffle_diagram()
    upsampling_strategies()
    regressor_results()
    posterior_toy()
