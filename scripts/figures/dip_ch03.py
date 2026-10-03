"""Figures for DIP Chapter 3 - Intensity Transformations and Spatial Filtering.

Run from the repo root:
    .venv/bin/python scripts/figures/dip_ch03.py
Writes PNGs to src/assets/dip/ch03/. All inputs are skimage.data images or synthetic arrays.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from skimage import color, data, exposure, img_as_float, img_as_ubyte

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "dip" / "ch03"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
RNG = np.random.default_rng(3)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    # Shrink the file: palette PNG (gray figures stay visually identical).
    im = Image.open(path).convert("RGB")
    if im.width > 1600:
        im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
    im.quantize(colors=128, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    print(f"{path.relative_to(OUT.parents[3])}: {path.stat().st_size / 1024:.0f} KB")


def show(ax, img, title, vmin=0, vmax=1):
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")


cam = img_as_float(data.camera())  # 512x512, [0, 1]


# --- Figure 3.1: transformation curves -------------------------------------
def fig_curves():
    r = np.linspace(0, 1, 512)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
    ax = axs[0]
    ax.plot(r, r, "k--", label="identity")
    ax.plot(r, 1 - r, label="negative")
    ax.plot(r, np.log1p(255 * r) / np.log(256), label="log (c=1/log 256)")
    ax.plot(r, (np.exp(r * np.log(256)) - 1) / 255, label="inverse log")
    ax.set_title("Basic transformations")
    ax.legend(fontsize=9, loc="lower right")
    ax = axs[1]
    for g in [0.1, 0.25, 0.5, 1, 2, 4, 10]:
        ax.plot(r, r ** g, label=f"γ = {g}")
    ax.set_title("Power law s = r^γ")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax = axs[2]
    r1, s1, r2, s2 = 0.3, 0.1, 0.7, 0.9
    ax.plot(r, np.interp(r, [0, r1, r2, 1], [0, s1, s2, 1]), label="contrast stretch")
    ax.plot(r, (r >= 0.5).astype(float), label="threshold at 0.5")
    ax.plot(r, np.where((r > 0.55) & (r < 0.75), 0.95, r), label="slice (keep rest)")
    ax.plot([r1, r2], [s1, s2], "ko", ms=4)
    ax.annotate("(r1, s1)", (r1, s1), (r1 + 0.03, s1 - 0.06), fontsize=9)
    ax.annotate("(r2, s2)", (r2, s2), (r2 - 0.25, s2 + 0.03), fontsize=9)
    ax.set_title("Piecewise-linear")
    ax.legend(fontsize=8, loc="lower right")
    for ax in axs:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("input r (normalized)")
        ax.set_aspect("equal")
    axs[0].set_ylabel("output s (normalized)")
    fig.tight_layout()
    save(fig, "curves.png")


# --- Figure 3.2: negative, gamma, log on a spectrum -----------------------
def fig_basic():
    spec = np.abs(np.fft.fftshift(np.fft.fft2(cam)))
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    show(axs[0, 0], cam, "Input r")
    show(axs[0, 1], 1 - cam, "Negative 1 - r")
    show(axs[0, 2], cam ** 0.4, "Gamma γ = 0.4 (brighter)")
    show(axs[1, 0], cam ** 2.5, "Gamma γ = 2.5 (darker)")
    show(axs[1, 1], spec / spec.max(), "|FFT| scaled linearly")
    lg = np.log1p(spec)
    show(axs[1, 2], lg / lg.max(), "log(1 + |FFT|)")
    fig.tight_layout()
    save(fig, "basic-transforms.png")


# --- Figure 3.3: bit planes -----------------------------------------------
def fig_bitplanes():
    img = data.camera()
    fig, axs = plt.subplots(3, 3, figsize=(8, 8.4))
    for k in range(8):
        ax = axs.flat[7 - k]
        show(ax, (img >> k) & 1, f"bit plane {k}" + (" (MSB)" if k == 7 else " (LSB)" if k == 0 else ""))
    recon = ((img >> 4) << 4).astype(float) / 255
    show(axs.flat[8], recon, "planes 7-4 only")
    fig.tight_layout()
    save(fig, "bit-planes.png")


# --- Figure 3.4 and compare pair: histogram equalization ------------------
def low_contrast():
    # Squeeze the moon image into a narrow dark band so equalization has work to do.
    m = img_as_float(data.moon())
    m = (m - m.min()) / (m.max() - m.min())
    return img_as_ubyte(0.15 + 0.25 * m)


def equalize(img, L=256):
    hist = np.bincount(img.ravel(), minlength=L)
    cdf = np.cumsum(hist) / img.size
    lut = np.round((L - 1) * cdf).astype(np.uint8)
    return lut[img], lut


def fig_histeq():
    lo = low_contrast()
    he, lut = equalize(lo)
    cl = img_as_ubyte(exposure.equalize_adapthist(lo, kernel_size=64, clip_limit=0.02))
    cv2.imwrite(str(OUT / "hist-eq-before.png"), lo)
    cv2.imwrite(str(OUT / "hist-eq-after.png"), he)
    print("wrote hist-eq-before.png / hist-eq-after.png")

    fig, axs = plt.subplots(2, 4, figsize=(10.5, 5.2), gridspec_kw={"height_ratios": [1.4, 1]})
    for j, (im, t) in enumerate([(lo, "Low-contrast input"), (he, "Global equalization"), (cl, "CLAHE (64 px tiles)")]):
        show(axs[0, j], im, t, 0, 255)
        ax = axs[1, j]
        h = np.bincount(im.ravel(), minlength=256)
        ax.bar(np.arange(256), h / h.max(), width=1.0, color="0.25")
        ax2 = ax.twinx()
        ax2.plot(np.cumsum(h) / im.size, color="tab:red", lw=1.5)
        ax2.set_ylim(0, 1.05)
        ax2.set_yticks([0, 0.5, 1])
        ax2.tick_params(labelsize=9, colors="tab:red")
        ax.set_xlim(0, 255)
        ax.set_yticks([])
        ax.set_xlabel("intensity")
        ax.set_title("histogram (bars), CDF (red)", fontsize=10)
    ax = axs[0, 3]
    ax.plot(np.arange(256), lut, color="tab:blue")
    ax.plot([0, 255], [0, 255], "k--", lw=0.8)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_aspect("equal")
    ax.set_xlabel("input r")
    ax.set_ylabel("output s")
    ax.set_title("Equalization T(r)")
    axs[1, 3].axis("off")
    fig.tight_layout()
    save(fig, "hist-eq.png")


# --- Figure 3.5: smoothing filters on impulse noise -----------------------
def fig_smoothing():
    clean = cam[96:352, 160:416]
    noisy = clean.copy()
    u = RNG.random(clean.shape)
    noisy[u < 0.05] = 0.0
    noisy[u > 0.95] = 1.0
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    show(axs[0, 0], noisy, "Salt-and-pepper noise (10%)")
    show(axs[0, 1], cv2.blur(noisy, (5, 5)), "Box 5x5")
    show(axs[0, 2], cv2.GaussianBlur(noisy, (0, 0), 1.5), "Gaussian σ = 1.5")
    show(axs[1, 0], ndi.median_filter(noisy, 3), "Median 3x3")
    show(axs[1, 1], ndi.minimum_filter(noisy, 3), "Min 3x3")
    show(axs[1, 2], ndi.maximum_filter(noisy, 3), "Max 3x3")
    fig.tight_layout()
    save(fig, "smoothing.png")


# --- Figure 3.6: derivatives of a 1-D profile -----------------------------
def fig_derivatives():
    f = np.full(60, 6.0)
    f[8:20] = np.linspace(6, 1, 12)  # downward ramp
    f[20:44] = 1
    f[32] = 5  # thin bright line (one sample)
    f[44:] = 7  # step up
    x = np.arange(60)
    d1 = np.zeros_like(f)
    d1[:-1] = f[1:] - f[:-1]
    d2 = np.zeros_like(f)
    d2[1:-1] = f[2:] + f[:-2] - 2 * f[1:-1]
    fig, axs = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)
    axs[0].step(x, f, where="mid", color="k")
    axs[0].set_ylabel("f(x)")
    axs[0].set_title("Ramp (x=8..19), thin line (x=32), step (x=44)")
    axs[1].stem(x, d1, basefmt="0.6")
    axs[1].set_ylabel("first derivative")
    axs[2].stem(x, d2, basefmt="0.6", linefmt="C3-", markerfmt="C3o")
    axs[2].set_ylabel("second derivative")
    axs[2].set_xlabel("x")
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "derivatives.png")


# --- Figure 3.7: sharpening -----------------------------------------------
def fig_sharpening():
    # A slightly soft cat: the cat image blurred a little, so sharpening has something to do.
    f = cv2.GaussianBlur(color.rgb2gray(data.chelsea())[:, 60:360], (0, 0), 1.2)
    lap_k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
    lap = cv2.filter2D(f.astype(np.float32), -1, lap_k, borderType=cv2.BORDER_REFLECT)
    sharp = np.clip(f - lap, 0, 1)
    blur = cv2.GaussianBlur(f, (0, 0), 2)
    mask = f - blur
    unsharp = np.clip(f + 1.0 * mask, 0, 1)
    boost = np.clip(f + 3.0 * mask, 0, 1)
    gx = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    show(axs[0, 0], f, "Input I (slightly blurred)")
    a = np.abs(lap).max()
    show(axs[0, 1], lap, "Laplacian ∇²I (gray = 0)", -a / 2, a / 2)
    show(axs[0, 2], sharp, "I - ∇²I")
    show(axs[1, 0], unsharp, "Unsharp mask (α = 1, σ = 2)")
    show(axs[1, 1], boost, "Highboost (α = 3, σ = 2)")
    show(axs[1, 2], mag / mag.max(), "Sobel gradient magnitude")
    fig.tight_layout()
    save(fig, "sharpening.png")


# --- Figure 3.8: combined pipeline ----------------------------------------
def fig_combined():
    g = color.rgb2gray(data.astronaut())[0:320, 80:400]
    dark = g ** 2.2 * 0.6  # underexposed
    noisy = np.clip(dark + RNG.normal(0, 0.01, dark.shape), 0, 1)
    u = RNG.random(dark.shape)
    noisy[u < 0.02] = 0
    noisy[u > 0.98] = 1
    s1 = ndi.median_filter(noisy, 3)  # step 1: impulses
    s2 = (s1 / s1.max()) ** 0.45  # step 2: gamma lifts shadows
    blur = cv2.GaussianBlur(s2, (0, 0), 2)
    detail = s2 - blur
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    edge = np.hypot(gx, gy)
    w = np.clip(edge / np.percentile(edge, 90), 0, 1)  # step 3: edge weight in [0, 1]
    s4 = np.clip(s2 + 2.0 * w * detail, 0, 1)  # step 4: gated unsharp mask
    plain = np.clip(s2 + 2.0 * detail, 0, 1)
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    show(axs[0, 0], noisy, "(a) Dark, noisy input")
    show(axs[0, 1], s1, "(b) Median 3x3")
    show(axs[0, 2], s2, "(c) Gamma γ = 0.45")
    show(axs[1, 0], w, "(d) Edge weight from smoothed Sobel")
    show(axs[1, 1], plain, "(e) Plain unsharp (noise boosted)")
    show(axs[1, 2], s4, "(f) Edge-gated unsharp")
    fig.tight_layout()
    save(fig, "combined.png")


if __name__ == "__main__":
    fig_curves()
    fig_basic()
    fig_bitplanes()
    fig_histeq()
    fig_smoothing()
    fig_derivatives()
    fig_sharpening()
    fig_combined()
