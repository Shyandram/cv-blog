"""Figures for DIP Chapter 6 - Color Image Processing.

Run:  .venv/bin/python scripts/figures/dip_ch06.py
Writes PNGs only to src/assets/dip/ch06/.
All inputs are skimage.data images or synthetic arrays.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, data, filters, util

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "dip" / "ch06"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 150
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def off(ax):
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------- helpers
def rgb_to_hsi(rgb):
    """rgb: float array in [0, 1], shape (..., 3). Returns H in [0, 1), S, I."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-12
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    H = np.where(B > G, 2 * np.pi - theta, theta) / (2 * np.pi)
    I = (R + G + B) / 3.0
    S = 1.0 - np.min(rgb, axis=-1) / (I + 1e-12)
    S = np.where(I > 0, S, 0.0)
    H = np.where(S > 1e-6, H, 0.0)  # hue is undefined for grays; set to 0
    return H, S, I


def cmf_wyman(lam):
    """CIE 1931 2-degree colour-matching functions, multi-lobe Gaussian fit
    of Wyman, Sloan & Shirley, JCGT 2(2), 2013 (Listing 1)."""
    def g(x, mu, s1, s2):
        t = (x - mu) * np.where(x < mu, s1, s2)
        return np.exp(-0.5 * t * t)
    xb = (0.362 * g(lam, 442.0, 0.0624, 0.0374) + 1.056 * g(lam, 599.8, 0.0264, 0.0323)
          - 0.065 * g(lam, 501.1, 0.0490, 0.0382))
    yb = 0.821 * g(lam, 568.8, 0.0213, 0.0247) + 0.286 * g(lam, 530.9, 0.0613, 0.0322)
    zb = 1.217 * g(lam, 437.0, 0.0845, 0.0278) + 0.681 * g(lam, 459.0, 0.0385, 0.0725)
    return xb, yb, zb


# ------------------------------------------------- 6.1 chromaticity diagram
def fig_chromaticity():
    # The analytic fit's far tails (below ~430 nm, above ~650 nm) are too small
    # to give accurate chromaticities, so the locus is drawn over 430-650 nm and
    # closed with a straight line (an approximation of the line of purples).
    lam = np.arange(430.0, 651.0, 1.0)
    X, Y, Z = cmf_wyman(lam)
    s = X + Y + Z
    x, y = X / s, Y / s

    # Fill the horseshoe with an (approximate) sRGB rendering of each chromaticity.
    n = 400
    gx, gy = np.meshgrid(np.linspace(0, 0.8, n), np.linspace(0, 0.9, n))
    from matplotlib.path import Path as MPath
    locus = MPath(np.c_[np.r_[x, x[0]], np.r_[y, y[0]]])
    inside = locus.contains_points(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    gz = 1 - gx - gy
    XYZ = np.stack([gx / np.maximum(gy, 1e-6), np.ones_like(gx),
                    gz / np.maximum(gy, 1e-6)], -1)
    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]])
    lin = np.clip(XYZ @ M.T, 0, None)
    lin = lin / np.maximum(lin.max(-1, keepdims=True), 1e-6)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    img = np.ones(gx.shape + (3,))
    img[inside] = srgb[inside]

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(img, origin="lower", extent=(0, 0.8, 0, 0.9))
    ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], "k-", lw=1.2)
    for wl in (460, 480, 500, 520, 540, 560, 580, 600, 620):
        i = int(wl - 430)
        ax.plot(x[i], y[i], "ko", ms=3)
        dx = -0.065 if wl < 520 else 0.015
        ax.text(x[i] + dx, y[i], f"{wl}", fontsize=9)
    prim = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.64, 0.33]])
    ax.plot(prim[:, 0], prim[:, 1], "w-", lw=2.2)
    ax.plot(prim[:, 0], prim[:, 1], "k--", lw=1.0, label="sRGB / BT.709 gamut")
    ax.plot(0.3127, 0.3290, "k+", ms=10, mew=2, label="D65 white (0.3127, 0.3290)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.9)
    ax.set_title("CIE 1931 xy chromaticity (analytic CMF fit)")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "chromaticity.png")


# ------------------------------------------------------- 6.2 RGB channels
def fig_rgb_split():
    img = util.img_as_float(data.astronaut())
    fig, ax = plt.subplots(1, 4, figsize=(10.5, 3.0))
    ax[0].imshow(img)
    ax[0].set_title("RGB image")
    for k, name in enumerate(["R", "G", "B"]):
        ax[k + 1].imshow(img[..., k], cmap="gray", vmin=0, vmax=1)
        ax[k + 1].set_title(f"{name} channel")
    for a in ax:
        off(a)
    save(fig, "rgb-split.png")


# ------------------------------------------------------- 6.3 HSI channels
def fig_hsi():
    img = util.img_as_float(data.astronaut())
    H, S, I = rgb_to_hsi(img)
    fig, ax = plt.subplots(1, 4, figsize=(10.5, 3.0))
    ax[0].imshow(img)
    ax[0].set_title("RGB image")
    ax[1].imshow(H, cmap="hsv", vmin=0, vmax=1)
    ax[1].set_title("Hue H (cyclic map)")
    ax[2].imshow(S, cmap="gray", vmin=0, vmax=1)
    ax[2].set_title("Saturation S")
    ax[3].imshow(I, cmap="gray", vmin=0, vmax=1)
    ax[3].set_title("Intensity I")
    for a in ax:
        off(a)
    save(fig, "hsi.png")


# ------------------------------------------------------- 6.4 pseudocolor
def fig_pseudocolor():
    g = util.img_as_float(data.moon())
    lo, hi = np.percentile(g, (1, 99))
    g = np.clip((g - lo) / (hi - lo), 0, 1)           # contrast stretch first
    edges = np.linspace(0, 1, 9)
    lvl = np.digitize(g, edges[1:-1])
    pal = plt.get_cmap("tab10")(np.arange(8))[:, :3]
    sliced = pal[lvl]
    fig = plt.figure(figsize=(10.5, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.7])
    titles = ["Grayscale", "Intensity slicing (8 bands)", "Rainbow (jet)", "viridis"]
    ims = [(g, "gray"), (sliced, None), (g, "jet"), (g, "viridis")]
    for k, ((im, cm), t) in enumerate(zip(ims, titles)):
        a = fig.add_subplot(gs[0, k])
        a.imshow(im, cmap=cm, vmin=0, vmax=1) if cm else a.imshow(im)
        a.set_title(t, fontsize=10)
        off(a)
    a = fig.add_subplot(gs[1, :])
    t = np.linspace(0, 1, 256)
    for name, ls in (("gray", ":"), ("jet", "-"), ("viridis", "--")):
        rgb = plt.get_cmap(name)(t)[:, :3]
        L = color.rgb2lab(rgb[None])[0, :, 0]
        a.plot(t, L, ls, lw=2, label=name)
    a.set_xlabel("data value (0 to 1)")
    a.set_ylabel("CIE L* (lightness)")
    a.set_title("Lightness along each colormap: jet is not monotonic")
    a.legend(loc="upper left")
    a.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "pseudocolor.png")


# ------------------------------------------------------- 6.5 color slicing
def fig_color_slicing():
    img = util.img_as_float(data.astronaut())
    a = np.array([0.85, 0.40, 0.20])     # orange of the suit (prototype colour)
    r0 = 0.25
    d = np.linalg.norm(img - a, axis=-1)
    keep = d <= r0
    gray = color.rgb2gray(img)[..., None].repeat(3, -1) * 0.5 + 0.25
    out = np.where(keep[..., None], img, gray)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6))
    ax[0].imshow(img)
    ax[0].set_title("Input")
    ax[1].imshow(d, cmap="magma")
    ax[1].set_title("Distance to prototype")
    ax[2].imshow(out)
    ax[2].set_title(f"Slice: keep ||z - a|| <= {r0}")
    for x in ax:
        off(x)
    save(fig, "color-slicing.png")


# --------------------------------------- 6.6 segmentation in RGB vector space
def fig_segmentation():
    img = util.img_as_float(data.immunohistochemistry())
    # Training patch inside a brown (DAB-stained) region, chosen by eye.
    patch = img[60:100, 60:100].reshape(-1, 3)
    m = patch.mean(0)
    C = np.cov(patch.T) + 1e-6 * np.eye(3)
    Ci = np.linalg.inv(C)
    z = img.reshape(-1, 3) - m
    d_e = np.sqrt((z ** 2).sum(1)).reshape(img.shape[:2])
    d_m = np.sqrt(np.einsum("ij,jk,ik->i", z, Ci, z)).reshape(img.shape[:2])
    seg_e = d_e <= 0.25
    seg_m = d_m <= 3.0
    fig = plt.figure(figsize=(10.5, 6.8))
    gs = fig.add_gridspec(2, 3)
    a0 = fig.add_subplot(gs[0, 0])
    a0.imshow(img)
    a0.add_patch(plt.Rectangle((60, 60), 40, 40, fill=False, ec="lime", lw=2))
    a0.set_title("Input + training patch")
    a1 = fig.add_subplot(gs[0, 1])
    a1.imshow(seg_e, cmap="gray")
    a1.set_title("Euclidean, D <= 0.25")
    a2 = fig.add_subplot(gs[0, 2])
    a2.imshow(seg_m, cmap="gray")
    a2.set_title("Mahalanobis, D <= 3")
    for a in (a0, a1, a2):
        off(a)
    rng = np.random.default_rng(0)
    idx = rng.choice(img.shape[0] * img.shape[1], 6000, replace=False)
    px = img.reshape(-1, 3)[idx]
    ax3 = fig.add_subplot(gs[1, :2], projection="3d")
    ax3.scatter(px[:, 0], px[:, 1], px[:, 2], c=px, s=2)
    ax3.scatter(*m, c="lime", s=80, edgecolors="k", depthshade=False)
    ax3.set_xlabel("R")
    ax3.set_ylabel("G")
    ax3.set_zlabel("B")
    ax3.set_title("Pixels as points in RGB space (green dot = mean a)")
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(d_e.ravel(), bins=80, color="0.4")
    ax4.axvline(0.25, color="r", ls="--")
    ax4.set_xlabel("Euclidean distance to a")
    ax4.set_ylabel("pixel count")
    fig.tight_layout()
    save(fig, "segmentation-rgb.png")


# -------------------------------------------- 6.7 white balance
CAST = np.array([0.75, 0.95, 1.25])               # synthetic bluish illuminant


def angular_error(est, true):
    est = est / np.linalg.norm(est)
    true = true / np.linalg.norm(true)
    return np.degrees(np.arccos(np.clip(est @ true, -1, 1)))


def estimate(img, method):
    px = img.reshape(-1, 3)
    if method == "gray-world":
        return px.mean(0)
    if method == "white-patch":                   # robust max-RGB
        return np.percentile(px, 99.5, axis=0)
    raise ValueError(method)


def correct(img, e):
    gain = e.mean() / e                           # diagonal (per-channel) gains
    out = img * gain
    return np.clip(out / np.percentile(out, 99.5), 0, 1)


def fig_white_balance():
    # (a) before/after pair for the ImageCompare slider (raw images, same size)
    img = util.img_as_float(data.astronaut()) * 0.8
    biased = np.clip(img * CAST, 0, 1)
    fixed = correct(biased, estimate(biased, "white-patch"))
    plt.imsave(OUT / "wb-before.png", biased)
    plt.imsave(OUT / "wb-after.png", fixed)
    print("wrote wb-before.png / wb-after.png")

    # (b) gray-world vs white-patch on an image whose average is not gray
    cat = util.img_as_float(data.chelsea()) * 0.8
    cb = np.clip(cat * CAST, 0, 1)
    fig, ax = plt.subplots(1, 4, figsize=(10.5, 2.4))
    ax[0].imshow(cat / 0.8)
    ax[0].set_title("Original", fontsize=10)
    ax[1].imshow(cb)
    ax[1].set_title("Bluish cast added", fontsize=10)
    for k, m in enumerate(["gray-world", "white-patch"]):
        e = estimate(cb, m)
        ax[k + 2].imshow(correct(cb, e))
        ax[k + 2].set_title(f"{m}: error {angular_error(e, CAST):.1f} deg",
                            fontsize=10)
    for a in ax:
        off(a)
    save(fig, "white-balance.png")


# --------------------------------------------- 6.8 color edges (Di Zenzo)
def fig_color_edges():
    h, w = 160, 240
    img = np.zeros((h, w, 3))
    # Left: red; right: green chosen to have (almost) the same luminance.
    # Rec. 601 luma: Y = 0.299 R + 0.587 G + 0.114 B
    red = np.array([0.8, 0.0, 0.0])
    green = np.array([0.0, 0.299 * 0.8 / 0.587, 0.0])
    img[:, : w // 2] = red
    img[:, w // 2:] = green
    img[40:120, 30:90] = [0.2, 0.2, 0.9]           # a blue square
    luma = img @ np.array([0.299, 0.587, 0.114])
    g_luma = np.hypot(filters.sobel_h(luma), filters.sobel_v(luma))
    gx = np.stack([filters.sobel_v(img[..., k]) for k in range(3)], -1)
    gy = np.stack([filters.sobel_h(img[..., k]) for k in range(3)], -1)
    gxx = (gx * gx).sum(-1)
    gyy = (gy * gy).sum(-1)
    gxy = (gx * gy).sum(-1)
    lam = 0.5 * (gxx + gyy + np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2))
    g_dz = np.sqrt(lam)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9))
    ax[0].imshow(img)
    ax[0].set_title("Input (red | green, equal luma)")
    ax[1].imshow(g_luma, cmap="gray", vmin=0, vmax=g_dz.max())
    ax[1].set_title("Gradient of luma only")
    ax[2].imshow(g_dz, cmap="gray", vmin=0, vmax=g_dz.max())
    ax[2].set_title("Di Zenzo vector gradient")
    for a in ax:
        off(a)
    save(fig, "color-edges.png")


if __name__ == "__main__":
    fig_chromaticity()
    fig_rgb_split()
    fig_hsi()
    fig_pseudocolor()
    fig_color_slicing()
    fig_segmentation()
    fig_white_balance()
    fig_color_edges()
