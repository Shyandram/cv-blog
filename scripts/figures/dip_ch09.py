"""Figures for DIP Chapter 9 - Morphological Image Processing.

Run:  .venv/bin/python scripts/figures/dip_ch09.py
Writes PNGs to src/assets/dip/ch09/ only. Re-runnable and deterministic.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import data, draw, filters, morphology, segmentation, io, util

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "dip" / "ch09"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})
DPI = 150


def save(fig, name):
    assert fig.get_size_inches()[0] <= 10.5, "keep figures <= 1600 px wide at 150 dpi"
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def show(ax, img, title, cmap="gray", **kw):
    ax.imshow(img, cmap=cmap, interpolation="nearest", **kw)
    ax.set_title(title)
    ax.axis("off")


# ---------------------------------------------------------------- 1. four basic ops
def fig_basic_ops():
    rng = np.random.default_rng(0)
    horse = ~data.horse()                     # True = horse (foreground)
    horse = horse[::2, ::2]                   # 164 x 200, keeps figure small
    noisy = horse.copy()
    flip = rng.random(horse.shape) < 0.04     # salt-and-pepper style speckle
    noisy[flip] = ~noisy[flip]
    se = morphology.disk(2)
    ops = [
        (noisy, "Noisy input A"),
        (morphology.binary_erosion(noisy, se), "Erosion  A ⊖ B"),
        (morphology.binary_dilation(noisy, se), "Dilation  A ⊕ B"),
        (morphology.binary_opening(noisy, se), "Opening  A ∘ B"),
        (morphology.binary_closing(noisy, se), "Closing  A • B"),
        (morphology.binary_closing(morphology.binary_opening(noisy, se), se),
         "Open, then close"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, (im, t) in zip(axes.ravel(), ops):
        show(ax, im, t)
    fig.suptitle("Binary morphology with a disk of radius 2 (B)", y=1.0)
    save(fig, "basic-ops.png")


# ---------------------------------------------------------------- 2. hit-or-miss
def fig_hit_or_miss():
    img = np.zeros((60, 90), bool)
    img[8:22, 8:30] = True          # rectangle
    img[30:52, 12:24] = True        # tall rectangle
    img[10:40, 45:52] = True        # L shape, part 1
    img[33:40, 45:75] = True        # L shape, part 2
    rr, cc = draw.disk((20, 74), 8)
    img[rr, cc] = True              # disk (no sharp corners)
    img[50, 40] = True              # isolated pixel
    img[48, 80] = True              # isolated pixel

    # Upper-left corner detector: foreground at centre/right/below,
    # background above and to the left. 0 = don't care.
    hit = np.array([[0, 0, 0],
                    [0, 1, 1],
                    [0, 1, 0]], bool)
    miss = np.array([[0, 1, 0],
                     [1, 0, 0],
                     [0, 0, 0]], bool)
    corners = ndi.binary_hit_or_miss(img, structure1=hit, structure2=miss)

    # Isolated-point detector: centre foreground, all 8 neighbours background.
    iso_hit = np.zeros((3, 3), bool); iso_hit[1, 1] = True
    iso_miss = ~iso_hit
    isolated = ndi.binary_hit_or_miss(img, structure1=iso_hit, structure2=iso_miss)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    show(axes[0], img, "Input A")
    for ax, res, title, col in [(axes[1], corners, "HMT: upper-left corners", "red"),
                                (axes[2], isolated, "HMT: isolated pixels", "orange")]:
        ax.imshow(img, cmap="gray", interpolation="nearest", alpha=0.45)
        ys, xs = np.nonzero(res)
        ax.scatter(xs, ys, s=60, facecolors="none", edgecolors=col, linewidths=2)
        ax.set_title(f"{title} ({len(xs)})")
        ax.axis("off")
    save(fig, "hit-or-miss.png")
    return hit, miss


# ---------------------------------------------------------------- 3. skeleton etc.
def fig_skeleton():
    horse = ~data.horse()
    boundary = horse & ~morphology.binary_erosion(horse, morphology.square(3))
    skel = morphology.skeletonize(horse)
    medial, dist = morphology.medial_axis(horse, return_distance=True)
    hull = morphology.convex_hull_image(horse)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    show(axes[0], boundary, "Boundary  A − (A ⊖ B)")
    rgb = np.ones(horse.shape + (3,))
    rgb[hull] = (0.62, 0.79, 0.95)          # hull: light blue
    rgb[horse] = (0.2, 0.2, 0.2)            # shape: dark grey
    axes[1].imshow(rgb, interpolation="nearest")
    axes[1].set_title("Convex hull (blue)"); axes[1].axis("off")
    axes[2].imshow(horse, cmap="gray", alpha=0.35, interpolation="nearest")
    ys, xs = np.nonzero(skel)
    axes[2].scatter(xs, ys, s=0.6, c="red")
    axes[2].set_title("Skeleton (thinning)"); axes[2].axis("off")
    show(axes[3], morphology.dilation(np.where(medial, dist, 0), morphology.disk(2)),
         "Medial axis, coloured by distance", cmap="magma")
    save(fig, "skeleton.png")


# ---------------------------------------------------------------- 4. reconstruction
def make_shapes():
    """Synthetic binary scene: big shapes with thin parts, small specks,
    holes, and objects touching the border."""
    rng = np.random.default_rng(4)
    img = np.zeros((160, 260), bool)
    img[30:80, 20:80] = True                            # big square ...
    img[52:56, 80:130] = True                           # ... with a thin arm
    rr, cc = draw.disk((110, 160), 32); img[rr, cc] = True   # big disk ...
    for (r, c, rad) in [(100, 150, 6), (120, 172, 5), (98, 172, 4)]:
        rr, cc = draw.disk((r, c), rad); img[rr, cc] = False  # ... with holes
    rr, cc = draw.polygon([0, 0, 45], [200, 255, 255]); img[rr, cc] = True  # touches border
    img[120:160, 30:44] = True                          # touches bottom border
    img[40:44, 150:200] = True                          # thin bar, no big part
    for _ in range(25):                                 # small specks
        r, c = rng.integers(3, 157), rng.integers(3, 257)
        rr, cc = draw.disk((r, c), rng.integers(1, 3), shape=img.shape)
        img[rr, cc] = True
    return img


def fig_reconstruction():
    a = make_shapes()
    se = morphology.disk(6)
    marker = morphology.binary_erosion(a, se)
    plain_open = morphology.binary_opening(a, se)
    rec = morphology.reconstruction(marker.astype(np.uint8), a.astype(np.uint8),
                                    method="dilation").astype(bool)
    filled = ndi.binary_fill_holes(a)
    holes = filled & ~a
    cleared = segmentation.clear_border(a)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5))
    show(axes[0, 0], a, "Input A")
    show(axes[0, 1], marker, "Marker  A ⊖ disk(6)")
    show(axes[0, 2], plain_open, "Ordinary opening")
    show(axes[1, 0], rec, "Opening by reconstruction")
    rgb = np.dstack([a | holes, a, a & ~holes]).astype(float)
    axes[1, 1].imshow(rgb, interpolation="nearest")
    axes[1, 1].set_title("Holes filled (red)"); axes[1, 1].axis("off")
    show(axes[1, 2], cleared, "Border-touching objects removed")
    save(fig, "reconstruction.png")


# ---------------------------------------------------------------- 5. top-hat
def make_grains(seed=1, size=256):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    background = 0.15 + 0.55 * (xx / size) * (0.6 + 0.4 * yy / size)  # uneven light
    grains = np.zeros((size, size))
    for _ in range(70):
        r0, c0 = rng.integers(8, size - 8, 2)
        a, b = rng.uniform(3, 6), rng.uniform(1.8, 3)
        rr, cc = draw.ellipse(r0, c0, a, b, shape=grains.shape,
                              rotation=rng.uniform(0, np.pi))
        grains[rr, cc] = 0.3
    img = background + grains + rng.normal(0, 0.015, grains.shape)
    return np.clip(img, 0, 1), grains > 0


def fig_tophat():
    img, truth = make_grains()
    se = morphology.disk(8)
    bg = morphology.opening(img, se)
    th = morphology.white_tophat(img, se)
    seg_raw = img > filters.threshold_otsu(img)
    seg_th = th > filters.threshold_otsu(th)

    def iou(m):
        return (m & truth).sum() / (m | truth).sum()

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7), gridspec_kw={"wspace": 0.35})
    show(axes[0, 0], img, "Input f (uneven light)", vmin=0, vmax=1)
    show(axes[0, 1], bg, "Opening f ∘ b  (background)", vmin=0, vmax=1)
    show(axes[0, 2], th, "Top-hat  f − f ∘ b")
    show(axes[1, 0], seg_raw, f"Otsu on f  (IoU {iou(seg_raw):.2f})")
    show(axes[1, 1], seg_th, f"Otsu on top-hat  (IoU {iou(seg_th):.2f})")
    axes[1, 2].plot(img[128], label="row 128 of f")
    axes[1, 2].plot(bg[128], label="opening")
    axes[1, 2].plot(th[128], label="top-hat")
    axes[1, 2].set_xlabel("column"); axes[1, 2].set_ylabel("intensity")
    axes[1, 2].legend(fontsize=9); axes[1, 2].set_title("One row, profiled")
    save(fig, "tophat.png")

    # Same-size pair for the ImageCompare slider.
    def to_u8(a):
        a = (a - a.min()) / (a.max() - a.min())
        return util.img_as_ubyte(a)
    io.imsave(OUT / "tophat-before.png", to_u8(img), check_contrast=False)
    io.imsave(OUT / "tophat-after.png", to_u8(th), check_contrast=False)
    print("wrote tophat-before.png, tophat-after.png")
    return iou(seg_raw), iou(seg_th)


# ---------------------------------------------------------------- 6. granulometry
def fig_granulometry():
    rng = np.random.default_rng(3)
    size = 256
    img = np.zeros((size, size))
    for radius, count in [(5, 30), (12, 9)]:
        placed = 0
        while placed < count:
            r0, c0 = rng.integers(radius + 2, size - radius - 2, 2)
            rr, cc = draw.disk((r0, c0), radius, shape=img.shape)
            if img[rr, cc].max() == 0:           # no overlap: clean demo
                img[rr, cc] = 1.0
                placed += 1
    img = np.clip(filters.gaussian(img, 0.6) + rng.normal(0, 0.02, img.shape), 0, 1)
    radii = np.arange(0, 17)
    volume = np.array([morphology.opening(img, morphology.disk(r)).sum() if r else img.sum()
                       for r in radii])
    volume = volume / volume[0]
    spectrum = -np.diff(volume)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), gridspec_kw={"wspace": 0.45})
    show(axes[0], img, "Input: small and large discs")
    axes[1].plot(radii, volume, "o-")
    axes[1].set_xlabel("opening radius r"); axes[1].set_ylabel("normalized sum of f ∘ b_r")
    axes[1].set_title("Granulometric curve")
    axes[2].bar(radii[1:], spectrum, color="tab:orange")
    axes[2].set_xlabel("opening radius r"); axes[2].set_ylabel("loss between r−1 and r")
    axes[2].set_title("Pattern spectrum (size histogram)")
    for ax in axes[1:]:
        ax.grid(alpha=0.3)
    save(fig, "granulometry.png")


# ---------------------------------------------------------------- 7. grayscale ops
def fig_gray_ops():
    img = util.img_as_float(data.camera())[::2, ::2]
    se = morphology.disk(3)
    ero = morphology.erosion(img, se)
    dil = morphology.dilation(img, se)
    grad = dil - ero
    smooth = morphology.closing(morphology.opening(img, se), se)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.4))
    for ax, im, t in zip(axes.ravel(), [img, ero, dil, smooth, grad, img - ero],
                         ["Input f", "Erosion f ⊖ b", "Dilation f ⊕ b",
                          "Smoothing (open→close)", "Gradient (f⊕b) − (f⊖b)",
                          "Internal gradient f − (f⊖b)"]):
        show(ax, im, t)
    save(fig, "gray-ops.png")


if __name__ == "__main__":
    fig_basic_ops()
    fig_hit_or_miss()
    fig_skeleton()
    fig_reconstruction()
    print("top-hat IoU (raw, top-hat):", fig_tophat())
    fig_granulometry()
    fig_gray_ops()
