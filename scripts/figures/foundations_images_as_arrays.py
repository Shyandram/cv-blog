"""Figures for foundations/images-as-arrays; all from skimage.data, no downloads.

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/foundations_images_as_arrays.py
Writes only src/assets/foundations/images-as-arrays/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from skimage import data

OUT = Path(__file__).resolve().parents[2] / "src/assets/foundations/images-as-arrays"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, RED = "#2a6fdb", "#e4572e", "#17a398", "#d62728"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name, f"{(OUT / name).stat().st_size / 1e3:.0f} kB")


def pixel_grid():
    """A photo, a highlighted 8 x 8 patch, and the patch as the numbers it really is."""
    gray = data.camera()
    r0, c0 = 160, 250
    patch = gray[r0:r0 + 8, c0:c0 + 8]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    a.imshow(gray, cmap="gray", vmin=0, vmax=255)
    a.add_patch(Rectangle((c0 - .5, r0 - .5), 8, 8, fill=False, ec=RED, lw=2))
    a.annotate("8 × 8 patch", (c0 + 8, r0), (c0 + 60, r0 - 60), color=RED,
               arrowprops=dict(arrowstyle="->", color=RED))
    a.set(title=f"gray.shape = {gray.shape}  (rows, columns)", xlabel="column index c", ylabel="row index r")
    b.imshow(patch, cmap="gray", vmin=0, vmax=255)
    for (i, j), v in np.ndenumerate(patch):
        b.text(j, i, str(v), ha="center", va="center", fontsize=10,
               color="white" if v < 128 else "black")
    b.set_xticks(range(8), [str(c0 + j) for j in range(8)])
    b.set_yticks(range(8), [str(r0 + i) for i in range(8)])
    b.set(title=f"gray[{r0}:{r0 + 8}, {c0}:{c0 + 8}]", xlabel="column index c", ylabel="row index r")
    b.xaxis.tick_top(); b.xaxis.set_label_position("top")
    fig.tight_layout()
    save(fig, "pixel-grid.png")


def channels():
    """RGB image, its three channel planes, and the weighted gray conversion."""
    rgb = data.astronaut()
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    fig, axes = plt.subplots(1, 5, figsize=(10.4, 2.6))
    axes[0].imshow(rgb); axes[0].set_title("rgb  (H, W, 3)")
    for k, (name, col) in enumerate([("R", RED), ("G", GREEN), ("B", BLUE)], start=1):
        axes[k].imshow(rgb[..., k - 1], cmap="gray", vmin=0, vmax=255)
        axes[k].set_title(f"rgb[..., {k - 1}]  = {name}", color=col)
    axes[4].imshow(gray, cmap="gray", vmin=0, vmax=255); axes[4].set_title("gray  (H, W)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    save(fig, "channels.png")


def quantize(img, bits):
    step = 256 // 2 ** bits
    return (img // step) * step + step // 2


def sampling_quantization():
    """Row 1: fewer gray levels. Row 2: fewer samples (point sampling, shown at equal size)."""
    gray = data.camera()
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.5))
    for ax, b in zip(axes[0], (8, 4, 2, 1)):
        ax.imshow(quantize(gray, b), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{b} bit{'s' if b > 1 else ''}: {2 ** b} levels")
    for ax, s in zip(axes[1], (1, 2, 4, 8)):
        small = gray[::s, ::s]
        ax.imshow(small, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"stride {s}: {small.shape[0]} × {small.shape[1]}")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    save(fig, "sampling-quantization.png")


def overflow():
    """Brightening a uint8 image by +60: wraparound versus clipping."""
    gray = data.camera()
    wrapped = gray + np.uint8(60)
    clipped = np.clip(gray.astype(np.int16) + 60, 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    for ax, img, title in zip(axes[:3], (gray, wrapped, clipped),
                              ("original", "gray + 60 (uint8 wraps)", "clip(int16(gray) + 60)")):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255); ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    v = np.arange(256)
    axes[3].plot(v, (v + 60) % 256, color=ORANGE, label="wraps (uint8)")
    axes[3].plot(v, np.clip(v + 60, 0, 255), color=GREEN, label="clipped")
    axes[3].set(xlabel="input value", ylabel="output value", xlim=(0, 255), ylim=(0, 255))
    axes[3].legend(fontsize=10, loc="center right"); axes[3].grid(alpha=.2)
    fig.tight_layout()
    save(fig, "overflow.png")


if __name__ == "__main__":
    pixel_grid()
    channels()
    sampling_quantization()
    overflow()
