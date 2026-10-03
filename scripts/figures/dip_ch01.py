"""Figures for DIP Chapter 1 (Introduction).

Run:  .venv/bin/python scripts/figures/dip_ch01.py
Writes PNGs to src/assets/dip/ch01/. Re-runnable; uses only skimage.data
images and synthetic arrays.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from skimage import data, feature, filters, measure, morphology, segmentation  # noqa: E402
from skimage.transform import iradon, radon, rescale  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "dip" / "ch01"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
DPI = 150


def save(fig, name, max_w=1600):
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    w = plt.imread(path).shape[1]
    if w > max_w:  # re-save at a lower dpi so the PNG stays <= max_w px wide
        fig.savefig(path, dpi=int(DPI * max_w / w) - 1, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(OUT.parents[3])} ({path.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------- 1. pixel grid
def fig_pixel_grid():
    img = data.camera()
    r0, c0, n = 88, 248, 10  # a patch on the photographer's face/camera edge
    patch = img[r0:r0 + n, c0:c0 + n]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axes[0]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), n, n, fill=False, ec="red", lw=2))
    ax.set_title("512 x 512 grayscale image f(x, y)")
    ax.set_xlabel("column y")
    ax.set_ylabel("row x")

    ax = axes[1]
    ax.imshow(patch, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    for i in range(n):
        for j in range(n):
            v = int(patch[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v < 128 else "black")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="red", lw=0.4)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(0, n, 2))
    ax.set_xticklabels([str(c0 + k) for k in range(0, n, 2)])
    ax.set_yticks(range(0, n, 2))
    ax.set_yticklabels([str(r0 + k) for k in range(0, n, 2)])
    ax.set_title("10 x 10 zoom: each square is one pixel (0 = black, 255 = white)")
    fig.tight_layout()
    save(fig, "pixel-grid.png")


# ------------------------------------------------------- 2. processing levels
def fig_levels():
    img = data.coins()
    # low level: denoise + contrast stretch (image in, image out)
    smooth = filters.gaussian(img, sigma=1.5)
    lo, hi = np.percentile(smooth, (2, 98))
    low = np.clip((smooth - lo) / (hi - lo), 0, 1)
    # mid level: segmentation into regions (image in, regions/attributes out)
    from scipy import ndimage as ndi
    elevation = filters.sobel(img)
    markers = np.zeros_like(img, dtype=int)
    markers[img < 30] = 1   # sure background
    markers[img > 150] = 2  # sure coin
    seg = segmentation.watershed(elevation, markers) == 2
    seg = ndi.binary_fill_holes(seg)
    seg = morphology.remove_small_objects(seg, 60)
    labels = measure.label(seg)
    regions = [r for r in measure.regionprops(labels) if r.area > 300]
    # high level: interpretation (count + size classes)
    areas = np.array([r.area for r in regions])
    thr = np.median(areas)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.4))
    axes = axes.ravel()
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input image")
    axes[1].imshow(low, cmap="gray")
    axes[1].set_title("Low level: smoothed +\ncontrast-stretched (image out)")
    axes[2].imshow(segmentation.mark_boundaries(low, labels, color=(1, 0, 0)))
    axes[2].set_title(f"Mid level: {len(regions)} regions\n+ area, centroid (attributes out)")
    axes[3].imshow(img, cmap="gray")
    for r in regions:
        y, x = r.centroid
        big = r.area >= thr
        axes[3].text(x, y, "L" if big else "S", color="yellow" if big else "cyan",
                     ha="center", va="center", fontsize=12, fontweight="bold")
    axes[3].set_title(f"High level: '{len(regions)} coins,\n"
                      f"{int((areas >= thr).sum())} large / {int((areas < thr).sum())} small'")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    save(fig, "processing-levels.png")


# ---------------------------------------------------------- 3. Bartlane levels
def fig_gray_levels():
    img = data.camera().astype(float) / 255.0
    img = rescale(img, 0.5, anti_aliasing=True)

    def quant(x, L):
        return np.round(x * (L - 1)) / (L - 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for a, (L, t) in zip(axes, [(5, "5 gray levels"), (15, "15 gray levels"), (256, "256 gray levels (8 bit)")]):
        a.imshow(quant(img, L), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        a.set_title(t)
        a.axis("off")
    fig.tight_layout()
    save(fig, "gray-levels.png")


# -------------------------------------------------------------- 4. EM spectrum
def fig_spectrum():
    # wavelength boundaries in metres (approximate, conventions vary)
    bands = [
        ("Gamma", 1e-14, 1e-11, "#5b2a86"),
        ("X-ray", 1e-11, 1e-8, "#3c5aa6"),
        ("UV", 1e-8, 3.8e-7, "#7b6fd0"),
        ("Visible", 3.8e-7, 7.5e-7, None),
        ("Infrared", 7.5e-7, 1e-3, "#c0392b"),
        ("Microwave", 1e-3, 1e-1, "#d68910"),
        ("Radio", 1e-1, 1e3, "#7d8c2e"),
    ]
    uses = {
        "Gamma": "PET, astronomy",
        "X-ray": "radiography, CT",
        "UV": "fluorescence\nmicroscopy",
        "Visible": "photography",
        "Infrared": "thermal,\nremote sensing",
        "Microwave": "radar (SAR)",
        "Radio": "MRI,\nradio astronomy",
    }
    h, c, eV = 6.626e-34, 2.998e8, 1.602e-19

    fig, ax = plt.subplots(figsize=(13, 3.6))
    for name, a, b, col in bands:
        la, lb = np.log10(a), np.log10(b)
        if name == "Visible":
            # draw a rainbow inside the (narrow) visible band
            xs = np.linspace(la, lb, 30)
            cmap = plt.get_cmap("nipy_spectral")
            for k in range(len(xs) - 1):
                ax.add_patch(Rectangle((xs[k], 0), xs[k + 1] - xs[k], 1,
                                       color=cmap(0.12 + 0.75 * k / (len(xs) - 1))))
            ax.annotate("Visible (380-750 nm)\nphotography", xy=((la + lb) / 2, 1), xytext=((la + lb) / 2, 1.55),
                        ha="center", fontsize=10, arrowprops=dict(arrowstyle="-", lw=0.8))
        else:
            ax.add_patch(Rectangle((la, 0), lb - la, 1, color=col, alpha=0.85))
            ax.text((la + lb) / 2, 0.5, name, ha="center", va="center", color="white",
                    fontsize=11, fontweight="bold")
        if name != "Visible":
            ax.text((la + lb) / 2, -0.35, uses[name], ha="center", va="top", fontsize=9.5)
    ax.set_xlim(-14, 3)
    ax.set_ylim(-1.3, 2.1)
    ax.set_yticks([])
    ticks = np.arange(-14, 4, 2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"1e{t}" for t in ticks])
    ax.set_xlabel("Wavelength (m, log scale)    shorter = higher frequency = more energetic photons")
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    top = ax.secondary_xaxis("top", functions=(lambda lg: lg, lambda lg: lg))
    top.set_xticks(ticks)
    top.set_xticklabels([f"{h * c / 10.0 ** t / eV:.0e}" for t in ticks], fontsize=9)
    top.set_xlabel("Photon energy E = hc / wavelength (eV)")
    fig.tight_layout()
    save(fig, "em-spectrum.png")


# ------------------------------------------------------------- 5. CT example
def fig_ct():
    from skimage.data import shepp_logan_phantom
    ph = rescale(shepp_logan_phantom(), 0.5, anti_aliasing=True)  # 200 x 200
    theta = np.linspace(0.0, 180.0, 180, endpoint=False)
    sino = radon(ph, theta=theta)
    rec = iradon(sino, theta=theta, filter_name="ramp")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    axes[0].imshow(ph, cmap="gray")
    axes[0].set_title("Hidden slice (synthetic phantom)")
    axes[1].imshow(sino, cmap="gray", aspect="auto",
                   extent=(0, 180, -sino.shape[0] / 2, sino.shape[0] / 2))
    axes[1].set_title("What the scanner measures:\nprojections at 180 angles")
    axes[1].set_xlabel("projection angle (degrees)")
    axes[1].set_ylabel("detector position")
    axes[2].imshow(rec, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Computed reconstruction\n(filtered back-projection)")
    for a in (axes[0], axes[2]):
        a.axis("off")
    fig.tight_layout()
    save(fig, "ct-reconstruction.png")


# --------------------------------------------------------- diagram helpers
def box(ax, x, y, w, h, text, fc="#eef2f7", ec="#34495e", fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, p, q, color="#34495e", style="-|>", lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=color, lw=lw, ls=ls))


# -------------------------------------------------------- 6. fundamental steps
def fig_steps():
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    out_img = "#e8f1fb"
    out_attr = "#fdf0e3"
    # left column: outputs are images
    left = [
        ("Ch 2  Image acquisition\n& fundamentals", 5.8),
        ("Ch 3  Intensity transforms\n& spatial filtering", 4.85),
        ("Ch 4  Frequency-domain\nfiltering", 3.9),
        ("Ch 5  Restoration &\nreconstruction", 2.95),
        ("Ch 6  Color image\nprocessing", 2.0),
        ("Ch 7  Wavelets & other\ntransforms", 1.05),
    ]
    right = [
        ("Ch 8  Compression &\nwatermarking", 5.8),
        ("Ch 9  Morphological\nprocessing", 4.85),
        ("Ch 10  Segmentation I\n(edges, thresholds, regions)", 3.9),
        ("Ch 11  Segmentation II\n(active contours)", 2.95),
        ("Ch 12  Feature\nextraction", 2.0),
        ("Ch 13  Pattern\nclassification", 1.05),
    ]
    for t, y in left:
        box(ax, 3.0, y, 3.6, 0.78, t, fc=out_img)
    for t, y in right:
        box(ax, 10.0, y, 3.8, 0.78, t, fc=out_attr)
    for (_, y1), (_, y2) in zip(left[:-1], left[1:]):
        arrow(ax, (3.0, y1 - 0.4), (3.0, y2 + 0.4))
    for (_, y1), (_, y2) in zip(right[:-1], right[1:]):
        arrow(ax, (10.0, y1 - 0.4), (10.0, y2 + 0.4))

    box(ax, 6.5, 3.4, 2.2, 1.3, "Knowledge\nbase\n(Ch 1 roadmap)", fc="#fff8d6", ec="#b7950b", bold=True)
    arrow(ax, (5.4, 3.4), (4.85, 3.9), color="#b7950b", style="<|-|>")
    arrow(ax, (7.6, 3.4), (8.1, 3.9), color="#b7950b", style="<|-|>")

    ax.text(0.25, 6.45, "Problem domain  ->", fontsize=11, fontweight="bold")
    ax.text(3.0, 0.3, "Outputs are generally IMAGES", ha="center", fontsize=11, color="#1f4e79",
            fontweight="bold")
    ax.text(10.0, 0.3, "Outputs are generally image ATTRIBUTES", ha="center", fontsize=11,
            color="#a0522d", fontweight="bold")
    save(fig, "fundamental-steps.png")


# ------------------------------------------------------- 7. system components
def fig_system():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    box(ax, 6.6, 4.9, 6.8, 0.7, "Network / cloud  (sharing, remote compute, storage)", fc="#eaf7ee")
    box(ax, 1.3, 2.8, 2.0, 1.0, "Scene /\nproblem domain", fc="#f4f4f4")
    box(ax, 3.9, 2.8, 2.0, 1.0, "Image sensor\n+ digitizer", fc="#e8f1fb")
    box(ax, 6.5, 2.8, 2.2, 1.0, "Specialized\nhardware\n(ISP, GPU, FPGA)", fc="#e8f1fb")
    box(ax, 9.3, 2.8, 2.2, 1.0, "Computer\n+ image processing\nsoftware", fc="#e8f1fb", bold=False)
    box(ax, 9.3, 0.8, 2.2, 0.8, "Mass storage\n(RAM, SSD, archive)", fc="#fdf0e3")
    box(ax, 12.0, 3.6, 1.8, 0.75, "Displays", fc="#fdf0e3")
    box(ax, 12.0, 2.0, 1.8, 0.75, "Hardcopy\n(printer, film)", fc="#fdf0e3")
    arrow(ax, (2.3, 2.8), (2.9, 2.8))
    ax.text(2.6, 3.55, "light,\nX-rays...", ha="center", fontsize=9)
    arrow(ax, (4.9, 2.8), (5.4, 2.8))
    arrow(ax, (7.6, 2.8), (8.2, 2.8))
    arrow(ax, (9.3, 2.3), (9.3, 1.2), style="<|-|>")
    arrow(ax, (10.4, 3.1), (11.1, 3.5))
    arrow(ax, (10.4, 2.5), (11.1, 2.1))
    for x in (3.9, 6.5, 9.3):
        arrow(ax, (x, 3.3), (x if x != 6.5 else 6.5, 4.55), style="<|-|>", color="#27ae60", lw=1.0)
    save(fig, "system-components.png")


# ------------------------------------------------- 8. classic vs learned ISP
def fig_isp():
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    stages = ["Black level\n& gain", "Demosaic", "Denoise", "White\nbalance", "Color\ncorrection",
              "Tone map\n& gamma", "Sharpen &\ncompress"]
    box(ax, 0.7, 3.2, 1.1, 0.8, "RAW\nsensor", fc="#f4f4f4", bold=True)
    xs = np.linspace(2.15, 10.85, len(stages))
    for k, (x, s) in enumerate(zip(xs, stages)):
        box(ax, x, 3.2, 1.25, 0.8, s, fc="#e8f1fb", fs=9.5)
        arrow(ax, ((0.7 + 0.55) if k == 0 else xs[k - 1] + 0.63, 3.2), (x - 0.63, 3.2))
    box(ax, 12.3, 3.2, 1.1, 0.8, "sRGB\nphoto", fc="#f4f4f4", bold=True)
    arrow(ax, (xs[-1] + 0.63, 3.2), (11.75, 3.2))
    ax.text(6.5, 4.05, "Classic ISP: a chain of hand-designed, separately tuned stages",
            ha="center", fontsize=11, fontweight="bold")

    box(ax, 0.7, 1.0, 1.1, 0.8, "RAW\nsensor", fc="#f4f4f4", bold=True)
    box(ax, 6.5, 1.0, 9.0, 0.8, "One neural network trained on (RAW, target photo) pairs", fc="#fdf0e3")
    box(ax, 12.3, 1.0, 1.1, 0.8, "sRGB\nphoto", fc="#f4f4f4", bold=True)
    arrow(ax, (1.25, 1.0), (2.0, 1.0))
    arrow(ax, (11.0, 1.0), (11.75, 1.0))
    ax.text(6.5, 1.85, "Learned ISP: the same mapping learned end to end from data",
            ha="center", fontsize=11, fontweight="bold")
    save(fig, "isp-classic-vs-learned.png")


if __name__ == "__main__":
    fig_pixel_grid()
    fig_levels()
    fig_gray_levels()
    fig_spectrum()
    fig_ct()
    fig_steps()
    fig_system()
    fig_isp()
