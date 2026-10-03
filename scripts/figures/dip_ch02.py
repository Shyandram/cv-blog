"""Figures for DIP Chapter 2 - Digital Image Fundamentals.

Run:  .venv/bin/python scripts/figures/dip_ch02.py
Writes only to src/assets/dip/ch02/.
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from skimage import data, img_as_float  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "dip" / "ch02"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})
DPI = 150


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


camera = data.camera()  # 512x512 uint8


# 1. Mach bands and simultaneous contrast ------------------------------------
def fig_perception():
    fig = plt.figure(figsize=(10, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1.2], hspace=0.45, wspace=0.25)

    steps = np.repeat(np.linspace(0.15, 0.85, 8), 40)
    stair = np.tile(steps, (120, 1))
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(stair, cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Staircase of 8 flat gray bands")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(steps.size)
    ax.plot(x, steps, "k-", lw=2, label="actual intensity")
    # a sketch of what observers report: over/undershoot at each edge
    perceived = steps.copy()
    k = np.exp(-np.arange(0, 12) / 3.0) * 0.04
    for e in range(40, steps.size, 40):
        perceived[e:e + 12] += k
        perceived[e - 12:e] -= k[::-1]
    ax.plot(x, perceived, "--", color="tab:red", lw=1.5, label="perceived (sketch)")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("brightness")
    ax.legend(fontsize=10, loc="upper left")

    ax = fig.add_subplot(gs[:, 1])
    canvas = np.ones((200, 420))
    canvas[:, :200] = 0.15
    canvas[:, 220:] = 0.85
    canvas[:, 200:220] = 1.0
    canvas[70:130, 70:130] = 0.5
    canvas[70:130, 290:350] = 0.5
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Simultaneous contrast:\nboth inner squares are 0.5")
    ax.axis("off")
    save(fig, "mach-simcontrast.png")


# 2. Illumination-reflectance model ------------------------------------------
def fig_formation():
    r = img_as_float(data.coins())  # reflectance proxy in [0, 1]
    h, w = r.shape
    yy, xx = np.mgrid[0:h, 0:w]
    i = 0.15 + 0.85 * np.exp(-(((xx - 0.25 * w) ** 2) + ((yy - 0.3 * h) ** 2)) / (2 * (0.45 * w) ** 2))
    f = i * r
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, im, t in zip(axs, [i, r, f], ["illumination i(x, y)", "reflectance r(x, y)", "image f = i · r"]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(t)
        ax.axis("off")
    save(fig, "image-formation.png")


# 3. Intensity quantization sweep -------------------------------------------
def quantize(img, levels):
    step = 256 // levels
    return (img // step) * step + step // 2


def fig_quantization():
    levels = [256, 32, 8, 4, 2]
    crop = camera[40:360, 120:440]
    fig, axs = plt.subplots(1, 5, figsize=(13, 3.0))
    for ax, L in zip(axs, levels):
        q = crop if L == 256 else quantize(crop, L)
        ax.imshow(q, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{L} levels ({int(np.log2(L))} bits)")
        ax.axis("off")
    save(fig, "quantization.png")


# 4. Spatial resolution sweep -----------------------------------------------
def fig_spatial():
    sizes = [512, 128, 64, 32]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.3))
    for ax, n in zip(axs, sizes):
        small = cv2.resize(camera, (n, n), interpolation=cv2.INTER_AREA)
        ax.imshow(small, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{n} × {n}")
        ax.axis("off")
    save(fig, "spatial-resolution.png")


# 5. Neighborhoods and m-adjacency ------------------------------------------
def draw_grid(ax, n=5):
    for k in range(n + 1):
        ax.plot([0, n], [k, k], color="0.6", lw=0.8)
        ax.plot([k, k], [0, n], color="0.6", lw=0.8)
    ax.set_xlim(-0.1, n + 0.1)
    ax.set_ylim(n + 0.1, -0.1)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_neighbors():
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    sets = {
        "4-neighbors N4(p)": [(-1, 0), (1, 0), (0, -1), (0, 1)],
        "diagonal neighbors ND(p)": [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        "8-neighbors N8(p)": [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
    }
    for ax, (title, offs) in zip(axs[:3], sets.items()):
        draw_grid(ax)
        ax.add_patch(Rectangle((2, 2), 1, 1, color="black"))
        ax.text(2.5, 2.5, "p", color="white", ha="center", va="center", fontsize=13)
        for dy, dx in offs:
            ax.add_patch(Rectangle((2 + dx, 2 + dy), 1, 1, color="tab:orange", alpha=0.8))
        ax.set_title(title)

    # m-adjacency example on a 3x3 binary patch
    ax = axs[3]
    patch = np.array([[0, 1, 1], [0, 1, 0], [0, 0, 1]])
    draw_grid(ax, 3)
    for y in range(3):
        for x in range(3):
            if patch[y, x]:
                ax.add_patch(Rectangle((x, y), 1, 1, color="0.75"))
            ax.text(x + 0.12, y + 0.18, str(patch[y, x]), ha="left", va="center", fontsize=12)
    c = lambda y, x: (x + 0.5, y + 0.5)  # noqa: E731
    # 8-paths: ambiguous loop (0,1)-(0,2)-(1,1) all mutually 8-adjacent
    for a, b in [((0, 1), (0, 2)), ((0, 1), (1, 1)), ((1, 1), (2, 2))]:
        ax.plot(*zip(c(*a), c(*b)), color="tab:blue", lw=2.5)
    ax.plot(*zip(c(0, 2), c(1, 1)), color="tab:red", lw=2, ls="--")
    ax.set_title("m-adjacency: red diagonal\nis dropped (4-path exists)")
    save(fig, "neighbors.png")


# 6. Distance measures -------------------------------------------------------
def fig_distances():
    n = 9
    yy, xx = np.mgrid[-(n // 2):n // 2 + 1, -(n // 2):n // 2 + 1]
    maps = {
        "Euclidean  D_E": np.sqrt(xx ** 2 + yy ** 2),
        "city-block  D_4": np.abs(xx) + np.abs(yy),
        "chessboard  D_8": np.maximum(np.abs(xx), np.abs(yy)),
    }
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (t, m) in zip(axs, maps.items()):
        ax.imshow(np.minimum(m, 4.5), cmap="viridis")
        for y in range(n):
            for x in range(n):
                v = m[y, x]
                ax.text(x, y, f"{v:.1f}" if "E" in t else f"{int(v)}", ha="center", va="center",
                        fontsize=10, color="white" if v < 3 else "black")
        ax.set_title(t)
        ax.axis("off")
    save(fig, "distances.png")


# 7. Interpolation comparison + ImageCompare pair ---------------------------
def fig_interp():
    crop = camera[150:214, 230:294]  # 64x64 around the camera and hand
    up = 4
    methods = [("nearest", cv2.INTER_NEAREST), ("bilinear", cv2.INTER_LINEAR), ("bicubic", cv2.INTER_CUBIC)]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, (name, flag) in zip(axs, methods):
        big = cv2.resize(crop, None, fx=up, fy=up, interpolation=flag)
        ax.imshow(big, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{name}  (×{up})")
        ax.axis("off")
    save(fig, "interpolation.png")

    crop2 = camera[130:226, 210:306]  # 96x96 -> 384x384
    for name, flag in [("nearest", cv2.INTER_NEAREST), ("bicubic", cv2.INTER_CUBIC)]:
        big = cv2.resize(crop2, None, fx=4, fy=4, interpolation=flag)
        cv2.imwrite(str(OUT / f"upsample-{name}.png"), big)
        print("wrote", f"upsample-{name}.png", big.shape)


# 8. Averaging noisy images -------------------------------------------------
def fig_averaging():
    rng = np.random.default_rng(0)
    g = img_as_float(data.camera())[40:360, 120:440]
    sigma = 0.25
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, K in zip(axs, [1, 4, 16, 64]):
        avg = np.mean([g + rng.normal(0, sigma, g.shape) for _ in range(K)], axis=0)
        ax.imshow(np.clip(avg, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"K = {K}  (noise std ≈ {sigma / np.sqrt(K):.3f})")
        ax.axis("off")
    save(fig, "averaging.png")


if __name__ == "__main__":
    fig_perception()
    fig_formation()
    fig_quantization()
    fig_spatial()
    fig_neighbors()
    fig_distances()
    fig_interp()
    fig_averaging()
