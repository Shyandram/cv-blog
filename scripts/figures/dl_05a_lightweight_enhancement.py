"""Figures for DLCV 05A - Lightweight Image Enhancement.

Run:  .venv/bin/python scripts/figures/dl_05a_lightweight_enhancement.py
Writes PNGs to src/assets/deep-learning/dl-05a-lightweight-enhancement/ only.
All inputs come from skimage.data or synthetic arrays; nothing is downloaded.
Everything is NumPy/SciPy arithmetic and runs in a few seconds on a CPU.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from skimage import data, img_as_float  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio as psnr  # noqa: E402
from skimage.transform import resize  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-05a-lightweight-enhancement"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.titlesize": 10})


def save(fig, name, quantize=False):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(path).convert("RGB")
    if quantize:
        im = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    im.save(path, optimize=True)
    w, h = Image.open(path).size
    print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB, {w}x{h})")


def save_rgb(arr, name):
    a = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
    path = OUT / name
    Image.fromarray(a).save(path, optimize=True)
    print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB) {a.shape}")


def box(ax, xy, w, h, text, fc, fontsize=10):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=fc, ec="#333333",
                                lw=1.1))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, p, q, color="#333333"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.2))


# ---------------------------------------------------------------- shared toy code (same as the tutorial)
luma = lambda x: x @ np.array([0.299, 0.587, 0.114])  # noqa: E731


def local_brighten(x):
    Y = luma(x)
    base = gaussian_filter(Y, x.shape[1] / 40)
    Y2 = Y ** (0.35 + 0.9 * base)
    out = x * (Y2 / np.maximum(Y, 1e-3))[..., None]
    return np.clip(Y2[..., None] + 1.3 * (out - Y2[..., None]), 0, 1)


def corners(x, shape):
    H, W = x.shape[:2]
    gy, gx, gz = shape
    rows, cols = np.meshgrid(np.linspace(0, gy - 1, H), np.linspace(0, gx - 1, W), indexing="ij")
    u = np.stack([rows, cols, luma(x) * (gz - 1)], -1)
    i0 = np.minimum(u.astype(int), np.array(shape) - 2)
    d = u - i0
    for c in np.ndindex(2, 2, 2):
        w = np.prod(np.where(c, d, 1 - d), axis=-1)
        yield tuple((i0 + c).reshape(-1, 3).T), w.ravel()


def fit_grid(x, y, shape=(9, 13, 8), lam=1e-4):
    X = np.concatenate([x.reshape(-1, 3), np.ones((x[..., 0].size, 1))], 1)
    Y = y.reshape(-1, 3)
    XtX, XtY = np.zeros(shape + (4, 4)), np.zeros(shape + (4, 3))
    for idx, w in corners(x, shape):
        np.add.at(XtX, idx, w[:, None, None] * X[:, :, None] * X[:, None, :])
        np.add.at(XtY, idx, w[:, None, None] * X[:, :, None] * Y[:, None, :])
    eye = np.eye(4)[:, :3]
    return np.linalg.solve(XtX + lam * np.eye(4), XtY + lam * eye)


def slice_apply(A, x):
    X = np.concatenate([x.reshape(-1, 3), np.ones((x[..., 0].size, 1))], 1)
    Ap = sum(w[:, None, None] * A[idx] for idx, w in corners(x, A.shape[:3]))
    return np.clip(np.einsum("nk,nkc->nc", X, Ap).reshape(x.shape), 0, 1)


def fit_curves(x, y, bins=64):
    curves = []
    for c in range(3):
        k = np.minimum((x[..., c] * bins).astype(int), bins - 1).ravel()
        n = np.bincount(k, minlength=bins)
        mean = np.bincount(k, y[..., c].ravel(), bins) / np.maximum(n, 1)
        centers = (np.arange(bins) + 0.5) / bins
        curves.append((centers[n > 0], mean[n > 0]))
    return lambda z: np.stack([np.interp(z[..., c], *curves[c]) for c in range(3)], -1)


# ---------------------------------------------------------------- Figure 1: LUT lattice and trilinear interpolation
def fig_lut():
    fig = plt.figure(figsize=(10.6, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.35, 1])
    # (a) the lattice
    ax = fig.add_subplot(gs[0], projection="3d")
    M = 5
    g = np.linspace(0, 1, M)
    R, G, B = np.meshgrid(g, g, g, indexing="ij")
    ax.scatter(R.ravel(), G.ravel(), B.ravel(), s=6, c=np.stack([R, G, B], -1).reshape(-1, 3), depthshade=False)
    p = np.array([0.6, 0.35, 0.45])
    i0 = np.floor(p * (M - 1)) / (M - 1)
    h = 1 / (M - 1)
    for c in np.ndindex(2, 2, 2):
        q = i0 + np.array(c) * h
        ax.scatter(*q, s=40, c="k")
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color="#555555", lw=0.8)
    ax.scatter(*p, s=60, c="#d62728", marker="*")
    ax.set_xlabel("R", labelpad=-8)
    ax.set_ylabel("G", labelpad=-8)
    ax.set_zlabel("B", labelpad=-8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("(a) M x M x M lattice (M = 5 shown)")
    ax.view_init(elev=22, azim=-58)

    # (b) one cell, shown as its two faces (blue index k and k + 1)
    ax = fig.add_subplot(gs[1])
    dr, dg = 0.4, 0.65
    for z, (x0, lab) in enumerate([(0.0, "B index k"), (2.4, "B index k + 1")]):
        ax.add_patch(Rectangle((x0, 0), 1, 1, fc="#f2f2f2", ec="#333333"))
        for cr in (0, 1):
            for cg in (0, 1):
                ax.plot(x0 + cr, cg, "ko", ms=6)
                ax.text(x0 + cr + (0.07 if cr else -0.07), cg + (0.07 if cg else -0.16),
                        f"({cr},{cg},{z})", ha="left" if cr else "right", fontsize=9)
        ax.plot(x0 + dr, dg, marker="*", color="#d62728", ms=12)
        ax.text(x0 + 0.5, -0.42, lab, ha="center", fontsize=9.5)
        ax.annotate("", xy=(x0 + dr, -0.06), xytext=(x0, -0.06), arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax.text(x0 + dr / 2, -0.2, "dr", ha="center", fontsize=9)
    ax.text(1.7, 1.3, "corner weight = product over R, G, B of\nd (offset 1) or 1 - d (offset 0)",
            ha="center", fontsize=9)
    ax.text(1.7, 0.45, "db", ha="center", fontsize=9)
    ax.annotate("", xy=(2.35, 0.6), xytext=(1.05, 0.6), arrowprops=dict(arrowstyle="->", lw=0.8, color="#555555"))
    ax.set_xlim(-0.65, 4.05)
    ax.set_ylim(-0.6, 1.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(b) one cell: 8 corners, 8 weights")

    # (c) 1D view: lattice size vs a steep curve
    ax = fig.add_subplot(gs[2])
    x = np.linspace(0, 1, 1000)
    ax.plot(x, x ** (1 / 2.2), color="k", lw=1.8, label="exact x^(1/2.2)")
    for M, col in [(5, "#d62728"), (9, "#1f77b4")]:
        nodes = np.linspace(0, 1, M)
        ax.plot(x, np.interp(x, nodes, nodes ** (1 / 2.2)), color=col, lw=1.2, label=f"lattice M = {M}")
        ax.plot(nodes, nodes ** (1 / 2.2), "o", color=col, ms=4)
    ax.set_xlabel("input value")
    ax.set_ylabel("output value")
    ax.set_title("(c) linear pieces between nodes")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "lut-trilinear.png")


# ---------------------------------------------------------------- Figure 2: bilateral grid and slicing
def fig_grid():
    full = (0.15 * img_as_float(data.coffee()) ** 2.2) ** (1 / 2.2)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), gridspec_kw=dict(width_ratios=[1.05, 1]))
    ax = axes[0]
    row = 200
    Yrow = luma(full[row])
    xs = np.arange(full.shape[1])
    gx, gz = 13, 8
    for k in range(gz):
        ax.axhline(k / (gz - 1), color="#bbbbbb", lw=0.7)
    for j in range(gx):
        ax.axvline(j * (full.shape[1] - 1) / (gx - 1), color="#bbbbbb", lw=0.7)
    ax.plot(xs, Yrow, color="#1f77b4", lw=1.0, label="guide g = luma along one row")
    px = 330
    sx = (full.shape[1] - 1) / (gx - 1)                  # pixels per spatial cell
    j0, k0 = int(px / sx), int(Yrow[px] * (gz - 1))      # lower corner of the enclosing cell
    ax.add_patch(Rectangle((j0 * sx, k0 / (gz - 1)), sx, 1 / (gz - 1), fc="#ff7f0e", alpha=0.3, ec="#ff7f0e"))
    for jj in (j0, j0 + 1):
        for kk in (k0, k0 + 1):
            ax.plot(jj * sx, kk / (gz - 1), "o", color="#ff7f0e", ms=6, mec="k")
    ax.plot(px, Yrow[px], "*", color="#d62728", ms=12, label="pixel (x, g(x)) picks its cell")
    ax.set_xlabel("x position (pixels)")
    ax.set_ylabel("guide value g")
    ax.set_ylim(0, 1)
    ax.set_title("(a) grid nodes over (position, intensity); a pixel\nblends the affine maps of the nodes around its cell")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, (0.1, 4.4), 2.2, 1.0, "full-res\ninput", "#e8f0fb")
    box(ax, (3.0, 4.4), 2.2, 1.0, "downsample\n(e.g. 256 x 256)", "#f2f2f2")
    box(ax, (5.9, 4.4), 3.9, 1.0, "small CNN or least squares\n-> grid of 3x4 affine maps", "#fff2df")
    box(ax, (0.1, 1.4), 2.2, 1.0, "guide map g\n(per pixel)", "#e8f0fb")
    box(ax, (3.0, 1.4), 2.6, 1.0, "slice: trilinear\nlookup at (x, y, g)", "#fde8e8")
    box(ax, (6.3, 1.4), 1.6, 1.0, "apply\nA [r g b 1]", "#e9f6e9")
    box(ax, (8.4, 1.4), 1.5, 1.0, "full-res\noutput", "#e8f0fb")
    arrow(ax, (2.3, 4.9), (3.0, 4.9))
    arrow(ax, (5.2, 4.9), (5.9, 4.9))
    arrow(ax, (1.2, 4.4), (1.2, 2.4))
    arrow(ax, (2.3, 1.9), (3.0, 1.9))
    arrow(ax, (7.85, 4.4), (4.6, 2.4))
    arrow(ax, (5.6, 1.9), (6.3, 1.9))
    arrow(ax, (7.9, 1.9), (8.4, 1.9))
    arrow(ax, (1.6, 4.4), (6.8, 2.4), color="#888888")
    ax.text(5.0, 0.55, "expensive part: low resolution, fixed size\ncheap part: a few hundred MACs per full-res pixel",
            ha="center", fontsize=9.5)
    ax.set_title("(b) predict the transform at low resolution, apply it at full resolution")
    fig.tight_layout()
    save(fig, "bilateral-grid.png")


# ---------------------------------------------------------------- Figure 3: cost versus resolution (pure arithmetic)
def conv(cin, cout, k=3, scale=1.0, fixed=None):
    return dict(params=k * k * cin * cout + cout, macs=k * k * cin * cout, cout=cout, scale=scale, fixed=fixed)


def cost(layers, H, W, per_pixel_apply=0):
    px = [l["fixed"] or H * W * l["scale"] for l in layers]
    macs = sum(l["macs"] * p for l, p in zip(layers, px)) + per_pixel_apply * H * W
    peak = max(2 * l["cout"] * p for l, p in zip(layers, px))
    return sum(l["params"] for l in layers), macs, peak


def fig_cost():
    tiny = [conv(3, 32)] + [conv(32, 32) for _ in range(5)] + [conv(32, 24)]
    unet = [conv(3, 32), conv(32, 32), conv(32, 64, scale=1 / 4), conv(64, 64, scale=1 / 4),
            conv(64, 128, scale=1 / 16), conv(128, 128, scale=1 / 16),
            conv(128, 256, scale=1 / 64), conv(256, 256, scale=1 / 64),
            conv(256 + 128, 128, scale=1 / 16), conv(128, 128, scale=1 / 16),
            conv(128 + 64, 64, scale=1 / 4), conv(64, 64, scale=1 / 4),
            conv(64 + 32, 32), conv(32, 32), conv(32, 3, k=1)]
    grid_net = [conv(3, 8, fixed=128 ** 2), conv(8, 16, fixed=64 ** 2), conv(16, 32, fixed=32 ** 2),
                conv(32, 64, fixed=16 ** 2), conv(64, 64, fixed=16 ** 2), conv(64, 64, fixed=16 ** 2),
                conv(64, 96, k=1, fixed=16 ** 2)]
    apply = 9 + 48 + 8 * 12 + 12
    sizes = [(480, 640), (720, 1280), (1080, 1920), (1440, 2560), (2160, 3840), (3000, 4000), (4320, 7680)]
    mp = np.array([h * w / 1e6 for h, w in sizes])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for label, layers, extra, col in [("U-Net-like (1.9M params)", unet, 0, "#d62728"),
                                      ("tiny full-res CNN (54k params)", tiny, 0, "#ff7f0e"),
                                      ("low-res grid + slicing (105k params)", grid_net, apply, "#2ca02c")]:
        c = [cost(layers, h, w, extra) for h, w in sizes]
        axes[0].plot(mp, [m / 1e9 for _, m, _ in c], "o-", color=col, label=label)
        axes[1].plot(mp, [pk / 2 ** 20 for _, _, pk in c], "o-", color=col, label=label)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("image size (megapixels, log scale)")
        ax.grid(alpha=0.3, which="both")
        for x in (2.07, 8.29):
            ax.axvline(x, color="#888888", ls="--", lw=0.8)
    axes[0].set_ylabel("GMACs per image (log scale)")
    axes[0].set_title("(a) compute grows with pixels unless the\nnetwork runs at a fixed low resolution")
    axes[1].set_ylabel("largest feature map, fp16 (MiB, log)")
    axes[1].set_title("(b) memory traffic: one 32-channel map at 4K\nis about half a gigabyte")
    axes[0].legend(fontsize=9, loc="upper left")
    axes[1].text(0.3, 400, "U-Net-like and tiny CNN coincide:\nboth keep 32 channels at full resolution",
                 fontsize=9)
    for ax, top in zip(axes, (False, True)):
        lo, hi = ax.get_ylim()
        for x, t in [(2.07, "1080p"), (8.29, "4K")]:
            ax.text(x * 1.05, hi / 2.2 if top else lo * 1.25, t, fontsize=9, color="#555555")
    fig.tight_layout()
    save(fig, "cost-vs-resolution.png")


# ---------------------------------------------------------------- Figure 4: the toy comparison
def fig_toy():
    full = (0.15 * img_as_float(data.coffee()) ** 2.2) ** (1 / 2.2)
    low = resize(full, (100, 150), anti_aliasing=True)
    target_low = local_brighten(low)
    reference = local_brighten(full)
    outs = {
        "upsampled low-res output": resize(target_low, full.shape, order=1),
        "global tone curve": fit_curves(low, target_low)(full),
        "bilateral grid 9x13x8": slice_apply(fit_grid(low, target_low), full),
    }
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 4.6))
    axes[0, 0].imshow(full)
    axes[0, 0].set_title("dark input (15% light)")
    axes[1, 0].imshow(reference)
    axes[1, 0].set_title("operator at full resolution\n(reference)")
    for i, (name, out) in enumerate(outs.items(), start=1):
        axes[0, i].imshow(out)
        axes[0, i].set_title(f"{name}\nPSNR {psnr(reference, out, data_range=1):.2f} dB")
        err = np.abs(out - reference).mean(-1)
        axes[1, i].imshow(np.clip(err * 5, 0, 1), cmap="magma", vmin=0, vmax=1)
        axes[1, i].set_title("|error| x 5")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    save(fig, "toy-comparison.png", quantize=True)
    save_rgb(full, "toy-input.png")
    save_rgb(outs["bilateral grid 9x13x8"], "toy-grid.png")


# ---------------------------------------------------------------- Figure 5: banding
def fig_banding():
    W, H = 900, 60
    ramp = np.tile(np.linspace(0, 0.1, W), (H, 1))
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 3.2))
    for ax, bits in zip(axes, (8, 10, 12)):
        q = np.round(ramp * (2 ** bits - 1)) / (2 ** bits - 1)
        out = np.round(q ** (1 / 2.2) * 255) / 255
        ax.imshow(out, cmap="gray", vmin=0, vmax=0.4, aspect="auto")
        n = len(np.unique(out))
        ax.set_title(f"{bits}-bit dark gradient after a strong brightening curve: {n} distinct output levels",
                     fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    save(fig, "banding.png")


if __name__ == "__main__":
    fig_lut()
    fig_grid()
    fig_cost()
    fig_toy()
    fig_banding()
