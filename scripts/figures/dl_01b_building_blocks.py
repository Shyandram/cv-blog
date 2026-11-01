"""Figures for DLCV 01B - Building Blocks and How to Compose Them.

Run:  .venv/bin/python scripts/figures/dl_01b_building_blocks.py
Writes only to src/assets/deep-learning/dl-01b-building-blocks/.
Block diagrams are drawn with matplotlib; the data figures use synthetic arrays and
skimage.data.camera(); nothing is downloaded.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from scipy.ndimage import correlate, gaussian_filter  # noqa: E402
from skimage import data  # noqa: E402
from skimage.transform import resize  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-01b-building-blocks"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})

INK, MUTED = "#0b0b0b", "#52514e"
KIND = {  # fill colour per operation type
    "conv": "#cfe0f7",   # dense convolution / linear layer
    "dw": "#c9eedc",     # depthwise convolution
    "norm": "#e4e3df",   # normalization
    "act": "#fde2c4",    # activation
    "gate": "#f7d0e0",   # gating / multiplicative attention
    "attn": "#ddd5f5",   # token mixing by attention
    "other": "#fff1b8",  # pooling, scaling, resampling
}
LEGEND = [("dense conv / linear", "conv"), ("depthwise conv", "dw"), ("normalization", "norm"),
          ("activation", "act"), ("gate (multiplicative)", "gate"), ("attention", "attn"),
          ("pool / scale / resample", "other")]
SKIP = "#d9541e"


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    kb = path.stat().st_size / 1024
    w = plt.imread(path).shape[1]
    print(f"{name}: {kb:.0f} KB, width {w} px")
    assert kb < 1024 and w <= 1600


# ------------------------------------------------------------------ figure 1: block catalogue
def draw_block(ax, title, items, subtitle=""):
    """items: list of (label, kind) boxes, 'ADD' for a residual addition, 'MUL' for a product node.
    A skip arrow runs from the block input (or the previous ADD) to each ADD."""
    ax.set_xlim(0, 10)
    n = len(items)
    top = n + 1.2
    ax.set_ylim(-0.6, top + 0.9)
    ax.axis("off")
    ax.text(5, top + 0.55, title, ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
    if subtitle:
        ax.text(5, top + 0.05, subtitle, ha="center", va="center", fontsize=10, color=MUTED)
    y = top - 0.6
    ax.text(5.75, y + 0.15, "x", ha="center", va="center", fontsize=11, color=INK, style="italic")
    src = y
    prev = y - 0.1
    for it in items:
        y -= 1.0
        ax.add_patch(FancyArrowPatch((5.75, prev), (5.75, y + 0.32), arrowstyle="-|>", mutation_scale=9,
                                     color=MUTED, lw=1.0))
        if it == "ADD":
            ax.add_patch(plt.Circle((5.75, y), 0.3, fc="white", ec=INK, lw=1.2))
            ax.text(5.75, y, "+", ha="center", va="center", fontsize=12, color=INK)
            ax.add_patch(FancyArrowPatch((0.9, src), (5.75 - 0.3, y), connectionstyle="angle,angleA=-90,angleB=180,rad=0",
                                         arrowstyle="-|>", mutation_scale=9, color=SKIP, lw=1.4))
            ax.plot([5.75, 0.9], [src, src], color=SKIP, lw=1.4)
            src = y
            prev = y - 0.3
        else:
            label, kind = it
            ax.add_patch(FancyBboxPatch((1.6, y - 0.3), 8.3, 0.6, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        fc=KIND[kind], ec="#8a8984", lw=0.8))
            ax.text(5.75, y, label, ha="center", va="center", fontsize=10, color=INK)
            prev = y - 0.3
    ax.add_patch(FancyArrowPatch((5.75, prev), (5.75, prev - 0.45), arrowstyle="-|>", mutation_scale=9,
                                 color=MUTED, lw=1.0))


def fig_catalogue():
    blocks = [
        ("(a) Basic residual", "ResNet-18/34", [
            ("3×3 conv", "conv"), ("BN", "norm"), ("ReLU", "act"), ("3×3 conv", "conv"), ("BN", "norm"),
            "ADD", ("ReLU", "act")]),
        ("(b) Bottleneck", "ResNet-50 and deeper", [
            ("1×1 conv, C → C/4", "conv"), ("BN, ReLU", "norm"), ("3×3 conv", "conv"), ("BN, ReLU", "norm"),
            ("1×1 conv, C/4 → C", "conv"), ("BN", "norm"), "ADD", ("ReLU", "act")]),
        ("(c) Inverted residual", "MobileNetV2", [
            ("1×1 conv, C → tC", "conv"), ("BN, ReLU6", "norm"), ("3×3 depthwise", "dw"), ("BN, ReLU6", "norm"),
            ("1×1 conv, tC → C", "conv"), ("BN (no activation)", "norm"), "ADD"]),
        ("(d) ConvNeXt block", "ConvNeXt", [
            ("7×7 depthwise", "dw"), ("LayerNorm", "norm"), ("1×1 conv, C → 4C", "conv"), ("GELU", "act"),
            ("1×1 conv, 4C → C", "conv"), ("× layer scale", "other"), "ADD"]),
        ("(e) SE and CBAM gates", "inside any residual branch", [
            ("conv layers F(x)", "conv"), ("channel gate (SE, CBAM)", "gate"), ("pool → MLP → sigmoid", "gate"),
            ("spatial gate (CBAM)", "gate"), ("7×7 conv → sigmoid", "gate"), "ADD"]),
        ("(f) Pre-norm Transformer", "ViT, Swin, SwinIR", [
            ("LayerNorm", "norm"), ("self-attention", "attn"), ("(global or windowed)", "attn"), "ADD",
            ("LayerNorm", "norm"), ("MLP: C → 4C, GELU", "conv"), ("MLP: 4C → C", "conv"), "ADD"]),
        ("(g) NAFNet block", "no activation functions", [
            ("LayerNorm", "norm"), ("1×1 conv, C → 2C", "conv"), ("3×3 depthwise", "dw"), ("SimpleGate → C", "gate"),
            ("simplified ch. attention", "gate"), ("1×1 conv", "conv"), ("× β (zero init)", "other"), "ADD",
            ("LayerNorm", "norm"), ("1×1 conv, C → 2C", "conv"), ("SimpleGate → C", "gate"), ("1×1 conv", "conv"),
            ("× γ (zero init)", "other"), "ADD"]),
        ("(h) Restormer block", "MDTA + GDFN", [
            ("LayerNorm", "norm"), ("Q, K, V: 1×1 + 3×3 dw", "dw"), ("C × C channel attention", "attn"),
            ("1×1 conv", "conv"), "ADD", ("LayerNorm", "norm"), ("two paths: 1×1 + 3×3 dw", "dw"),
            ("GELU(path 1) ⊙ path 2", "gate"), ("1×1 conv", "conv"), "ADD"]),
    ]
    rows = [blocks[:4], blocks[4:]]
    heights = [max(len(b[2]) for b in r) + 2.7 for r in rows]
    fig = plt.figure(figsize=(10.6, 0.37 * sum(heights) + 0.6))
    gs = fig.add_gridspec(2, 4, height_ratios=heights, hspace=0.04, wspace=0.02)
    for r, row in enumerate(rows):
        for c, (title, sub, items) in enumerate(row):
            ax = fig.add_subplot(gs[r, c])
            draw_block(ax, title, items, sub)
            ax.set_ylim(len(items) + 2.1 - heights[r], len(items) + 2.1)
    handles = [Patch(fc=KIND[k], ec="#8a8984", label=lab) for lab, k in LEGEND]
    handles.append(Patch(fc="white", ec=SKIP, label="skip (identity) path"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, "block-catalogue.png")


# ------------------------------------------------------------------ figure 2: macro-architectures
def box(ax, x, y, w, h, text, kind="conv", fs=10, rot=0):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=KIND[kind], ec="#8a8984", lw=0.8))
    ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs, color=INK, rotation=rot)


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, color=color, lw=lw,
                                 linestyle=ls))


def plus(ax, x, y, r=0.27):
    ax.add_patch(plt.Circle((x, y), r, fc="white", ec=INK, lw=1.2))
    ax.text(x, y, "+", ha="center", va="center", fontsize=12, color=INK)


def fig_macro():
    fig, axes = plt.subplots(3, 1, figsize=(10.6, 9.8), gridspec_kw={"height_ratios": [1, 1, 1.35]})
    for ax in axes:
        ax.set_xlim(0, 20)
        ax.axis("off")

    # (a) classification: a funnel
    ax = axes[0]
    ax.set_ylim(-2.2, 2.4)
    ax.text(0, 2.15, "(a) Classification: shrink the map, widen the features, pool, classify (ResNet, Swin, ConvNeXt)",
            fontsize=11.5, fontweight="bold", color=INK)
    box(ax, 0.1, 0, 1.7, 3.0, "image\nH × W × 3", "other")
    arrow(ax, 1.85, 0, 2.15, 0)
    box(ax, 2.2, 0, 1.4, 2.3, "stem\n(÷4)", "conv")
    x = 4.0
    for i, (t, h) in enumerate([("stage 1\nH/4, C", 1.9), ("stage 2\nH/8, 2C", 1.45),
                                ("stage 3\nH/16, 4C", 1.05), ("stage 4\nH/32, 8C", 0.8)]):
        arrow(ax, x - 0.38, 0, x - 0.02, 0)
        box(ax, x, 0, 1.6, h, t, "conv")
        if i > 0:
            ax.text(x - 0.2, -1.35, "↓2", ha="center", fontsize=10, color=MUTED)
        x += 2.0
    arrow(ax, x - 0.38, 0, x - 0.02, 0)
    box(ax, x, 0, 1.9, 0.9, "global\navg pool", "other")
    arrow(ax, x + 1.95, 0, x + 2.35, 0)
    box(ax, x + 2.4, 0, 1.3, 0.7, "linear", "conv")
    arrow(ax, x + 3.75, 0, x + 4.1, 0)
    ax.text(x + 4.15, 0, "class\nscores", va="center", fontsize=10, color=INK)
    ax.text(10, -1.95, "box height ~ spatial size: resolution falls 32-fold, channels grow 8-fold, the receptive field covers the image",
            ha="center", fontsize=10, color=MUTED)

    # (b) restoration, plain full resolution
    ax = axes[1]
    ax.set_ylim(-2.2, 2.4)
    ax.text(0, 2.15, "(b) Restoration, plain: stay at full resolution (EDSR, RCAN, SwinIR)",
            fontsize=11.5, fontweight="bold", color=INK)
    box(ax, 0.1, 0, 1.7, 2.3, "input\nH × W × 3", "other")
    arrow(ax, 1.85, 0, 2.15, 0)
    box(ax, 2.2, 0, 1.9, 2.3, "conv\n(shallow\nfeatures)", "conv")
    x = 4.55
    for i in range(4):
        arrow(ax, x - 0.43, 0, x - 0.02, 0)
        box(ax, x, 0, 1.5, 2.3, f"residual\ngroup {i + 1}" if i < 3 else "residual\ngroup K", "conv")
        x += 1.95
    arrow(ax, x - 0.43, 0, x - 0.02, 0)
    box(ax, x, 0, 1.0, 2.3, "conv", "conv")
    p1 = x + 1.55
    arrow(ax, x + 1.02, 0, p1 - 0.29, 0)
    plus(ax, p1, 0)
    ax.plot([3.15, 3.15, p1], [1.17, 1.6, 1.6], color=SKIP, lw=1.4)
    arrow(ax, p1, 1.6, p1, 0.29, color=SKIP, lw=1.4)
    ax.text(8.2, 1.73, "long skip: shallow features bypass the body", fontsize=10, color=SKIP, ha="center")
    u = p1 + 0.55
    arrow(ax, p1 + 0.28, 0, u - 0.02, 0)
    box(ax, u, 0, 2.0, 1.4, "upsampler\n(SR only)", "other")
    p2 = u + 2.55
    arrow(ax, u + 2.02, 0, p2 - 0.29, 0)
    plus(ax, p2, 0)
    arrow(ax, p2 + 0.28, 0, p2 + 0.7, 0)
    ax.text(p2 + 0.75, 0, "output", fontsize=10, color=INK, va="center")
    ax.plot([0.95, 0.95, p2], [-1.17, -1.55, -1.55], color=SKIP, lw=1.4, ls="--")
    arrow(ax, p2, -1.55, p2, -0.29, color=SKIP, lw=1.4, ls="--")
    ax.text(10, -1.98, "global residual (dashed), used when output and input have the same size: output = input + predicted correction",
            ha="center", fontsize=10, color=MUTED)

    # (c) restoration, U-shaped
    ax = axes[2]
    ax.set_ylim(-3.7, 2.9)
    ax.text(0, 2.65, "(c) Restoration, U-shaped: go down for context, come back for detail (U-Net, NAFNet, Restormer)",
            fontsize=11.5, fontweight="bold", color=INK)
    lv = [0.7, -0.35, -1.3, -2.15]
    hs = [1.5, 1.15, 0.85, 0.85]
    box(ax, 0.1, lv[0], 1.5, 1.5, "input", "other")
    arrow(ax, 1.65, lv[0], 1.95, lv[0])
    xs_enc = [2.0, 4.1, 6.2]
    xs_dec = [14.9, 12.8, 11.0]
    for i in range(3):
        box(ax, xs_enc[i], lv[i], 1.7, hs[i], f"enc {i + 1}\n" + ["H, C", "H/2, 2C", "H/4, 4C"][i], "conv")
        box(ax, xs_dec[i], lv[i], 1.7, hs[i], f"dec {i + 1}\n" + ["H, C", "H/2, 2C", "H/4, 4C"][i], "conv")
        nx = xs_enc[i + 1] if i < 2 else 8.1
        arrow(ax, xs_enc[i] + 1.72, lv[i] - 0.15, nx + 0.05, lv[i + 1] + 0.2)
        sx = xs_dec[i + 1] + 1.65 if i < 2 else 10.62
        arrow(ax, sx, lv[i + 1] + 0.2, xs_dec[i] - 0.03, lv[i] - 0.15)
        top = 2.25 - 0.3 * i
        ax.plot([xs_enc[i] + 0.85] * 2, [lv[i] + hs[i] / 2, top], color=SKIP, lw=1.3)
        ax.plot([xs_enc[i] + 0.85, xs_dec[i] + 0.85], [top, top], color=SKIP, lw=1.3)
        arrow(ax, xs_dec[i] + 0.85, top, xs_dec[i] + 0.85, lv[i] + hs[i] / 2 + 0.02, color=SKIP, lw=1.3)
    box(ax, 8.1, lv[3], 2.5, 0.85, "bottleneck\nH/8, 8C", "conv")
    arrow(ax, 16.63, lv[0], 16.97, lv[0])
    box(ax, 17.0, lv[0], 1.0, 0.9, "conv", "conv")
    plus(ax, 18.55, lv[0])
    arrow(ax, 18.02, lv[0], 18.26, lv[0])
    arrow(ax, 18.83, lv[0], 19.25, lv[0])
    ax.text(19.15, lv[0] + 0.45, "output", fontsize=10, color=INK, ha="center")
    ax.plot([0.85, 0.85, 18.55], [lv[0] - 0.77, -3.05, -3.05], color=SKIP, lw=1.4, ls="--")
    arrow(ax, 18.55, -3.05, 18.55, lv[0] - 0.28, color=SKIP, lw=1.4, ls="--")
    ax.text(10, -3.55, "skips (solid): add (NAFNet) or concatenate (U-Net; Restormer then applies a 1×1 conv); "
            "global residual (dashed)", ha="center", fontsize=10, color=MUTED)
    save(fig, "macro-architectures.png")


# ------------------------------------------------------------------ figure 3: receptive field vs depth and compute
def analyze(layers, H=256):
    """Exact 1-D receptive field (max over 16 output positions) and MACs, as in the chapter's snippet."""
    macs, size = 0, H
    for layer in layers:
        if layer[0] == "conv":
            _, k, s, cin, cout = layer
            size //= s
            macs += size * size * k * k * cin * cout
        else:
            size *= layer[1]
    rf = 0
    for p0 in range(16):
        lo = hi = p0
        for layer in reversed(layers):
            if layer[0] == "conv":
                _, k, s, _, _ = layer
                pad = (k - 1) // 2
                lo, hi = s * lo - pad, s * hi - pad + k - 1
            else:
                lo, hi = lo // layer[1], hi // layer[1]
        rf = max(rf, hi - lo + 1)
    return rf, macs, size


def networks():
    resnet = [("conv", 7, 2, 3, 64), ("conv", 3, 2, 64, 0)]
    cin = 64
    for width, nb, stride in [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]:
        for b in range(nb):
            s = stride if b == 0 else 1
            resnet += [("conv", 1, 1, cin, width), ("conv", 3, s, width, width), ("conv", 1, 1, width, 4 * width)]
            if b == 0:
                resnet += [("conv", 1, 1, cin, 4 * width)]
            cin = 4 * width
    plain = [("conv", 3, 1, 3, 64)] + [("conv", 3, 1, 64, 64)] * 32 + [("conv", 3, 1, 64, 3)]
    w = [32, 64, 128, 256, 512]
    unet = [("conv", 3, 1, 3, w[0])]
    for i in range(4):
        unet += [("conv", 3, 1, w[i], w[i])] * 2 + [("conv", 2, 2, w[i], w[i + 1])]
    unet += [("conv", 3, 1, w[4], w[4])] * 2
    for i in reversed(range(4)):
        unet += [("up", 2), ("conv", 1, 1, w[i + 1], w[i]), ("conv", 3, 1, 2 * w[i], w[i]), ("conv", 3, 1, w[i], w[i])]
    unet += [("conv", 3, 1, w[0], 3)]
    return {"ResNet-50 (classification)": resnet, "plain, 64 ch, 34 convs (restoration)": plain,
            "4-level U-Net (restoration)": unet}


def fig_receptive_field():
    nets = networks()
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    for (name, net), col in zip(nets.items(), colors):
        xs, rfs, macs = [], [], []
        for i in range(1, len(net) + 1):
            if net[i - 1][0] == "conv":
                rf, m, _ = analyze(net[:i])
                xs.append(sum(1 for L in net[:i] if L[0] == "conv" and L[4] > 0))
                rfs.append(rf)
                macs.append(m / 1e9)
        a1.plot(xs, rfs, color=col, lw=2, label=name)
        a2.plot(macs, rfs, color=col, lw=2, label=name)
        a2.plot(macs[-1], rfs[-1], "o", color=col)
        off, ha = ((-10, 8), "right") if macs[-1] > 40 else ((9, -4), "left")
        a2.annotate(f"{rfs[-1]} px, {macs[-1]:.1f} GMACs", (macs[-1], rfs[-1]), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=10, color=col)
    a1.set_xlabel("convolution layers so far")
    a1.set_ylabel("receptive field (pixels, one side)")
    a1.set_yscale("log")
    a1.set_title("Receptive field vs depth")
    a2.set_xlabel("cumulative GMACs on a 256 × 256 input")
    a2.set_yscale("log")
    a2.set_title("Receptive field vs compute")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    a1.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    save(fig, "receptive-field-growth.png")


# ------------------------------------------------------------------ figure 4: checkerboard toy
def transposed_conv(x, w, s):
    k, (H, W) = w.shape[0], x.shape
    out = np.zeros((s * (H - 1) + k, s * (W - 1) + k))
    for i in range(H):
        for j in range(W):
            out[s * i:s * i + k, s * j:s * j + k] += x[i, j] * w
    return out


def resize_conv(x, w, s):
    return correlate(np.kron(x, np.ones((s, s))), w, mode="nearest")


def subpixel_conv(x, ws, s):
    out = np.zeros((s * x.shape[0], s * x.shape[1]))
    for n, w in enumerate(ws):
        out[n // s::s, n % s::s] = correlate(x, w, mode="nearest")
    return out


def fig_checkerboard():
    rng = np.random.default_rng(0)            # same draws, in the same order, as the chapter's snippet
    w3u = np.ones((3, 3))
    w4r = rng.uniform(0.5, 1.5, (4, 4))
    wsp = rng.uniform(0.5, 1.5, (4, 3, 3))
    wrc = rng.uniform(0.5, 1.5, (3, 3))
    methods = [
        ("transposed conv\nk=3, s=2, uniform", lambda x: transposed_conv(x, w3u / 2.25, 2)[1:-2, 1:-2]),
        ("transposed conv\nk=4, s=2, random", lambda x: transposed_conv(x, w4r / (w4r.sum() / 4), 2)[1:-3, 1:-3]),
        ("sub-pixel conv\n4 random 3×3", lambda x: subpixel_conv(x, wsp / wsp.sum(axis=(1, 2)).mean(), 2)),
        ("nearest resize + conv\nrandom 3×3", lambda x: resize_conv(x, wrc / wrc.sum(), 2)),
    ]
    flat = np.ones((16, 16))
    cam = data.camera().astype(float) / 255
    small = resize(cam[60:316, 150:406], (64, 64), anti_aliasing=True)
    fig, axes = plt.subplots(2, 4, figsize=(10.6, 5.9))
    for j, (title, f) in enumerate(methods):
        y = [transposed_conv(flat, w3u, 2), transposed_conv(flat, w4r, 2),
             subpixel_conv(flat, wsp, 2), resize_conv(flat, wrc, 2)][j][4:-4, 4:-4]   # the snippet's crops
        axes[0, j].imshow(y / y.mean(), cmap="gray", vmin=0.4, vmax=1.6, interpolation="nearest")
        axes[0, j].set_title(title, fontsize=10.5)
        axes[0, j].set_xlabel(f"flat input: spread {np.ptp(y) / y.mean():.2f}", fontsize=10)
        z = f(small)[8:-8, 8:-8]
        axes[1, j].imshow(z[:72, :72], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1, j].set_xlabel("camera crop, upsampled ×2", fontsize=10)
        for a in axes[:, j]:
            a.set_xticks([])
            a.set_yticks([])
    fig.tight_layout()
    save(fig, "checkerboard.png")


# ------------------------------------------------------------------ figure 5: BN versus per-sample normalization
def fig_normalization():
    rng = np.random.default_rng(0)            # identical to the chapter's snippet

    def image(level, size=32):
        t = gaussian_filter(rng.standard_normal((size, size)), 2)
        return level + 0.15 * t / t.std()

    C = 4
    W1 = rng.normal(0.1, 0.2, (C, 3, 3))
    w2 = rng.normal(0.0, 0.05, C)

    def features(batch):
        return np.stack([[correlate(x, W1[c], mode="nearest") for c in range(C)] for x in batch])

    def branch(batch, norm, running=None):
        f = features(batch)
        if norm == "bn-train":
            mu, var = f.mean((0, 2, 3), keepdims=True), f.var((0, 2, 3), keepdims=True)
        elif norm == "bn-eval":
            mu, var = running
        else:
            mu, var = f.mean((2, 3), keepdims=True), f.var((2, 3), keepdims=True)
        h = np.maximum((f - mu) / np.sqrt(var + 1e-5), 0)
        return np.tensordot(w2, h, axes=(0, 1))

    train = np.stack([image(rng.uniform(0.3, 0.7)) for _ in range(64)])
    f = features(train)
    running = (f.mean((0, 2, 3), keepdims=True), f.var((0, 2, 3), keepdims=True))
    A = image(0.5)
    dark = np.stack([image(0.15) for _ in range(7)])
    bright = np.stack([image(0.85) for _ in range(7)])
    panels = [("input A", A)]
    for norm, label in [("bn-train", "BN (train mode)"), ("instance", "instance norm")]:
        for name, others in [("dark", dark), ("bright", bright)]:
            out = A + branch(np.concatenate([A[None], others]), norm, running)[0]
            if norm == "instance" and name == "bright":
                continue
            tag = f"{label}\nwith 7 {name} images" if norm == "bn-train" else f"{label}\n(any batch)"
            panels.append((tag, out))
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.3))
    for a, (t, im) in zip(axes, panels):
        a.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        a.set_title(t, fontsize=10.5)
        a.set_xlabel(f"mean {im.mean():.3f}", fontsize=10.5)
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    save(fig, "normalization-range.png")


if __name__ == "__main__":
    fig_catalogue()
    fig_macro()
    fig_receptive_field()
    fig_checkerboard()
    fig_normalization()
