"""Figures for DL 05 - Low-Light Image Enhancement.

Run from the repo root:
    .venv/bin/python scripts/figures/dl_05_low_light_enhancement.py
Writes PNGs to src/assets/deep-learning/dl-05-low-light-enhancement/.
All inputs are skimage.data images or synthetic arrays; the zero-reference toy
trains from scratch on CPU (capped at ~50 s).
"""
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, data, exposure, img_as_float
from skimage.transform import resize

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-05-low-light-enhancement"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
RNG = np.random.default_rng(7)
torch.manual_seed(0)
torch.set_num_threads(1)


def save(fig, name, colors=256):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(path).convert("RGB")
    if im.width > 1600:
        im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
    if colors:
        im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    else:  # photographic panels: keep full color to avoid banding
        im.save(path, optimize=True)
    print(f"{name}: {path.stat().st_size / 1024:.0f} KB")


def save_rgb(img, name):
    """Save a float RGB image at its own size, without palette quantization."""
    path = OUT / name
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, optimize=True)
    print(f"{name}: {path.stat().st_size / 1024:.0f} KB")


def show(ax, img, title, cmap=None):
    ax.imshow(np.clip(img, 0, 1), cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


# --- Low-light synthesis ------------------------------------------------------
GAMMA = 2.2  # simple stand-in for the sRGB transfer curve


def to_linear(x):
    return np.clip(x, 0, 1) ** GAMMA


def to_display(x):
    return np.clip(x, 0, 1) ** (1 / GAMMA)


def synth_low_light(clean, k=0.05, peak=2000.0, read_std=0.0008, rng=RNG):
    """Darken in linear light, add Poisson-Gaussian noise, re-encode, quantize to 8 bits."""
    lin = k * to_linear(clean)
    noisy = rng.poisson(lin * peak) / peak + rng.normal(0.0, read_std, lin.shape)
    out = to_display(noisy)
    return np.round(out * 255) / 255


clean = img_as_float(data.coffee())                # 400 x 600 x 3
dark_clean = to_display(0.05 * to_linear(clean))     # darkened, no noise
dark_noisy = synth_low_light(clean)                  # darkened + noise


def luma(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114])


# --- Figure 1: synthetic low-light degradation ---------------------------------
def fig_degradation():
    naive = to_display(to_linear(dark_noisy) / 0.05)   # undo the exposure only
    imgs = [(clean, "Clean (normal light)"),
            (dark_clean, "x 0.05 light (linear)"),
            (dark_noisy, "+ Poisson-Gaussian noise"),
            (naive, "Noisy input x 20 (linear)")]
    fig, axs = plt.subplots(2, 4, figsize=(14, 5.6), gridspec_kw={"height_ratios": [3, 1.4]})
    for j, (im, t) in enumerate(imgs):
        show(axs[0, j], im, t)
        ax = axs[1, j]
        ax.hist(luma(np.clip(im, 0, 1)).ravel(), bins=64, range=(0, 1), color="0.35")
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("luma Y")
    fig.tight_layout()
    save(fig, "degradation.png", colors=None)


# --- Figure 2: brightness channels in different color spaces ----------------------
def rgb_to_hsi_intensity(rgb):
    return rgb.mean(axis=2)


def enhance_channel(rgb, space, g=0.4):
    """Apply s = r^g to one brightness channel, keep the chroma channels."""
    if space == "RGB":
        return rgb ** g
    if space == "HSV":
        hsv = color.rgb2hsv(rgb)
        hsv[..., 2] = hsv[..., 2] ** g
        return color.hsv2rgb(hsv)
    if space == "HSI":
        i = rgb.mean(axis=2, keepdims=True)
        return np.clip(rgb * (i ** g) / np.maximum(i, 1e-6), 0, 1)
    if space == "YCbCr":
        y = luma(rgb)
        cb = (rgb[..., 2] - y) / 1.772
        cr = (rgb[..., 0] - y) / 1.402
        y2 = y ** g
        r = y2 + 1.402 * cr
        b = y2 + 1.772 * cb
        gg = (y2 - 0.299 * r - 0.114 * b) / 0.587
        return np.clip(np.dstack([r, gg, b]), 0, 1)
    if space == "Lab":
        lab = color.rgb2lab(rgb)
        lab[..., 0] = 100 * (lab[..., 0] / 100) ** g
        return np.clip(color.lab2rgb(lab), 0, 1)
    raise ValueError(space)


def fig_color_spaces():
    x = dark_clean
    chans = [(x, "Input (RGB)"),
             (x.max(axis=2), "HSV: V = max(R,G,B)"),
             (rgb_to_hsi_intensity(x), "HSI: I = mean(R,G,B)"),
             (luma(x), "YCbCr: Y (BT.601)"),
             (color.rgb2lab(x)[..., 0] / 100, "Lab: L* / 100")]
    outs = ["RGB", "HSV", "HSI", "YCbCr", "Lab"]
    fig, axs = plt.subplots(2, 5, figsize=(16, 5.2))
    for j, (c, t) in enumerate(chans):
        show(axs[0, j], c * (1 if j == 0 else 3), t + ("" if j == 0 else "  (x3 for display)"),
             cmap=None if j == 0 else "gray")
        names = {"RGB": "gamma 0.4 on R, G, B", "HSV": "gamma 0.4 on V only",
                 "HSI": "gamma 0.4 on I only", "YCbCr": "gamma 0.4 on Y only",
                 "Lab": "gamma 0.4 on L* only"}
        show(axs[1, j], enhance_channel(x, outs[j]), names[outs[j]])
    for ax in axs[0, 1:]:
        ax.title.set_fontsize(10)
    fig.tight_layout()
    save(fig, "color-spaces.png", colors=None)


# --- Figure 3 + 4: classic methods and Retinex ---------------------------------
def single_scale_retinex(rgb, sigma=40, eps=1e-3):
    """SSR: log I - log(G_sigma * I), per channel, then a robust stretch to [0, 1]."""
    log_i = np.log(rgb + eps)
    log_l = np.log(ndi.gaussian_filter(rgb, sigma=(sigma, sigma, 0)) + eps)
    r = log_i - log_l
    lo, hi = np.percentile(r, [1, 99])
    return np.clip((r - lo) / (hi - lo), 0, 1)


def multi_scale_retinex(rgb, sigmas=(15, 80, 250), eps=1e-3):
    log_i = np.log(rgb + eps)
    r = np.mean([log_i - np.log(ndi.gaussian_filter(rgb, sigma=(s, s, 0)) + eps) for s in sigmas], axis=0)
    lo, hi = np.percentile(r, [1, 99])
    return np.clip((r - lo) / (hi - lo), 0, 1)


def he_on_y(rgb):
    ycc = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB) / 255.0


def clahe_on_l(rgb, clip=0.01):
    lab = color.rgb2lab(rgb)
    lab[..., 0] = 100 * exposure.equalize_adapthist(lab[..., 0] / 100, clip_limit=clip)
    return np.clip(color.lab2rgb(lab), 0, 1)


def fig_classic():
    x = dark_noisy
    panels = [(x, "Low-light input (noisy)"),
              (x ** 0.4, "Gamma 0.4 (RGB)"),
              (he_on_y(x), "Global HE on Y"),
              (clahe_on_l(x), "CLAHE on L*"),
              (single_scale_retinex(x), "SSR (sigma = 40)"),
              (multi_scale_retinex(x), "MSR (15, 80, 250)")]
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.2))
    for ax, (im, t) in zip(axs.flat, panels):
        show(ax, im, t)
    fig.tight_layout()
    save(fig, "classic-methods.png")


def fig_retinex():
    x = dark_clean
    l0 = x.max(axis=2)                                   # LIME-style initial illumination
    l = ndi.gaussian_filter(l0, 6)                         # crude smoothing stands in for LIME's refinement
    l = np.maximum(l, 0.02)
    r = np.clip(x / l[..., None], 0, 1)
    enhanced = np.clip(r * (l ** 0.35)[..., None], 0, 1)   # brighten L only, recombine
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.4))
    show(axs[0], x, "Input S (dark)")
    show(axs[1], l0 * 3, "max(R,G,B)  (x3)", cmap="gray")
    show(axs[2], l * 3, "Smoothed L  (x3)", cmap="gray")
    show(axs[3], r, "Reflectance R = S / L")
    show(axs[4], enhanced, "R * L^0.35")
    fig.tight_layout()
    save(fig, "retinex-decomposition.png", colors=None)


# --- Figure 5: LE curves ------------------------------------------------------
def fig_le_curves():
    x = np.linspace(0, 1, 400)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axs[0]
    cmap = plt.get_cmap("coolwarm")
    for a in np.linspace(-1, 1, 9):
        ax.plot(x, x + a * x * (1 - x), color=cmap((a + 1) / 2), label=f"α = {a:+.2f}")
    ax.set_title("One LE curve: x + α x (1 - x)")
    ax.legend(fontsize=10, ncol=2, loc="upper left")
    ax = axs[1]
    for n, ls in zip([1, 2, 4, 8], ["-", "--", "-.", ":"]):
        y = x.copy()
        for _ in range(n):
            y = y + 1.0 * y * (1 - y)
        ax.plot(x, y, ls, color="C3", label=f"α = 1, n = {n}")
        y = x.copy()
        for _ in range(n):
            y = y + 0.4 * y * (1 - y)
        ax.plot(x, y, ls, color="C0", label=f"α = 0.4, n = {n}")
    ax.set_title("Iterated curve LE_n (same α each step)")
    ax.legend(fontsize=10, ncol=2, loc="lower right")
    for ax in axs:
        ax.plot(x, x, color="0.6", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("input x")
        ax.set_aspect("equal")
    axs[0].set_ylabel("output")
    fig.tight_layout()
    save(fig, "le-curves.png")


# --- Figure 6: taxonomy diagram ------------------------------------------------
def box(ax, x, y, w, h, text, fc, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec="0.25", lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def fig_taxonomy():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5.4))
    panels = [
        ("What training signal is available?", [
            "SL: paired dark / reference images",
            "RL: rewards for adjustment actions",
            "UL: unrelated dark / normal-light sets",
            "ZSL: low-light inputs, no bright targets",
            "SSL: paired and unpaired stages",
        ]),
        ("What does the model predict?", [
            "Curve parameters: Zero-DCE / ++",
            "Illumination: Retinex-Net, RUAS, SCI",
            "Restored image: CNNs / Transformers",
            "Conditional distribution: flow / diffusion",
            "Retouching transform: HDRNet / 3D LUT",
        ]),
    ]
    for ax, (title, rows) in zip(axs, panels):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=16)
        for i, label in enumerate(rows):
            y = 0.87 - i * 0.17
            ax.text(0.04, y, label, fontsize=10.5, va="center",
                    bbox=dict(boxstyle="round,pad=0.65", facecolor="#edf2f8", edgecolor="#c9d5e4"))
    fig.text(0.5, 0.025, "These are independent axes. Retinex is a design model; diffusion is a model family.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "taxonomy.png")


def fig_timeline():
    events = {
        2017: ["LLNet", "LIME (non-DL)"],
        2018: ["Retinex-Net + LOL", "SID (raw)"],
        2019: ["KinD"],
        2020: ["Zero-DCE", "MIRNet"],
        2021: ["EnlightenGAN (TIP)", "RUAS", "KinD++", "MAET"],
        2022: ["URetinex-Net", "SNR-aware", "SCI", "LLFlow", "Zero-DCE++ (TPAMI)"],
        2023: ["Retinexformer", "Diff-Retinex", "GSAD", "CLIP-LIT", "FeatEnHancer"],
        2024: ["RetinexMamba", "LightenDiffusion", "InstructIR", "NTIRE 2024 LLIE"],
        2025: ["HVI-CIDNet", "Reti-Diff", "GT-Mean loss", "NTIRE 2025 LLIE"],
    }
    fig, ax = plt.subplots(figsize=(16, 5.4))
    years = list(events)
    ax.plot([years[0] - 0.5, years[-1] + 0.5], [0, 0], color="0.3", lw=2)
    colors = {"Other": "0.15", "Retinex": "#c77d0a", "Zero": "#6a4fb3", "Diff": "#1f7fa8", "NTIRE": "#888888"}
    for y in years:
        ax.plot(y, 0, "o", color="0.2", ms=7)
        ax.text(y, -0.35, str(y), ha="center", va="top", fontsize=11, fontweight="bold")
        for k, name in enumerate(events[y]):
            c = "0.15"
            if "Retinex" in name or name in ("KinD", "KinD++", "RUAS", "URetinex-Net"):
                c = colors["Retinex"]
            if name.startswith(("Zero", "SCI", "CLIP", "EnlightenGAN")):
                c = colors["Zero"]
            if name.startswith(("Diff", "GSAD", "Lighten", "Reti-Diff", "LLFlow")):
                c = colors["Diff"]
            if "NTIRE" in name:
                c = colors["NTIRE"]
            ax.text(y, 0.35 + 0.5 * k, name, ha="center", va="bottom", fontsize=10, color=c)
    handles = [plt.Line2D([], [], color=v, lw=4) for v in colors.values()]
    ax.legend(handles, ["Other supervised / task-driven", "Retinex-based / unrolled", "Unpaired or zero-reference",
                        "Flow / diffusion", "Challenge"], loc="upper left", fontsize=10, frameon=False)
    ax.set_xlim(years[0] - 0.7, years[-1] + 0.7)
    ax.set_ylim(-0.9, 3.2)
    ax.axis("off")
    save(fig, "timeline.png")


# --- Figure 8 + compare pair: a zero-reference toy -------------------------------
def darken(img, k):
    return to_display(k * to_linear(img))


class TinyDCE(nn.Module):
    """A small Zero-DCE-style network: 8 iterations x 3 channels of alpha maps."""

    def __init__(self, n_iter=8, width=16):
        super().__init__()
        self.n_iter = n_iter
        self.body = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, 3 * n_iter, 3, padding=1), nn.Tanh())

    def forward(self, x):
        alphas = self.body(x).chunk(self.n_iter, dim=1)
        y = x
        for a in alphas:
            y = y + a * y * (1 - y)
        return y, torch.cat(alphas, 1)


def L_exp(y, E=0.6, p=16):
    return (F.avg_pool2d(y.mean(1, keepdim=True), p) - E).pow(2).mean()


def L_col(y):
    r, g, b = y.mean((2, 3)).unbind(1)
    return ((r - g) ** 2 + (r - b) ** 2 + (g - b) ** 2).mean()


def L_tv(a):
    return (a[..., 1:, :] - a[..., :-1, :]).pow(2).mean() + (a[..., :, 1:] - a[..., :, :-1]).pow(2).mean()


def L_spa(y, x, p=4):
    Y = F.avg_pool2d(y.mean(1, keepdim=True), p)
    X = F.avg_pool2d(x.mean(1, keepdim=True), p)
    return sum(((torch.diff(Y, dim=d) - torch.diff(X, dim=d)) ** 2).mean() for d in (2, 3))


def train_toy(steps=600):
    rng = np.random.default_rng(0)
    train = [resize(img_as_float(f()), (256, 256), anti_aliasing=True)
             for f in (data.astronaut, data.chelsea, data.rocket, data.immunohistochemistry)]

    def batch(n=4, size=64):
        out = []
        for _ in range(n):
            img = train[rng.integers(len(train))]
            i, j = rng.integers(0, 256 - size, 2)
            crop = darken(img[i:i + size, j:j + size], rng.uniform(0.02, 0.3))
            out.append(torch.tensor(crop, dtype=torch.float32).permute(2, 0, 1))
        return torch.stack(out)

    net = TinyDCE()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t0, hist = time.time(), []
    for step in range(steps):
        x = batch()
        y, a = net(x)
        loss = L_spa(y, x) + 10 * L_exp(y) + 5 * L_col(y) + 200 * L_tv(a)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
    print(f"toy: {len(hist)} steps in {time.time() - t0:.1f}s, final loss {hist[-1]:.4f}")
    return net, hist


def run(net, img):
    with torch.no_grad():
        y, a = net(torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)[None])
    return y[0].permute(1, 2, 0).numpy(), a[0].numpy()


def fig_toy():
    net, hist = train_toy()
    out_clean, alphas = run(net, dark_clean)
    out_noisy, _ = run(net, dark_noisy)
    save_rgb(dark_clean, "toy-input.png")
    save_rgb(out_clean, "toy-output.png")

    fig, axs = plt.subplots(2, 3, figsize=(14, 6.4))
    show(axs[0, 0], dark_clean, "Dark input (no noise)")
    show(axs[0, 1], out_clean, "Toy zero-reference output")
    show(axs[0, 2], clean, "Clean reference (never seen)")
    show(axs[1, 0], dark_noisy, "Dark input + noise")
    show(axs[1, 1], out_noisy, "Same network: noise amplified")
    ax = axs[1, 2]
    ax.plot(np.convolve(hist, np.ones(20) / 20, mode="valid"), color="C0")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_title("Total loss (moving average)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "zero-reference-toy.png")
    print("mean alpha per iteration:", np.round(alphas.reshape(8, 3, -1).mean(axis=(1, 2)), 3))


if __name__ == "__main__":
    fig_degradation()
    fig_color_spaces()
    fig_classic()
    fig_retinex()
    fig_le_curves()
    fig_taxonomy()
    fig_timeline()
    fig_toy()
