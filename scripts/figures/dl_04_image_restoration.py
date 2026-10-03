"""Figures for DL 04 - Deep Image Restoration: Low-Level Vision.

Run:  .venv/bin/python scripts/figures/dl_04_image_restoration.py
Writes PNGs to src/assets/deep-learning/dl-04-image-restoration/ only.
All inputs come from skimage.data or synthetic arrays; nothing is downloaded.
The tiny denoiser is trained from scratch on CPU (about a minute or less).
"""
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage import data, img_as_float32, transform  # noqa: E402
from skimage.color import rgb2gray  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio as psnr  # noqa: E402
from skimage.metrics import structural_similarity as ssim  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-04-image-restoration"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white"})
rng = np.random.default_rng(4)
torch.manual_seed(0)
torch.set_num_threads(2)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(path).convert("RGB")
    im = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    im.save(path, optimize=True)
    print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB)")


def save_array(arr, name, scale=2):
    """Save a [0,1] grayscale array as a PNG of exact pixel size (for ImageCompare)."""
    a = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
    a = np.kron(a, np.ones((scale, scale), dtype=np.uint8))  # nearest upscaling
    path = OUT / name
    Image.fromarray(a, mode="L").save(path, optimize=True)
    print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB) {a.shape}")


# ------------------------------------------------------------ 1. degradation gallery
def jpeg(img, q):
    u8 = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", u8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255


def gallery():
    x = img_as_float32(data.astronaut())[20:276, 140:396]  # 256x256 RGB crop
    H, W, _ = x.shape
    blur = np.stack([ndi.gaussian_filter(x[..., c], 2.5) for c in range(3)], -1)
    noisy = np.clip(x + rng.normal(0, 0.08, x.shape), 0, 1)
    # haze: depth increases toward the top, t = exp(-beta d)
    d = np.linspace(1.0, 0.2, H)[:, None] * np.ones((1, W))
    t = np.exp(-1.6 * d)[..., None]
    A = 0.9
    hazy = x * t + A * (1 - t)
    # rain: sparse impulses smeared along a slanted line
    seeds = (rng.random((H, W)) > 0.985).astype(np.float32)
    k = np.zeros((15, 15), np.float32)
    for i in range(15):
        k[i, int(round(4 + i * 0.45))] = 1
    streaks = np.clip(ndi.convolve(seeds, k / 4.0), 0, 1)
    rain = np.clip(x + 0.7 * streaks[..., None], 0, 1)
    # low light: S = R * L with a dim, smooth illumination, plus sensor noise
    L = 0.12 + 0.08 * np.linspace(0, 1, W)[None, :] * np.ones((H, 1))
    dark = np.clip(x * L[..., None] + rng.normal(0, 0.01, x.shape), 0, 1)
    # SR: blur + x4 downsample, shown with nearest upsampling
    lr = transform.resize(blur, (H // 4, W // 4), anti_aliasing=False)
    lr_up = np.kron(lr, np.ones((4, 4, 1)))
    jp = jpeg(x, 8)
    panels = [
        (x, "Clean  x"),
        (blur, "Blur  h * x"),
        (noisy, "Noise  x + n"),
        (lr_up, "Downsample  (h * x)↓4"),
        (hazy, "Haze  J t + A(1 - t)"),
        (rain, "Rain  B + R"),
        (dark, "Low light  R · L"),
        (jp, "JPEG, quality 8"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.4))
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    save(fig, "degradation-gallery.png")


# ------------------------------------------------------------ helpers for diagrams
def box(ax, xy, w, h, text, fc, fontsize=10, ec="#333333"):
    b = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2)
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def arrow(ax, p, q, color="#333333", style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, color=color, lw=1.2))


# ------------------------------------------------------------ 2. taxonomy
def taxonomy():
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.45, 5.6)
    ax.axis("off")
    box(ax, (3.9, 4.75), 2.7, 0.7, "Image restoration\n(low-level vision)", "#e8e8e8", 11)
    groups = [
        ("Sensor / optics\n(y = D(x) + n)", "#dbe9f6", 0.1,
         ["Denoising", "Deblurring", "Super-resolution", "JPEG artifact\nremoval"]),
        ("Weather / medium\n(physical scattering)", "#dff0d8", 2.75,
         ["Dehazing", "Deraining\n(streak / drop)", "Desnowing", "Underwater"]),
        ("Illumination", "#fcf3cf", 5.4,
         ["Low-light\nenhancement", "Exposure\ncorrection"]),
        ("Missing / multiple\ninputs", "#f6dbe0", 7.9,
         ["Inpainting", "Matting", "Image fusion\n(IR-visible,\nexposure, focus)"]),
    ]
    for title, fc, x0, items in groups:
        box(ax, (x0, 3.45), 2.4, 0.8, title, fc, 10)
        arrow(ax, (5.25, 4.75), (x0 + 1.2, 4.25))
        for i, it in enumerate(items):
            y = 2.55 - i * 0.82
            box(ax, (x0 + 0.25, y), 1.9, 0.68, it, "white", 9.5)
            if i == 0:
                arrow(ax, (x0 + 1.2, 3.45), (x0 + 1.2, y + 0.68))
            else:
                arrow(ax, (x0 + 1.2, y + 0.82), (x0 + 1.2, y + 0.68))
    ax.text(5.25, -0.35, "All-in-one models try to cover several boxes with one network", ha="center",
            fontsize=10, style="italic")
    save(fig, "task-taxonomy.png")


# ------------------------------------------------------------ 3. timeline
def timeline():
    sr = [(2014, "SRCNN", 0), (2016, "FSRCNN\nESPCN", 0), (2017, "EDSR\nSRGAN", 0), (2018, "RCAN\nESRGAN", 0),
          (2021, "SwinIR\nReal-ESRGAN", 0), (2023, "HAT\nDiffBIR", 0), (2024, "SUPIR\nOSEDiff\nMambaIR", 0),
          (2025.2, "MambaIRv2", 0)]
    dn = [(2017, "DnCNN", 0), (2018, "FFDNet\nSIDD (data)", 0), (2020, "MIRNet", 0), (2021.8, "Restormer\nNAFNet", 0),
          (2022.4, "AirNet\nTransWeather", 1), (2023.25, "PromptIR", 2), (2024, "DA-CLIP\nInstructIR", 0),
          (2025, "AdaIR", 0)]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axhline(0, color="#333333", lw=1.5)
    for yr in range(2014, 2026):
        ax.plot([yr, yr], [-0.08, 0.08], color="#333333", lw=1)
        ax.text(yr, -0.25, str(yr), ha="center", va="top", fontsize=9.5)
    for yr, lab, lv in sr:
        top = 0.55 + 0.6 * lv
        ax.plot([yr, yr], [0.1, top], color="#2c6fbb", lw=1)
        ax.plot(yr, 0.1, "o", color="#2c6fbb", ms=5)
        ax.text(yr, top + 0.05, lab, ha="center", va="bottom", fontsize=9.5, color="#1b4f8a")
    for yr, lab, lv in dn:
        bot = -0.9 - 0.75 * lv
        ax.plot([yr, yr], [-0.55, bot], color="#c0504d", lw=1)
        ax.plot(yr, -0.55, "s", color="#c0504d", ms=5)
        ax.text(yr, bot - 0.05, lab, ha="center", va="top", fontsize=9.5, color="#8b2f2c")
    ax.text(2013.4, 1.15, "Super-resolution", color="#1b4f8a", fontsize=11, weight="bold")
    ax.text(2013.4, -3.05, "Denoising, general backbones and all-in-one", color="#8b2f2c", fontsize=11,
            weight="bold")
    ax.set_xlim(2013.3, 2025.9)
    ax.set_ylim(-3.15, 1.35)
    ax.axis("off")
    save(fig, "milestones-timeline.png")


# ------------------------------------------------------------ 4. perception-distortion
def perception_distortion():
    D = np.linspace(0.12, 1.0, 200)
    P = 0.02 + 0.06 / (D - 0.08)  # convex, decreasing illustrative bound
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.fill_between(D, 0, P, color="#d9d9d9", label="unattainable region")
    ax.plot(D, P, color="#333333", lw=2, label="perception-distortion bound")
    pts = {"MSE-trained\n(e.g. EDSR-like)": (0.2, 0.85), "GAN / perceptual\n(e.g. ESRGAN-like)": (0.42, 0.28),
           "Diffusion prior\n(e.g. SUPIR-like)": (0.58, 0.17), "Bicubic": (0.8, 0.8)}
    colors = ["#2c6fbb", "#c0504d", "#7a5195", "#888888"]
    for (lab, (dx, py)), c in zip(pts.items(), colors):
        ax.plot(dx, py, "o", color=c, ms=8)
        ax.annotate(lab, (dx, py), xytext=(8, 4), textcoords="offset points", fontsize=9.5, color=c)
    ax.set_xlabel("Distortion  (e.g. MSE, lower is better)")
    ax.set_ylabel("Perceptual index  (lower = more natural)")
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9.5, frameon=False)
    ax.set_title("Illustration (schematic, not measured data)", fontsize=10)
    save(fig, "perception-distortion.png")


# ------------------------------------------------------------ 5. all-in-one architecture
def all_in_one():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    box(ax, (0.1, 1.9), 1.4, 0.9, "Degraded\ninput y\n(rain? haze?\nnoise? blur?)", "#f2f2f2", 9.5)
    # encoder-decoder
    enc = [(1.9, 1.5, 1.6), (2.8, 1.75, 1.1), (3.7, 1.95, 0.7)]
    for x0, y0, h in enc:
        box(ax, (x0, y0), 0.7, h, "Enc", "#dbe9f6", 9.5)
    box(ax, (4.6, 2.05), 0.9, 0.5, "Latent", "#c9dbef", 9.5)
    dec = [(5.7, 1.95, 0.7), (6.6, 1.75, 1.1), (7.5, 1.5, 1.6)]
    for x0, y0, h in dec:
        box(ax, (x0, y0), 0.7, h, "Dec", "#dbe9f6", 9.5)
    box(ax, (8.95, 1.9), 1.45, 0.9, "Restored\noutput x̂", "#f2f2f2", 9.5)
    arrow(ax, (1.5, 2.35), (1.9, 2.35))
    arrow(ax, (8.2, 2.35), (8.95, 2.35))
    for a, b in [(2.6, 2.8), (3.5, 3.7), (4.4, 4.6), (5.5, 5.7), (6.4, 6.6), (7.3, 7.5)]:
        arrow(ax, (a, 2.35), (b, 2.35))
    # skip connections
    for (xe, ye, he), (xd, yd, hd) in zip(enc, reversed(dec)):
        ax.annotate("", xy=(xd + 0.35, yd + hd), xytext=(xe + 0.35, ye + he),
                    arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1, ls="--",
                                    connectionstyle="arc3,rad=-0.25"))
    # degradation-aware branch
    box(ax, (1.7, 0.1), 2.6, 0.75, "Degradation encoder\n(contrastive / CLIP / prompts)", "#fcf3cf", 9.5)
    box(ax, (6.0, 0.1), 2.4, 0.75, "Text instruction\n\"remove the rain\"", "#dff0d8", 9.5)
    box(ax, (4.55, 0.1), 1.2, 0.75, "Condition\nvector c", "#f6dbe0", 9.5)
    arrow(ax, (0.8, 1.9), (2.0, 0.85))
    arrow(ax, (4.3, 0.47), (4.55, 0.47))
    arrow(ax, (6.0, 0.47), (5.75, 0.47))
    for xd, yd in ((6.05, 1.95), (6.95, 1.75), (7.85, 1.5)):
        arrow(ax, (5.15, 0.85), (xd, yd), color="#c0504d")
    ax.text(5.15, 4.25, "One network, many degradations: the condition c modulates the decoder blocks",
            ha="center", fontsize=10, style="italic")
    ax.text(8.5, 0.95, "modulate\n(prompts, FiLM,\ncross-attention)", fontsize=9, color="#c0504d")
    save(fig, "all-in-one.png")


# ------------------------------------------------------------ 6. tiny denoiser + ImageCompare pair
class TinyDnCNN(nn.Module):
    def __init__(self, depth=5, ch=32):
        super().__init__()
        layers = [nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(ch, 1, 3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, y):
        return y - self.body(y)


def denoiser():
    sigma = 25 / 255
    train = [img_as_float32(im) for im in (data.camera(), data.coins(), data.moon(), data.page())]
    test = img_as_float32(rgb2gray(data.astronaut()))

    def patches(n=32, p=32):
        out = []
        for _ in range(n):
            im = train[rng.integers(len(train))]
            i, j = rng.integers(0, im.shape[0] - p), rng.integers(0, im.shape[1] - p)
            out.append(im[i:i + p, j:j + p])
        return torch.from_numpy(np.stack(out))[:, None]

    net = TinyDnCNN()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    losses = []
    t0 = time.time()
    for _ in range(300):
        x = patches()
        y = x + sigma * torch.randn_like(x)
        loss = nn.functional.l1_loss(net(y), x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"denoiser trained in {time.time() - t0:.0f} s")
    net.eval()
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        x = torch.from_numpy(test)[None, None]
        y = (x + sigma * torch.randn(x.shape, generator=gen)).clamp(0, 1)
        xh = net(y).clamp(0, 1)
    x, y, xh = x[0, 0].numpy(), y[0, 0].numpy(), xh[0, 0].numpy()
    gauss = ndi.gaussian_filter(y, 1.0)
    for name, im in [("noisy", y), ("gaussian", gauss), ("tiny CNN", xh)]:
        print(f"  {name:9s} PSNR {psnr(x, im, data_range=1):.2f} dB  SSIM {ssim(x, im, data_range=1):.3f}")
    # ImageCompare crop (face region), upscaled x2 with nearest
    sl = (slice(30, 230), slice(140, 340))
    save_array(y[sl], "compare-noisy.png")
    save_array(xh[sl], "compare-denoised.png")
    # summary figure
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})
    axes[0].plot(np.convolve(losses, np.ones(10) / 10, mode="valid"), color="#2c6fbb")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("L1 loss (smoothed)")
    axes[0].set_title("Training on 32x32 patches", fontsize=10)
    for ax, im, t in [(axes[1], y, "Noisy (σ = 25)"), (axes[2], gauss, "Gaussian filter"),
                      (axes[3], xh, "Tiny residual CNN")]:
        ax.imshow(im[sl], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{t}\n{psnr(x, im, data_range=1):.1f} dB / SSIM {ssim(x, im, data_range=1):.2f}",
                     fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    save(fig, "tiny-denoiser.png")


if __name__ == "__main__":
    gallery()
    taxonomy()
    timeline()
    perception_distortion()
    all_in_one()
    denoiser()
