"""Extra teaching figures for ML 08 (autoregressive vision); synthetic data, CPU, no downloads.

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/ml_08_autoregressive_vision_extra.py
Writes only to src/assets/ml/ml-08-autoregressive-vision/. Figures show mechanisms on
toy problems; they are not outputs of any published model.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT = Path(__file__).resolve().parents[2] / "src/assets/ml/ml-08-autoregressive-vision"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, GRAY = "#2a6fdb", "#e4572e", "#17a398", "#8a8f98"
torch.set_num_threads(1)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name)


# ---------------------------------------------------------------- masked conv
class MaskedConv2d(nn.Conv2d):
    def __init__(self, cin, cout, k, mask_type):
        super().__init__(cin, cout, k, padding=k // 2, bias=False)
        mask = torch.ones(k, k)
        c = k // 2
        mask[c, c + (mask_type == "B"):] = 0
        mask[c + 1:] = 0
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, padding=self.padding)


def masked_conv_figure():
    torch.manual_seed(0)
    layers = [MaskedConv2d(1, 8, 3, "A"), nn.Tanh()]
    for _ in range(4):
        layers += [MaskedConv2d(8, 8, 3, "B"), nn.Tanh()]
    net = nn.Sequential(*layers, MaskedConv2d(8, 1, 1, "B"))
    x = torch.rand(1, 1, 9, 9, requires_grad=True)
    net(x)[0, 0, 4, 4].backward()
    dep = (x.grad[0, 0].abs() > 0).numpy()
    earlier = np.zeros((9, 9), bool)
    earlier[:4] = True
    earlier[4, :4] = True
    blind = earlier & ~dep

    def mask_img(kind):
        m = np.ones((5, 5))
        m[2, 2 + (kind == "B"):] = 0
        m[3:] = 0
        return m

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, kind in zip(axes[:2], "AB"):
        ax.imshow(mask_img(kind), cmap="Blues", vmin=-.3, vmax=1.2)
        ax.add_patch(Rectangle((1.5, 1.5), 1, 1, fill=False, ec=ORANGE, lw=2.5))
        for (i, j), v in np.ndenumerate(mask_img(kind)):
            ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=11)
        ax.set_title(f"5x5 mask type {kind}")
        ax.set_xticks([]); ax.set_yticks([])
    rgb = np.ones((9, 9, 3))
    rgb[dep] = matplotlib.colors.to_rgb(BLUE)
    rgb[blind] = matplotlib.colors.to_rgb(ORANGE)
    rgb[4, 4] = (0.15, 0.15, 0.15)
    axes[2].imshow(rgb)
    axes[2].set_title("5 stacked 3x3 masked layers")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[2].text(4, 4, "x", color="white", ha="center", va="center", fontsize=12)
    handles = [Rectangle((0, 0), 1, 1, color=BLUE), Rectangle((0, 0), 1, 1, color=ORANGE),
               Rectangle((0, 0), 1, 1, color=(.15, .15, .15))]
    axes[2].legend(handles, ["seen by output", "blind spot", "current pixel"],
                   loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, frameon=False)
    save(fig, "masked-conv.png")


# ---------------------------------------------------------------- VQ pipeline diagram
def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="black", lw=1, alpha=.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.3))


def vq_pipeline_figure():
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.2); ax.axis("off")
    box(ax, 0.1, 2.4, 1.3, 1.1, "image\n256x256x3", "#dce8fb")
    box(ax, 1.9, 2.4, 1.3, 1.1, "CNN/ViT\nencoder", "#fde2d9")
    box(ax, 3.7, 2.4, 1.5, 1.1, "features\n16x16xd", "#dce8fb")
    box(ax, 5.7, 2.4, 1.6, 1.1, "nearest code\nin codebook", "#d6f2ee")
    box(ax, 7.8, 2.4, 1.2, 1.1, "decoder", "#fde2d9")
    box(ax, 9.3, 2.4, 1.1, 1.1, "recon.\nimage", "#dce8fb")
    for x0, x1 in [(1.4, 1.9), (3.2, 3.7), (5.2, 5.7), (7.3, 7.8), (9.0, 9.3)]:
        arrow(ax, x0, 2.95, x1, 2.95)
    box(ax, 5.55, 0.25, 1.9, 1.15, "256 integer indices\n(each in 0..K-1)", "#fff4cc")
    arrow(ax, 6.5, 2.4, 6.5, 1.4)
    box(ax, 7.9, 0.25, 2.5, 1.15, "stage 2: AR / masked\ntransformer models\np(indices)", "#ece0f8")
    arrow(ax, 7.45, 0.82, 7.9, 0.82)
    ax.text(0.1, 1.55, "Stage 1 (tokenizer): reconstruction + codebook +\n"
            "commitment (+ perceptual / adversarial for VQGAN).\n"
            "Stage 2 never sees pixels, only indices.", fontsize=10, va="top")
    ax.text(5.25, 3.85, "65,536 pixel positions  ->  256 tokens (factor-16 downsampling)",
            ha="center", fontsize=10)
    save(fig, "vq-pipeline.png")


# ---------------------------------------------------------------- VQ codebook toy
def vq_codebook_figure():
    torch.manual_seed(0)
    centres = torch.tensor([[2., 0.], [-2., 0.], [0., 2.], [0., -2.]])
    encoder, decoder = nn.Linear(2, 2), nn.Linear(2, 2)
    codebook = nn.Parameter(torch.randn(8, 2) * .1)
    opt = torch.optim.Adam([*encoder.parameters(), *decoder.parameters(), codebook], lr=.03)
    for _ in range(800):
        x = centres[torch.randint(4, (256,))] + .1 * torch.randn(256, 2)
        z = encoder(x)
        idx = torch.cdist(z, codebook).argmin(1)
        e = codebook[idx]
        zq = z + (e - z).detach()
        loss = (F.mse_loss(decoder(zq), x) + F.mse_loss(e, z.detach())
                + .25 * F.mse_loss(z, e.detach()))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x = centres.repeat_interleave(150, 0) + .1 * torch.randn(600, 2)
        z = encoder(x)
        idx = torch.cdist(z, codebook).argmin(1)
        cb = codebook.detach()
        used = torch.zeros(len(cb), dtype=torch.bool)
        used[idx.unique()] = True
        lo, hi = z.min(0).values - 1, z.max(0).values + 1
        g0, g1 = torch.meshgrid(torch.linspace(lo[0], hi[0], 300),
                                torch.linspace(lo[1], hi[1], 300), indexing="xy")
        grid = torch.stack([g0.flatten(), g1.flatten()], 1)
        cell = torch.cdist(grid, cb).argmin(1).reshape(300, 300)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.imshow(cell.numpy(), origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]],
              cmap="Pastel2", alpha=.8, aspect="auto")
    ax.scatter(*z.T.numpy(), s=6, color=BLUE, alpha=.5, label="encoder outputs z_e(x)")
    ax.scatter(*cb[used].T.numpy(), s=120, marker="*", color=ORANGE, ec="black",
               label="code used")
    ax.scatter(*cb[~used].T.numpy(), s=70, marker="X", color=GRAY, ec="black",
               label="code never selected")
    ax.set(title="Toy VQ: each region of feature space maps to one code",
           xlabel="feature 1", ylabel="feature 2")
    ax.legend(fontsize=9, loc="best")
    save(fig, "vq-codebook.png")


# ---------------------------------------------------------------- MaskGIT schedule + toy
def maskgit_figure():
    gamma = lambda r: math.cos(math.pi * r / 2)
    torch.manual_seed(0)
    N, MASK = 16, 2
    net = nn.Sequential(nn.Flatten(), nn.Linear(3 * N, 128), nn.ReLU(), nn.Linear(128, N))
    logits = lambda tok: net(F.one_hot(tok, 3).float())
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(1500):
        x = torch.randint(2, (256, 4)).repeat(1, 4)
        ratio = torch.tensor([gamma(r) for r in torch.rand(256).tolist()])
        n_mask = (ratio * N).ceil().clamp(min=1)
        masked = torch.rand(256, N).argsort(1) < n_mask[:, None]
        inp = torch.where(masked, MASK, x)
        loss = F.binary_cross_entropy_with_logits(logits(inp)[masked], x[masked].float())
        opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def valid_fraction(T, n=1000):
        tok = torch.full((n, N), MASK)
        for t in range(1, T + 1):
            p = logits(tok).sigmoid()
            s = torch.bernoulli(p).long()
            conf = torch.where(s == 1, p, 1 - p)
            still = tok == MASK
            conf = torch.where(still, conf, torch.inf)
            prop = torch.where(still, s, tok)
            k = math.floor(gamma(t / T) * N)
            remask = torch.zeros_like(still)
            remask.scatter_(1, conf.argsort(1)[:, :k], True)
            tok = torch.where(remask, MASK, prop)
        rows = tok.reshape(-1, 4, 4)
        return (rows == rows[:, :1]).all(2).all(1).float().mean().item()

    Ts = [1, 2, 4, 8, 16]
    valid = [valid_fraction(T) for T in Ts]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
    fig.subplots_adjust(wspace=.38)
    r = np.linspace(0, 1, 200)
    axes[0].plot(r, np.cos(np.pi * r / 2), color=BLUE, label="cosine")
    axes[0].plot(r, 1 - r, color=ORANGE, label="linear")
    axes[0].plot(r, 1 - r ** 2, color=GREEN, label="square")
    axes[0].set(title="Mask schedule gamma(r)", xlabel="progress r = t/T",
                ylabel="fraction still masked")
    axes[0].legend(fontsize=9)
    T, Ntok = 8, 256
    left = [Ntok] + [math.floor(gamma(t / T) * Ntok) for t in range(1, T + 1)]
    revealed = [left[t - 1] - left[t] for t in range(1, T + 1)]
    axes[1].bar(range(1, T + 1), revealed, color=BLUE)
    axes[1].set(title="Tokens fixed per step (N=256, T=8)", xlabel="decoding step",
                ylabel="tokens fixed")
    axes[2].plot(Ts, valid, "o-", color=GREEN)
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(Ts, [str(t) for t in Ts])
    axes[2].set(title="Toy 4x4 task: valid samples", xlabel="decoding steps T",
                ylabel="fraction with all rows equal", ylim=(-.05, 1.05))
    save(fig, "maskgit-schedule.png")


# ---------------------------------------------------------------- prediction units
def prediction_units_figure():
    """Which sequential step produces each final 4x4 token, for three generation orders."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    raster = np.arange(1, 17).reshape(4, 4)
    rng = np.random.default_rng(1)
    gamma = lambda r: math.cos(math.pi * r / 2)
    T, N = 4, 16
    left = [N] + [math.floor(gamma(t / T) * N) for t in range(1, T + 1)]
    order = rng.permutation(N)                       # stand-in for confidence order
    masked_step = np.zeros(N, int)
    pos = 0
    for t in range(1, T + 1):
        k = left[t - 1] - left[t]
        masked_step[order[pos:pos + k]] = t
        pos += k
    masked_step = masked_step.reshape(4, 4)
    for ax, a, title, vmax in [(axes[0], raster, "Next token (raster): 16 steps", 16),
                               (axes[2], masked_step, "Masked parallel (T=4): 4 steps", 4)]:
        ax.imshow(a, cmap="Blues", vmin=0, vmax=vmax * 1.25)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        for (i, j), v in np.ndenumerate(a):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=11)
    ax = axes[1]
    ax.set_xlim(-.2, 8.4); ax.set_ylim(-1.2, 4.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Next scale: 3 steps", fontsize=11)
    colors = ["#c6dbef", "#6baed6", "#2171b5"]
    x0 = 0
    for step, n in enumerate([1, 2, 4], start=1):
        cell = 1.0 if n < 4 else 1.0
        for i in range(n):
            for j in range(n):
                ax.add_patch(Rectangle((x0 + j * cell, 4 - (i + 1) * cell), cell, cell,
                                       fc=colors[step - 1], ec="white", lw=1.5))
                ax.text(x0 + (j + .5) * cell, 4 - (i + .5) * cell, str(step),
                        ha="center", va="center", fontsize=10,
                        color="white" if step == 3 else "black")
        ax.text(x0 + n * cell / 2, -.6, f"{n}x{n}", ha="center", fontsize=10)
        x0 += n * cell + .2
    save(fig, "prediction-units.png")


if __name__ == "__main__":
    prediction_units_figure()
    masked_conv_figure()
    vq_pipeline_figure()
    vq_codebook_figure()
    maskgit_figure()
