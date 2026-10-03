"""Figures for DL 02 · Deep Image Classification.

Run:  .venv/bin/python scripts/figures/dl_02_image_classification.py
Writes PNGs to src/assets/deep-learning/dl-02-image-classification/ and prints
the numbers quoted in the tutorial text. CPU only, about one minute.
"""
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src/assets/deep-learning/dl-02-image-classification"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})
torch.set_num_threads(4)

BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = "#3b6fb6", "#e08a2c", "#3a9a5b", "#c8453c", "#7b5ea7", "#666666"


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p.name}  {p.stat().st_size / 1024:.0f} KB")


def box(ax, x, y, w, h, text, fc="#e8eef8", ec=BLUE, fs=10, lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1, color="black", style="-|>", lw=1.4, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------- diagrams
def fig_timeline():
    # (year of first public version, name, idea, label x, label y, color)
    items = [
        (1998, "LeNet-5", "conv + pool,\nbackprop", -2.2, 1.0, GRAY),
        (2012, "AlexNet", "ReLU, dropout,\nGPUs", 0.0, -1.2, BLUE),
        (2014, "VGG", "stacked 3×3", 1.6, 1.0, BLUE),
        (2014, "GoogLeNet", "multi-branch,\n1×1 reduce", 2.4, 2.4, BLUE),
        (2015, "ResNet", "residual\nshortcuts", 3.0, -1.2, BLUE),
        (2016, "DenseNet", "concatenate\nall features", 4.1, 1.0, BLUE),
        (2017, "MobileNet", "depthwise\nseparable", 5.3, 2.4, BLUE),
        (2017, "SENet", "channel\nattention", 5.0, -1.2, ORANGE),
        (2019, "EfficientNet", "compound\nscaling", 6.6, 1.0, BLUE),
        (2020, "RegNet", "design-space\nsearch", 6.6, -2.6, BLUE),
        (2020, "ViT", "patches +\nself-attention", 8.0, -1.2, PURPLE),
        (2020, "DeiT", "data-efficient\ntraining", 7.9, 2.4, PURPLE),
        (2021, "Swin", "shifted\nwindows", 9.3, 1.0, PURPLE),
        (2021, "CLIP · DINO · MAE", "language and self-\nsupervised pre-training", 9.6, 3.8, GREEN),
        (2022, "ConvNeXt", "modernized\nCNN", 10.2, -1.2, BLUE),
        (2023, "DINOv2", "frozen general-\npurpose features", 11.4, 2.4, GREEN),
        (2025, "DINOv3", "scaled SSL\nbackbone", 13.0, -1.2, GREEN),
    ]

    def X(y):
        return (y - 2012) if y >= 2012 else -2.2
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.axhline(0, color="black", lw=1.2, zorder=1)
    for year, name, idea, lx, ly, c in items:
        x = X(year)
        ty = ly - 0.25 if ly > 0 else ly + 0.45
        ax.plot([x, lx], [0, ty], color=c, lw=0.8, ls=":", zorder=1)
        ax.plot(x, 0, "o", color=c, ms=7, zorder=3)
        ax.text(lx, ly + 0.2, name, ha="center", va="center", fontsize=10, weight="bold", color=c,
                bbox=dict(fc="white", ec="none", pad=0.5), zorder=4)
        ax.text(lx, ly - 0.35, idea, ha="center", va="center", fontsize=8.5,
                bbox=dict(fc="white", ec="none", pad=0.5), zorder=4)
    for y in [2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026]:
        ax.text(X(y), -0.18, str(y), ha="center", va="top", fontsize=9, color=GRAY)
    ax.text(X(1998), -0.18, "1998", ha="center", va="top", fontsize=9, color=GRAY)
    ax.text(-1.1, 0.0, "//", ha="center", va="center", fontsize=12, bbox=dict(fc="white", ec="none"))
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
           [(BLUE, "CNN"), (ORANGE, "attention inside a CNN"), (PURPLE, "vision transformer"),
            (GREEN, "pre-training paradigm"), (GRAY, "precursor")]]
    ax.legend(handles=leg, loc="lower center", fontsize=9, frameon=False, ncol=5,
              bbox_to_anchor=(0.5, -0.08))
    ax.set_xlim(-3.2, 14.4)
    ax.set_ylim(-3.4, 4.6)
    ax.axis("off")
    ax.set_title("Image-classification backbones and the idea each introduced (year of first public version)")
    save(fig, "timeline.png")


def fig_blocks():
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.2))
    titles = ["(a) plain block", "(b) basic residual block", "(c) bottleneck block"]
    for ax, t in zip(axes, titles):
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(t)
    # (a) plain
    ax = axes[0]
    box(ax, 1, 8.6, 2, 0.8, "x  (C channels)", fc="white", ec=GRAY)
    box(ax, 1, 6.6, 2, 0.9, "3×3 conv, BN, ReLU")
    box(ax, 1, 4.6, 2, 0.9, "3×3 conv, BN, ReLU")
    box(ax, 1, 2.4, 2, 0.8, "H(x)", fc="white", ec=GRAY)
    arrow(ax, 2, 8.6, 2, 7.5); arrow(ax, 2, 6.6, 2, 5.5); arrow(ax, 2, 4.6, 2, 3.2)
    ax.text(2, 1.3, "layers must learn H(x)\ndirectly", ha="center", fontsize=9.5, color=GRAY)
    # (b) residual
    ax = axes[1]
    box(ax, 0.8, 8.6, 2, 0.8, "x  (C channels)", fc="white", ec=GRAY)
    box(ax, 0.8, 6.6, 2, 0.9, "3×3 conv, BN, ReLU")
    box(ax, 0.8, 4.6, 2, 0.9, "3×3 conv, BN")
    ax.add_patch(plt.Circle((1.8, 3.6), 0.3, fc="white", ec="black"))
    ax.text(1.8, 3.6, "+", ha="center", va="center", fontsize=14)
    box(ax, 0.8, 1.6, 2, 0.8, "ReLU → y", fc="white", ec=GRAY)
    arrow(ax, 1.8, 8.6, 1.8, 7.5); arrow(ax, 1.8, 6.6, 1.8, 5.5); arrow(ax, 1.8, 4.6, 1.8, 3.9)
    arrow(ax, 1.8, 3.3, 1.8, 2.4)
    ax.plot([2.8, 3.5, 3.5], [9.0, 9.0, 3.6], color=RED, lw=2)
    ax.annotate("", xy=(2.1, 3.6), xytext=(3.5, 3.6),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(3.6, 6.3, "identity\nshortcut", color=RED, fontsize=9.5, ha="left", va="center")
    ax.text(0.15, 5.6, "F(x)", color=BLUE, fontsize=11, ha="center")
    ax.text(1.8, 0.7, "y = F(x) + x", ha="center", fontsize=10.5)
    # (c) bottleneck
    ax = axes[2]
    box(ax, 0.8, 8.6, 2, 0.8, "x  (4d channels)", fc="white", ec=GRAY)
    box(ax, 0.8, 7.0, 2, 0.8, "1×1 conv → d", fc="#fdf1e3", ec=ORANGE)
    box(ax, 0.8, 5.5, 2, 0.8, "3×3 conv, d → d")
    box(ax, 0.8, 4.0, 2, 0.8, "1×1 conv → 4d", fc="#fdf1e3", ec=ORANGE)
    ax.add_patch(plt.Circle((1.8, 3.0), 0.3, fc="white", ec="black"))
    ax.text(1.8, 3.0, "+", ha="center", va="center", fontsize=14)
    box(ax, 0.8, 1.2, 2, 0.8, "ReLU → y", fc="white", ec=GRAY)
    arrow(ax, 1.8, 8.6, 1.8, 7.8); arrow(ax, 1.8, 7.0, 1.8, 6.3); arrow(ax, 1.8, 5.5, 1.8, 4.8)
    arrow(ax, 1.8, 4.0, 1.8, 3.3); arrow(ax, 1.8, 2.7, 1.8, 2.0)
    ax.plot([2.8, 3.5, 3.5], [9.0, 9.0, 3.0], color=RED, lw=2)
    ax.annotate("", xy=(2.1, 3.0), xytext=(3.5, 3.0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(1.8, 0.4, "reduce → process → expand", ha="center", fontsize=9.5, color=GRAY)
    save(fig, "residual-blocks.png")


def fig_depthwise():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1, 1.5]})
    cols_in = ["#7fa7d9", "#e3a96b", "#8cc79f"]

    def stack(ax, x, y, n, w, h, cols, label=None, dx=0.12, dy=0.12):
        for i in range(n):
            ax.add_patch(Rectangle((x + i * dx, y + i * dy), w, h, fc=cols[i % len(cols)], ec="black", lw=0.8))
        if label:
            ax.text(x + w / 2 + n * dx / 2, y - 0.35, label, ha="center", va="top", fontsize=9.5)

    ax = axes[0]
    ax.set_title("(a) standard convolution")
    stack(ax, 0.2, 1.0, 3, 1.4, 1.4, cols_in, "input\nH×W×M")
    for k in range(4):
        stack(ax, 2.55 + k * 0.05, 3.2 - k * 0.55, 3, 0.35, 0.35, cols_in, dx=0.06, dy=0.06)
    ax.text(2.9, 0.75, "N kernels,\neach K×K×M", ha="center", va="top", fontsize=9.5)
    stack(ax, 3.7, 1.0, 4, 1.4, 1.4, ["#bbbbbb", "#999999", "#cccccc", "#aaaaaa"], "output\nH×W×N")
    arrow(ax, 1.95, 1.7, 2.45, 1.7); arrow(ax, 3.25, 1.7, 3.65, 1.7)
    ax.text(2.6, -0.65, "cost ∝ H·W·K²·M·N", ha="center", fontsize=10.5, color=RED)
    ax.set_xlim(0, 5.6); ax.set_ylim(-1.0, 4.2); ax.axis("off")

    ax = axes[1]
    ax.set_title("(b) depthwise separable convolution")
    stack(ax, 0.2, 1.0, 3, 1.4, 1.4, cols_in, "input\nH×W×M")
    for i in range(3):
        ax.add_patch(Rectangle((2.3, 2.6 - i * 0.6), 0.35, 0.35, fc=cols_in[i], ec="black", lw=0.8))
    ax.text(2.48, 0.75, "M kernels\nK×K×1\n(one per channel)", ha="center", va="top", fontsize=9)
    stack(ax, 3.2, 1.0, 3, 1.4, 1.4, cols_in, "H×W×M")
    for k in range(4):
        ax.add_patch(Rectangle((5.25, 2.6 - k * 0.45), 0.12, 0.3, fc="#dddddd", ec="black", lw=0.8))
    ax.text(5.31, 0.75, "N kernels\n1×1×M", ha="center", va="top", fontsize=9)
    stack(ax, 6.1, 1.0, 4, 1.4, 1.4, ["#bbbbbb", "#999999", "#cccccc", "#aaaaaa"], "output\nH×W×N")
    arrow(ax, 1.95, 1.7, 2.2, 1.7); arrow(ax, 2.8, 1.7, 3.15, 1.7)
    arrow(ax, 4.95, 1.7, 5.2, 1.7); arrow(ax, 5.45, 1.7, 6.05, 1.7)
    ax.text(4.2, -0.65, "cost ∝ H·W·K²·M  +  H·W·M·N", ha="center", fontsize=10.5, color=GREEN)
    ax.set_xlim(0, 8.0); ax.set_ylim(-1.0, 4.2); ax.axis("off")
    save(fig, "depthwise-separable.png")


def fig_vit():
    from skimage import data, transform
    img = transform.resize(data.astronaut(), (128, 128), anti_aliasing=True)
    P = 32
    fig = plt.figure(figsize=(10.2, 3.9))
    ax0 = fig.add_axes([0.0, 0.12, 0.25, 0.76])
    ax0.imshow(img)
    for k in range(1, 4):
        ax0.axhline(k * P - 0.5, color="white", lw=1.5); ax0.axvline(k * P - 0.5, color="white", lw=1.5)
    ax0.set_title("image split into 4×4 patches")
    ax0.axis("off")
    ax = fig.add_axes([0.27, 0.0, 0.73, 1.0])
    ax.set_xlim(0, 20); ax.set_ylim(0, 8); ax.axis("off")
    # patch strip
    patches = [img[i * P:(i + 1) * P, j * P:(j + 1) * P] for i in range(4) for j in range(4)]
    x0 = 1.5
    for n, p in enumerate(patches):
        ax.imshow(p, extent=(x0 + n * 1.05, x0 + n * 1.05 + 0.9, 6.0, 6.9))
    ax.text(0.2, 6.45, "flatten", fontsize=9.5, va="center")
    # token boxes
    ax.add_patch(Rectangle((0.3, 3.6), 0.9, 1.2, fc="#f3e0f7", ec=PURPLE, lw=1.4))
    ax.text(0.75, 4.2, "[CLS]", ha="center", va="center", fontsize=8.5, rotation=90)
    for n in range(16):
        x = x0 + n * 1.05
        arrow(ax, x + 0.45, 5.95, x + 0.45, 4.85, lw=0.8)
        ax.add_patch(Rectangle((x, 3.6), 0.9, 1.2, fc="#e8eef8", ec=BLUE, lw=1.0))
        ax.text(x + 0.45, 4.2, f"z{n + 1}", ha="center", va="center", fontsize=8)
    ax.text(10, 5.4, "linear projection E  (P²·C → D)", ha="center", fontsize=9.5, color=BLUE,
            bbox=dict(fc="white", ec="none", alpha=0.95), zorder=5)
    ax.text(10, 3.25, "+ learned position embeddings  E_pos", ha="center", fontsize=9.5, color=ORANGE)
    box(ax, 0.3, 1.3, 18.6, 1.2, "Transformer encoder  ×L   (multi-head self-attention + MLP, LayerNorm, residuals)",
        fc="#eef6f0", ec=GREEN, fs=10.5)
    arrow(ax, 10, 3.0, 10, 2.55)
    arrow(ax, 0.75, 1.25, 0.75, 0.45, color=PURPLE)
    ax.text(1.0, 0.25, "class-token output → MLP head → logits", fontsize=9.5, color=PURPLE, va="center")
    save(fig, "vit-patches.png")


# ---------------------------------------------------------------- data + model
def digits_32():
    d = load_digits()
    x = torch.tensor(d.images, dtype=torch.float32)[:, None] / 16.0      # (1797,1,8,8) in [0,1]
    x = F.interpolate(x, size=32, mode="bilinear", align_corners=False)  # 32x32 like CIFAR
    x = (x - 0.5) / 0.5
    return x, torch.tensor(d.target)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Identity()
        if stride != 1 or cin != cout:   # projection shortcut when shapes change
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.short(x))   # y = F(x) + x


class TinyResNet(nn.Module):
    def __init__(self, num_classes=10, w=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        self.layer1 = BasicBlock(w, w)            # 32x32
        self.layer2 = BasicBlock(w, 2 * w, 2)     # 16x16
        self.layer3 = BasicBlock(2 * w, 4 * w, 2)  # 8x8
        self.fc = nn.Linear(4 * w, num_classes)

    def features(self, x):
        return self.layer3(self.layer2(self.layer1(self.stem(x))))

    def forward(self, x):
        return self.fc(self.features(x).mean(dim=(2, 3)))   # global average pooling


def augment(x, max_rot=15, scale=(0.8, 1.15), max_shift=0.12):
    """Random rotation, scale and shift (a random-resized-crop-like affine), batch-wise."""
    n = x.shape[0]
    ang = (torch.rand(n) * 2 - 1) * math.radians(max_rot)
    s = torch.empty(n).uniform_(*scale)
    t = (torch.rand(n, 2) * 2 - 1) * max_shift
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1), torch.stack([sin, cos, t[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


def warmup_cosine(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


def train(xtr, ytr, xva, yva, epochs, aug, label_smoothing=0.0, seed=0, bs=32, lr=3e-3, wd=0.05):
    torch.manual_seed(seed)
    model = TinyResNet()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = math.ceil(len(xtr) / bs)
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, total, warmup=2 * steps_per_epoch))
    hist = {"train": [], "val": []}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            xb, yb = xtr[idx], ytr[idx]
            if aug:
                xb = augment(xb)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=label_smoothing)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist["train"].append(accuracy(model, xtr, ytr))
        hist["val"].append(accuracy(model, xva, yva))
    return model, hist


@torch.no_grad()
def logits_of(model, x):
    model.eval()
    return torch.cat([model(x[i:i + 256]) for i in range(0, len(x), 256)])


def accuracy(model, x, y):
    return (logits_of(model, x).argmax(1) == y).float().mean().item()


def ece(probs, y, n_bins=10):
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = (correct[m].mean() - conf[m].mean()).abs()
            total += m.float().mean() * gap
            rows.append((lo.item(), hi.item(), correct[m].mean().item(), conf[m].mean().item(), int(m.sum())))
        else:
            rows.append((lo.item(), hi.item(), float("nan"), float("nan"), 0))
    return float(total), rows


def fit_temperature(logits, y, iters=300):
    log_t = torch.zeros(1, requires_grad=True)          # T = exp(log_t) > 0
    opt = torch.optim.Adam([log_t], lr=0.05)
    for _ in range(iters):
        loss = F.cross_entropy(logits / log_t.exp(), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return log_t.exp().item()


def grad_cam(model, x, target=None):
    """Grad-CAM on the last residual stage. x: (1,1,H,W)."""
    model.eval()
    feats = model.features(x)                 # (1, K, h, w)
    feats.retain_grad()
    logits = model.fc(feats.mean(dim=(2, 3)))
    c = logits.argmax(1).item() if target is None else target
    logits[0, c].backward()
    alpha = feats.grad.mean(dim=(2, 3), keepdim=True)          # (1, K, 1, 1) channel weights
    cam = F.relu((alpha * feats).sum(1, keepdim=True))         # (1, 1, h, w)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return (cam / (cam.max() + 1e-8))[0, 0].detach(), c


def fgsm(model, x, y, eps):
    x = x.clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    return (x + eps * x.grad.sign()).clamp(-1, 1).detach()


# ---------------------------------------------------------------- experiments
def experiments():
    x, y = digits_32()
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(x), generator=g)
    tr, cal, te = perm[:100], perm[100:600], perm[600:]
    xtr, ytr, xcal, ycal, xte, yte = x[tr], y[tr], x[cal], y[cal], x[te], y[te]
    print("split sizes", len(tr), len(cal), len(te))
    nparams = sum(p.numel() for p in TinyResNet().parameters())
    print("TinyResNet params", nparams)

    EP = 40
    m_plain, h_plain = train(xtr, ytr, xte, yte, EP, aug=False)
    m_aug, h_aug = train(xtr, ytr, xte, yte, EP, aug=True, label_smoothing=0.1)
    print(f"no aug     : train {h_plain['train'][-1]:.3f}  test {h_plain['val'][-1]:.3f}")
    print(f"aug + LS0.1: train {h_aug['train'][-1]:.3f}  test {h_aug['val'][-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    ep = np.arange(1, EP + 1)
    for ax, h, t in [(axes[0], h_plain, "no augmentation"),
                     (axes[1], h_aug, "augmentation + label smoothing 0.1")]:
        ax.plot(ep, h["train"], color=BLUE, lw=2, label="train")
        ax.plot(ep, h["val"], color=ORANGE, lw=2, label="test")
        ax.set_title(f"{t}\nfinal test acc {h['val'][-1]:.3f}")
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy"); axes[0].set_ylim(0.0, 1.02); axes[0].legend(loc="lower right")
    fig.suptitle("TinyResNet trained on 100 upsampled digits (AdamW, warmup + cosine)", y=1.06)
    save(fig, "training-curves.png")

    # ---- calibration with temperature scaling (the plain model: no augmentation, no smoothing)
    m_cal = m_plain
    lc, lt = logits_of(m_cal, xcal), logits_of(m_cal, xte)
    T = fit_temperature(lc, ycal)
    e0, rows0 = ece(lt.softmax(1), yte)
    e1, rows1 = ece((lt / T).softmax(1), yte)
    nll0 = F.cross_entropy(lt, yte).item(); nll1 = F.cross_entropy(lt / T, yte).item()
    acc_c = (lt.argmax(1) == yte).float().mean().item()
    print(f"calibration model: test acc {acc_c:.3f}  T={T:.2f}  ECE {e0:.3f}->{e1:.3f}  NLL {nll0:.3f}->{nll1:.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, rows, e, t in [(axes[0], rows0, e0, "before (T = 1)"), (axes[1], rows1, e1, f"after temperature scaling (T = {T:.2f})")]:
        centers = [(a + b) / 2 for a, b, *_ in rows]
        accs = [r[2] if r[4] else 0 for r in rows]
        confs = [r[3] if r[4] else c for r, c in zip(rows, centers)]
        ax.bar(centers, accs, width=0.095, color=BLUE, alpha=0.85, label="accuracy in bin", edgecolor="white")
        for c, a, cf, r in zip(centers, accs, confs, rows):
            if r[4]:
                ax.bar(c, cf - a, bottom=a, width=0.095, color="none", edgecolor=RED, hatch="//", lw=0.8)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.set_title(f"{t}\nECE = {e:.3f}")
        ax.set_xlabel("confidence (max softmax probability)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    from matplotlib.patches import Patch
    h, l = axes[0].get_legend_handles_labels()
    h.append(Patch(fc="none", ec=RED, hatch="//")); l.append("gap |accuracy - confidence|")
    axes[0].set_ylabel("accuracy"); axes[0].legend(h, l, loc="upper left", fontsize=9)
    save(fig, "reliability.png")

    # ---- Grad-CAM
    fig, axes = plt.subplots(2, 6, figsize=(11, 4.0))
    for k in range(6):
        i = int((yte == k * 1 + (k >= 3) * 3).nonzero()[0])  # digits 0,1,2,6,7,8
        xi = xte[i:i + 1]
        cam, c = grad_cam(m_aug, xi)
        img = (xi[0, 0] * 0.5 + 0.5).numpy()
        axes[0, k].imshow(img, cmap="gray"); axes[0, k].set_title(f"label {int(yte[i])}", fontsize=10)
        axes[1, k].imshow(img, cmap="gray"); axes[1, k].imshow(cam.numpy(), cmap="jet", alpha=0.45)
        axes[1, k].set_title(f"Grad-CAM, pred {c}", fontsize=10)
    for a in axes.ravel():
        a.axis("off")
    save(fig, "grad-cam.png")

    # ---- FGSM
    eps_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    accs = {}
    for name, m in [("no aug", m_plain), ("aug + LS", m_aug)]:
        accs[name] = []
        for eps in eps_list:
            m.eval()
            xa = torch.cat([fgsm(m, xte[i:i + 256], yte[i:i + 256], eps) for i in range(0, len(xte), 256)])
            accs[name].append(accuracy(m, xa, yte))
        print("FGSM", name, [f"{a:.3f}" for a in accs[name]])
    fig = plt.figure(figsize=(10.4, 3.4))
    i = int((yte == 3).nonzero()[0])
    xi, yi = xte[i:i + 1], yte[i:i + 1]
    xa = fgsm(m_aug, xi, yi, 0.2)
    pc = logits_of(m_aug, xi).softmax(1)[0]; pa = logits_of(m_aug, xa).softmax(1)[0]
    panels = [(xi, f"clean: pred {int(pc.argmax())} ({pc.max():.2f})"),
              ((xa - xi) / 0.4 + 0.0, "ε·sign(∇x L)   (ε = 0.2)"),
              (xa, f"adversarial: pred {int(pa.argmax())} ({pa.max():.2f})")]
    for k, (im, t) in enumerate(panels):
        ax = fig.add_axes([0.0 + k * 0.17, 0.05, 0.16, 0.8])
        ax.imshow((im[0, 0] * 0.5 + 0.5).numpy() if k != 1 else im[0, 0].numpy(), cmap="gray",
                  vmin=0 if k != 1 else -0.6, vmax=1 if k != 1 else 0.6)
        ax.set_title(t, fontsize=9.5); ax.axis("off")
    ax = fig.add_axes([0.58, 0.15, 0.4, 0.72])
    for (name, a), c in zip(accs.items(), [BLUE, ORANGE]):
        ax.plot(eps_list, a, "o-", color=c, lw=2, label=name)
    ax.set_xlabel("ε (inputs scaled to [-1, 1])"); ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(); ax.set_title("FGSM accuracy vs. ε")
    save(fig, "fgsm.png")


if __name__ == "__main__":
    import sys
    fig_timeline()
    fig_blocks()
    fig_depthwise()
    fig_vit()
    if "--diagrams-only" not in sys.argv:
        experiments()
