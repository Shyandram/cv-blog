"""Figures for DLCV 03B - Image Segmentation: Semantic, Instance and Panoptic.

Run:  .venv/bin/python scripts/figures/dl_03b_image_segmentation.py
Writes only to src/assets/deep-learning/dl-03b-image-segmentation/.
All images are synthetic (drawn shapes and label maps); nothing is downloaded.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-03b-image-segmentation"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    kb = path.stat().st_size / 1024
    w = plt.imread(path).shape[1]
    print(f"{name}: {kb:.0f} KB, width {w} px")
    assert kb < 1024 and w <= 1600


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def outlines(seg):
    """Boolean map of pixels where the segment id changes (4-neighbourhood)."""
    e = np.zeros(seg.shape, bool)
    e[:-1] |= seg[:-1] != seg[1:]
    e[1:] |= seg[:-1] != seg[1:]
    e[:, :-1] |= seg[:, :-1] != seg[:, 1:]
    e[:, 1:] |= seg[:, :-1] != seg[:, 1:]
    return e


# ------------------------------------------------------------------ figure 1: tasks
def street_scene(H=200, W=320, seed=3):
    """Synthetic street: stuff = sky, building, grass, road; things = 3 cars, 2 people."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    SKY, BUILDING, GRASS, ROAD, CAR, PERSON = range(6)
    sem = np.full((H, W), SKY)
    sem[(yy >= 15) & (yy < 112) & (xx >= 222)] = BUILDING
    sem[(yy >= 80) & (yy < 112) & (xx < 222)] = GRASS
    sem[yy >= 112] = ROAD
    inst = np.zeros((H, W), int)

    def rounded_rect(r0, r1, c0, c1, rad):
        m = (yy >= r0) & (yy < r1) & (xx >= c0) & (xx < c1)
        for (cy, cx) in [(r0 + rad, c0 + rad), (r0 + rad, c1 - rad), (r1 - rad, c0 + rad), (r1 - rad, c1 - rad)]:
            corner = ((yy < r0 + rad) | (yy >= r1 - rad)) & ((xx < c0 + rad) | (xx >= c1 - rad))
            near = (np.abs(yy - cy) <= rad) & (np.abs(xx - cx) <= rad)
            m &= ~(corner & near & ((yy - cy) ** 2 + (xx - cx) ** 2 > rad ** 2))
        return m

    def car(r0, r1, c0, c1):
        body = rounded_rect(r0 + (r1 - r0) // 3, r1, c0, c1, 6)
        cab = rounded_rect(r0, r0 + (r1 - r0) // 2, c0 + (c1 - c0) // 5, c1 - (c1 - c0) // 4, 6)
        return body | cab

    def person(cx, top, h):
        head = (yy - (top + h * 0.09)) ** 2 + (xx - cx) ** 2 <= (h * 0.09) ** 2
        body = ((yy - (top + h * 0.45)) / (h * 0.28)) ** 2 + ((xx - cx) / (h * 0.11)) ** 2 <= 1
        legs = (yy >= top + h * 0.6) & (yy < top + h) & (np.abs(xx - cx) <= h * 0.08)
        return head | body | legs

    things = [(CAR, car(118, 162, 28, 112)), (CAR, car(126, 170, 96, 186)),
              (CAR, car(98, 128, 246, 300)), (PERSON, person(203, 78, 72)),
              (PERSON, person(226, 84, 66))]
    for k, (c, m) in enumerate(things, start=1):
        sem[m] = c
        inst[m] = k
    base = {SKY: "#b9d7ee", BUILDING: "#b5aea4", GRASS: "#7fb36a", ROAD: "#6f6f6f"}
    inst_col = {1: "#c0392b", 2: "#2e86c1", 3: "#f1c40f", 4: "#8e44ad", 5: "#16a085"}
    img = np.zeros((H, W, 3))
    for c, h in base.items():
        img[sem == c] = hex2rgb(h)
    for k, h in inst_col.items():
        img[inst == k] = hex2rgb(h)
    img[(sem == BUILDING) & ((yy % 20 < 8) & (xx % 18 < 9)) & (yy > 22)] *= 0.75   # windows
    img[(sem == ROAD) & (np.abs(yy - 185) < 2) & (xx % 40 < 22)] = 0.95             # lane marks
    img *= 1 - 0.25 * (yy / H)[..., None] * (sem == SKY)[..., None]
    img += 0.035 * rng.standard_normal(img.shape)
    img = np.clip(ndi.gaussian_filter(img, (0.7, 0.7, 0)), 0, 1)
    return img, sem, inst


def fig_tasks():
    img, sem, inst = street_scene()
    names = ["sky", "building", "grass", "road", "car", "person"]
    sem_cols = ["#9ecae1", "#a3a3a3", "#74c476", "#525252", C[0], C[1]]
    inst_cols = {1: "#2a78d6", 2: "#1baf7a", 3: "#eda100", 4: "#e87ba4", 5: "#4a3aa7"}
    H, W = sem.shape
    sem_rgb = np.array([hex2rgb(h) for h in sem_cols])[sem]
    ins_rgb = np.ones((H, W, 3)) * 0.93
    for k, h in inst_cols.items():
        ins_rgb[inst == k] = hex2rgb(h)
    pan_rgb = sem_rgb.copy()
    for k, h in inst_cols.items():
        pan_rgb[inst == k] = hex2rgb(h)
    seg_id = sem * 10 + inst
    pan_rgb[outlines(seg_id)] = 1.0
    ins_rgb[outlines(inst) & (inst > 0)] = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    panels = [(img, "(a) Image"), (sem_rgb, "(b) Semantic: a class for every pixel"),
              (ins_rgb, "(c) Instance: a mask for every object"),
              (pan_rgb, "(d) Panoptic: class + instance id everywhere")]
    for ax, (im, t) in zip(axes.ravel(), panels):
        ax.imshow(im, interpolation="nearest")
        ax.set_title(t, fontsize=12, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
    axes[1, 0].text(6, 14, "stuff (sky, road, ...) is not labeled", fontsize=10, color=MUTED)
    axes[0, 1].text(20, 196, "the two touching cars become one blob", fontsize=10, color="white")
    handles = [Patch(fc=c, ec="none", label=n) for c, n in zip(sem_cols, names)]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "tasks.png")


# ------------------------------------------------------------------ figure 2: architectures
def fig_architectures():
    fig, axes = plt.subplots(3, 1, figsize=(10.4, 8.6))
    blue, orange, green, yellow = "#cfe1f5", "#fbd9c7", "#cdeedf", "#fbe7b0"

    def block(ax, x, h, color, label, w=0.55, sub=None):
        ax.add_patch(FancyBboxPatch((x, -h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=color, ec=INK, lw=1.0))
        ax.text(x + w / 2, -h / 2 - 0.25, label, ha="center", va="top", fontsize=10)
        if sub:
            ax.text(x + w / 2, 0, sub, ha="center", va="center", fontsize=9.5)

    def arrow(ax, x0, x1, y0=0.0, y1=0.0, rad=0.0, color=INK, ls="-", text=None, ty=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                     lw=1.3, color=color, ls=ls, connectionstyle=f"arc3,rad={rad}"))
        if text:
            ax.text((x0 + x1) / 2, ty if ty is not None else (y0 + y1) / 2 + 0.12, text,
                    ha="center", va="bottom", fontsize=9.5, color=color)

    def setup(ax, title):
        ax.set_xlim(-0.3, 15.3); ax.set_ylim(-2.15, 2.6); ax.axis("off")
        ax.set_title(title, loc="left", fontsize=12)

    H = {1: 2.6, 2: 2.0, 4: 1.5, 8: 1.1, 16: 0.8, 32: 0.55}
    # (a) FCN
    ax = axes[0]; setup(ax, "(a) FCN: classify at stride 32, upsample, fuse a few finer score maps")
    xs = [0, 1.2, 2.4, 3.6, 4.8, 6.0]
    for x, s in zip(xs, [1, 2, 4, 8, 16, 32]):
        block(ax, x, H[s], blue, "image" if s == 1 else f"1/{s}")
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + 0.6, b - 0.05)
    block(ax, 7.6, H[32], orange, "scores 1/32", w=0.7)
    arrow(ax, 6.6, 7.55)
    block(ax, 9.2, H[16], orange, "fused 1/16", w=0.7)
    block(ax, 10.8, H[8], orange, "fused 1/8", w=0.7)
    block(ax, 13.6, H[1], green, "per-pixel\nscores", w=0.8)
    arrow(ax, 8.35, 9.15, text="×2")
    arrow(ax, 9.95, 10.75, text="×2")
    arrow(ax, 11.55, 13.55, text="×8 bilinear")
    arrow(ax, 5.1, 9.5, 0.45, 0.45, rad=-0.25, color=C[0], ls="--")
    arrow(ax, 3.9, 11.1, 0.6, 0.6, rad=-0.22, color=C[0], ls="--")
    ax.text(7.1, 1.85, "skip: add a score map from a finer stage", color=C[0], fontsize=9.5, ha="center")

    # (b) U-Net style encoder-decoder
    ax = axes[1]; setup(ax, "(b) Encoder–decoder (U-Net style): upsample step by step, concatenate skips")
    enc = [(0, 1), (1.3, 2), (2.6, 4), (3.9, 8), (5.2, 16)]
    dec = [(6.6, 8), (7.9, 4), (9.2, 2), (10.5, 1)]
    for x, s in enc:
        block(ax, x, H[s], blue, "image" if s == 1 else f"1/{s}")
    for a, b in zip(enc[:-1], enc[1:]):
        arrow(ax, a[0] + 0.6, b[0] - 0.05)
    for x, s in dec:
        block(ax, x, H[s], yellow, f"1/{s}" if s > 1 else "1")
    arrow(ax, 5.8, 6.55, text="up ×2")
    for a, b in zip(dec[:-1], dec[1:]):
        arrow(ax, a[0] + 0.6, b[0] - 0.05)
    for (xe, s), (xd, _) in zip(enc[:-1][::-1], dec):
        arrow(ax, xe + 0.3, xd + 0.25, H[s] / 2 + 0.05, H[s] / 2 + 0.05, rad=-0.16, color=C[2])
    ax.text(7.0, -1.95, "green arcs: copy the encoder map of the same size and concatenate it (skip connection)",
            color=C[2], fontsize=9.5, ha="center", va="center")
    block(ax, 13.6, H[1], green, "per-pixel\nscores", w=0.8)
    arrow(ax, 11.1, 13.55, text="1×1 conv")

    # (c) DeepLab-style
    ax = axes[2]; setup(ax, "(c) DeepLab style: atrous convs keep stride 16, ASPP adds context, light decoder")
    xs = [0, 1.2, 2.4, 3.6, 4.8, 6.0]
    strides = [1, 2, 4, 8, 16, 16]
    for x, s in zip(xs, strides):
        block(ax, x, H[s], blue, "image" if s == 1 else f"1/{s}")
    ax.text(6.3, 0.55, "atrous,\nno stride", ha="center", va="bottom", fontsize=9.5, color=C[1])
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + 0.6, b - 0.05)
    # ASPP
    ax.add_patch(FancyBboxPatch((7.3, -1.25), 1.7, 2.5, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="white", ec=C[1], lw=1.4))
    for i, t in enumerate(["1×1 conv", "3×3, rate 6", "3×3, rate 12", "3×3, rate 18", "image pooling"]):
        ax.add_patch(Rectangle((7.4, 0.85 - i * 0.48), 1.5, 0.38, fc=orange, ec="none"))
        ax.text(8.15, 1.04 - i * 0.48, t, ha="center", va="center", fontsize=9)
    ax.text(8.15, -1.4, "ASPP (concat)", ha="center", va="top", fontsize=10)
    arrow(ax, 6.6, 7.25)
    block(ax, 10.2, H[4], yellow, "1/4", w=0.6)
    arrow(ax, 9.05, 10.15, text="×4")
    arrow(ax, 2.7, 10.5, H[4] / 2 + 0.05, H[4] / 2 + 0.05, rad=-0.26, color=C[2])
    ax.text(6.6, 2.35, "low-level features at 1/4 (concatenate)", color=C[2], fontsize=9.5, ha="center")
    block(ax, 13.6, H[1], green, "per-pixel\nscores", w=0.8)
    arrow(ax, 10.85, 13.55, text="3×3 convs, ×4")
    fig.tight_layout()
    save(fig, "architectures.png")


# ------------------------------------------------------------------ figure 3: dilation
def fig_dilation():
    fig = plt.figure(figsize=(10.4, 4.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.3, 1, 1, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    n = 21
    c = n // 2
    rows = {0: "input", 1: "layer 1 (rate 1)", 2: "layer 2 (rate 2)", 3: "layer 3 (rate 4)"}
    for y in range(4):
        ax.scatter(np.arange(n), np.full(n, y), s=22, color=GRID, zorder=2)
        ax.text(-1.2, y, rows[y], ha="right", va="center", fontsize=10)
    active = {3: {c}}
    for y, r in [(3, 4), (2, 2), (1, 1)]:
        nxt = set()
        for p in active[y]:
            for d in (-1, 0, 1):
                q = p + d * r
                nxt.add(q)
                ax.plot([p, q], [y, y - 1], color=C[0], lw=1.0, alpha=0.7, zorder=1)
        active[y - 1] = nxt
    for y, pts in active.items():
        ax.scatter(sorted(pts), [y] * len(pts), s=34, color=C[0] if y < 3 else C[1], zorder=3)
    lo, hi = min(active[0]), max(active[0])
    ax.annotate("", xy=(lo - 0.3, -0.55), xytext=(hi + 0.3, -0.55),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    ax.text(c, -0.75, f"receptive field = {hi - lo + 1} inputs, no downsampling", ha="center", va="top",
            fontsize=10)
    ax.set_xlim(-9, n); ax.set_ylim(-1.3, 3.5); ax.axis("off")
    ax.set_title("(a) Three 3-tap layers, rates 1, 2, 4", loc="left", fontsize=11.5)

    for i, r in enumerate([1, 2, 4]):
        ax = fig.add_subplot(gs[i + 1])
        g = np.zeros((11, 11))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                g[5 + dy * r, 5 + dx * r] = 1
        ax.imshow(g, cmap=ListedColormap(["#f3f2ef", C[0]]), vmin=0, vmax=1)
        ax.set_xticks(np.arange(-0.5, 11, 1), minor=True); ax.set_yticks(np.arange(-0.5, 11, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1)
        ax.tick_params(which="both", length=0); ax.set_xticks([]); ax.set_yticks([])
        k = 3 + 2 * (r - 1)
        ax.set_title(f"({'bcd'[i]}) 3×3, rate {r}\nspan {k}×{k}, 9 weights", fontsize=10.5)
    save(fig, "dilation.png")


# ------------------------------------------------------------------ figure 4: upsampling
def toy_labels():
    H = W = 256
    yy, xx = np.mgrid[0:H, 0:W]
    gt = np.zeros((H, W), int)
    gt[(yy - 120) ** 2 + (xx - 90) ** 2 < 60 ** 2] = 1
    gt[40:216, 178:184] = 2
    gt[196:216, 60:80] = 3
    return gt


def coarse_predictions(gt, s, K=4):
    H, W = gt.shape
    onehot = np.stack([(gt == k).astype(float) for k in range(K)])
    coarse = onehot.reshape(K, H // s, s, W // s, s).mean(axis=(2, 4))
    lab_nn = np.kron(coarse.argmax(0), np.ones((s, s), int))
    up = np.stack([ndi.zoom(c, s, order=1, mode="nearest", grid_mode=True) for c in coarse])
    return lab_nn, up.argmax(0)


def miou(gt, pred, K=4):
    return np.mean([((gt == k) & (pred == k)).sum() / ((gt == k) | (pred == k)).sum() for k in range(K)])


def fig_upsampling():
    gt = toy_labels()
    cmap = ListedColormap(["#f3f2ef", C[0], C[1], C[2]])
    lab_nn, lab_bi = coarse_predictions(gt, 16)
    fig = plt.figure(figsize=(10.4, 3.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.35], wspace=0.3)
    for i, (m, t) in enumerate([(gt, "(a) Ground truth"), (lab_nn, "(b) Stride 16:\ndecide, then enlarge"),
                                (lab_bi, "(c) Stride 16: bilinear\nscores, then decide")]):
        ax = fig.add_subplot(gs[i])
        ax.imshow(m, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        if i > 0:
            err = (m != gt)
            ov = np.zeros(gt.shape + (4,)); ov[err] = [0.1, 0.1, 0.1, 0.55]
            ax.imshow(ov, interpolation="nearest")
        ax.set_title(t, fontsize=10.5, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[3])
    strides = [4, 8, 16, 32]
    nn = [miou(gt, coarse_predictions(gt, s)[0]) for s in strides]
    bi = [miou(gt, coarse_predictions(gt, s)[1]) for s in strides]
    ax.plot(strides, nn, "o-", color=C[1], label="decide, then enlarge")
    ax.plot(strides, bi, "s-", color=C[0], label="bilinear scores")
    ax.set_xscale("log", base=2); ax.set_xticks(strides); ax.set_xticklabels(strides)
    ax.set_xlabel("output stride"); ax.set_ylabel("mIoU")
    ax.set_ylim(0.3, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=9.5, frameon=False, loc="lower left")
    ax.set_title("(d) Ideal coarse scores", fontsize=10.5, loc="left")
    save(fig, "upsampling.png")


# ------------------------------------------------------------------ figure 5: PQ toy
def pq_scene(boxes, sky_rows, H=100, W=200):
    seg = np.full((H, W), 2); seg[:sky_rows] = 1
    cls = {1: 0, 2: 1}
    for sid, (c, r0, r1, c0, c1) in enumerate(boxes, start=3):
        seg[r0:r1, c0:c1] = sid; cls[sid] = c
    return seg, cls


def panoptic_quality(gt_seg, gt_cls, pr_seg, pr_cls):
    """Same function as in the tutorial text."""
    out = {}
    for c in sorted(set(gt_cls.values()) | set(pr_cls.values())):
        g_ids = [i for i, k in gt_cls.items() if k == c]
        p_ids = [i for i, k in pr_cls.items() if k == c]
        ious, matched_p = [], set()
        for gi in g_ids:
            G = gt_seg == gi
            for pj in p_ids:
                P = (pr_seg == pj) & (gt_seg != 0)
                iou = (G & P).sum() / (G | P).sum()
                if iou > 0.5:
                    ious.append(iou); matched_p.add(pj)
        tp = len(ious); fn = len(g_ids) - tp; fp = len(p_ids) - len(matched_p)
        sq = np.mean(ious) if tp else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn)
        out[c] = (sq * rq, sq, rq, tp, fp, fn)
    return out


def fig_pq():
    SKY, ROAD, CAR, PERSON = 0, 1, 2, 3
    gt_seg, gt_cls = pq_scene([(CAR, 50, 80, 10, 55), (CAR, 50, 80, 65, 110),
                               (CAR, 38, 52, 125, 150), (PERSON, 30, 80, 165, 185)], 30)
    pr_seg, pr_cls = pq_scene([(CAR, 52, 80, 13, 55), (CAR, 55, 80, 65, 96), (CAR, 84, 97, 110, 150),
                               (PERSON, 30, 52, 165, 185), (PERSON, 52, 80, 165, 185)], 33)
    res = panoptic_quality(gt_seg, gt_cls, pr_seg, pr_cls)
    stuff_col = {SKY: "#cfe3f3", ROAD: "#bdbdbd"}
    thing_col = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]

    def render(seg, cls):
        rgb = np.zeros(seg.shape + (3,))
        k = 0
        for sid in sorted(cls):
            if cls[sid] in stuff_col:
                rgb[seg == sid] = hex2rgb(stuff_col[cls[sid]])
            else:
                rgb[seg == sid] = hex2rgb(thing_col[k % len(thing_col)]); k += 1
        rgb[outlines(seg)] = 1
        return rgb

    def iou(a, b):
        return (a & b).sum() / (a | b).sum()

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    axes[0].imshow(render(gt_seg, gt_cls), interpolation="nearest")
    axes[1].imshow(render(pr_seg, pr_cls), interpolation="nearest")
    axes[0].set_title("(a) Ground truth", loc="left", fontsize=11)
    axes[1].set_title("(b) Prediction", loc="left", fontsize=11)
    gl = {1: "sky", 2: "road", 3: "car A", 4: "car B", 5: "car C", 6: "person"}
    pos = {1: (15, 100), 2: (90, 100)}
    for sid, t in gl.items():
        r, c = pos.get(sid, np.argwhere(gt_seg == sid).mean(0))
        col = INK if sid <= 2 else "white"
        axes[0].text(c, r, t, ha="center", va="center", fontsize=8.5, color=col, weight="bold",
                     rotation=90 if t == "person" else 0)
    for sid in range(3, 8):
        P = pr_seg == sid
        best = max(iou(gt_seg == g, P) for g in gt_cls if gt_cls[g] == pr_cls[sid])
        t = "TP" if best > 0.5 else "FP"
        rc = np.argwhere(P)
        r, c = rc.mean(0)
        sep = "\n" if np.ptp(rc[:, 1]) < 30 else " "
        axes[1].text(c, r, f"{t}{sep}{best:.2f}", ha="center", va="center", fontsize=8, color="white",
                     weight="bold")
    axes[1].text(137, 45, "car C\nmissed\n(FN)", ha="center", va="center", fontsize=8, color=INK)
    for ax in axes[:2]:
        ax.set_xticks([]); ax.set_yticks([])
    ax = axes[2]; ax.axis("off")
    names = ["sky", "road", "car", "person"]
    rows = [("class", "SQ", "RQ", "PQ")] + [(names[c], f"{v[1]:.3f}", f"{v[2]:.3f}", f"{v[0]:.3f}")
                                          for c, v in res.items()]
    for i, row in enumerate(rows):
        for j, t in enumerate(row):
            ax.text(0.0 + j * 0.27, 0.9 - i * 0.13, t, fontsize=10.5, weight="bold" if i == 0 else "normal",
                    transform=ax.transAxes)
    pq_all = np.mean([v[0] for v in res.values()])
    ax.text(0.0, 0.9 - 5 * 0.13, f"PQ (mean of 4 classes) = {pq_all:.3f}", fontsize=10.5, transform=ax.transAxes)
    ax.text(0.0, 0.9 - 6 * 0.13, "labels in (b): IoU with the best\nground-truth segment of that class",
            fontsize=9.5, color=MUTED, transform=ax.transAxes, va="top")
    ax.set_title("(c) Panoptic quality", loc="left", fontsize=11)
    fig.tight_layout()
    save(fig, "pq-toy.png")


if __name__ == "__main__":
    fig_tasks()
    fig_architectures()
    fig_dilation()
    fig_upsampling()
    fig_pq()
