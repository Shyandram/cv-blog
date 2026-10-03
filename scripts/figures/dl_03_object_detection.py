"""Figures for DL 03 - Object Detection: From R-CNN to DETR.

Run:  .venv/bin/python scripts/figures/dl_03_object_detection.py
Writes only to src/assets/deep-learning/dl-03-object-detection/.
All images are synthetic (drawn shapes, random boxes); nothing is downloaded.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-03-object-detection"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})

# categorical palette, fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    kb = path.stat().st_size / 1024
    print(f"{name}: {kb:.0f} KB")
    assert kb < 1024


# ---------------------------------------------------------------- helpers
def box_iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def draw_box(ax, b, color, label=None, lw=2.2, ls="-"):
    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                           ec=color, lw=lw, ls=ls))
    if label:
        ax.text(b[0] + 1, b[1] - 2, label, color="white", fontsize=9, va="bottom",
                bbox=dict(fc=color, ec="none", pad=1.5))


def make_scene(H=200, W=300):
    """Synthetic scene: two disks ('ball'), one square ('box'), one triangle ('cone')."""
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.ones((H, W, 3)) * np.array([0.93, 0.93, 0.90])
    inst = np.zeros((H, W), int)
    objs = []
    # disks
    for k, (cx, cy, r) in enumerate([(70, 120, 38), (125, 105, 32)]):
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        inst[m] = k + 1
        objs.append(("ball", [cx - r, cy - r, cx + r, cy + r]))
    # square
    m = (xx >= 185) & (xx <= 255) & (yy >= 40) & (yy <= 110)
    inst[m] = 3
    objs.append(("box", [185, 40, 255, 110]))
    # triangle
    m = (yy >= 120) & (yy <= 185) & (np.abs(xx - 230) <= (yy - 120) * 0.55)
    inst[m] = 4
    objs.append(("cone", [230 - 65 * 0.55, 120, 230 + 65 * 0.55, 185]))
    cls_of = {1: 0, 2: 0, 3: 1, 4: 2}
    base = [np.array([0.16, 0.47, 0.84]), np.array([0.92, 0.41, 0.20]), np.array([0.11, 0.69, 0.48])]
    for k, c in cls_of.items():
        shade = 0.85 if k == 2 else 1.0
        img[inst == k] = base[c] * shade
    return img, inst, objs, cls_of


# ---------------------------------------------------------------- 1. tasks
def fig_tasks():
    img, inst, objs, cls_of = make_scene()
    fig, axs = plt.subplots(1, 4, figsize=(10.4, 2.6))
    titles = ["(a) Classification", "(b) Detection", "(c) Semantic segmentation", "(d) Instance segmentation"]
    for ax, t in zip(axs, titles):
        ax.set_title(t, fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
    axs[0].imshow(img)
    axs[0].text(150, 190, '"ball"  (one label per image)', ha="center", fontsize=9.5,
                bbox=dict(fc="white", ec=MUTED, pad=2))
    axs[1].imshow(img)
    scores = [0.97, 0.91, 0.88, 0.76]
    for (name, b), s in zip(objs, scores):
        col = C[{"ball": 6, "box": 4, "cone": 3}[name]]
        draw_box(axs[1], b, col, lw=1.8)
        lx, ly, va = (b[0] + 1, b[3] + 2, "top") if b[0] < 40 else (b[0] + 1, b[1] - 2, "bottom")
        axs[1].text(lx, ly, f"{name} {s:.2f}", color="white", fontsize=9, va=va,
                    bbox=dict(fc=col, ec="none", pad=1.5))
    sem = np.ones(img.shape) * 0.97
    clscol = [np.array([0.16, 0.47, 0.84]), np.array([0.92, 0.41, 0.20]), np.array([0.11, 0.69, 0.48])]
    for k, c in cls_of.items():
        sem[inst == k] = clscol[c]
    axs[2].imshow(sem)
    ins = np.ones(img.shape) * 0.97
    inscol = ["#2a78d6", "#4a3aa7", "#eb6834", "#1baf7a"]
    for k in range(1, 5):
        ins[inst == k] = np.array(matplotlib.colors.to_rgb(inscol[k - 1]))
    axs[3].imshow(ins)
    axs[2].text(150, 192, "both balls share one class color", ha="center", fontsize=9)
    axs[3].text(150, 192, "each object gets its own mask", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "tasks.png")


# ---------------------------------------------------------------- 2. anchors + FPN
def fig_anchors_fpn():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.4), gridspec_kw=dict(width_ratios=[1, 1.25]))
    # left: grid with 9 anchors at one location
    S = 8
    for i in range(S + 1):
        a1.plot([0, S], [i, i], color=GRID, lw=0.8); a1.plot([i, i], [0, S], color=GRID, lw=0.8)
    cx, cy = 4.5, 3.5
    a1.add_patch(Rectangle((4, 3), 1, 1, fc="#fde7c2", ec="none"))
    scales, ratios = [1.2, 2.0, 3.0], [0.5, 1.0, 2.0]
    for si, s in enumerate(scales):
        for r in ratios:
            w, h = s * np.sqrt(1 / r) * 0.9, s * np.sqrt(r) * 0.9
            a1.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec=C[[0, 1, 6][si]], lw=1.6))
    a1.plot(cx, cy, "o", color=INK, ms=4)
    a1.set_xlim(-0.2, S + 0.2); a1.set_ylim(S + 0.2, -0.2); a1.set_aspect("equal")
    a1.set_xticks([]); a1.set_yticks([])
    a1.set_title("9 anchors (3 scales x 3 aspect ratios)\ncentered on one feature-map cell", fontsize=10.5)
    a1.legend(handles=[plt.Line2D([], [], color=C[[0, 1, 6][i]], lw=2, label=f"scale {i + 1}") for i in range(3)],
              loc="lower left", fontsize=9, frameon=True)

    # right: FPN diagram
    a2.set_xlim(0, 11); a2.set_ylim(0, 10); a2.axis("off")
    levels = [("C2", "P2", 4, 3.2), ("C3", "P3", 8, 2.5), ("C4", "P4", 16, 1.8), ("C5", "P5", 32, 1.2)]
    ys = [1.2, 3.6, 5.9, 8.0]
    for (cn, pn, stride, size), y in zip(levels, ys):
        # bottom-up (backbone)
        a2.add_patch(FancyBboxPatch((2.2 - size / 2, y - size / 4), size, size / 2,
                                    boxstyle="round,pad=0.02", fc="#dbe8f8", ec=C[0], lw=1.4))
        a2.text(2.2, y, cn, ha="center", va="center", fontsize=10)
        # top-down
        a2.add_patch(FancyBboxPatch((6.6 - size / 2, y - size / 4), size, size / 2,
                                    boxstyle="round,pad=0.02", fc="#fde0d3", ec=C[1], lw=1.4))
        a2.text(6.6, y, pn, ha="center", va="center", fontsize=10)
        a2.annotate("", xy=(6.6 - size / 2 - 0.08, y), xytext=(2.2 + size / 2 + 0.08, y),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
        a2.text(4.4, y + 0.15, "1x1 lateral", fontsize=8, color=MUTED, ha="center", va="bottom")
        a2.text(9.0, y, f"stride {stride}\npredict here", fontsize=8.5, va="center")
    for y0, y1 in zip(ys[:-1], ys[1:]):
        a2.annotate("", xy=(2.2, y1 - 0.45), xytext=(2.2, y0 + 0.45),
                    arrowprops=dict(arrowstyle="->", color=C[0], lw=1.4))
        a2.annotate("", xy=(6.6, y0 + 0.45), xytext=(6.6, y1 - 0.45),
                    arrowprops=dict(arrowstyle="->", color=C[1], lw=1.4))
    a2.text(0.8, 9.6, "bottom-up backbone", color=C[0], fontsize=10)
    a2.text(5.2, 9.6, "top-down + 2x upsample", color=C[1], fontsize=10)
    a2.text(2.2, 0.05, "image", ha="center", fontsize=9, color=MUTED)
    a2.set_title("Feature Pyramid Network: small objects on fine levels,\nlarge objects on coarse levels",
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, "anchors-fpn.png")


# ---------------------------------------------------------------- 3. NMS
def nms(boxes, scores, thr):
    order = np.argsort(-scores); keep = []
    while order.size:
        i = order[0]; keep.append(i)
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= thr]
    return keep


def soft_nms(boxes, scores, sigma=0.5):
    s = scores.copy(); idx = list(range(len(s))); out = {}
    while idx:
        j = max(idx, key=lambda k: s[k]); out[j] = s[j]; idx.remove(j)
        for k in idx:
            s[k] *= np.exp(-box_iou(boxes[j:j + 1], boxes[k:k + 1])[0, 0] ** 2 / sigma)
    return out


def fig_nms():
    rng = np.random.default_rng(3)
    gts = np.array([[30, 30, 110, 170], [95, 40, 170, 175], [210, 60, 280, 140]], float)
    boxes, scores = [], []
    for g, top in zip(gts, [0.95, 0.88, 0.9]):
        w, h = g[2] - g[0], g[3] - g[1]
        for k in range(6):
            j = rng.normal(0, 0.07, 4) * np.array([w, h, w, h])
            boxes.append(g + j); scores.append(top - 0.08 * k - rng.uniform(0, 0.03))
    boxes, scores = np.array(boxes), np.clip(np.array(scores), 0.05, 1)
    keep = nms(boxes, scores, 0.5)
    soft = soft_nms(boxes, scores)

    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.3))
    for ax in axs:
        ax.set_xlim(0, 300); ax.set_ylim(200, -18); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(Rectangle((0, -18), 300, 218, fc="#f4f4f1", ec="none"))
    owner = box_iou(boxes, gts).argmax(1)
    for b, s, o in zip(boxes, scores, owner):
        draw_box(axs[0], b, C[[0, 1, 6][o]], lw=1.0 + 1.5 * s)
    axs[0].set_title(f"(a) Raw detections: {len(boxes)} boxes", fontsize=10.5)
    for i in keep:
        draw_box(axs[1], boxes[i], C[[0, 1, 6][owner[i]]], f"{scores[i]:.2f}", lw=2.2)
    axs[1].set_title(f"(b) Greedy NMS, IoU > 0.5: {len(keep)} kept", fontsize=10.5)
    for i, s in soft.items():
        if s > 0.05:
            draw_box(axs[2], boxes[i], C[[0, 1, 6][owner[i]]], lw=0.4 + 2.6 * s)
    axs[2].set_title("(c) Soft-NMS: scores decayed\n(line width = new score)", fontsize=10.5)
    fig.tight_layout()
    save(fig, "nms.png")


# ---------------------------------------------------------------- 4. pipelines
def block(ax, x, y, w, h, text, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, color=INK)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=MUTED, lw=1.3))


def fig_pipelines():
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    ax.set_xlim(0, 16.4); ax.set_ylim(0, 9.4); ax.axis("off")
    rows = [
        ("Two-stage\n(Faster R-CNN)", 7.3, [("Backbone\n+ FPN", 1.7), ("RPN:\n~1000s of\nproposals", 1.7),
                                            ("RoIPool /\nRoIAlign 7x7", 1.7), ("Head: class\n+ box refine", 1.8),
                                            ("NMS", 1.1), ("Boxes", 1.2)]),
        ("One-stage\n(RetinaNet / YOLO)", 4.2, [("Backbone\n+ neck", 1.7), ("Dense head at\nevery location\n(+ anchors)", 2.3),
                                                ("Score\nthreshold", 1.5), ("NMS", 1.1), ("Boxes", 1.2)]),
        ("Set prediction\n(DETR)", 1.1, [("Backbone", 1.4), ("Transformer\nencoder", 1.7),
                                         ("Decoder with\nN object\nqueries", 1.9), ("N (class, box)\npairs", 1.8),
                                         ("Boxes\n(no NMS)", 1.4)]),
    ]
    cols = [("#dbe8f8", C[0]), ("#fde0d3", C[1]), ("#d4f0e4", C[2])]
    for (name, y, blocks), (fc, ec) in zip(rows, cols):
        ax.text(0.05, y + 0.6, name, fontsize=10.5, va="center", fontweight="bold", color=ec)
        x = 3.4
        for k, (t, w) in enumerate(blocks):
            hl = t.startswith("NMS") or "no NMS" in t
            block(ax, x, y, w, 1.2, t, "#fff4d6" if hl else fc, C[3] if hl else ec)
            if k < len(blocks) - 1:
                arrow(ax, x + w + 0.03, y + 0.6, x + w + 0.37, y + 0.6)
            x += w + 0.4
    ax.text(11.1, 0.35, "training: Hungarian matching gives each object exactly one query",
            fontsize=9, color=MUTED, ha="center")
    ax.text(10.6, 3.55, "training: anchors/points labeled by IoU or center rules",
            fontsize=9, color=MUTED, ha="center")
    ax.text(10.6, 6.65, "training: proposals labeled by IoU with ground truth",
            fontsize=9, color=MUTED, ha="center")
    save(fig, "pipelines.png")


# ---------------------------------------------------------------- 5. PR curve
def synthetic_pr(seed=0, jitter=0.06):
    rng = np.random.default_rng(seed)
    gts, dets = [], []
    for _ in range(50):
        n = rng.integers(1, 5)
        xy = rng.uniform(0, 200, (n, 2)); wh = rng.uniform(20, 80, (n, 2))
        g = np.hstack([xy, xy + wh]); gts.append(g)
        keep = rng.random(n) > 0.15
        jit = rng.normal(0, jitter, (n, 4)) * np.hstack([wh, wh])
        db = (g + jit)[keep]; ds = rng.uniform(0.4, 1.0, keep.sum())
        m = rng.integers(0, 3)
        fxy = rng.uniform(0, 200, (m, 2)); fwh = rng.uniform(20, 80, (m, 2))
        db = np.vstack([db, np.hstack([fxy, fxy + fwh])]); ds = np.concatenate([ds, rng.uniform(0, 0.7, m)])
        dets.append((db, ds))
    return gts, dets


def match(dets, gts, thr):
    recs = sorted([(ds[k], i, k) for i, (db, ds) in enumerate(dets) for k in range(len(ds))], key=lambda r: -r[0])
    used = [np.zeros(len(g), bool) for g in gts]; tp = np.zeros(len(recs))
    for n, (s, i, k) in enumerate(recs):
        ious = box_iou(dets[i][0][k:k + 1], gts[i])[0]; ious[used[i]] = -1
        j = ious.argmax()
        if ious[j] >= thr:
            tp[n] = 1; used[i][j] = True
    ctp = np.cumsum(tp)
    return ctp / sum(len(g) for g in gts), ctp / np.arange(1, len(tp) + 1)


def fig_pr():
    gts, dets = synthetic_pr()
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for ax, thr in zip(axs, [0.5, 0.75]):
        rec, prec = match(dets, gts, thr)
        r = np.concatenate([[0], rec, [rec[-1]]]); p = np.concatenate([[prec[0]], prec, [0]])
        env = np.maximum.accumulate(p[::-1])[::-1]
        ax.fill_between(r, env, step="post", color=C[0], alpha=0.18, lw=0)
        ax.plot(rec, prec, color=MUTED, lw=1.2, label="raw precision")
        ax.step(r, env, where="post", color=C[0], lw=2, label="interpolated (envelope)")
        pts = np.linspace(0, 1, 11)
        pi = [env[r >= t].max() if np.any(r >= t) else 0 for t in pts]
        ax.plot(pts, pi, "o", color=C[1], ms=6, label="VOC 11-point samples")
        i = np.where(r[1:] != r[:-1])[0]
        apv = np.sum((r[i + 1] - r[i]) * env[i + 1])
        ax.set_title(f"IoU threshold {thr}: AP (shaded area) = {apv:.3f}", fontsize=10.5)
        ax.set_xlabel("recall"); ax.set_ylabel("precision")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05); ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    save(fig, "pr-curve.png")


# ---------------------------------------------------------------- 6. IoU thresholds
def fig_iou():
    g = np.array([40, 40, 140, 140], float)

    def shift_for(target):
        # shift a same-size box diagonally until IoU == target (solve on a grid)
        for d in np.linspace(0, 100, 20001):
            b = g + np.array([d, d * 0.4, d, d * 0.4])
            if box_iou(g[None], b[None])[0, 0] <= target:
                return b
    fig, axs = plt.subplots(1, 4, figsize=(10.4, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.35]))
    for ax, t in zip(axs[:3], [0.5, 0.75, 0.9]):
        b = shift_for(t)
        ax.set_xlim(20, 200); ax.set_ylim(185, 20); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ix = [max(g[0], b[0]), max(g[1], b[1]), min(g[2], b[2]), min(g[3], b[3])]
        ax.add_patch(Rectangle((ix[0], ix[1]), ix[2] - ix[0], ix[3] - ix[1], fc=C[3], alpha=0.35, ec="none"))
        draw_box(ax, g, C[5], lw=2.2)
        draw_box(ax, b, C[1], lw=2.2, ls="--")
        ax.set_title(f"IoU = {t}", fontsize=10.5)
    axs[0].legend(handles=[plt.Line2D([], [], color=C[5], lw=2, label="ground truth"),
                           plt.Line2D([], [], color=C[1], lw=2, ls="--", label="prediction")],
                  loc="lower right", fontsize=8.5)
    # IoU of same-size squares vs relative shift
    ax = axs[3]
    s = np.linspace(0, 1, 200)
    iou_1d = (1 - s) / (1 + s)                       # shift along one axis
    iou_2d = (1 - s) ** 2 / (2 - (1 - s) ** 2)      # equal shift along both axes
    ax.plot(s, iou_1d, color=C[0], lw=2, label="shift along x")
    ax.plot(s, iou_2d, color=C[6], lw=2, label="shift along x and y")
    for t in (0.5, 0.75):
        ax.axhline(t, color=GRID, lw=1, ls="--")
        ax.text(1.0, t + 0.02, f"{t}", ha="right", fontsize=9, color=MUTED)
    ax.set_xlabel("shift / box side"); ax.set_ylabel("IoU")
    ax.set_title("IoU of two equal squares", fontsize=10.5)
    ax.legend(fontsize=8.5); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "iou-thresholds.png")


# ---------------------------------------------------------------- 7. timeline
def fig_timeline():
    ev = [
        (2001, "Viola-Jones", 0), (2005, "HOG + SVM", 0), (2010, "DPM (journal)", 0),
        (2014, "R-CNN", 1), (2015, "Fast R-CNN", 1), (2015, "Faster R-CNN", 1),
        (2016, "YOLO", 2), (2016, "SSD", 2), (2017, "FPN", 1), (2017, "Mask R-CNN", 1),
        (2017, "RetinaNet", 2), (2018, "Cascade R-CNN", 1), (2018, "CornerNet", 3),
        (2019, "FCOS", 3), (2019, "CenterNet", 3), (2020, "DETR", 4), (2021, "Deformable DETR", 4),
        (2022, "DINO", 4), (2022, "OWL-ViT", 5), (2023, "YOLOv7", 2),
        (2024, "RT-DETR", 4), (2024, "Grounding DINO", 5), (2024, "YOLO-World", 5),
        (2024, "YOLOv9 / v10", 2), (2025, "YOLOv12 (arXiv)", 2), (2025, "SAM 3 (arXiv)", 5),
    ]
    fams = ["Classical", "Two-stage", "One-stage", "Anchor-free", "DETR family", "Open-vocabulary"]
    fcol = [MUTED, C[0], C[1], C[2], C[6], C[4]]
    years = sorted({y for y, _, _ in ev})
    cell = {}
    for y, name, f in ev:
        cell.setdefault((y, f), []).append(name)
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    colx = np.arange(len(fams)) * 1.75
    for j, (f, c) in enumerate(zip(fams, fcol)):
        ax.text(colx[j], -0.9, f, ha="center", va="center", fontsize=10, color=c, fontweight="bold")
    for i, y in enumerate(years):
        ax.axhline(i, color=GRID, lw=0.6, zorder=0)
        ax.text(-1.0, i, str(y), ha="right", va="center", fontsize=10, color=INK)
    for (y, f), names in cell.items():
        i = years.index(y)
        ax.text(colx[f], i, "\n".join(names) if len(names) > 1 and len(" / ".join(names)) > 18 else " / ".join(names),
                ha="center", va="center", fontsize=8.8, color=INK,
                bbox=dict(fc="white", ec=fcol[f], lw=1.2, boxstyle="round,pad=0.25"))
    ax.annotate("", xy=(-1.55, len(years) - 0.5), xytext=(-1.55, -0.5),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.set_xlim(-1.8, colx[-1] + 0.95); ax.set_ylim(len(years) - 0.4, -1.4)
    ax.axis("off")
    ax.set_title("Detection milestones, by year of the publication cited in this tutorial\n"
                 "(years without listed milestones are omitted)", fontsize=10.5)
    save(fig, "timeline.png")


if __name__ == "__main__":
    fig_tasks()
    fig_anchors_fpn()
    fig_nms()
    fig_pipelines()
    fig_pr()
    fig_iou()
    fig_timeline()
