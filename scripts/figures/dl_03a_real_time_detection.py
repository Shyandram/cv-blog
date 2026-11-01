"""Figures for DLCV 03A - One-Stage Detection, Read Through YOLOv4.

Run:  .venv/bin/python scripts/figures/dl_03a_real_time_detection.py [figure names]
Writes only to src/assets/deep-learning/dl-03a-real-time-detection/.
Everything is synthetic (diagrams, random boxes, toy distributions, coloured rectangles);
nothing is downloaded.
"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-03a-real-time-detection"
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
    print(f"{name}: {kb:.0f} KB, {w} px wide")
    assert kb < 1024 and w <= 1600


def block(ax, x, y, w, h, text, fc, ec, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, x0, y0, x1, y1, color=MUTED, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=1.3, linestyle=ls))


def iou_matrix(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


# ---------------------------------------------------------------- 1. lineage / timeline
def fig_timeline():
    # (label, year of first public release, lane, kind, y-offset)
    # kind: "peer" = peer-reviewed venue, "arxiv" = arXiv / tech report only,
    #       "soft" = software release without a paper at release time, "announced" = not released
    items = [
        ("YOLO v1", 2015, 0, "peer", 0), ("YOLOv2", 2016, 0, "peer", 0), ("YOLOv3", 2018, 0, "arxiv", 0),
        ("YOLOv4", 2020, 1, "arxiv", 0), ("YOLOv7", 2022, 1, "peer", 0), ("YOLOv9", 2024, 1, "peer", 0),
        ("YOLOv5", 2020, 2, "soft", 0), ("YOLOv8", 2023, 2, "soft", 0), ("YOLO11", 2024, 2, "soft", 0),
        ("YOLO26", 2026, 2, "soft", -0.25), ("YOLO27\n(announced)", 2026, 2, "announced", 0.3),
        ("YOLOX", 2021, 3, "arxiv", 0), ("YOLOv6", 2022, 3, "arxiv", 0), ("YOLOv10", 2024, 3, "peer", 0),
        ("YOLOv12", 2025, 3, "peer", -0.25), ("YOLOv13", 2025, 3, "arxiv", 0.3),
        ("RT-DETR", 2023, 4, "peer", 0), ("RT-DETRv2", 2024, 4, "arxiv", -0.5), ("D-FINE", 2024, 4, "peer", 0.05),
        ("DEIM", 2024, 4, "peer", 0.45), ("RF-DETR", 2025, 4, "peer", -0.25), ("DEIMv2", 2025, 4, "arxiv", 0.3),
    ]
    lanes = ["Redmon & Farhadi", "Bochkovskiy, Wang & Liao", "Ultralytics releases",
             "Other research groups", "Real-time DETRs"]
    style = {"peer": dict(marker="o", mfc=C[0], mec=C[0]), "arxiv": dict(marker="s", mfc="white", mec=C[1]),
             "soft": dict(marker="D", mfc=C[2], mec=C[2]), "announced": dict(marker="D", mfc="white", mec=MUTED)}
    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    gap = 1.6                                            # vertical distance between lanes
    for k in range(len(lanes)):
        ax.axhline(k * gap, color=GRID, lw=1, zorder=0)
    for label, yr, lane, kind, dy in items:
        y = lane * gap + dy
        ax.plot(yr, y, ms=9, mew=1.8, zorder=3, **style[kind])
        below = dy > 0.2                                 # lower item of a stacked pair: label underneath
        ax.text(yr, y + 0.13 if below else y - 0.13, label, ha="center", va="top" if below else "bottom",
                fontsize=10, color=MUTED if kind == "announced" else INK)
    ax.set_yticks([k * gap for k in range(len(lanes))], lanes)
    ax.set_xticks(range(2015, 2027))
    ax.set_xlim(2014.4, 2026.8); ax.set_ylim(-0.75, (len(lanes) - 1) * gap + 1.0)
    ax.invert_yaxis()
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("year of first public release (arXiv preprint or software)")
    handles = [Line2D([], [], ls="", ms=9, mew=1.8, **style[k]) for k in ["peer", "arxiv", "soft", "announced"]]
    ax.legend(handles, ["peer-reviewed paper (venue may be a year later)", "arXiv / technical report only",
                        "software release (no paper at release)", "announced, not released"],
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=10)
    save(fig, "yolo-timeline.png")


# ---------------------------------------------------------------- 2. backbone - neck - head
def fig_anatomy():
    fig, ax = plt.subplots(figsize=(10.0, 4.9))
    ax.set_xlim(0, 18); ax.set_ylim(0.4, 8.4); ax.axis("off")
    blue, orange, green, yellow = "#dceafa", "#fde3d6", "#d6f1e6", "#fdf0cc"
    block(ax, 0.1, 3.3, 1.5, 1.8, "Input\n640 x 640", "#f1f0ee", MUTED)
    ax.text(3.3, 8.05, "Backbone", ha="center", fontsize=11, weight="bold")
    ax.text(7.3, 8.05, "Neck (FPN + PAN)", ha="center", fontsize=11, weight="bold")
    ax.text(11.3, 8.05, "Head (decoupled)", ha="center", fontsize=11, weight="bold")
    ys = [5.9, 3.7, 1.5]
    names = [("C3", "stride 8", "80 x 80"), ("C4", "stride 16", "40 x 40"), ("C5", "stride 32", "20 x 20")]
    for y, (c, s, g) in zip(ys, names):
        block(ax, 2.4, y, 1.8, 1.3, f"{c}\n{s}", blue, C[0])
        block(ax, 6.4, y, 1.8, 1.3, f"P{c[1]}", orange, C[1])
        block(ax, 10.0, y + 0.68, 2.6, 0.6, "class branch", green, C[2])
        block(ax, 10.0, y + 0.02, 2.6, 0.6, "box branch", green, C[2])
        arrow(ax, 4.2, y + 0.65, 6.4, y + 0.65)
        arrow(ax, 8.2, y + 0.65, 10.0, y + 0.98)
        arrow(ax, 8.2, y + 0.65, 10.0, y + 0.32)
        ax.text(12.75, y + 0.65, f"{g}\ncells", va="center", fontsize=10)
    arrow(ax, 1.6, 4.8, 2.4, 6.3); arrow(ax, 3.3, 5.9, 3.3, 5.0); arrow(ax, 3.3, 3.7, 3.3, 2.8)
    # FPN top-down (left of P blocks) and PAN bottom-up (right side)
    arrow(ax, 6.7, 2.8, 6.7, 3.7, C[1]); arrow(ax, 6.7, 5.0, 6.7, 5.9, C[1])
    arrow(ax, 7.9, 5.9, 7.9, 5.0, C[4]); arrow(ax, 7.9, 3.7, 7.9, 2.8, C[4])
    ax.text(5.6, 0.75, "orange: top-down (FPN)\npink: bottom-up (PAN)", fontsize=10, color=MUTED)
    block(ax, 15.1, 4.3, 2.7, 1.8, "8,400\ncandidates\n(box + scores)", yellow, C[3])
    for y in ys:
        arrow(ax, 14.2, y + 0.65, 15.1, 5.2)
    ax.text(16.45, 3.9, "one-to-many head:\nthreshold + NMS\n\none-to-one head:\ntop-k, no NMS",
            ha="center", va="top", fontsize=10)
    save(fig, "anatomy.png")


# ---------------------------------------------------------------- 3. one-to-many vs one-to-one
def toy_assignment():
    rng = np.random.default_rng(3)
    gt = np.array([[20, 20, 60, 70], [55, 30, 95, 80], [10, 80, 40, 100]], float)
    ys, xs = np.mgrid[0:8, 0:8]
    pts = np.stack([(xs.ravel() + 0.5) * 12.5, (ys.ravel() + 0.5) * 12.5], axis=1)
    inside = ((pts[:, None, 0] > gt[None, :, 0]) & (pts[:, None, 0] < gt[None, :, 2]) &
              (pts[:, None, 1] > gt[None, :, 1]) & (pts[:, None, 1] < gt[None, :, 3]))
    near = inside.argmax(axis=1)
    pred = gt[near] + rng.normal(0, 5, (len(pts), 4))
    p = np.where(inside.any(axis=1), rng.uniform(0.3, 0.9, len(pts)), rng.uniform(0, 0.2, len(pts)))
    m = inside * p[:, None] ** 0.5 * iou_matrix(pred, gt) ** 6.0
    o2m = {}
    for j in range(len(gt)):
        best = np.argsort(-m[:, j])[:4]
        o2m[j] = best[m[best, j] > 0].tolist()
    rows, cols = linear_sum_assignment(-m)
    o2o = {int(c): [int(r)] for r, c in zip(rows, cols)}
    return gt, pts, pred, o2m, o2o


def fig_assignment():
    gt, pts, pred, o2m, o2o = toy_assignment()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))
    for ax, assign, title in [(axes[0], o2m, "(a) one-to-many: top-4 points per object"),
                              (axes[1], o2o, "(b) one-to-one: exactly one point per object")]:
        ax.scatter(pts[:, 0], pts[:, 1], s=12, color=GRID, zorder=1)
        for j, g in enumerate(gt):
            ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec=C[j], lw=2.6))
            for i in assign[j]:
                b = pred[i]
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=C[j],
                                       lw=1.1, ls="--"))
                ax.scatter(*pts[i], s=70, color=C[j], edgecolor=INK, zorder=3)
        ax.set_xlim(0, 100); ax.set_ylim(105, -2); ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [Line2D([], [], color=INK, lw=2.6, label="ground-truth box"),
               Line2D([], [], color=INK, lw=1.1, ls="--", label="box predicted by a positive point"),
               Line2D([], [], marker="o", ls="", color=GRID, label="grid point (negative)"),
               Line2D([], [], marker="o", ls="", mfc=C[0], mec=INK, ms=8, label="positive point")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "assignment.png")


# ---------------------------------------------------------------- 4. NMS cost
def nms_count(boxes, scores, thr=0.5):
    order = np.argsort(-scores)
    keep, n_iou = [], 0
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        n_iou += rest.size
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0]); y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2]); y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        iou = inter / (area(boxes[i]) + area(boxes[rest]) - inter)
        order = rest[iou <= thr]
    return len(keep), n_iou


def random_boxes(n, n_objects, rng, size=640):
    centres = rng.uniform(40, size - 40, (n_objects, 2))
    owner = rng.integers(0, n_objects, n)
    c = centres[owner] + rng.normal(0, 6, (n, 2))
    wh = rng.uniform(30, 60, (n, 2))
    return np.hstack([c - wh / 2, c + wh / 2]), rng.random(n)


def fig_nms_cost():
    ns = np.array([50, 100, 200, 400, 800, 1600, 3200, 6400, 8400])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3))
    for k, n_objects in enumerate([5, 50, 500]):
        rng = np.random.default_rng(0)
        times, ious = [], []
        for n in ns:
            b, s = random_boxes(n, n_objects, rng)
            best = np.inf
            for _ in range(3):                      # best of 3 runs to reduce timing noise
                t0 = time.perf_counter(); _, n_iou = nms_count(b, s); best = min(best, time.perf_counter() - t0)
            times.append(best * 1e3); ious.append(n_iou)
        axes[0].loglog(ns, times, "o-", color=C[k], label=f"{n_objects} objects in the image")
        axes[1].loglog(ns, ious, "o-", color=C[k], label=f"{n_objects} objects in the image")
    ref = ns.astype(float) ** 2 / 2
    axes[1].loglog(ns, ref, "--", color=MUTED, label="worst case n(n-1)/2")
    axes[0].set_ylabel("greedy NMS time on one CPU core (ms)")
    axes[1].set_ylabel("IoU computations")
    for ax, t in zip(axes, ["(a) wall-clock time (machine dependent)", "(b) work done (deterministic)"]):
        ax.set_xlabel("boxes entering NMS")
        ax.set_title(t, fontsize=11)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    save(fig, "nms-cost.png")


# ---------------------------------------------------------------- 5. DFL distribution
def fig_dfl():
    K = 16
    bins = np.arange(K)
    sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()  # noqa: E731
    dists = [("untrained head (uniform)", sm(np.zeros(K))),
             ("ideal DFL target: 0.7 at bin 5, 0.3 at bin 6", sm(np.where(bins == 5, np.log(0.7),
                                                                   np.where(bins == 6, np.log(0.3), -10.0)))),
             ("a broad, uncertain prediction", sm(-0.5 * ((bins - 6.0) / 2.2) ** 2))]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), sharey=True)
    for ax, (title, S), col in zip(axes, dists, C):
        ax.bar(bins, S, color=col, width=0.8)
        e = S @ bins
        ax.axvline(e, color=INK, lw=1.6, ls="--")
        ax.axvline(5.3, color=C[3], lw=1.8)
        ax.set_title(title, fontsize=10)
        ax.text(0.98, 0.95, f"expected value {e:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=10)
        ax.set_xlabel("bin (distance in stride units)")
        ax.set_xticks([0, 5, 10, 15])
    axes[0].set_ylabel("probability")
    fig.legend(handles=[Line2D([], [], color=INK, ls="--", lw=1.6, label="decoded distance (expected value)"),
                        Line2D([], [], color=C[3], lw=1.6, label="true distance 5.3")],
               loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "dfl.png")


# ---------------------------------------------------------------- 6. Mosaic augmentation
def mosaic(images, boxes, size=416, rng=None, min_keep=0.25):
    """Same procedure as the Mosaic snippet in the chapter: four images meet at a random split
    point; parts outside each quadrant are cut; boxes keeping < min_keep of their area are dropped."""
    xc, yc = (int(v) for v in rng.uniform(0.3, 0.7, 2) * size)
    canvas = np.full((size, size, 3), 114, np.uint8)
    kept, dropped = [], []
    for k, (img, bx) in enumerate(zip(images, boxes)):
        h, w = img.shape[:2]
        ox = xc - w if k in (0, 2) else xc
        oy = yc - h if k in (0, 1) else yc
        rx0, rx1 = (0, xc) if k in (0, 2) else (xc, size)
        ry0, ry1 = (0, yc) if k in (0, 1) else (yc, size)
        x0, x1, y0, y1 = max(rx0, ox), min(rx1, ox + w), max(ry0, oy), min(ry1, oy + h)
        canvas[y0:y1, x0:x1] = img[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        shifted = bx + [ox, oy, ox, oy]
        clipped = np.clip(shifted, [x0, y0, x0, y0], [x1, y1, x1, y1])
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # noqa: E731
        ok = area(clipped) >= min_keep * area(shifted)
        cut = area(clipped) < area(shifted)
        kept += [(b, k, c) for b, c in zip(clipped[ok], cut[ok])]
        dropped += [(b, k) for b in clipped[~ok]]
    return canvas, kept, dropped, (xc, yc)


def fig_mosaic():
    rng = np.random.default_rng(0)
    backgrounds = [(222, 235, 247), (253, 230, 214), (220, 242, 230), (247, 236, 205)]
    imgs, bxs = [], []
    for k in range(4):                                       # four synthetic 256 x 256 "photos"
        img = np.empty((256, 256, 3), np.uint8)
        img[:] = backgrounds[k]
        b = []
        for _ in range(3):
            x, y = rng.integers(0, 200, 2)
            w, h = rng.integers(30, 90, 2)
            x2, y2 = min(x + w, 256), min(y + h, 256)
            rng.integers(0, 256, 3)        # keeps the random sequence identical to the chapter's snippet
            img[y:y2, x:x2] = np.array(matplotlib.colors.to_rgb(C[k])) * 255 * 0.85
            b.append([x, y, x2, y2])
        imgs.append(img)
        bxs.append(np.array(b, float))
    canvas, kept, dropped, (xc, yc) = mosaic(imgs, bxs, rng=rng)

    fig = plt.figure(figsize=(10.4, 4.9))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.15, 2.2], wspace=0.08, hspace=0.12)
    for k in range(4):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.imshow(imgs[k])
        for x1, y1, x2, y2 in bxs[k]:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=INK, lw=1.2))
        ax.set_xlim(-0.5, 255.5)
        ax.set_ylim(255.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"image {k + 1}", fontsize=10)
    ax = fig.add_subplot(gs[:, 3])
    ax.imshow(canvas)
    for (x1, y1, x2, y2), k, cut in kept:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=INK, lw=1.8,
                               ls="--" if cut else "-"))
    for (x1, y1, x2, y2), k in dropped:
        ax.add_patch(Rectangle((x1, y1), max(x2 - x1, 1), max(y2 - y1, 1), fill=False, ec=C[4], lw=2.6))
    ax.set_xlim(-0.5, canvas.shape[1] - 0.5)
    ax.set_ylim(canvas.shape[0] - 0.5, -0.5)
    ax.axvline(xc, color=MUTED, lw=1, ls=":")
    ax.axhline(yc, color=MUTED, lw=1, ls=":")
    ax.plot([xc], [yc], "o", color=INK, ms=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"one 416 x 416 Mosaic: {len(kept)} boxes kept, {len(dropped)} dropped", fontsize=10)
    fig.legend(handles=[Line2D([], [], color=INK, lw=1.8, label="box kept whole"),
                        Line2D([], [], color=INK, lw=1.8, ls="--", label="box cut at a quadrant edge"),
                        Line2D([], [], color=C[4], lw=2.0, label="box dropped (< 25% left)"),
                        Line2D([], [], color=INK, marker="o", lw=0, label="random split point")],
               loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.1)
    save(fig, "mosaic.png")


FIGURES = {"timeline": fig_timeline, "anatomy": fig_anatomy, "assignment": fig_assignment,
           "nms": fig_nms_cost, "dfl": fig_dfl, "mosaic": fig_mosaic}

if __name__ == "__main__":
    # Run all figures, or only the named ones (e.g. `... dl_03a_real_time_detection.py mosaic`).
    # nms-cost.png contains wall-clock timings, so re-running it changes the PNG on every machine.
    for name in (sys.argv[1:] or FIGURES):
        FIGURES[name]()
