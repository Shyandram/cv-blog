"""Figures for DLCV 09 - 3D Object Detection: LiDAR, Cameras and Bird's-Eye View.

Run:  .venv/bin/python scripts/figures/dl_09_3d_object_detection.py
Writes only to src/assets/deep-learning/dl-09-3d-object-detection/.
All data are synthetic (ray-cast toy LiDAR scans, random boxes); nothing is downloaded.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge  # noqa: E402
from sklearn.cluster import DBSCAN  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-09-3d-object-detection"
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


# ---------------------------------------------------------------- geometry helpers
def bev_corners(x, y, l, w, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[l, w], [-l, w], [-l, -w], [l, -w]]) / 2
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


def clip(subject, clipper):
    """Sutherland-Hodgman clipping of a polygon by a convex counter-clockwise polygon."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def cross(p, q, a, b):
        d1, d2 = q - p, b - a
        t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / (d1[0] * d2[1] - d1[1] * d2[0])
        return p + t * d1

    out = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        pts, out = out, []
        for j in range(len(pts)):
            p, q = pts[j], pts[(j + 1) % len(pts)]
            if inside(q, a, b):
                if not inside(p, a, b):
                    out.append(cross(p, q, a, b))
                out.append(q)
            elif inside(p, a, b):
                out.append(cross(p, q, a, b))
        if not out:
            break
    return np.array(out)


def poly_area(p):
    if len(p) < 3:
        return 0.0
    return 0.5 * abs(p[:, 0] @ np.roll(p[:, 1], -1) - p[:, 1] @ np.roll(p[:, 0], -1))


def bev_iou(a, b):
    """a, b = (x, y, l, w, yaw)."""
    inter = poly_area(clip(bev_corners(*a), bev_corners(*b)))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


# toy spinning LiDAR: 32 beams, 0.2 degree azimuth step, mounted 1.8 m above the ground
ELEV = np.deg2rad(np.linspace(-16, 10, 32))
AZIM = np.deg2rad(np.arange(0, 360, 0.2))
_E, _A = np.meshgrid(ELEV, AZIM, indexing="ij")
DIRS = np.stack([np.cos(_E) * np.cos(_A), np.cos(_E) * np.sin(_A), np.sin(_E)], -1).reshape(-1, 3)
ORIGIN = np.array([0.0, 0.0, 1.8])


def raycast(boxes, max_range=80.0):
    """First hit of every ray with the ground (z = 0) or with yawed boxes (x, y, l, w, h, yaw)."""
    t_best = np.full(len(DIRS), np.inf)
    hit_id = np.full(len(DIRS), -1)
    down = DIRS[:, 2] < 0
    t_ground = np.where(down, -ORIGIN[2] / np.where(down, DIRS[:, 2], -1), np.inf)
    t_best = np.minimum(t_best, t_ground)
    for k, (x, y, l, w, h, yaw) in enumerate(boxes):
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        o = (ORIGIN - [x, y, h / 2]) @ R            # ray origin in the box frame
        d = DIRS @ R
        half = np.array([l, w, h]) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - o) / d, (half - o) / d
        tn = np.nanmax(np.minimum(t1, t2), axis=1)
        tf = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tf >= tn) & (tf > 0) & (tn < t_best)
        t_best[hit] = tn[hit]
        hit_id[hit] = k
    keep = t_best < max_range
    pts = ORIGIN + DIRS[keep] * t_best[keep, None]
    return pts, hit_id[keep]


# ---------------------------------------------------------------- Figure 1: sensors and frames
def fig_sensors():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.6, 4.6), gridspec_kw=dict(width_ratios=[1.05, 1]))
    # (a) bird's-eye view of the ego vehicle
    rng = np.random.default_rng(3)
    for r in np.linspace(3, 14, 9):                          # LiDAR rings on the ground
        t = np.linspace(0, 2 * np.pi, 300)
        ax.plot(r * np.cos(t), r * np.sin(t), color=GRID, lw=0.8, zorder=0)
    cams = [(0, 70, "front"), (55, 70, ""), (-55, 70, ""), (110, 70, ""), (-110, 70, ""), (180, 110, "back")]
    for ang, fov, _ in cams:
        ax.add_patch(Wedge((0, 0), 12, ang - fov / 2, ang + fov / 2, fc=C[0], alpha=0.07, ec=C[0], lw=0.6))
    for ang in (0, 60, -60, 135, -135):
        ax.add_patch(Wedge((0, 0), 15.5, ang - 10, ang + 10, fc="none", ec=C[3], lw=1.2, ls="--"))
    ax.add_patch(Rectangle((-2.3, -0.95), 4.6, 1.9, fc="white", ec=INK, lw=1.6, zorder=3))
    ax.add_patch(Circle((0, 0), 0.35, fc=C[1], ec=INK, zorder=4))
    ax.annotate("", xy=(4.5, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8), zorder=5)
    ax.annotate("", xy=(0, 4.5), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.8), zorder=5)
    ax.text(4.7, 0.4, "x (forward)", fontsize=10, bbox=dict(fc="white", ec="none", pad=1))
    ax.text(0.3, 4.6, "y (left)", fontsize=10, bbox=dict(fc="white", ec="none", pad=1))
    ax.text(-2.3, -3.3, "z: up, out of the page", fontsize=10, color=MUTED, bbox=dict(fc="white", ec="none", pad=1))
    # a target object with its 3D box seen from above
    tgt = bev_corners(9, 6.5, 4.5, 1.9, 0.5)
    ax.add_patch(Polygon(tgt, closed=True, fc=C[2], alpha=0.25, ec=C[2], lw=2, zorder=4))
    ax.plot([9, 9 + 2.6 * np.cos(0.5)], [6.5, 6.5 + 2.6 * np.sin(0.5)], color=C[2], lw=2, zorder=4)
    ax.text(9, 10.2, "3D box (x, y, z, l, w, h, yaw)", ha="center", fontsize=10, color=C[2],
            bbox=dict(fc="white", ec="none", pad=1), zorder=5)
    th = np.linspace(0, 0.5, 30)
    ax.plot(9 + 1.8 * np.cos(th), 6.5 + 1.8 * np.sin(th), color=C[2], lw=1, zorder=4)
    ax.plot([9, 11.4], [6.5, 6.5], color=C[2], lw=1, ls=":", zorder=4)
    ax.text(11.5, 6.0, "yaw", fontsize=10, color=C[2], bbox=dict(fc="white", ec="none", pad=1), zorder=5)
    for p in rng.uniform([-14, -14], [14, 14], (60, 2)):
        if 3 < np.hypot(*p) < 14:
            ax.plot(*p, ".", color=MUTED, ms=2)
    ax.plot([], [], color=C[1], marker="o", ls="", label="LiDAR (360 degrees)")
    ax.add_patch(Rectangle((0, 0), 0, 0, fc=C[0], alpha=0.25, label="camera fields of view"))
    ax.plot([], [], color=C[3], ls="--", label="radar fields of view")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax.set_xlim(-16, 16); ax.set_ylim(-16, 16); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) Ego frame seen from above", fontsize=11)

    # (b) pinhole camera: a near small object and a far large object look the same
    f = 1.0
    bx.plot([0, 0], [-0.6, 0.6], color=INK, lw=2)                    # image plane at x = -f (drawn at 0)
    bx.plot(-f, 0, "o", color=INK)
    bx.text(-f - 0.1, -0.25, "camera\ncentre", ha="right", va="top", fontsize=10)
    bx.text(0.0, -0.75, "image\nplane", ha="center", va="top", fontsize=10)
    for Z, Hh, col, lab in [(4.0, 1.0, C[0], "near, small"), (8.0, 2.0, C[1], "far, large")]:
        bx.add_patch(Rectangle((Z - f + 0.0, 0), 0.5, Hh, fc=col, alpha=0.3, ec=col, lw=1.5))
        bx.plot([-f, Z - f], [0, Hh], color=col, lw=1.2, ls="--")
        bx.plot([-f, Z - f], [0, 0], color=col, lw=0.8)
        bx.text(Z - f + 0.25, Hh + 0.12, lab, ha="center", fontsize=10, color=col)
    bx.plot([0, 0], [0, 0.25], color=C[6], lw=5, solid_capstyle="butt")
    bx.annotate("same image height h", xy=(0.05, 0.2), xytext=(0.4, 1.6), fontsize=10, color=C[6],
                arrowprops=dict(arrowstyle="->", color=C[6]))
    bx.annotate("", xy=(7.6, -0.45), xytext=(0.3, -0.45), arrowprops=dict(arrowstyle="->", color=MUTED))
    bx.text(4.0, -0.62, "depth Z (lost by projection)", ha="center", va="top", fontsize=10, color=MUTED)
    bx.set_xlim(-1.6, 8.0); bx.set_ylim(-1.4, 2.6); bx.set_aspect("equal"); bx.axis("off")
    bx.set_title("(b) Side view of one camera: h = f H / Z", fontsize=11)
    save(fig, "sensors-frames.png")


# ---------------------------------------------------------------- Figure 2: density versus range
def fig_density():
    boxes = [(8, 3, 4.5, 1.8, 1.5, 0.2), (18, -7, 4.5, 1.8, 1.5, -0.4), (32, 10, 4.5, 1.8, 1.5, 1.2),
             (48, -12, 4.5, 1.8, 1.5, 0.1), (65, 5, 4.5, 1.8, 1.5, 0.6)]
    pts, hid = raycast(boxes)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    g = hid < 0
    ax.scatter(pts[g, 0], pts[g, 1], s=0.15, color=MUTED, rasterized=True)
    ax.scatter(pts[~g, 0], pts[~g, 1], s=1.5, color=C[1], rasterized=True)
    for (x, y, l, w, h, yaw), n in zip(boxes, [np.sum(hid == k) for k in range(len(boxes))]):
        ax.add_patch(Polygon(bev_corners(x, y, l, w, yaw), closed=True, fill=False, ec=C[0], lw=1.2))
        ax.text(x, y + (3.2 if y >= 0 else -4.6), f"{n} pts", ha="center", fontsize=10, color=C[0],
                bbox=dict(fc="white", ec="none", pad=1))
    ax.plot(0, 0, "^", color=INK, ms=8)
    ax.set_xlim(-5, 72); ax.set_ylim(-22, 22); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("(a) One toy scan from above: same car, fewer points with range", fontsize=11)

    ranges = np.array([5, 7, 10, 14, 20, 28, 40, 56, 80])
    counts = []
    for r in ranges:
        _, h = raycast([(r, 0, 4.5, 1.8, 1.5, 0.0)], max_range=120)
        counts.append(np.sum(h == 0))
    counts = np.array(counts)
    bx.loglog(ranges, counts, "o-", color=C[1], label="toy LiDAR (32 beams)")
    ref = counts[2] * (ranges / ranges[2]) ** -2.0
    bx.loglog(ranges, ref, "--", color=MUTED, label=r"reference $\propto 1/r^2$")
    bx.set_xticks([5, 10, 20, 40, 80]); bx.set_xticklabels(["5", "10", "20", "40", "80"])
    bx.minorticks_off()
    bx.set_xlabel("range r (m)"); bx.set_ylabel("points on a 4.5 m car")
    bx.grid(True, which="both", color=GRID, lw=0.6)
    bx.legend(fontsize=10)
    bx.set_title("(b) Points per object versus range", fontsize=11)
    save(fig, "density-range.png")


# ---------------------------------------------------------------- Figure 3: classical pipeline
def box_surface(rng, cx, cy, l, w, h, yaw, n):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = []
    for nx, ny, half, along in [(1, 0, l / 2, w), (-1, 0, l / 2, w), (0, 1, w / 2, l), (0, -1, w / 2, l)]:
        normal = np.array([c * nx - s * ny, s * nx + c * ny])
        mid = np.array([cx, cy]) + half * normal
        if normal @ mid >= 0:
            continue
        t = rng.uniform(-along / 2, along / 2, n)
        xy = mid + t[:, None] * np.array([-normal[1], normal[0]])
        pts.append(np.column_stack([xy, rng.uniform(0, h, n)]))
    return np.vstack(pts)


def fig_classical():
    rng = np.random.default_rng(0)
    r, a = rng.uniform(2, 60, 20000), rng.uniform(-np.pi / 2, np.pi / 2, 20000)
    ground = np.column_stack([r * np.cos(a), r * np.sin(a), rng.normal(0, 0.03, r.size)])
    objects = [(12, -4, 4.5, 1.8, 1.5, 0.3), (25, 6, 4.2, 1.7, 1.4, -1.0), (40, -10, 0.8, 0.6, 1.7, 0.0)]
    cloud = [ground] + [box_surface(rng, *o, n=int(40000 / np.hypot(o[0], o[1]) ** 2) + 5) for o in objects]
    pts = np.vstack(cloud) + rng.normal(0, 0.02, (sum(len(c) for c in cloud), 3))
    best = None
    for _ in range(100):
        p = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((pts - p[0]) @ n) < 0.1
        if best is None or inl.sum() > best.sum():
            best = inl
    above = pts[~best]
    labels = DBSCAN(eps=0.7, min_samples=5).fit_predict(above[:, :2])
    fits = []
    for k in range(labels.max() + 1):
        q = above[labels == k]
        best_score, best_fit = np.inf, None
        for yaw in np.deg2rad(np.arange(0, 180, 1)):
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            uv = q[:, :2] @ R
            lo, hi = uv.min(0), uv.max(0)
            d = np.minimum(uv - lo, hi - uv)
            score = np.minimum(d[:, 0], d[:, 1]).mean()
            if score < best_score and hi[0] - lo[0] >= hi[1] - lo[1]:
                best_score, best_fit = score, (R @ ((lo + hi) / 2), hi - lo, yaw)
        fits.append(best_fit)

    fig, axs = plt.subplots(1, 3, figsize=(10.4, 4.3), gridspec_kw=dict(width_ratios=[1.35, 1, 1]),
                            layout="constrained")
    ax = axs[0]
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=np.where(pts[:, 2] > 0.15, 3, 0.3), cmap="viridis",
                    vmin=-0.1, vmax=1.7, rasterized=True)
    ax.plot(0, 0, "^", color=INK, ms=7)
    ax.set_xlim(-2, 45); ax.set_ylim(-18, 18); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("(a) Raw points (colour = height z, m)", fontsize=11)
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    for j, (o, ax) in enumerate(zip(objects[:2], axs[1:])):
        x0, y0 = o[0], o[1]
        ax.scatter(pts[best, 0], pts[best, 1], s=1.0, color=GRID, rasterized=True, label="ground (RANSAC)")
        for k in range(labels.max() + 1):
            q = above[labels == k]
            ax.scatter(q[:, 0], q[:, 1], s=4, color=C[k % len(C)], rasterized=True)
        q = above[labels == -1]
        ax.scatter(q[:, 0], q[:, 1], s=8, marker="x", color=MUTED, lw=0.8, label="DBSCAN noise")
        for (ctr, size, yaw), o2 in zip(fits, objects):
            ax.add_patch(Polygon(bev_corners(ctr[0], ctr[1], size[0], size[1], yaw), closed=True,
                                 fill=False, ec=INK, lw=1.6))
            ax.add_patch(Polygon(bev_corners(o2[0], o2[1], o2[2], o2[3], o2[5]), closed=True,
                                 fill=False, ec=C[4], lw=1.6, ls="--"))
        ax.plot([], [], color=INK, lw=1.6, label="fitted box")
        ax.plot([], [], color=C[4], lw=1.6, ls="--", label="true box")
        ax.set_xlim(x0 - 4, x0 + 4); ax.set_ylim(y0 - 4, y0 + 4); ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(f"({'bc'[j]}) Close-up of object {j}", fontsize=11)
        if j == 0:
            fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=10,
                       markerscale=2)
    save(fig, "classical-pipeline.png")


# ---------------------------------------------------------------- Figure 4: pillars
def fig_pillars():
    boxes = [(9, 2.5, 4.5, 1.8, 1.5, 0.15), (16, -4, 4.5, 1.8, 1.5, -0.5), (6, -3, 0.7, 0.7, 1.7, 0.0)]
    pts, hid = raycast(boxes, max_range=60)
    vx = 0.16
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw=dict(width_ratios=[1, 1.15]))
    # (a) close-up with the grid and occupied pillars
    xl, yl = (5.0, 12.0), (-1.0, 5.0)
    sel = (pts[:, 0] > xl[0]) & (pts[:, 0] < xl[1]) & (pts[:, 1] > yl[0]) & (pts[:, 1] < yl[1])
    q = pts[sel]
    ij = np.unique(np.floor(q[:, :2] / vx).astype(int), axis=0)
    for i, j in ij:
        ax.add_patch(Rectangle((i * vx, j * vx), vx, vx, fc=C[3], alpha=0.35, ec="none"))
    for gx in np.arange(xl[0], xl[1] + 1e-6, vx):
        ax.axvline(gx, color=GRID, lw=0.3, zorder=0)
    for gy in np.arange(yl[0], yl[1] + 1e-6, vx):
        ax.axhline(gy, color=GRID, lw=0.3, zorder=0)
    ax.scatter(q[:, 0], q[:, 1], s=2, color=INK, zorder=3)
    for x, y, l, w, h, yaw in boxes:
        ax.add_patch(Polygon(bev_corners(x, y, l, w, yaw), closed=True, fill=False, ec=C[0], lw=1.5, ls="--",
                             zorder=4))
    ax.text(9.0, 3.65, "car (dashed outline)", ha="center", fontsize=10, color=C[0],
            bbox=dict(fc="white", ec="none", pad=1), zorder=5)
    nx_cells = int(round((xl[1] - xl[0]) / vx)) * int(round((yl[1] - yl[0]) / vx))
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"(a) 0.16 m pillars: {len(ij)} of {nx_cells} cells occupied", fontsize=11)
    # (b) point count per pillar over the whole grid
    W, H = 440, 500
    ix = np.floor(pts[:, 0] / vx).astype(int)
    iy = np.floor((pts[:, 1] + 40) / vx).astype(int)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    cnt = np.zeros((H, W))
    np.add.at(cnt, (iy[ok], ix[ok]), 1)
    occ = (cnt > 0).mean()
    im = bx.imshow(np.log1p(cnt), origin="lower", extent=[0, W * vx, -40, H * vx - 40], cmap="magma_r",
                   interpolation="nearest")
    bx.set_xlim(0, 60); bx.set_ylim(-30, 30)
    bx.set_xlabel("x (m)"); bx.set_ylabel("y (m)")
    bx.set_title(f"(b) Points per pillar, full grid: {100 * (1 - occ):.1f}% empty", fontsize=11)
    cb = fig.colorbar(im, ax=bx, fraction=0.04, pad=0.02)
    cb.set_label("log(1 + points)", fontsize=10)
    save(fig, "pillars.png")


# ---------------------------------------------------------------- Figure 5: rotated IoU
def fig_rotated_iou():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    a = (0.0, 0.0, 4.5, 1.8, 0.0)
    b = (0.6, 0.3, 4.5, 1.8, np.deg2rad(25))
    pa, pb = bev_corners(*a), bev_corners(*b)
    inter = clip(pa, pb)
    ax.add_patch(Polygon(pa, closed=True, fill=False, ec=C[2], lw=2, label="ground truth"))
    ax.add_patch(Polygon(pb, closed=True, fill=False, ec=C[1], lw=2, ls="--", label="prediction"))
    ax.add_patch(Polygon(inter, closed=True, fc=C[0], alpha=0.35, ec=C[0], lw=1, label="intersection"))
    ax.plot(inter[:, 0], inter[:, 1], "o", color=C[0], ms=4)
    ax.set_xlim(-3.4, 3.8); ax.set_ylim(-2.6, 2.8); ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"(a) Rotated boxes: BEV IoU = {bev_iou(a, b):.3f}", fontsize=11)
    errs = np.linspace(0, 90, 181)
    for (l, w), col, lab in [((4.5, 1.8), C[1], "car 4.5 x 1.8 m"), ((0.7, 0.7), C[0], "pedestrian 0.7 x 0.7 m"),
                             ((12.0, 2.5), C[6], "bus 12 x 2.5 m")]:
        bx.plot(errs, [bev_iou((0, 0, l, w, 0), (0, 0, l, w, np.deg2rad(e))) for e in errs], color=col, lw=2,
                label=lab)
    bx.axhline(0.7, color=MUTED, ls=":", lw=1)
    bx.axhline(0.5, color=MUTED, ls=":", lw=1)
    bx.text(91, 0.7, "0.7", va="center", fontsize=10, color=MUTED)
    bx.text(91, 0.5, "0.5", va="center", fontsize=10, color=MUTED)
    bx.set_xlabel("yaw error (degrees), centre and size exact"); bx.set_ylabel("BEV IoU")
    bx.set_xlim(0, 90); bx.set_ylim(0, 1.02)
    bx.grid(True, color=GRID, lw=0.6)
    bx.legend(fontsize=10)
    bx.set_title("(b) IoU lost to a yaw error alone", fontsize=11)
    save(fig, "rotated-iou.png")


# ---------------------------------------------------------------- Figure 6: IoU versus centre distance
def fig_matching():
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    d = np.linspace(0, 4.5, 400)
    for (l, w), col, lab in [((4.5, 1.8), C[1], "car"), ((0.7, 0.7), C[0], "pedestrian")]:
        along = [bev_iou((0, 0, l, w, 0), (x, 0, l, w, 0)) for x in d]
        across = [bev_iou((0, 0, l, w, 0), (0, x, l, w, 0)) for x in d]
        ax.plot(d, along, color=col, lw=2, label=f"{lab}, shift along length")
        if l != w:
            ax.plot(d, across, color=col, lw=2, ls="--", label=f"{lab}, shift along width")
    for t in (0.5, 1, 2, 4):
        ax.axvline(t, color=C[6], lw=1, ls=":")
        ax.text(t, 1.04, f"{t:g} m", ha="center", fontsize=10, color=C[6])
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(4.55, 0.5, "IoU 0.5", va="center", fontsize=10, color=MUTED)
    ax.set_xlim(0, 4.5); ax.set_ylim(0, 1.1)
    ax.set_xlabel("centre error (m)"); ax.set_ylabel("BEV IoU with the true box")
    ax.grid(True, color=GRID, lw=0.6)
    ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    ax.set_title("Same centre error, very different IoU (purple: nuScenes distance thresholds)", fontsize=11)
    save(fig, "iou-vs-distance.png")


# ---------------------------------------------------------------- Figure 7: timeline
def fig_timeline():
    ev = [
        (2012, "KITTI", 4), (2017, "PointNet / PointNet++", 0), (2017, "MV3D", 3),
        (2018, "VoxelNet / SECOND", 1), (2018, "Frustum PointNets", 3),
        (2019, "PointRCNN", 0), (2019, "PointPillars", 1), (2019, "Pseudo-LiDAR", 2),
        (2020, "PV-RCNN", 1), (2020, "LSS", 2), (2020, "PointPainting", 3), (2020, "nuScenes / Waymo", 4),
        (2021, "CenterPoint", 1), (2021, "DETR3D / FCOS3D / BEVDet", 2), (2021, "CenterFusion", 3),
        (2022, "SST", 1), (2022, "BEVFormer / PETR", 2), (2022, "TransFusion / BEVFusion", 3),
        (2023, "DSVT", 1), (2023, "BEVDepth / StreamPETR", 2), (2023, "BEVFusion (ICRA) / UniTR", 3),
        (2024, "LION / Voxel Mamba", 1), (2024, "Far3D", 2),
    ]
    fams = ["Point-based", "Voxel / pillar /\nhybrid / sequence", "Camera-only", "Fusion", "Datasets"]
    fcol = [C[0], C[1], C[2], C[6], MUTED]
    years = sorted({y for y, _, _ in ev})
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    colx = np.arange(len(fams)) * 2.1
    for j, (f, c) in enumerate(zip(fams, fcol)):
        ax.text(colx[j], -1.0, f, ha="center", va="center", fontsize=10.5, color=c, fontweight="bold")
    for i, y in enumerate(years):
        ax.axhline(i, color=GRID, lw=0.6, zorder=0)
        ax.text(-1.25, i, str(y), ha="right", va="center", fontsize=10, color=INK)
    for y, name, f in ev:
        i = years.index(y)
        ax.text(colx[f], i, name.replace(" / ", "\n") if len(name) > 18 else name, ha="center", va="center",
                fontsize=10, color=INK, bbox=dict(fc="white", ec=fcol[f], lw=1.2, boxstyle="round,pad=0.25"))
    ax.set_xlim(-1.9, colx[-1] + 1.1); ax.set_ylim(len(years) - 0.4, -1.6)
    ax.axis("off")
    ax.set_title("3D detection milestones, by year of the publication cited in this chapter\n"
                 "(years without a listed milestone are omitted)", fontsize=11)
    save(fig, "timeline.png")


if __name__ == "__main__":
    fig_sensors()
    fig_density()
    fig_classical()
    fig_pillars()
    fig_rotated_iou()
    fig_matching()
    fig_timeline()
