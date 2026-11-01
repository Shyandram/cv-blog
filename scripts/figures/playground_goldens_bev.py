"""Golden reference for the `bev-iou` lab (DLCV 09).

Writes scripts/tests/golden/bev-iou.json with values computed by the chapter's own NumPy code (Sutherland-Hodgman clipping and
the shoelace formula, copied from the "BEV IoU and 3D IoU of rotated boxes" snippet of dl-09-3d-object-detection.mdx):
  * the chapter's seven worked cases,
  * the rotated IoU and the IoU of the enclosing axis-aligned rectangles of 80 random pairs of boxes,
  * the IoU against yaw error of the three illustrative shapes (pedestrian, car, bus) at three centre offsets,
  * the yaw error at which each shape first falls below IoU 0.7 (scan in 0.05 degree steps).

Run with the project venv:  .venv/bin/python scripts/figures/playground_goldens_bev.py
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "tests" / "golden" / "bev-iou.json"


# ------------------------------------------------------------------ the chapter's snippet
def corners(box):
    """BEV corners (counter-clockwise) of box = (x, y, z, l, w, h, yaw); z is the box centre."""
    x, y, _, l, w, _, yaw = box
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[l, w], [-l, w], [-l, -w], [l, -w]]) / 2
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


def cross_point(p, q, a, b):
    d1, d2 = q - p, b - a
    t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / (d1[0] * d2[1] - d1[1] * d2[0])
    return p + t * d1


def clip(subject, clipper):
    out = list(subject)
    for i in range(len(clipper)):
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        inside = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0
        pts, out = out, []
        for j in range(len(pts)):
            p, q = pts[j], pts[(j + 1) % len(pts)]
            if inside(q):
                if not inside(p):
                    out.append(cross_point(p, q, a, b))
                out.append(q)
            elif inside(p):
                out.append(cross_point(p, q, a, b))
        if not out:
            break
    return np.array(out)


def area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))


def bev_iou(a, b):
    inter = area(clip(corners(a), corners(b)))
    return inter / (a[3] * a[4] + b[3] * b[4] - inter)


def aabb_iou(a, b):
    ca, cb = corners(a), corners(b)
    ra = (ca[:, 0].min(), ca[:, 0].max(), ca[:, 1].min(), ca[:, 1].max())
    rb = (cb[:, 0].min(), cb[:, 0].max(), cb[:, 1].min(), cb[:, 1].max())
    iw = max(0.0, min(ra[1], rb[1]) - max(ra[0], rb[0]))
    ih = max(0.0, min(ra[3], rb[3]) - max(ra[2], rb[2]))
    inter = iw * ih
    union = (ra[1] - ra[0]) * (ra[3] - ra[2]) + (rb[1] - rb[0]) * (rb[3] - rb[2]) - inter
    return inter / union


def box(x, y, l, w, yaw):
    return (x, y, 0.0, l, w, 1.0, yaw)


def pack(b):
    return {"x": b[0], "y": b[1], "l": b[3], "w": b[4], "yaw": b[6]}


# ------------------------------------------------------------------ goldens
car = box(0, 0, 4, 2, 0)
cases = {
    "identical": (car, car),
    "rotated by 180 deg": (car, box(0, 0, 4, 2, np.pi)),
    "rotated by 90 deg": (car, box(0, 0, 4, 2, np.pi / 2)),
    "unit square vs 45 deg": (box(0, 0, 1, 1, 0), box(0, 0, 1, 1, np.pi / 4)),
    "shifted 1 m along length": (car, box(1, 0, 4, 2, 0)),
    "far apart": (car, box(10, 0, 4, 2, 0.3)),
}
chapter = [{"name": k, "a": pack(a), "b": pack(b), "iou": bev_iou(a, b)} for k, (a, b) in cases.items()]

rng = np.random.default_rng(5)
random_pairs = []
for _ in range(80):
    l1, w1, l2, w2 = rng.uniform(0.5, 8, 4)
    a = box(rng.uniform(-2, 2), rng.uniform(-2, 2), l1, w1, rng.uniform(-np.pi, np.pi))
    b = box(rng.uniform(-3, 3), rng.uniform(-3, 3), l2, w2, rng.uniform(-np.pi, np.pi))
    random_pairs.append({"a": pack(a), "b": pack(b), "iou": bev_iou(a, b), "aabb": aabb_iou(a, b)})

SHAPES = {"pedestrian": (0.7, 0.7), "car": (4.5, 1.8), "bus": (12.0, 2.5)}
curves = {}
below = {}
for name, (l, w) in SHAPES.items():
    for off in (0.0, 0.5, 1.0):
        gt = box(0, 0, l, w, 0)
        vals = [bev_iou(gt, box(off, 0, l, w, np.deg2rad(d))) for d in range(0, 91, 5)]
        curves[f"{name}@{off}"] = vals
        first = None
        for d in np.arange(0, 90.0001, 0.05):
            if bev_iou(gt, box(off, 0, l, w, np.deg2rad(d))) < 0.7:
                first = round(float(d), 2)
                break
        below[f"{name}@{off}"] = first

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"chapter": chapter, "random": random_pairs, "curves": curves, "yawStepDeg": 5, "below07": below}))
print("wrote", OUT, "|", {k: v for k, v in below.items() if k.endswith("@0.0")})
