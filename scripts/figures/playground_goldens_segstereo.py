"""Golden references for the `segmentation-metrics` (DLCV 03B) and `stereo-matching` (DLCV 08A) labs.

Writes scripts/tests/golden/segmentation-metrics.json and scripts/tests/golden/stereo-matching.json:
  * segmentation: the chapter's own toy experiment (256 x 256 scene, ideal stride-s scores, nearest labels vs bilinear scores, per-class IoU
    exactly as printed in the chapter) and the lab's 128 x 128 scene for every stride and both enlarging methods, all in NumPy/SciPy;
  * stereo: the lab's whole synthetic pipeline (scene, occlusion, noise, SAD block matching with winner-take-all, left-right check, bad-1 and the
    other scores) re-implemented here in NumPy/SciPy, for several noise levels and every window size. The lab's mulberry32 generator and its
    Box-Muller normal field are ported (vectorised, since the state is a counter).

Run with the project venv:  .venv/bin/python scripts/figures/playground_goldens_segstereo.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

OUT = Path(__file__).resolve().parents[1] / "tests" / "golden"


# ----------------------------------------------------------------------------------------------- segmentation
def seg_scores(gt, pred, K):
    C = np.bincount(K * gt.ravel() + pred.ravel(), minlength=K * K).reshape(K, K)
    tp = np.diag(C).astype(float)
    iou = tp / (C.sum(1) + C.sum(0) - tp)
    return {"pixelAcc": tp.sum() / C.sum(), "mIoU": iou.mean(), "iou": iou.tolist(), "wrong": int(C.sum() - tp.sum())}


def enlarge(gt, K, s):
    """(nearest labels, bilinear scores) for the ideal stride-s scores: the class fractions of every s x s cell."""
    H, W = gt.shape
    onehot = np.stack([(gt == k).astype(float) for k in range(K)])
    coarse = onehot.reshape(K, H // s, s, W // s, s).mean(axis=(2, 4))
    nearest = np.kron(coarse.argmax(0), np.ones((s, s), int))
    up = np.stack([ndi.zoom(c, s, order=1, mode="nearest", grid_mode=True) for c in coarse])
    return nearest, up.argmax(0)


def chapter_scene():
    H = W = 256
    yy, xx = np.mgrid[0:H, 0:W]
    gt = np.zeros((H, W), int)
    gt[(yy - 120) ** 2 + (xx - 90) ** 2 < 60 ** 2] = 1
    gt[40:216, 178:184] = 2
    gt[196:216, 60:80] = 3
    return gt


def lab_scene():
    H = W = 128
    yy, xx = np.mgrid[0:H, 0:W]
    gt = np.zeros((H, W), int)
    gt[(xx - 38) ** 2 + (yy - 66) ** 2 < 21 ** 2] = 1
    gt[20:46, 80:104] = 2
    gt[30:118, 118:121] = 3
    for sx, sy in [(18, 100), (50, 108), (84, 84), (97, 110), (66, 76)]:
        gt[sy:sy + 6, sx:sx + 6] = 4
    return gt


def segmentation_golden():
    out = {"chapter": [], "lab": [], "labCounts": None, "labMaxFractionClass4At8": None}
    gt = chapter_scene()
    for s in [4, 8, 16, 32]:
        a, b = enlarge(gt, 4, s)
        out["chapter"].append({"stride": s, "nearest": seg_scores(gt, a, 4), "bilinear": seg_scores(gt, b, 4)})
    gt = lab_scene()
    out["labCounts"] = np.bincount(gt.ravel(), minlength=5).tolist()
    for s in [1, 2, 4, 8, 16, 32]:
        a, b = (gt, gt) if s == 1 else enlarge(gt, 5, s)
        out["lab"].append({"stride": s, "nearest": seg_scores(gt, a, 5), "bilinear": seg_scores(gt, b, 5)})
    onehot = np.stack([(gt == k).astype(float) for k in range(5)])
    out["labMaxFractionClass4At8"] = float(onehot.reshape(5, 16, 8, 16, 8).mean(axis=(2, 4))[4].max())
    return out


# ----------------------------------------------------------------------------------------------- stereo
W, H, D, D_BG, D_FG = 128, 96, 16, 4, 12
FG = (86, 44, 25)
FLAT = (20, 46, 62, 84)  # x0, x1, y0, y1
WINDOWS = list(range(3, 32, 2))


def mulberry32(seed, n):
    """First n outputs of the lab's mulberry32 (src/lib/imagekit/rng.ts), vectorised: the state is seed + k * 0x6d2b79f5."""
    k = np.arange(1, n + 1, dtype=np.uint64)
    a = ((np.uint64(seed) + k * np.uint64(0x6D2B79F5)) % np.uint64(2**32)).astype(np.uint32)
    t = a
    t = (t ^ (t >> np.uint32(15))) * (t | np.uint32(1))
    t = t ^ (t + (t ^ (t >> np.uint32(7))) * (t | np.uint32(61)))
    return ((t ^ (t >> np.uint32(14))).astype(np.float64)) / 4294967296.0


def normal_field(n, seed):
    r = mulberry32(seed, 2 * n)
    z = np.sqrt(-2 * np.log(np.maximum(1e-12, r[0::2]))) * np.cos(2 * np.pi * r[1::2])
    return z.astype(np.float32)


def texture(seed):
    t = ndi.gaussian_filter(normal_field(W * H, seed).reshape(H, W).astype(np.float64), 1.2, truncate=4.0).astype(np.float32)
    return ((t - t.mean()) / t.std()).astype(np.float32)


def stereo_scene():
    tex, fill = texture(31), texture(32)
    yy, xx = np.mgrid[0:H, 0:W]
    disc = (xx - FG[0]) ** 2 + (yy - FG[1]) ** 2 < FG[2] ** 2
    flat = ~disc & (xx >= FLAT[0]) & (xx < FLAT[1]) & (yy >= FLAT[2]) & (yy < FLAT[3])
    gt = np.where(disc, D_FG, D_BG)
    left = np.where(disc, 0.64 + 0.14 * tex, np.where(flat, 0.5, 0.36 + 0.1 * tex)).astype(np.float32)
    right = (0.36 + 0.1 * fill).astype(np.float32)
    occluded = np.zeros((H, W), bool)
    for y in range(H):
        owner = np.full(W, -1)
        for d in (D_BG, D_FG):
            for x in range(W):
                if gt[y, x] == d and x - d >= 0:
                    right[y, x - d] = left[y, x]
                    owner[x - d] = x
        for x in range(W):
            xr = x - gt[y, x]
            occluded[y, x] = xr < 0 or owner[xr] != x
    evaluated = ~occluded & (xx >= D)
    return {"left": left, "right": right, "gt": gt, "occluded": occluded, "flat": flat, "evaluated": evaluated,
            "nl": normal_field(W * H, 33).reshape(H, W), "nr": normal_field(W * H, 34).reshape(H, W)}


def box_means(ad, ok, win):
    """Mean of ad over the in-bounds window pixels that have a partner (clipped windows)."""
    s_ad = ndi.uniform_filter(ad, win, mode="constant", cval=0.0) * win * win
    s_ok = ndi.uniform_filter(ok, win, mode="constant", cval=0.0) * win * win
    return s_ad / np.maximum(s_ok, 1e-9)


def block_match(a, b, win, ref):
    """WTA disparity, reference image `a`; 'left' compares a(x) with b(x - d), 'right' compares a(x) with b(x + d)."""
    best = np.full((H, W), np.inf)
    arg = np.zeros((H, W), np.uint8)
    xs = np.arange(W)
    for d in range(D):
        sh = -d if ref == "left" else d
        xp = xs + sh
        ok_col = (xp >= 0) & (xp < W)
        ad = np.zeros((H, W))
        ad[:, ok_col] = np.abs(a[:, ok_col].astype(np.float64) - b[:, xp[ok_col]].astype(np.float64))
        ok = np.broadcast_to(ok_col.astype(float), (H, W)).copy()
        cost = box_means(ad, ok, win)
        cost[:, ~ok_col] = np.inf
        better = cost < best
        best[better] = cost[better]
        arg[better] = d
    return arg


def lr_check(dl, dr, tol=1):
    out = np.full((H, W), 255, np.uint8)
    for y in range(H):
        for x in range(W):
            d = int(dl[y, x]); xr = x - d
            if xr >= 0 and abs(d - int(dr[y, xr])) <= tol:
                out[y, x] = d
    return out


def stereo_scores(sc, disp):
    kept = disp != 255
    ev = sc["evaluated"]
    k = ev & kept
    wrong = np.abs(disp.astype(int) - sc["gt"]) > 1
    fk = k & sc["flat"]
    spill = (kept & (np.abs(disp.astype(int) - D_FG) <= 1) & ev & (sc["gt"] != D_FG)).sum()
    occ = sc["occluded"]
    return {
        "bad1": 100 * wrong[k].mean() if k.any() else None,
        "badFlat": 100 * wrong[fk].mean() if fk.any() else None,
        "fgSpill": 100 * spill / (sc["gt"] == D_FG).sum(),
        "invalid": 100 * (ev.sum() - k.sum()) / ev.sum(),
        "invalidOccluded": 100 * (~kept & occ).sum() / occ.sum(),
    }


def stereo_golden():
    sc = stereo_scene()
    out = {"counts": {"evaluated": int(sc["evaluated"].sum()), "occluded": int(sc["occluded"].sum()), "flat": int(sc["flat"].sum())}, "runs": []}
    for sigma in [0.03, 0.06]:
        left = (sc["left"] + sigma * sc["nl"]).astype(np.float32)
        right = (sc["right"] + sigma * sc["nr"]).astype(np.float32)
        run = {"sigma": sigma, "off": [], "on": [], "maps": {}}
        for win in WINDOWS:
            dl, dr = block_match(left, right, win, "left"), block_match(right, left, win, "right")
            run["off"].append(stereo_scores(sc, dl))
            run["on"].append(stereo_scores(sc, lr_check(dl, dr)))
            if sigma == 0.03 and win in (5, 21):
                run["maps"][str(win)] = {"left": "".join(chr(48 + int(v)) for v in dl.ravel())}
        out["runs"].append(run)
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "segmentation-metrics.json").write_text(json.dumps(segmentation_golden()))
    (OUT / "stereo-matching.json").write_text(json.dumps(stereo_golden()))
    print("wrote", OUT / "segmentation-metrics.json", OUT / "stereo-matching.json")
