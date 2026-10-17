"""Golden reference for the `tracking-association` lab (DLCV 06).

Writes scripts/tests/golden/tracking.json:
  * Hungarian assignments from scipy.optimize.linear_sum_assignment on random cost matrices (square and rectangular),
  * IoU of random boxes,
  * the chapter's KalmanBox (cx, cy, w, h, vx, vy; q = 1, r = 4) run on a measurement sequence,
  * the whole lab pipeline (synthetic scene, rendering, detection jitter, colour-histogram embeddings, Kalman prediction,
    cost, Hungarian, identity switches) re-implemented here in NumPy/SciPy, for a grid of settings and for many seeds.
Everything is deterministic: the lab's mulberry32 generator is ported (vectorised, since its state is a counter).

Run with the project venv:  .venv/bin/python scripts/figures/playground_goldens_tracking.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

OUT = Path(__file__).resolve().parents[1] / "tests" / "golden" / "tracking.json"

# ----------------------------------------------------------------------------------------------- generators (ports)
def mulberry32(seed, n):
    """First n outputs of the lab's mulberry32 (src/lib/imagekit/rng.ts), vectorised: the state is seed + k * 0x6d2b79f5."""
    k = np.arange(1, n + 1, dtype=np.uint64)
    a = ((np.uint64(seed) + k * np.uint64(0x6D2B79F5)) % np.uint64(2**32)).astype(np.uint32)
    t = a
    t = (t ^ (t >> np.uint32(15))) * (t | np.uint32(1))
    t = t ^ (t + (t ^ (t >> np.uint32(7))) * (t | np.uint32(61)))
    return ((t ^ (t >> np.uint32(14))).astype(np.float64)) / 4294967296.0


def normal_field(n, seed):
    """normalField(n, seed) of the lab kit (Box-Muller), as float32 like the Float32Array it returns."""
    r = mulberry32(seed, 2 * n)
    r1, r2 = r[0::2], r[1::2]
    return (np.sqrt(-2 * np.log(np.maximum(1e-12, r1))) * np.cos(2 * np.pi * r2)).astype(np.float32)


# ----------------------------------------------------------------------------------------------- scene (mirror of the lab)
W, H, T, K = 160, 128, 30, 3
BOX_W, BOX_H = 30.0, 40.0
SPEED, MEET = 4.0, 14                                         # pixels per frame; the pair meets and turns back at frame MEET
Y_A, Y_B, Y_C = 40.0, 48.0, 100.0
BACKGROUND = np.array([0.86, 0.86, 0.84])
TROUSERS = np.array([[0.45, 0.30, 0.10], [0.10, 0.15, 0.62], [0.15, 0.15, 0.20]])
SHIRTS = np.array([[0.90, 0.55, 0.12], [0.15, 0.40, 0.85], [0.20, 0.65, 0.30]])   # orange, blue, green
SHIRT_ROWS = 22
PIXEL_NOISE = 0.03
RENDER_SEED = 2024
BINS = 4
Q, R, ALPHA = 1.0, 4.0, 0.9
SEED = 16                                                     # detection-jitter seed of the lab


def truth_boxes():
    """(T, K, 4) boxes [cx, cy, w, h]. Objects 0 and 1 walk toward each other along one line, meet at the same x at frame MEET
    and turn back; object 2 walks along the bottom and never touches them."""
    f = np.arange(T, dtype=float)
    xa = np.where(f <= MEET, 30 + SPEED * f, 30 + SPEED * MEET - SPEED * (f - MEET))
    xb = np.where(f <= MEET, 30 + 2 * SPEED * MEET - SPEED * f, 30 + SPEED * MEET + SPEED * (f - MEET))
    out = np.zeros((T, K, 4))
    out[:, 0] = np.c_[xa, np.full(T, Y_A), np.full(T, BOX_W), np.full(T, BOX_H)]
    out[:, 1] = np.c_[xb, np.full(T, Y_B), np.full(T, BOX_W), np.full(T, BOX_H)]
    out[:, 2] = np.c_[20 + SPEED * f, np.full(T, Y_C), np.full(T, BOX_W), np.full(T, BOX_H)]
    return out


def px_range(c, size, limit):
    lo = int(np.floor(c - size / 2 + 0.5)); hi = int(np.floor(c + size / 2 + 0.5))
    return max(0, lo), min(limit, hi)


def render(f, boxes):
    """RGB float32 frame (H, W, 3): background, then objects 0, 1, 2 in this order (later ones in front), plus pixel noise."""
    img = np.tile(BACKGROUND, (H, W, 1))
    for k in range(K):
        b = boxes[f, k]
        x0, x1 = px_range(b[0], b[2], W); y0, y1 = px_range(b[1], b[3], H)
        ytop = int(np.floor(b[1] - b[3] / 2 + 0.5))
        for y in range(y0, y1):
            img[y, x0:x1] = SHIRTS[k] if y - ytop < SHIRT_ROWS else TROUSERS[k]
    noise = normal_field(W * H * 3, RENDER_SEED + f).reshape(H, W, 3).astype(np.float64)
    return np.clip(img + PIXEL_NOISE * noise, 0, 1).astype(np.float32)


def embedding(img, box):
    x0, x1 = px_range(box[0], box[2], W); y0, y1 = px_range(box[1], box[3], H)
    crop = img[y0:y1, x0:x1].astype(np.float64).reshape(-1, 3)
    if len(crop) == 0:
        return np.zeros(BINS ** 3)
    idx = np.minimum(BINS - 1, np.floor(crop * BINS).astype(int))
    h = np.bincount(idx[:, 0] * BINS * BINS + idx[:, 1] * BINS + idx[:, 2], minlength=BINS ** 3).astype(np.float64)
    return h / np.linalg.norm(h)


def iou(a, b):
    a1 = a[:2] - a[2:] / 2; a2 = a[:2] + a[2:] / 2; b1 = b[:2] - b[2:] / 2; b2 = b[:2] + b[2:] / 2
    wh = np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, None)
    inter = wh[0] * wh[1]
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


class KalmanBox:
    """Chapter DLCV 06, section 'Motion model: the Kalman filter' (same code)."""
    def __init__(self, z, q=Q, r=R):
        self.x = np.r_[z, 0.0, 0.0]
        self.P = np.diag([r, r, r, r, 100.0, 100.0])
        self.F = np.eye(6); self.F[0, 4] = self.F[1, 5] = 1.0
        self.H = np.eye(4, 6)
        self.Q = q * np.diag([1, 1, 0.1, 0.1, 0.5, 0.5])
        self.R = r * np.eye(4)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].copy()

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        Kg = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ (z - self.H @ self.x)
        self.P = (np.eye(6) - Kg @ self.H) @ self.P


def detections(f, boxes, sigma, noise):
    """Detection slot j of frame f holds object (j + f) mod 3: the detector does not report identities. Returns (boxes, truth ids)."""
    out, truth = [], []
    for j in range(K):
        k = (j + f) % K
        n = noise[(f * K + k) * 4:(f * K + k) * 4 + 4].astype(np.float64)
        out.append(boxes[f, k] + sigma * np.array([n[0], n[1], n[2] / 2, n[3] / 2])); truth.append(k)
    return out, truth


def track(lam, sigma, noise_seed, frames=None, boxes=None):
    boxes = truth_boxes() if boxes is None else boxes
    frames = [render(f, boxes) for f in range(T)] if frames is None else frames
    noise = normal_field(T * K * 4, noise_seed)
    kf, mem, owner = None, None, {}
    ids = np.zeros((T, K), dtype=int)           # ids[f, k] = id of the track assigned to object k in frame f
    out = np.zeros((T, K, 4)); costs = []
    for f in range(T):
        dets, truth = detections(f, boxes, sigma, noise)
        emb = [embedding(frames[f], d) for d in dets]
        if f == 0:
            kf = [None] * K; mem = [None] * K
            for j in range(K):
                kf[truth[j]] = KalmanBox(dets[j]); mem[truth[j]] = emb[j].copy(); ids[0, truth[j]] = truth[j]
                out[0, truth[j]] = dets[j]
            costs.append(None)
            continue
        pred = [t.predict() for t in kf]
        Ci = np.array([[1 - iou(pred[k], dets[j]) for j in range(K)] for k in range(K)])
        Ca = np.array([[1 - mem[k] @ emb[j] for j in range(K)] for k in range(K)])
        C = lam * Ci + (1 - lam) * Ca
        rows, cols = linear_sum_assignment(C)
        for k, j in zip(rows, cols):
            kf[k].update(dets[j])
            mem[k] = ALPHA * mem[k] + (1 - ALPHA) * emb[j]; mem[k] = mem[k] / np.linalg.norm(mem[k])
            ids[f, truth[j]] = k; out[f, k] = kf[k].x[:4]
        costs.append(dict(C=C, Ci=Ci, Ca=Ca, rows=rows, cols=cols, truth=truth))
    return ids, out, costs


def summarise(ids, out, boxes):
    sw = 0; first = None
    for f in range(1, T):
        for k in range(K):
            if ids[f, k] != ids[f - 1, k]:
                sw += 1
                if first is None: first = f
    wrong = int(np.sum(ids != np.arange(K)[None, :]))
    miou = float(np.mean([iou(out[f, k], boxes[f, k]) for f in range(1, T) for k in range(K)]))
    return dict(switches=sw, firstSwitchFrame=first, wrongObjectFrames=wrong, meanIou=miou, mota=1 - sw / (T * K))


def main():
    rng = np.random.default_rng(7)
    g = {}
    # Hungarian
    cases = []
    for n, m in [(2, 2), (3, 3), (3, 3), (4, 4), (5, 5), (8, 8), (12, 12), (3, 5), (5, 3), (2, 7), (7, 2), (6, 9), (9, 6), (1, 4), (4, 1)]:
        for rep in range(8):
            C = np.round(rng.random((n, m)) * 10, 6) if rep % 2 else rng.random((n, m))
            r, c = linear_sum_assignment(C)
            row2col = [-1] * n
            for a, b in zip(r, c): row2col[a] = int(b)
            cases.append(dict(cost=C.tolist(), rowToCol=row2col, total=float(C[r, c].sum())))
    g["hungarian"] = cases
    # IoU
    ious = []
    for _ in range(60):
        a = np.r_[rng.uniform(0, 50, 2), rng.uniform(5, 30, 2)]; b = a + np.r_[rng.normal(0, 8, 2), rng.normal(0, 4, 2)]; b[2:] = np.abs(b[2:]) + 1
        ious.append(dict(a=a.tolist(), b=b.tolist(), iou=float(iou(a, b))))
    g["iou"] = ious
    # Kalman: chapter's class on a short noisy sequence (constant velocity (3, -1) plus noise)
    zs = [np.r_[40 + 3 * t, 60 - t, 30, 40] + rng.normal(0, 2, 4) for t in range(12)]
    kf = KalmanBox(zs[0]); states = []
    for z in zs[1:]:
        p = kf.predict(); kf.update(z); states.append(dict(pred=p.tolist(), x=kf.x.tolist(), P=np.diag(kf.P).tolist()))
    g["kalman"] = dict(z=[z.tolist() for z in zs], steps=states)
    # mulberry / normal field / rendering checksums
    g["rng"] = dict(mulberry=mulberry32(12345, 6).tolist(), normal=normal_field(6, 12345).astype(np.float64).tolist())
    boxes = truth_boxes(); frames = [render(f, boxes) for f in range(T)]
    g["scene"] = dict(boxes=[[round(float(v), 9) for v in boxes[f, k]] for f in (0, 14, 15, 29) for k in range(K)],
                      pixelSums=[float(frames[f].astype(np.float64).sum()) for f in (0, 15, 29)],
                      embedding=embedding(frames[15], boxes[15, 0]).tolist())
    # pipeline
    grid = []
    for lam in (1.0, 0.75, 0.5, 0.25, 0.0):
        for sigma in (0.0, 2.5, 5.0):
            ids, out, costs = track(lam, sigma, SEED, frames, boxes)
            grid.append(dict(lam=lam, sigma=sigma, seed=SEED, ids=ids.tolist(), **summarise(ids, out, boxes)))
    g["pipelineSeed"] = SEED
    g["grid"] = grid
    # one frame's cost matrices (default settings), frame to inspect given below
    ids, out, costs = track(1.0, 2.5, SEED, frames, boxes)
    g["frameCosts"] = {str(f): dict(C=costs[f]["C"].tolist(), Ci=costs[f]["Ci"].tolist(), Ca=costs[f]["Ca"].tolist(), rows=costs[f]["rows"].tolist(), cols=costs[f]["cols"].tolist(), truth=costs[f]["truth"]) for f in (5, 15, 18, 25)}
    # robustness over seeds
    rob = {}
    for name, lam, sigma in [("motion", 1.0, 2.5), ("mixed", 0.5, 2.5), ("appearance", 0.0, 2.5), ("motionQuiet", 1.0, 0.0), ("motionLoud", 1.0, 5.0), ("mixedLoud", 0.5, 5.0)]:
        sw = [summarise(*track(lam, sigma, s, frames, boxes)[:2], boxes)["switches"] for s in range(200)]
        rob[name] = dict(lam=lam, sigma=sigma, seeds=200, withSwitch=int(sum(1 for v in sw if v > 0)), switches=sw)
    g["robust"] = rob
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(g))
    print({k: (v["withSwitch"], v["seeds"]) for k, v in rob.items()})
    print([(d["lam"], d["sigma"], d["switches"], d["firstSwitchFrame"], round(d["meanIou"], 3)) for d in grid])
    print("cos(A,B) at frame 0:", float(embedding(frames[0], boxes[0, 0]) @ embedding(frames[0], boxes[0, 1])))


if __name__ == "__main__":
    main()
