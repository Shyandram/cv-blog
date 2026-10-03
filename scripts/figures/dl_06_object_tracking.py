"""Figures for DL 06 · Object Tracking (deep-learning/dl-06-object-tracking).

Re-runnable: writes PNGs only to src/assets/deep-learning/dl-06-object-tracking/.
Contains a small NumPy SORT-style tracker (constant-velocity Kalman filter, IoU +
Hungarian association, optional BYTE low-score pass and appearance cost) and
MOTA / IDF1 / HOTA on a synthetic scene.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------- 1. constant-velocity Kalman filter ----------
class KalmanBox:
    """State x = [cx, cy, w, h, vx, vy]; measurement z = [cx, cy, w, h]."""
    def __init__(self, z, q=1.0, r=4.0):
        self.x = np.r_[z, 0.0, 0.0]
        self.P = np.diag([r, r, r, r, 100.0, 100.0])    # unknown velocity: large variance
        self.F = np.eye(6); self.F[0, 4] = self.F[1, 5] = 1.0   # dt = 1 frame
        self.H = np.eye(4, 6)
        self.Q = q * np.diag([1, 1, 0.1, 0.1, 0.5, 0.5])
        self.R = r * np.eye(4)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)          # Kalman gain
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P

# ---------- 2. IoU and Hungarian association ----------
def iou_matrix(a, b):
    """a: (N,4), b: (M,4) boxes as [cx, cy, w, h] -> (N,M) IoU."""
    a1, a2 = a[:, None, :2] - a[:, None, 2:] / 2, a[:, None, :2] + a[:, None, 2:] / 2
    b1, b2 = b[None, :, :2] - b[None, :, 2:] / 2, b[None, :, :2] + b[None, :, 2:] / 2
    wh = np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area = lambda x: x[..., 2] * x[..., 3]
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)

def associate(cost, max_cost):
    """Hungarian assignment; pairs whose cost exceeds max_cost are rejected."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    pairs = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= max_cost]
    mr = {r for r, _ in pairs}; mc = {c for _, c in pairs}
    return (pairs, [r for r in range(cost.shape[0]) if r not in mr],
            [c for c in range(cost.shape[1]) if c not in mc])

# ---------- 3. synthetic scene ----------
def make_scene(T=80, seed=0):
    rng = np.random.default_rng(seed)
    W, H = 30.0, 70.0
    t = np.arange(T, dtype=float)
    tr = {}
    # pair 1: plain crossing, B is behind A (occluded while they overlap)
    tr[0] = np.c_[80 + 5 * t, np.full(T, 120.0)]
    tr[1] = np.c_[480 - 5 * t, np.full(T, 128.0)]
    # pair 2: they meet and bounce back (non-linear motion, like dancers)
    m = 40
    x2 = np.where(t < m, 120 + 4 * t, 120 + 4 * m - 4 * (t - m))
    x3 = np.where(t < m, 440 - 4 * t, 440 - 4 * m + 4 * (t - m))
    tr[2] = np.c_[x2 + 10, np.full(T, 260.0)]
    tr[3] = np.c_[x3 - 10, np.full(T, 266.0)]
    # walker passing behind a pole (frames 30-37): detections become low-score
    tr[4] = np.c_[60 + 3 * t, 40 + 1.0 * t]
    gt = []                                   # per frame list of (id, box)
    dets = []                                 # per frame list of (box, score, gt_id)
    feats = {k: (lambda v: v / np.linalg.norm(v))(rng.normal(size=16)) for k in tr}
    for f in range(T):
        g, d = [], []
        boxes = {k: np.r_[tr[k][f], W, H] for k in tr}
        for k, b in boxes.items():
            g.append((k, b))
            score = rng.uniform(0.6, 0.95)
            if k == 1 and iou_matrix(b[None], boxes[0][None])[0, 0] > 0.2:
                score = rng.uniform(0.15, 0.45)       # occluded by object 0
            if k == 3 and iou_matrix(b[None], boxes[2][None])[0, 0] > 0.2:
                score = rng.uniform(0.15, 0.45)       # occluded by object 2
            if k == 4 and 30 <= f <= 37:
                score = rng.uniform(0.15, 0.45)       # behind the pole
            if rng.random() < 0.05:
                continue                               # random missed detection
            noisy = b + np.r_[rng.normal(0, 2.0, 2), rng.normal(0, 1.0, 2)]
            e = feats[k] + 0.15 * rng.normal(size=16)
            d.append((noisy, score, k, e / np.linalg.norm(e)))
        if rng.random() < 0.3:                         # low-score false positive
            fp = np.r_[rng.uniform(50, 550), rng.uniform(50, 320), W, H]
            e = rng.normal(size=16)
            d.append((fp, rng.uniform(0.1, 0.4), -1, e / np.linalg.norm(e)))
        gt.append(g); dets.append(d)
    return gt, dets

# ---------- 4. SORT-style tracker with optional BYTE and appearance ----------
class Track:
    def __init__(self, tid, z, emb):
        self.id, self.kf, self.emb = tid, KalmanBox(z), emb
        self.lost = 0

def run_tracker(dets, use_low=False, use_app=False, hi=0.5, max_age=2,
                app_age=30, lam=0.5):
    tracks, out, next_id = [], [], 0
    for d in dets:
        for tk in tracks:
            tk.kf.predict()
        boxes = np.array([x[0] for x in d]).reshape(-1, 4)
        scores = np.array([x[1] for x in d])
        embs = np.array([x[3] for x in d]).reshape(-1, 16)
        high = np.where(scores >= hi)[0]
        low = np.where((scores < hi) & (scores >= 0.1))[0]
        active = [tk for tk in tracks if tk.lost <= max_age]
        pred = np.array([tk.kf.x[:4] for tk in active]).reshape(-1, 4)
        # stage 1: high-score detections vs active tracks
        cost = 1 - iou_matrix(pred, boxes[high])
        gate = 0.7
        if use_app and len(active) and len(high):
            tE = np.array([tk.emb for tk in active])
            app = 1 - tE @ embs[high].T            # cosine distance
            cost = np.where(cost > 0.9, 9.0, lam * cost + (1 - lam) * app)
            gate = 0.6
        pairs, ut, ud = associate(cost, gate)
        matched = [(active[r], high[c]) for r, c in pairs]
        rest_tracks = [active[r] for r in ut]
        rest_high = [high[c] for c in ud]
        # stage 2 (BYTE): remaining tracks vs low-score detections, IoU only
        if use_low and rest_tracks and len(low):
            p2 = np.array([tk.kf.x[:4] for tk in rest_tracks])
            pairs2, ut2, _ = associate(1 - iou_matrix(p2, boxes[low]), 0.5)
            matched += [(rest_tracks[r], low[c]) for r, c in pairs2]
            rest_tracks = [rest_tracks[r] for r in ut2]
        # stage 3 (appearance only): long-lost tracks vs leftover high-score dets
        if use_app and rest_high:
            lost = [tk for tk in tracks if max_age < tk.lost <= app_age]
            if lost:
                app = 1 - np.array([tk.emb for tk in lost]) @ embs[rest_high].T
                pairs3, _, ud3 = associate(app, 0.3)
                matched += [(lost[r], rest_high[c]) for r, c in pairs3]
                rest_high = [rest_high[c] for c in ud3]
        for tk, j in matched:
            tk.kf.update(boxes[j]); tk.lost = 0
            tk.emb = 0.9 * tk.emb + 0.1 * embs[j]; tk.emb /= np.linalg.norm(tk.emb)
        hit = {id(tk) for tk, _ in matched}
        for tk in tracks:
            if id(tk) not in hit:
                tk.lost += 1
        for j in rest_high:                           # birth: unmatched high-score
            tracks.append(Track(next_id, boxes[j], embs[j])); next_id += 1
        tracks = [tk for tk in tracks if tk.lost <= (app_age if use_app else max_age)]
        out.append([(tk.id, tk.kf.x[:4].copy()) for tk in tracks if tk.lost == 0])
    return out

# ---------- 5. metrics: MOTA, IDF1, HOTA ----------
def frame_match(g, p, thr):
    if not g or not p:
        return []
    iou = iou_matrix(np.array([b for _, b in g]), np.array([b for _, b in p]))
    r, c = linear_sum_assignment(-iou)
    return [(g[i][0], p[j][0]) for i, j in zip(r, c) if iou[i, j] >= thr]

def mota(gt, hyp, thr=0.5):
    fn = fp = idsw = 0; ngt = 0; last = {}
    for g, p in zip(gt, hyp):
        m = frame_match(g, p, thr)
        ngt += len(g); fn += len(g) - len(m); fp += len(p) - len(m)
        for gi, pi in m:
            if gi in last and last[gi] != pi:
                idsw += 1
            last[gi] = pi
    return 1 - (fn + fp + idsw) / ngt, idsw

def idf1(gt, hyp, thr=0.5):
    gids = sorted({i for g in gt for i, _ in g}); pids = sorted({i for p in hyp for i, _ in p})
    gi = {k: n for n, k in enumerate(gids)}; pi = {k: n for n, k in enumerate(pids)}
    ov = np.zeros((len(gids), len(pids)))
    for g, p in zip(gt, hyp):
        if g and p:
            iou = iou_matrix(np.array([b for _, b in g]), np.array([b for _, b in p]))
            for a in range(len(g)):
                for b in range(len(p)):
                    ov[gi[g[a][0]], pi[p[b][0]]] += iou[a, b] >= thr
    r, c = linear_sum_assignment(-ov)
    idtp = ov[r, c].sum()
    n_gt = sum(len(g) for g in gt); n_p = sum(len(p) for p in hyp)
    return 2 * idtp / (n_gt + n_p)

def hota(gt, hyp, alphas=np.arange(0.05, 0.96, 0.05)):
    res = []
    for a in alphas:
        pairs = []; n_gt = n_p = 0
        for g, p in zip(gt, hyp):
            pairs += frame_match(g, p, a); n_gt += len(g); n_p += len(p)
        tp = len(pairs)
        if tp == 0:
            res.append((0, 0, 0)); continue
        cnt = {}
        for q in pairs:
            cnt[q] = cnt.get(q, 0) + 1
        gcount = {}; pcount = {}
        for g in gt:
            for i, _ in g: gcount[i] = gcount.get(i, 0) + 1
        for p in hyp:
            for i, _ in p: pcount[i] = pcount.get(i, 0) + 1
        ass = sum(cnt[q] / (gcount[q[0]] + pcount[q[1]] - cnt[q]) for q in pairs) / tp
        det = tp / (n_gt + n_p - tp)
        res.append((np.sqrt(det * ass), det, ass))
    return np.array(res).mean(axis=0)   # HOTA, DetA, AssA averaged over alpha


# =====================================================================
# Figures
# =====================================================================
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "src", "assets", "deep-learning", "dl-06-object-tracking")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
COLORS = ["#2a6fdb", "#e8590c", "#2b8a3e", "#c2255c", "#7048e8", "#0b7285",
          "#e67700", "#5c940d", "#862e9c", "#495057", "#d6336c", "#1864ab"]


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, os.path.getsize(path) // 1024, "KB")


def box(ax, x, y, w, h, text, fc="#e7f0fd", ec="#2a6fdb", fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, color="#343a40", text=None, fs=9):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", va="bottom",
                fontsize=fs, color=color)


def fig_taxonomy():
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 3.75, 5.1, 3.0, 0.7, "Object tracking", fc="#fff3bf", ec="#e67700", fs=12, bold=True)
    # by target
    box(ax, 0.2, 3.7, 4.6, 0.6, "By target (what is followed)", fc="#f1f3f5", ec="#868e96", bold=True)
    box(ax, 0.2, 2.5, 2.2, 0.9, "SOT / VOT\none object,\ngiven in frame 1")
    box(ax, 2.6, 2.5, 2.2, 0.9, "MOT\nall objects of given\nclasses, with IDs")
    box(ax, 1.4, 0.9, 2.2, 1.0, "MTMCT\nmany targets,\nmany cameras", fc="#ffe3e3", ec="#c92a2a")
    # by method
    box(ax, 5.7, 3.7, 4.6, 0.6, "By method (how identity is kept)", fc="#f1f3f5", ec="#868e96", bold=True)
    box(ax, 5.7, 2.5, 2.2, 0.9, "Re-ID\nmatch by appearance,\nno time needed", fc="#ebfbee", ec="#2b8a3e")
    box(ax, 8.1, 2.5, 2.2, 0.9, "Temporal tracking\nmotion + appearance\nframe to frame", fc="#ebfbee", ec="#2b8a3e")
    box(ax, 7.0, 0.9, 2.2, 1.0, "Related: point\ntracking (TAP),\nvideo segmentation", fc="#f3f0ff", ec="#7048e8")
    arrow(ax, 4.6, 5.1, 2.5, 4.3); arrow(ax, 5.9, 5.1, 8.0, 4.3)
    arrow(ax, 1.3, 2.5, 2.2, 1.9); arrow(ax, 3.7, 2.5, 2.8, 1.9)
    arrow(ax, 6.2, 2.5, 3.65, 1.4, color="#c92a2a")
    ax.text(5.3, 1.35, "Re-ID links\ncameras", fontsize=9, color="#c92a2a", ha="center", va="top")
    arrow(ax, 9.2, 2.5, 8.4, 1.9, color="#7048e8")
    save(fig, "taxonomy.png")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.set_xlim(0, 10.5); ax.set_ylim(-0.5, 4.0); ax.axis("off")
    box(ax, 0.1, 2.8, 1.4, 0.8, "Frame t", fc="#f1f3f5", ec="#495057")
    box(ax, 1.9, 2.8, 1.7, 0.8, "Detector\nboxes + scores")
    box(ax, 4.0, 2.8, 1.8, 0.8, "Re-ID head\nembeddings", fc="#ebfbee", ec="#2b8a3e")
    box(ax, 1.9, 0.9, 1.7, 0.8, "Tracks at t−1", fc="#fff3bf", ec="#e67700")
    box(ax, 4.0, 0.9, 1.8, 0.8, "Kalman predict\nboxes at t", fc="#fff3bf", ec="#e67700")
    box(ax, 6.2, 1.3, 1.9, 1.9, "Cost matrix\n1−IoU and/or\ncosine distance", fc="#f3f0ff", ec="#7048e8")
    box(ax, 8.5, 2.6, 1.9, 0.8, "Hungarian\nassignment")
    box(ax, 8.5, 0.9, 1.9, 1.1, "update matched\nbirth: new IDs\ndeath: old tracks", fc="#ffe3e3", ec="#c92a2a", fs=9)
    arrow(ax, 1.5, 3.2, 1.9, 3.2)
    arrow(ax, 3.6, 3.2, 4.0, 3.2)
    arrow(ax, 5.8, 3.2, 6.2, 2.85, text="")
    arrow(ax, 3.6, 1.3, 4.0, 1.3)
    arrow(ax, 5.8, 1.3, 6.2, 1.65)
    arrow(ax, 8.1, 2.4, 8.5, 2.85)
    arrow(ax, 9.45, 2.6, 9.45, 2.0)
    ax.annotate("", xy=(2.75, 0.9), xytext=(9.45, 0.9),
                arrowprops=dict(arrowstyle="-|>", color="#c92a2a", lw=1.4,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(6.1, -0.15, "tracks carried to the next frame", color="#c92a2a", fontsize=9, ha="center")
    ax.text(5.9, 3.75, "observations", fontsize=9, color="#495057", ha="center")
    ax.text(5.9, 0.62, "state", fontsize=9, color="#495057", ha="center")
    save(fig, "tbd-pipeline.png")


def fig_kalman():
    rng = np.random.default_rng(1)
    T = 30
    t = np.arange(T)
    true = np.c_[20 + 6 * t, 60 + 25 * np.sin(t / 7.0)]
    meas = true + rng.normal(0, 3.0, true.shape)
    missing = set(range(13, 18))
    kf = KalmanBox(np.r_[meas[0], 30, 70], q=1.0, r=9.0)
    pri, post, sd_pri, sd_post = [], [], [], []
    for f in range(1, T):
        kf.predict(); pri.append(kf.x[:2].copy()); sd_pri.append(np.sqrt(kf.P[0, 0]))
        Ppri = kf.P[:2, :2].copy()
        if f not in missing:
            kf.update(np.r_[meas[f], 30, 70])
        post.append((f, kf.x[:2].copy(), Ppri, kf.P[:2, :2].copy()))
        sd_post.append(np.sqrt(kf.P[0, 0]))
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axs[0]
    ax.plot(true[:, 0], true[:, 1], color="#adb5bd", lw=2, label="true path")
    obs = [f for f in range(T) if f not in missing]
    ax.plot(meas[obs, 0], meas[obs, 1], "o", ms=4, color="#e8590c", label="detections")
    for f, x, Pp, Pu in post:
        for P, c in [(Pp, "#7048e8"), (Pu, "#2b8a3e")]:
            w, v = np.linalg.eigh(P)
            ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
            if f < 3:
                continue
            ax.add_patch(Ellipse(x if c == "#2b8a3e" else pri[f - 1], 2 * np.sqrt(w[1]),
                                 2 * np.sqrt(w[0]), angle=ang, fill=False, ec=c, lw=1, alpha=0.8))
    ax.plot(*np.array([p for _, p, _, _ in post]).T, "-", color="#2b8a3e", lw=1.2, label="filtered (after update)")
    ax.plot([], [], color="#7048e8", label="1σ ellipse: predicted (purple), updated (green)")
    ax.axvspan(true[13, 0] - 3, true[17, 0] + 3, color="#ffe3e3", alpha=0.6, label="missed detections")
    ax.set_xlabel("cx (pixels)"); ax.set_ylabel("cy (pixels)"); ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    ax.set_title("Predict grows, update shrinks")
    ax = axs[1]
    tt = np.repeat(np.arange(1, T), 2)
    sd = np.ravel(np.c_[sd_pri, sd_post])
    ax.plot(tt, sd, color="#343a40", lw=1.3)
    ax.axvspan(13, 17, color="#ffe3e3", alpha=0.6)
    ax.set_xlabel("frame"); ax.set_ylabel("std of cx (pixels)")
    ax.set_title("Uncertainty: saw-tooth, grows in gaps")
    save(fig, "kalman.png")


def id_switch_frames(gt, hyp):
    last, ev = {}, []
    for f, (g, p) in enumerate(zip(gt, hyp)):
        for gi, pi in frame_match(g, p, 0.5):
            if gi in last and last[gi] != pi:
                ev.append((f, gi))
            last[gi] = pi
    return ev


def fig_tracks(seed=3):
    gt, dets = make_scene(seed=seed)
    cfgs = [("SORT-style: IoU only", {}),
            ("+ second pass on low-score boxes (BYTE)", dict(use_low=True)),
            ("+ appearance cost and re-identification", dict(use_low=True, use_app=True))]
    fig, axs = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)
    stats = []
    for ax, (name, kw) in zip(axs, cfgs):
        hyp = run_tracker(dets, **kw)
        M, sw = mota(gt, hyp); F1 = idf1(gt, hyp); H = hota(gt, hyp)[0]
        nid = len({i for p in hyp for i, _ in p})
        stats.append((name, M, F1, H, sw, nid))
        for k in range(5):
            xs = [(f, b[0]) for f, g in enumerate(gt) for i, b in g if i == k]
            ax.plot(*np.array(xs).T, color="#dee2e6", lw=6, zorder=1)
        ids = sorted({i for p in hyp for i, _ in p})
        for n, i in enumerate(ids):
            pts = np.array([(f, b[0]) for f, p in enumerate(hyp) for j, b in p if j == i])
            ax.plot(pts[:, 0], pts[:, 1], ".", ms=4, color=COLORS[n % len(COLORS)], zorder=2)
        for f, gi in id_switch_frames(gt, hyp):
            x = [b[0] for i, b in gt[f] if i == gi][0]
            ax.plot(f, x, "x", ms=11, mew=2.2, color="black", zorder=3)
        ax.set_title(f"{name}   |   MOTA {M:.2f}  IDF1 {F1:.2f}  HOTA {H:.2f}  "
                     f"ID switches {sw}  IDs used {nid} (true 5)", fontsize=10)
        ax.set_ylabel("box centre x")
    axs[-1].set_xlabel("frame")
    axs[0].plot([], [], color="#dee2e6", lw=6, label="ground truth")
    axs[0].plot([], [], "x", color="black", label="ID switch")
    axs[0].legend(loc="upper left", fontsize=9, ncol=2)
    save(fig, "tracks-id-switches.png")
    return stats


def fig_hota_vs_mota():
    T = 100
    b = np.r_[100.0, 100.0, 30.0, 70.0]
    gt = [[(0, b)] for _ in range(T)]
    A = [[(f // 10, b)] for f in range(T)]                     # every frame, new ID every 10
    B = [[(0, b)] if f < 60 else [] for f in range(T)]          # 60 % of frames, one ID
    res = {}
    for n, h in [("A", A), ("B", B)]:
        M, _ = mota(gt, h); H, D, As = hota(gt, h)
        res[n] = (M, idf1(gt, h), H, D, As)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    ax.barh(2, T, left=0, color="#adb5bd", height=0.5)
    for k in range(10):
        ax.barh(1, 10, left=10 * k, color=COLORS[k], height=0.5, edgecolor="white")
    ax.barh(0, 60, left=0, color=COLORS[0], height=0.5)
    ax.set_yticks([2, 1, 0], ["ground truth", "tracker A", "tracker B"])
    ax.set_xlabel("frame"); ax.set_xlim(0, T)
    ax.set_title("One object, two trackers (colour = predicted ID)")
    ax = axs[1]
    names = ["MOTA", "IDF1", "HOTA", "DetA", "AssA"]
    x = np.arange(5)
    ax.bar(x - 0.18, res["A"], 0.36, color="#e8590c", label="A: finds every frame, 9 ID switches")
    ax.bar(x + 0.18, res["B"], 0.36, color="#2a6fdb", label="B: misses 40 %, never switches")
    for i in range(5):
        ax.text(i - 0.18, res["A"][i] + 0.02, f"{res['A'][i]:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.18, res["B"][i] + 0.02, f"{res['B'][i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, names); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1); ax.set_title("MOTA prefers A; IDF1 and HOTA prefer B")
    save(fig, "hota-vs-mota.png")
    return res


def fig_sot():
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax in axs:
        ax.set_xlim(0, 5); ax.set_ylim(0, 4.2); ax.axis("off")
    ax = axs[0]; ax.set_title("Two-stream Siamese (SiamFC, SiamRPN)")
    box(ax, 0.1, 2.9, 1.2, 0.8, "template z\n(frame 1)", fc="#fff3bf", ec="#e67700", fs=9)
    box(ax, 0.1, 0.6, 1.2, 0.8, "search x\n(frame t)", fc="#f1f3f5", ec="#495057", fs=9)
    box(ax, 1.7, 2.9, 1.0, 0.8, "CNN φ", fs=10)
    box(ax, 1.7, 0.6, 1.0, 0.8, "CNN φ\n(shared)", fs=9)
    box(ax, 3.0, 1.65, 0.9, 0.9, "cross-\ncorr. ⋆", fc="#f3f0ff", ec="#7048e8", fs=9)
    box(ax, 4.1, 1.65, 0.85, 0.9, "score\nmap", fc="#ebfbee", ec="#2b8a3e", fs=9)
    for y in (3.3, 1.0):
        arrow(ax, 1.3, y, 1.7, y)
    arrow(ax, 2.7, 3.3, 3.2, 2.55); arrow(ax, 2.7, 1.0, 3.2, 1.65); arrow(ax, 3.9, 2.1, 4.1, 2.1)
    ax.text(2.5, 0.1, "features extracted separately, compared once at the end", ha="center", fontsize=9)
    ax = axs[1]; ax.set_title("One-stream transformer (OSTrack)")
    box(ax, 0.1, 2.9, 1.2, 0.8, "template z\npatches", fc="#fff3bf", ec="#e67700", fs=9)
    box(ax, 0.1, 0.6, 1.2, 0.8, "search x\npatches", fc="#f1f3f5", ec="#495057", fs=9)
    box(ax, 1.6, 0.6, 1.8, 3.1, "ViT encoder\njoint attention\nover [z ; x]\ntokens\n\nextracts features\nand relates z to x\nin one pass",
        fc="#f3f0ff", ec="#7048e8", fs=9)
    box(ax, 3.7, 1.65, 1.25, 0.9, "box head on\nsearch tokens", fc="#ebfbee", ec="#2b8a3e", fs=9)
    arrow(ax, 1.3, 3.3, 1.6, 2.6); arrow(ax, 1.3, 1.0, 1.6, 1.7); arrow(ax, 3.4, 2.1, 3.7, 2.1)
    ax.text(2.5, 0.1, "template and search interact in every layer", ha="center", fontsize=9)
    save(fig, "sot-siamese-vs-onestream.png")


if __name__ == "__main__":
    fig_taxonomy()
    fig_pipeline()
    fig_kalman()
    for s in fig_tracks():
        print("%-45s MOTA %.3f IDF1 %.3f HOTA %.3f IDSW %d IDs %d" % s)
    print(fig_hota_vs_mota())
    fig_sot()
