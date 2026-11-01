"""Figures for DLCV 06A - Human Pose and Keypoint Estimation.

Run:  .venv/bin/python scripts/figures/dl_06a_pose_keypoints.py
Writes only to src/assets/deep-learning/dl-06a-pose-keypoints/.
All data are synthetic (hand-placed stick figures, Gaussian heatmaps); nothing is downloaded.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-06a-pose-keypoints"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})

BLUE, ORANGE, GREEN, YELLOW, PINK, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"
INK, MUTED, GRID, RED = "#0b0b0b", "#52514e", "#d9d8d4", "#c0392b"


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    kb = path.stat().st_size / 1024
    w = plt.imread(path).shape[1]
    print(f"{name}: {kb:.0f} KB, width {w} px")
    assert kb < 1024 and w <= 1600


# ---------------------------------------------------------------- COCO-style 17-keypoint person
NAMES = ["nose", "left eye", "right eye", "left ear", "right ear", "left shoulder", "right shoulder",
         "left elbow", "right elbow", "left wrist", "right wrist", "left hip", "right hip",
         "left knee", "right knee", "left ankle", "right ankle"]
SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
                   1.07, 1.07, .87, .87, .89, .89]) / 10.0
# limbs as drawn (1-based pairs from the COCO person skeleton, converted to 0-based)
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
            (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
GT = np.array([[100, 30], [106, 25], [94, 25], [112, 28], [88, 28], [125, 60], [75, 60],
               [135, 95], [65, 95], [140, 128], [60, 128], [115, 130], [85, 130],
               [117, 175], [83, 175], [118, 220], [82, 220]], float)


def side_color(i):
    if i == 0:
        return GREEN
    return BLUE if NAMES[i].startswith("left") else ORANGE


def limb_color(a, b):
    na, nb = NAMES[a], NAMES[b]
    if na.startswith("left") and nb.startswith("left"):
        return BLUE
    if na.startswith("right") and nb.startswith("right"):
        return ORANGE
    return MUTED


def draw_person(ax, P, vis=None, lw=2.5, ms=7, alpha=1.0, color=None):
    vis = np.full(len(P), 2) if vis is None else vis
    for a, b in SKELETON:
        if vis[a] > 0 and vis[b] > 0:
            ax.plot(*P[[a, b]].T, color=color or limb_color(a, b), lw=lw, alpha=alpha, zorder=2,
                    solid_capstyle="round")
    for i, p in enumerate(P):
        if vis[i] == 0:
            continue
        c = color or side_color(i)
        face = "white" if vis[i] == 1 else c
        ax.plot(*p, "o", ms=ms, mfc=face, mec=c, mew=2, alpha=alpha, zorder=3)


# ---------------------------------------------------------------- Figure 1: what a pose is
# a wider pose (arms out) so that every label has room; used only for this figure
POSE_FIG = np.array([[100, 45], [108, 37], [92, 37], [118, 42], [82, 42], [132, 80], [68, 80],
                     [168, 108], [32, 108], [192, 140], [8, 140], [122, 160], [78, 160],
                     [130, 210], [70, 210], [138, 262], [62, 262]], float)


def fig_skeleton():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.8), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    P = POSE_FIG
    vis = np.full(17, 2)
    vis[4] = 0      # right ear: not labeled
    vis[10] = 1     # right wrist: labeled but occluded
    draw_person(ax, P, vis)
    # label position, alignment; face labels use leader lines
    place = {0: ((100, 5), "center"), 1: ((150, 14), "left"), 3: ((150, 34), "left"),
             2: ((50, 14), "right"), 4: ((50, 34), "right"),
             5: ((140, 72), "left"), 6: ((60, 72), "right"), 7: ((176, 104), "left"), 8: ((24, 104), "right"),
             9: ((200, 140), "left"), 10: ((0, 140), "right"), 11: ((130, 160), "left"),
             12: ((70, 160), "right"), 13: ((138, 210), "left"), 14: ((62, 210), "right"),
             15: ((146, 262), "left"), 16: ((54, 262), "right")}
    for i, ((tx, ty), ha) in place.items():
        lab = f"{i}: {NAMES[i]}"
        if vis[i] == 1:
            lab = f"{i}: {NAMES[i]}\n(v=1, occluded)"
        if vis[i] == 0:
            lab += " (v=0, not labeled)"
        col = MUTED if vis[i] == 0 else INK
        ax.text(tx, ty, lab, fontsize=10, color=col, ha=ha, va="center")
        if i <= 4:
            ax.plot([P[i, 0], tx + (-3 if ha == "left" else 3 if ha == "right" else 0)],
                    [P[i, 1], ty + (4 if i == 0 else 0)], color=GRID, lw=1, zorder=1)
    ax.set_xlim(-95, 295)
    ax.set_ylim(285, -5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) 17 keypoints, limbs and visibility flags", fontsize=11)
    handles = [Line2D([], [], color=BLUE, lw=2.5, marker="o", label="person's left"),
               Line2D([], [], color=ORANGE, lw=2.5, marker="o", label="person's right"),
               Line2D([], [], color=ORANGE, lw=0, marker="o", mfc="white", mew=2,
                      label="labeled, not visible (v=1)")]
    ax.legend(handles=handles, loc="upper center", fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.02), ncol=2)

    ax = axes[1]
    stride, sigma = 4, 2.0
    Hh, Wh = 280 // stride, 200 // stride
    ys, xs = np.mgrid[0:Hh, 0:Wh]
    stack = np.zeros((17, Hh, Wh))
    for k, (x, y) in enumerate(P):
        if vis[k] > 0:
            stack[k] = np.exp(-((xs - x / stride) ** 2 + (ys - y / stride) ** 2) / (2 * sigma ** 2))
    ax.imshow(stack.max(0), cmap="magma", extent=[0, 200, 280, 0], interpolation="nearest")
    ax.set_title("(b) training target: one Gaussian heatmap per\nlabeled keypoint (stride 4, maximum of 16 maps)",
                 fontsize=11)
    ax.set_xlabel("x (input pixels)")
    ax.set_ylabel("y (input pixels)")
    fig.tight_layout()
    save(fig, "skeleton.png")


# ---------------------------------------------------------------- Figure 2: decoding error
def encode(xy, shape, stride, sigma=2.0):
    cx, cy = xy[0] / stride, xy[1] / stride
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))


def peak(h):
    y, x = np.unravel_index(np.argmax(h), h.shape)
    return x, y


def decode_argmax(h, stride):
    x, y = peak(h)
    return np.array([x, y], float) * stride


def decode_quarter(h, stride):
    x, y = peak(h)
    p = np.array([x, y], float)
    if 0 < x < h.shape[1] - 1:
        p[0] += 0.25 * np.sign(h[y, x + 1] - h[y, x - 1])
    if 0 < y < h.shape[0] - 1:
        p[1] += 0.25 * np.sign(h[y + 1, x] - h[y - 1, x])
    return p * stride


def decode_taylor(h, stride):
    x, y = peak(h)
    p = np.array([x, y], float)
    if 1 <= x < h.shape[1] - 1 and 1 <= y < h.shape[0] - 1:
        L = np.log(np.maximum(h[y - 1:y + 2, x - 1:x + 2], 1e-10))
        g = np.array([L[1, 2] - L[1, 0], L[2, 1] - L[0, 1]]) / 2
        hxx = L[1, 2] - 2 * L[1, 1] + L[1, 0]
        hyy = L[2, 1] - 2 * L[1, 1] + L[0, 1]
        hxy = (L[2, 2] - L[2, 0] - L[0, 2] + L[0, 0]) / 4
        Hs = np.array([[hxx, hxy], [hxy, hyy]])
        if np.linalg.det(Hs) > 0 and hxx < 0:
            p += -np.linalg.solve(Hs, g)
    return p * stride


def fig_decoding():
    # same random draws, in the same order, as the tested snippet in the chapter
    rng = np.random.default_rng(0)
    W, H = 192, 256
    pts = rng.uniform([16, 16], [W - 16, H - 16], (500, 2))

    def mean_error(decoder, s, noise=0.01):
        shape, e = (H // s, W // s), 0.0
        for xy in pts:
            h = np.clip(encode(xy, shape, s) + rng.normal(0, noise, shape), 1e-6, None)
            e += np.linalg.norm(decoder(h, s) - xy)
        return e / len(pts)

    strides = [1, 2, 4, 8]
    res = {"argmax": [], "quarter offset": [], "Taylor (log-parabola)": []}
    for s in strides:
        for k, d in zip(res, [decode_argmax, decode_quarter, decode_taylor]):
            res[k].append(mean_error(d, s))
    print("decoding errors:", {k: np.round(v, 2).tolist() for k, v in res.items()})

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    s, xy = 4, np.array([97.0, 130.6])
    h = encode(xy, (H // s, W // s), s)
    x0, y0 = peak(h)
    win = 4
    sub = h[y0 - win:y0 + win + 1, x0 - win:x0 + win + 1]
    ax.imshow(sub, cmap="magma", extent=[x0 - win - .5, x0 + win + .5, y0 + win + .5, y0 - win - .5])
    t = xy / s
    ax.plot(*t, "*", ms=17, color="white", mec=INK, label="true position")
    a = decode_argmax(h, s) / s
    q = decode_quarter(h, s) / s
    ty = decode_taylor(h, s) / s
    ax.plot(*a, "s", ms=10, mfc="none", mec=BLUE, mew=2.5, label="argmax")
    ax.plot(*q, "^", ms=10, mfc="none", mec=GREEN, mew=2.5, label="quarter offset")
    ax.plot(*ty, "o", ms=6, color=YELLOW, mec=INK, label="Taylor")
    ax.set_title("(a) heatmap cells near the peak (stride 4)", fontsize=11)
    ax.set_xlabel("x (heatmap cells)")
    ax.set_ylabel("y (heatmap cells)")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    ax = axes[1]
    for (k, v), c, m in zip(res.items(), [BLUE, GREEN, YELLOW], ["s", "^", "o"]):
        ax.plot(strides, v, marker=m, color=c, lw=2, ms=7, label=k)
    sg = np.linspace(1, 8, 50)
    ax.plot(sg, 0.3826 * sg, "--", color=MUTED, lw=1.5, label="theory for argmax: 0.383 x stride")
    ax.set_xticks(strides)
    ax.set_xlabel("heatmap stride (input pixels per heatmap cell)")
    ax.set_ylabel("mean keypoint error (input pixels)")
    ax.set_title("(b) decoding error vs stride (500 random keypoints)", fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    save(fig, "heatmap-decoding.png")


# ---------------------------------------------------------------- Figure 3: pipelines
def box(ax, x, y, w, h, text, fc, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=INK, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, lw=1.4, color=INK))


def fig_pipelines():
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 5.9)
    ax.axis("off")
    rows = [
        (4.55, "Top-down", ["image", "person\ndetector", "crop + resize\neach box", "single-person\nheatmap network",
                            "decode, map\nto the image"], ["#ffffff", "#dbe8f8", "#dbe8f8", "#fde1d3", "#e8e8e4"],
         "cost grows with the number of people; fails where the detector fails"),
        (2.65, "Bottom-up", ["image", "all keypoint\nheatmaps", "association cues\n(PAFs or tags)",
                             "group into\npeople", "poses"], ["#ffffff", "#fde1d3", "#fde1d3", "#d4f0e4", "#e8e8e4"],
         "one network pass for everyone; grouping is the hard part"),
        (0.75, "One-stage /\nset prediction", ["image", "backbone +\nshared head",
                                               "one pose per\ncenter or query", "score filter\n(often no NMS)",
                                               "poses"], ["#ffffff", "#ece6f8", "#ece6f8", "#e8e8e4", "#e8e8e4"],
         "person and keypoints predicted together"),
    ]
    w, h, gap, x0 = 1.62, 0.95, 0.36, 1.6
    for y, title, labels, colors, note in rows:
        ax.text(0.0, y + h / 2, title, ha="left", va="center", fontsize=11, fontweight="bold", color=INK)
        for i, (lab, fc) in enumerate(zip(labels, colors)):
            x = x0 + i * (w + gap)
            box(ax, x, y, w, h, lab, fc)
            if i:
                arrow(ax, x - gap + 0.02, y + h / 2, x - 0.02, y + h / 2)
        ax.text(x0, y - 0.2, note, fontsize=10, color=MUTED, ha="left", va="top", style="italic")
    save(fig, "pipelines.png")


# ---------------------------------------------------------------- Figure 4: OKS toy
def fig_oks():
    vis = np.full(17, 2)
    vis[3] = 0
    area = 0.6 * (70 * 200)
    s = np.sqrt(area)
    rng = np.random.default_rng(1)
    jitter = GT + rng.normal(0, 3, GT.shape)
    flipped = jitter.copy()
    flipped[[9, 10]] = flipped[[10, 9]]
    d2 = ((flipped - GT) ** 2).sum(1)
    ks = np.exp(-d2 / (2 * area * (2 * SIGMAS) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    for i, p in enumerate(GT):
        if vis[i] == 0:
            continue
        ax.add_patch(Circle(p, s * 2 * SIGMAS[i], fc=GREEN, alpha=0.15, ec=GREEN, lw=1))
    draw_person(ax, GT, vis, lw=2, ms=5, color=GREEN)
    draw_person(ax, flipped, vis, lw=1.5, ms=5, color=PURPLE, alpha=0.9)
    for i in (9, 10):
        ax.annotate(f"{ks[i]:.2f}", flipped[i], xytext=(flipped[i][0] + (-14 if i == 9 else 14), flipped[i][1] + 18),
                    fontsize=10, color=RED, ha="center", arrowprops=dict(arrowstyle="-", color=RED, lw=1))
    ax.set_xlim(20, 180)
    ax.set_ylim(240, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [Line2D([], [], color=GREEN, lw=2, marker="o", ms=5, label="ground truth"),
               Line2D([], [], color=PURPLE, lw=1.5, marker="o", ms=5, label="prediction (wrists swapped)"),
               Circle((0, 0), 1, fc=GREEN, alpha=0.15, ec=GREEN, label=r"radius $s\kappa_i$ (similarity 0.61)")]
    ax.legend(handles=handles, loc="lower center", fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.12))
    ax.set_title(f"(a) toy prediction: OKS = {ks[vis > 0].mean():.3f}", fontsize=11)

    ax = axes[1]
    d = np.linspace(0, 30, 200)
    for k, c, ls in [(0, PINK, "-"), (9, YELLOW, "-"), (5, BLUE, "-"), (11, ORANGE, "-")]:
        ax.plot(d, np.exp(-d ** 2 / (2 * area * (2 * SIGMAS[k]) ** 2)), color=c, lw=2.2, ls=ls,
                label=f"{NAMES[k]} ($\\sigma$ = {SIGMAS[k]:.3f})")
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(30, 0.52, "similarity 0.50", ha="right", fontsize=10, color=MUTED)
    ax.axhline(0.95, color=MUTED, lw=1, ls=":")
    ax.text(30, 0.965, "0.95", ha="right", fontsize=10, color=MUTED)
    ax.set_xlabel("distance from the ground-truth keypoint (pixels)")
    ax.set_ylabel("keypoint similarity")
    ax.set_title(f"(b) similarity vs error for this person (s = {s:.0f} px)", fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "oks-toy.png")


# ---------------------------------------------------------------- Figure 5: grouping toy
J13 = ["nose", "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri",
       "l_hip", "r_hip", "l_knee", "r_knee", "l_ank", "r_ank"]
LIMBS = [(0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 9), (9, 11), (8, 10), (10, 12)]
PA = np.array([[40, 20], [52, 38], [28, 38], [70, 50], [24, 62], [100, 74], [22, 86],
               [48, 80], [32, 80], [50, 110], [30, 110], [52, 140], [28, 140]], float)
PB = np.array([[72, 24], [84, 42], [62, 42], [98, 58], [56, 64], [74, 76], [52, 86],
               [80, 84], [66, 84], [82, 114], [64, 114], [84, 144], [62, 144]], float)
HG, WG, WIDTH = 160, 120, 3.0


def render_paf(c):
    ys, xs = np.mgrid[0:HG, 0:WG]
    field, count = np.zeros((HG, WG, 2)), np.zeros((HG, WG))
    for P in (PA, PB):
        a, b = P[LIMBS[c][0]], P[LIMBS[c][1]]
        v = (b - a) / np.linalg.norm(b - a)
        along = (xs - a[0]) * v[0] + (ys - a[1]) * v[1]
        across = np.abs(-(xs - a[0]) * v[1] + (ys - a[1]) * v[0])
        on = (along >= 0) & (along <= np.linalg.norm(b - a)) & (across <= WIDTH)
        field[on] += v
        count[on] += 1
    return field / np.maximum(count, 1)[..., None]


def limb_score(field, p, q, n=10):
    d = (q - p) / np.linalg.norm(q - p)
    u = np.linspace(0, 1, n)[:, None]
    pts = np.round(p + u * (q - p)).astype(int)
    return np.mean(field[pts[:, 1], pts[:, 0]] @ d)


def fig_grouping():
    rng = np.random.default_rng(0)
    cands = [np.stack([PA[j], PB[j]]) + rng.normal(0, 1.0, (2, 2)) for j in range(13)]
    links = {"paf": [], "dist": []}
    for c, (j1, j2) in enumerate(LIMBS):
        field = render_paf(c)
        S = np.array([[limb_score(field, p, q) for q in cands[j2]] for p in cands[j1]])
        D = np.linalg.norm(cands[j1][:, None] - cands[j2][None], axis=2)
        for key, M in (("paf", -S), ("dist", D)):
            r, k = linear_sum_assignment(M)
            for i, j in zip(r, k):
                links[key].append((cands[j1][i], cands[j2][j], i == j, i))

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 5.0))
    ax = axes[0]
    field = render_paf(3)
    ys, xs = np.mgrid[0:HG:3, 0:WG:3]
    F = field[::3, ::3]
    m = np.linalg.norm(F, axis=2) > 0
    ax.quiver(xs[m], ys[m], F[..., 0][m], F[..., 1][m], color=PURPLE, angles="xy", scale_units="xy",
              scale=0.35, width=0.008)
    for j in range(13):
        ax.plot(*cands[j].T, "o", ms=4, color=MUTED)
    for j, mk in ((3, "s"), (5, "D")):
        ax.plot(*cands[j][0], mk, ms=8, color=BLUE)
        ax.plot(*cands[j][1], mk, ms=8, color=ORANGE)
    ax.set_title("(a) all detected keypoints and the\nPAF of the left forearm", fontsize=11)
    handles = [Line2D([], [], color=BLUE, marker="s", lw=0, label="elbow of A"),
               Line2D([], [], color=ORANGE, marker="s", lw=0, label="elbow of B"),
               Line2D([], [], color=BLUE, marker="D", lw=0, label="wrist of A"),
               Line2D([], [], color=ORANGE, marker="D", lw=0, label="wrist of B")]
    ax.legend(handles=handles, fontsize=10, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.24))
    for ax, key, title in ((axes[1], "paf", "(b) matched by PAF line integral"),
                           (axes[2], "dist", "(c) matched by shortest distance")):
        nbad = 0
        for p, q, good, who in links[key]:
            col = (BLUE if who == 0 else ORANGE) if good else RED
            ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=3 if not good else 2.2, solid_capstyle="round")
            nbad += not good
        for j in range(13):
            ax.plot(*cands[j].T, "o", ms=4, color=INK)
        ax.set_title(f"{title}\n({len(links[key]) - nbad}/{len(links[key])} limbs correct)", fontsize=11)
    for ax in axes:
        ax.set_xlim(5, 115)
        ax.set_ylim(155, 5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[2].legend(handles=[Line2D([], [], color=BLUE, lw=2.2, label="person A"),
                            Line2D([], [], color=ORANGE, lw=2.2, label="person B"),
                            Line2D([], [], color=RED, lw=3, label="wrong limb")],
                   fontsize=10, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    save(fig, "grouping-toy.png")


if __name__ == "__main__":
    fig_skeleton()
    fig_decoding()
    fig_pipelines()
    fig_oks()
    fig_grouping()
