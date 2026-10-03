"""Figures for PR 06 (templates, DTW, structural recognition, HMMs and CRFs).

Writes only into src/assets/pattern-recognition/pr-06-templates-structured-models/.
Run: .venv/bin/python scripts/figures/pr_06_templates_structured_models_extra.py
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from skimage import data

OUT = Path(__file__).resolve().parents[2] / "src/assets/pattern-recognition/pr-06-templates-structured-models"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white"})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def template_matching():
    img = data.camera().astype(np.float32)
    r0, c0, h, w = 300, 200, 40, 40
    tpl = img[r0:r0 + h, c0:c0 + w].copy()
    scene = 0.5 * img + 20.0
    ncc = cv2.matchTemplate(scene, tpl, cv2.TM_CCOEFF_NORMED)
    ssd = cv2.matchTemplate(scene, tpl, cv2.TM_SQDIFF)
    rs, cs = np.unravel_index(ssd.argmin(), ssd.shape)
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.0), gridspec_kw={"width_ratios": [0.45, 1, 1, 1]})
    axes[0].imshow(tpl, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Template\n(original exposure)")
    axes[1].imshow(scene, cmap="gray", vmin=0, vmax=255)
    axes[1].add_patch(Rectangle((c0, r0), w, h, fill=False, ec="#2f855a", lw=2))
    axes[1].add_patch(Rectangle((cs, rs), w, h, fill=False, ec="#c53030", lw=2, ls="--"))
    axes[1].set_title("Scene: 0.5 x image + 20\ngreen = NCC, red = SSD")
    axes[2].imshow(ncc, cmap="magma")
    axes[2].plot(c0, r0, "c+", ms=14, mew=2)
    axes[2].set_title("NCC map (peak = 1.0\nat the true location)")
    axes[3].imshow(-ssd, cmap="magma")
    axes[3].plot(cs, rs, "c+", ms=14, mew=2)
    axes[3].set_title("Negative SSD map\n(peak at a wrong place)")
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    save(fig, "template-matching-ncc.png")


def dtw_path(x, t):
    n, m = len(x), len(t)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = abs(x[i - 1] - t[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return D[1:, 1:], path[::-1]


def dtw_figure():
    s = np.linspace(0, 2 * np.pi, 60)
    a = np.sin(s)
    b = np.sin(s - 0.6)
    D, path = dtw_path(a, b)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, title, pairs in [(axes[0], "Lock-step (Euclidean) pairing", [(i, i) for i in range(0, 60, 3)]),
                             (axes[1], "DTW alignment", path[::3])]:
        ax.plot(a + 1.5, color="#2b6cb0", lw=1.5, label="a")
        ax.plot(b - 1.5, color="#c53030", lw=1.5, label="b (shifted)")
        for i, j in pairs:
            ax.plot([i, j], [a[i] + 1.5, b[j] - 1.5], color="gray", lw=0.6)
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xlabel("time index")
    axes[0].legend(loc="upper right", fontsize=8)
    im = axes[2].imshow(D, origin="lower", cmap="viridis")
    p = np.array(path)
    axes[2].plot(p[:, 1], p[:, 0], color="white", lw=1.5)
    jj = np.arange(60)
    axes[2].fill_between(jj, np.clip(jj - 3, 0, 59), np.clip(jj + 3, 0, 59), color="white", alpha=0.25)
    axes[2].set_title("Cumulative cost D(i, j) and\nwarping path (band of width 3 shaded)")
    axes[2].set_xlabel("index j in b")
    axes[2].set_ylabel("index i in a")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    save(fig, "dtw-alignment.png")


def chain_codes():
    codes = {"square\n000111222333": "000111222333",
             "square + bump\n00011121232333": "00011121232333",
             "rectangle\n000001222223": "000001222223"}
    step = {"0": (1, 0), "1": (0, 1), "2": (-1, 0), "3": (0, -1)}
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for ax, (title, code) in zip(axes, codes.items()):
        x, y = [0], [0]
        for ch in code:
            dx, dy = step[ch]
            x.append(x[-1] + dx)
            y.append(y[-1] + dy)
        ax.plot(x, y, "-o", color="#2b6cb0", ms=4)
        ax.plot(0, 0, "s", color="#c53030", ms=7)
        ax.set_aspect("equal")
        ax.set_xlim(-0.7, 5.7)
        ax.set_ylim(-0.7, 4.7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(6))
        ax.set_yticks(range(5))
        ax.grid(alpha=0.3)
        ax.tick_params(labelbottom=False, labelleft=False)
    fig.suptitle("4-direction chain codes (0 right, 1 up, 2 left, 3 down; red square = start)", y=1.02)
    fig.tight_layout()
    save(fig, "chain-codes.png")


def hmm_figure():
    rng = np.random.default_rng(3)
    A = np.array([[0.8, 0.2, 0.0], [0.0, 0.8, 0.2], [0.2, 0.0, 0.8]])
    B = np.array([[0.7, 0.2, 0.1], [0.15, 0.7, 0.15], [0.1, 0.2, 0.7]])
    pi = np.ones(3) / 3
    T = 60
    z = np.zeros(T, int)
    x = np.zeros(T, int)
    z[0] = rng.choice(3, p=pi)
    x[0] = rng.choice(3, p=B[z[0]])
    for t in range(1, T):
        z[t] = rng.choice(3, p=A[z[t - 1]])
        x[t] = rng.choice(3, p=B[z[t]])
    al = np.zeros((T, 3))
    c = np.zeros(T)
    al[0] = pi * B[:, x[0]]
    c[0] = al[0].sum()
    al[0] /= c[0]
    for t in range(1, T):
        al[t] = B[:, x[t]] * (al[t - 1] @ A)
        c[t] = al[t].sum()
        al[t] /= c[t]
    be = np.ones((T, 3))
    for t in range(T - 2, -1, -1):
        be[t] = A @ (B[:, x[t + 1]] * be[t + 1]) / c[t + 1]
    g = al * be
    with np.errstate(divide="ignore"):
        lA, lB = np.log(A), np.log(B)
    d = np.log(pi) + lB[:, x[0]]
    bk = []
    for t in range(1, T):
        cand = d[:, None] + lA
        bk.append(cand.argmax(0))
        d = cand.max(0) + lB[:, x[t]]
    path = [d.argmax()]
    for b in reversed(bk):
        path.append(b[path[-1]])
    path = np.array(path[::-1])
    cols = ["#2b6cb0", "#2f855a", "#c53030"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [0.8, 0.8, 1.2, 1.2]})
    axes[0].step(range(T), z, where="mid", color="k", label="true state")
    axes[0].step(range(T), path + 0.08, where="mid", color="#dd6b20", ls="--", label="Viterbi path")
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_ylabel("state")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].scatter(range(T), x, c=[cols[k] for k in x], s=14)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_ylabel("symbol")
    for ax, P, title in [(axes[2], al, "filtered p(z_t | x_1..t)"), (axes[3], g, "smoothed p(z_t | x_1..T)")]:
        ax.stackplot(range(T), P.T, colors=cols, alpha=0.8, labels=["state 0", "state 1", "state 2"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("prob.")
        ax.set_title(title, fontsize=10, loc="left")
    axes[2].legend(loc="upper right", fontsize=8, ncol=3)
    axes[3].set_xlabel("time t")
    fig.tight_layout()
    save(fig, "hmm-filter-smooth-viterbi.png")
    return float((path == z).mean()), float((g.argmax(1) == z).mean()), float((al.argmax(1) == z).mean())


def node(ax, xy, text, shaded=False):
    ax.add_patch(Circle(xy, 0.32, fc="#cbd5e0" if shaded else "white", ec="k", lw=1.2))
    ax.text(*xy, text, ha="center", va="center", fontsize=11)


def arrow(ax, p, q, directed=True):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>" if directed else "-", mutation_scale=12,
                                 shrinkA=13, shrinkB=13, color="k", lw=1.1))


def graphs_figure():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.0))
    for ax, title, directed in [(axes[0], "HMM: generative, directed\np(z, x) = p(z_1) p(x_1|z_1) prod p(z_t|z_t-1) p(x_t|z_t)", True),
                                (axes[1], "Linear-chain CRF: discriminative, undirected\np(y | x) = exp(sum of potentials) / Z(x)", False)]:
        lab = "z" if directed else "y"
        for t in range(4):
            node(ax, (t * 1.6, 1.2), f"${lab}_{t + 1}$")
            if directed:
                node(ax, (t * 1.6, 0), f"$x_{t + 1}$", shaded=True)
                arrow(ax, (t * 1.6, 1.2), (t * 1.6, 0))
            if t:
                arrow(ax, ((t - 1) * 1.6, 1.2), (t * 1.6, 1.2), directed)
        if not directed:
            ax.add_patch(Rectangle((-0.5, -0.35), 3 * 1.6 + 1.0, 0.7, fc="#cbd5e0", ec="k"))
            ax.text(1.5 * 1.6, 0, "whole observed sequence  x", ha="center", va="center")
            for t in range(4):
                ax.plot([t * 1.6, t * 1.6], [1.2 - 0.32, 0.35], color="k", lw=1.1)
        ax.set_xlim(-0.7, 5.6)
        ax.set_ylim(-0.6, 1.7)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=9.5)
    fig.tight_layout()
    save(fig, "hmm-vs-crf.png")


if __name__ == "__main__":
    template_matching()
    dtw_figure()
    chain_codes()
    print("HMM accuracies (viterbi, smoothed, filtered):", hmm_figure())
    graphs_figure()
    for f in sorted(OUT.glob("*.png")):
        print(f.name, f.stat().st_size // 1024, "KB")
