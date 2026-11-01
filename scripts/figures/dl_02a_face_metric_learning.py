"""Figures for DLCV 02A - Face Recognition and Metric Learning.

Run:  .venv/bin/python scripts/figures/dl_02a_face_metric_learning.py
Writes only to src/assets/deep-learning/dl-02a-face-metric-learning/.
All data are synthetic (random embeddings, Gaussian score distributions, a small
NumPy network trained on synthetic clusters); no face images are used or downloaded.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Wedge  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-02a-face-metric-learning"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})

BLUE, ORANGE, GREEN, YELLOW, PINK, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"
INK, MUTED, GRID, RED = "#0b0b0b", "#52514e", "#d9d8d4", "#c0392b"
PALETTE = [BLUE, ORANGE, GREEN, YELLOW, PINK, PURPLE]


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    kb = path.stat().st_size / 1024
    w = plt.imread(path).shape[1]
    print(f"{name}: {kb:.0f} KB, width {w} px")
    assert kb < 1024 and w <= 1600


def unit(v, ax):
    return v / np.linalg.norm(v, axis=ax, keepdims=True)


# ---------------------------------------------------------------- 1. verification vs identification
def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal")


def arrow(ax, p, q, color=INK, text=None, off=(0, 0.05), ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.3, color=color, linestyle=ls))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha="center", va="bottom",
                fontsize=9, color=color)


def fig_open_closed():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    titles = ["(a) Verification (1:1)", "(b) Closed-set identification (1:N)", "(c) Open-set identification (1:N)"]
    for ax, t in zip(axes, titles):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(t, fontsize=12, weight="bold")

    ax = axes[0]
    box(ax, 0.2, 0.72, 0.34, 0.16, "probe\nembedding", fc="#eaf2fb")
    box(ax, 0.2, 0.30, 0.34, 0.16, "enrolled template\nof claimed ID", fc="#fdf0e9")
    box(ax, 0.72, 0.51, 0.36, 0.16, "score s:\ncosine similarity")
    arrow(ax, (0.37, 0.70), (0.54, 0.55))
    arrow(ax, (0.37, 0.32), (0.54, 0.47))
    box(ax, 0.72, 0.18, 0.40, 0.14, "s >= t: accept\ns < t: reject", fc="#eef8f3", fs=10)
    arrow(ax, (0.72, 0.43), (0.72, 0.26))
    ax.text(0.5, 0.94, "Is this person who they claim to be?", ha="center", fontsize=10, color=MUTED)

    for k, ax in enumerate(axes[1:]):
        box(ax, 0.135, 0.55, 0.22, 0.16, "probe\nembedding", fc="#eaf2fb")
        names = ["ID 1", "ID 2", "ID 3", "...", "ID N"]
        ys = np.linspace(0.84, 0.22, 5)
        scores = ["0.12", "0.71", "0.08", "", "0.05"] if k == 0 else ["0.12", "0.19", "0.08", "", "0.05"]
        for nm, yy, sc in zip(names, ys, scores):
            if nm == "...":
                ax.text(0.46, yy, "...", ha="center", va="center", fontsize=12)
                continue
            hl = (k == 0 and nm == "ID 2")
            box(ax, 0.46, yy, 0.17, 0.10, nm, fc="#fdf0e9" if not hl else "#ffe08a", bold=hl)
            arrow(ax, (0.26, 0.55), (0.37, yy), color=GRID)
            ax.text(0.575, yy, sc, ha="left", va="center", fontsize=9, color=MUTED)
        ax.text(0.46, 0.06, "gallery of N enrolled people", ha="center", fontsize=9, color=MUTED)
        if k == 0:
            box(ax, 0.85, 0.55, 0.26, 0.22, "answer:\nbest match\n(ID 2)", fc="#eef8f3", fs=10)
            ax.text(0.5, 0.94, "Which of the N people is this?", ha="center", fontsize=10, color=MUTED)
        else:
            box(ax, 0.85, 0.55, 0.26, 0.28, "best score\n0.19 < t:\n\"not in\ngallery\"", fc="#fbeaea", fs=10)
            ax.text(0.5, 0.94, "Is this anyone we know? If so, who?", ha="center", fontsize=10, color=MUTED)
    fig.tight_layout()
    save(fig, "open-closed-set.png")


# ---------------------------------------------------------------- 2. margin geometry on the circle
def target_logit(theta, kind):
    if kind == "softmax":
        return np.cos(theta)
    if kind == "sphereface":           # psi(theta) for m = 2
        m = 2
        k = np.floor(theta * m / np.pi)
        return (-1) ** k * np.cos(m * theta) - 2 * k
    if kind == "cosface":
        return np.cos(theta) - 0.35
    if kind == "arcface":
        return np.cos(theta + 0.5)
    raise ValueError(kind)


def fig_margin_geometry():
    phi = np.radians(80)                       # angle between the two class centres
    alphas = np.radians(np.linspace(-180, 180, 3601))
    kinds = [("softmax", "Normalized softmax (no margin)"), ("sphereface", "SphereFace, m = 2"),
             ("cosface", "CosFace, m = 0.35"), ("arcface", "ArcFace, m = 0.5")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.6))
    axes = axes.ravel()
    for ax, (kind, title) in zip(axes, kinds):
        th1 = np.arccos(np.cos(alphas))        # angle in [0, pi] to centre W1 (at 0 deg)
        th2 = np.arccos(np.cos(alphas - phi))  # angle in [0, pi] to centre W2 (at 80 deg)
        ok1 = target_logit(th1, kind) > np.cos(th2)
        ok2 = target_logit(th2, kind) > np.cos(th1)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=GRID, lw=1))
        for mask, col in [(ok1, BLUE), (ok2, ORANGE), (~ok1 & ~ok2, "#9a9994")]:
            a = np.where(mask, alphas, np.nan)
            ax.plot(np.cos(a), np.sin(a), color=col, lw=7, solid_capstyle="butt")
        for ang, col, lab in [(0, BLUE, "$W_1$"), (phi, ORANGE, "$W_2$")]:
            ax.annotate("", xy=(0.9 * np.cos(ang), 0.9 * np.sin(ang)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
            ax.text(0.55 * np.cos(ang) - 0.17 * np.sin(ang) - 0.05, 0.55 * np.sin(ang) + 0.13 * np.cos(ang) - 0.04,
                    lab, color=col, fontsize=13)
        gap = np.where(~ok1 & ~ok2 & (alphas > 0) & (alphas < phi), alphas, np.nan)
        width = np.degrees(np.nanmax(gap) - np.nanmin(gap)) if np.isfinite(gap).any() else 0.0
        ax.set_title(f"{title}\ngap between classes: {width:.0f} deg", fontsize=11)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
    handles = [plt.Line2D([], [], color=BLUE, lw=6), plt.Line2D([], [], color=ORANGE, lw=6),
               plt.Line2D([], [], color="#9a9994", lw=6)]
    fig.legend(handles, ["loss satisfied for class 1", "loss satisfied for class 2",
                         "neither class satisfied (margin zone)"],
               loc="lower center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "margin-geometry.png")


# ---------------------------------------------------------------- 3. toy embeddings, softmax vs ArcFace
def toy_data(seed=0, C=6, n_per=50, D=16, K=4):
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.normal(size=(D, D)))[0]
    ids = rng.normal(size=(C, K))
    y = np.repeat(np.arange(C), n_per)
    x = ids[y] @ U[:, :K].T + 1.5 * rng.normal(size=(len(y), D - K)) @ U[:, K:].T + 0.3 * rng.normal(size=(len(y), D))
    return x, y


def train(x, y, C, d=2, margin=None, s=16.0, steps=800, lr=0.2, H=64):
    """Same model and update rule as the chapter's code block."""
    r = np.random.default_rng(1)
    D = x.shape[1]
    W1 = r.normal(size=(D, H)) / 4
    b1 = np.zeros(H)
    W2 = r.normal(size=(H, d)) / 8
    W = r.normal(size=(d, C))
    Y = np.eye(C)[y]
    for _ in range(steps):
        h = np.maximum(x @ W1 + b1, 0)
        z = h @ W2
        if margin is None:
            L = z @ W
        else:
            f, w = unit(z, 1), unit(W, 0)
            th = np.arccos(np.clip(f @ w, -1 + 1e-7, 1 - 1e-7))
            L = s * np.where(Y > 0, np.cos(th + margin), np.cos(th))
        p = np.exp(L - L.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        g = (p - Y) / len(y)
        if margin is None:
            gz, gW = g @ W.T, z.T @ g
        else:
            gc = g * s * np.where(Y > 0, np.sin(th + margin) / np.sin(th), 1.0)
            gf, gw = gc @ w.T, f.T @ gc
            gz = (gf - (gf * f).sum(1, keepdims=True) * f) / np.linalg.norm(z, axis=1, keepdims=True)
            gW = (gw - (gw * w).sum(0, keepdims=True) * w) / np.linalg.norm(W, axis=0, keepdims=True)
        gh = (gz @ W2.T) * (h > 0)
        W1 -= lr * x.T @ gh
        b1 -= lr * gh.sum(0)
        W2 -= lr * h.T @ gz
        W -= lr * gW
    return (lambda xx: np.maximum(xx @ W1 + b1, 0) @ W2), W


def intra_angle(e, y):
    e = unit(e, 1)
    cen = unit(np.stack([e[y == j].mean(0) for j in np.unique(y)]), 1)
    return np.degrees(np.arccos(np.clip((e * cen[y]).sum(1), -1, 1))).mean()


def fig_toy_embeddings():
    x, y = toy_data()
    emb_s, _ = train(x, y, 6, margin=None)
    emb_a, W_a = train(x, y, 6, margin=0.5)
    zs, za = emb_s(x), emb_a(x)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.0))
    cols = np.array(PALETTE)[y]
    ax = axes[0]
    ax.scatter(zs[:, 0], zs[:, 1], c=cols, s=12, alpha=0.8)
    ax.set_title("(a) Softmax: raw 2-D features", fontsize=11)
    ax.set_aspect("equal")
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    for ax, z, title in [(axes[1], zs, "(b) Softmax features, normalized"), (axes[2], za, "(c) ArcFace (m = 0.5), normalized")]:
        f = unit(z, 1)
        rad = 1 + 0.06 * (y - 2.5) / 2.5          # small radial offset per class so overlaps stay visible
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color=GRID, lw=1))
        ax.scatter(f[:, 0] * rad, f[:, 1] * rad, c=cols, s=12, alpha=0.8)
        ax.set_title(f"{title}\nmean angle to class centre: {intra_angle(z, y):.1f} deg", fontsize=11)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")
    axes[0].set_title(f"(a) Softmax: raw 2-D features\nmean angle to class centre: {intra_angle(zs, y):.1f} deg", fontsize=11)
    fig.tight_layout()
    save(fig, "toy-embeddings.png")


# ---------------------------------------------------------------- 4/5. scores, ROC and DET
def scores(seed=0, mg=0.55, sg=0.13, mi=0.02, si=0.09, ng=5_000, ni=500_000):
    rng = np.random.default_rng(seed)
    return (np.clip(rng.normal(mg, sg, ng), -1, 1), np.clip(rng.normal(mi, si, ni), -1, 1))


def fig_histograms():
    gen, imp = scores()
    t2, t4 = np.quantile(imp, 1 - 1e-2), np.quantile(imp, 1 - 1e-4)
    bins = np.linspace(-0.45, 1.0, 146)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, logy in zip(axes, [False, True]):
        ax.hist(imp, bins=bins, density=True, color=ORANGE, alpha=0.55, label="impostor (different people)")
        ax.hist(gen, bins=bins, density=True, color=BLUE, alpha=0.55, label="genuine (same person)")
        for t, lab, ls in [(t2, "t for FAR = 1e-2", "--"), (t4, "t for FAR = 1e-4", "-")]:
            ax.axvline(t, color=INK, ls=ls, lw=1.4, label=f"{lab} ({t:.2f})")
        ax.set_xlabel("cosine similarity score")
        ax.set_ylabel("density" + (" (log scale)" if logy else ""))
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 10)
            ax.set_title("(b) Same data, log scale: the impostor tail decides t", fontsize=11)
        else:
            ax.set_title("(a) Genuine and impostor score distributions", fontsize=11)
            ax.legend(fontsize=9, frameon=False, loc="upper right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "score-histograms.png")


def fig_roc_det():
    sysA = scores()
    sysB = scores(seed=1, mg=0.45, sg=0.15)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for (gen, imp), col, lab in [(sysA, BLUE, "system A"), (sysB, ORANGE, "system B")]:
        yy = np.r_[np.ones(gen.size), np.zeros(imp.size)]
        fpr, tpr, _ = roc_curve(yy, np.r_[gen, imp])
        keep = fpr > 0
        axes[0].plot(fpr[keep], tpr[keep], color=col, lw=2, label=lab)
        i = np.searchsorted(fpr, 1e-3, side="right") - 1
        axes[0].plot(fpr[i], tpr[i], "o", color=col)
        axes[0].annotate(f"TAR@FAR=1e-3: {tpr[i]:.3f}", (fpr[i], tpr[i]), textcoords="offset points",
                         xytext=(8, -16), fontsize=9, color=col)
        fnr = 1 - tpr
        ok = (fpr > 1e-5) & (fnr > 1e-4) & (fpr < 0.5) & (fnr < 0.5)
        axes[1].plot(norm.ppf(fpr[ok]), norm.ppf(fnr[ok]), color=col, lw=2, label=lab)
    ax = axes[0]
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(0.5, 1.005)
    ax.set_xlabel("false accept rate (FAR), log scale")
    ax.set_ylabel("true accept rate (TAR)")
    ax.set_title("(a) ROC with a log FAR axis", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, loc="lower right")
    ax = axes[1]
    ticks = np.array([1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.4])
    ax.set_xticks(norm.ppf(ticks))
    ax.set_xticklabels(["1e-5", "1e-4", "1e-3", "1e-2", "0.05", "0.2", "0.4"])
    yt = np.array([1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.4])
    ax.set_yticks(norm.ppf(yt))
    ax.set_yticklabels(["1e-4", "1e-3", "1e-2", "0.05", "0.2", "0.4"])
    ax.set_xlim(norm.ppf(1e-5), norm.ppf(0.5))
    ax.set_ylim(norm.ppf(1e-4), norm.ppf(0.5))
    ax.plot([norm.ppf(1e-5), norm.ppf(0.5)], [norm.ppf(1e-5), norm.ppf(0.5)], color=GRID, lw=1, ls=":")
    ax.text(norm.ppf(0.012), norm.ppf(0.018), "EER line", color=MUTED, fontsize=9, rotation=40)
    ax.set_xlabel("false accept rate (normal-deviate scale)")
    ax.set_ylabel("false reject rate (normal-deviate scale)")
    ax.set_title("(b) DET curve: both error rates, lower-left is better", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "roc-det.png")


# ---------------------------------------------------------------- 6. subgroup thresholds
def fig_subgroups():
    rng = np.random.default_rng(0)
    n_imp, n_gen = 400_000, 4_000
    imp = {"A": rng.normal(0.00, 0.09, n_imp), "B": rng.normal(0.06, 0.09, n_imp)}
    gen = {"A": rng.normal(0.55, 0.13, n_gen), "B": rng.normal(0.55, 0.13, n_gen)}
    t_global = np.quantile(np.r_[imp["A"], imp["B"]], 1 - 1e-3)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    bins = np.linspace(-0.45, 1.0, 146)
    ax.hist(imp["A"], bins=bins, density=True, histtype="step", lw=2, color=BLUE, label="impostors, group A")
    ax.hist(imp["B"], bins=bins, density=True, histtype="step", lw=2, color=ORANGE, label="impostors, group B")
    ax.hist(np.r_[gen["A"], gen["B"]], bins=bins, density=True, color=GREEN, alpha=0.35, label="genuine (both groups)")
    ax.axvline(t_global, color=INK, lw=1.5, label=f"one global threshold ({t_global:.3f})")
    ax.set_xlabel("cosine similarity score")
    ax.set_ylabel("density")
    ax.set_title("(a) Group B's impostor scores sit slightly higher", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ts = np.linspace(0.15, 0.45, 301)
    for g, col in [("A", BLUE), ("B", ORANGE)]:
        s = np.sort(imp[g])
        fmr = 1 - np.searchsorted(s, ts) / s.size
        ax.plot(ts, np.maximum(fmr, 1e-6), color=col, lw=2, label=f"FMR, group {g}")
        f_at = (imp[g] >= t_global).mean()
        ax.plot(t_global, f_at, "o", color=col)
        ax.annotate(f"{f_at:.2e}", (t_global, f_at), textcoords="offset points", xytext=(8, 4), fontsize=9, color=col)
    ax.axhline(1e-3, color=MUTED, ls="--", lw=1, label="target FMR 1e-3 (pooled)")
    ax.axvline(t_global, color=INK, lw=1.5)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("threshold t")
    ax.set_ylabel("false match rate (log scale)")
    ax.set_title("(b) Same threshold, about 10 times more false matches in B", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    save(fig, "subgroup-thresholds.png")


# ---------------------------------------------------------------- 7. identification: CMC and FPIR
def fig_identification():
    rng = np.random.default_rng(0)
    dim, noise, n_probe, thr = 128, 1.1, 300, 0.36
    sizes = [300, 3_000, 30_000]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    fpirs = []
    for n, col in zip(sizes, [BLUE, ORANGE, PURPLE]):
        people = rng.normal(size=(n + n_probe, dim))
        gallery = unit(people[:n] + noise * rng.normal(size=(n, dim)), 1)
        mated = rng.choice(n, n_probe, replace=False)
        p_in = unit(people[mated] + noise * rng.normal(size=(n_probe, dim)), 1)
        p_out = unit(people[n:] + noise * rng.normal(size=(n_probe, dim)), 1)
        S_in = p_in @ gallery.T
        true_s = S_in[np.arange(n_probe), mated]
        rank = (S_in > true_s[:, None]).sum(1) + 1
        ks = np.arange(1, 21)
        cmc = [(rank <= k).mean() for k in ks]
        axes[0].plot(ks, cmc, "o-", color=col, ms=4, lw=2, label=f"gallery size {n:,}")
        fpirs.append((p_out @ gallery.T).max(1))
    axes[0].set_xlabel("rank k")
    axes[0].set_ylabel("CMC(k): true match within top k")
    axes[0].set_title("(a) Closed set: CMC falls as the gallery grows", fontsize=11)
    axes[0].set_xticks([1, 5, 10, 15, 20])
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False, loc="lower right")
    ts = np.linspace(0.25, 0.5, 251)
    for top, n, col in zip(fpirs, sizes, [BLUE, ORANGE, PURPLE]):
        axes[1].plot(ts, [(top >= t).mean() for t in ts], color=col, lw=2, label=f"gallery size {n:,}")
    axes[1].axvline(thr, color=INK, ls="--", lw=1.2, label=f"threshold {thr}")
    axes[1].set_xlabel("threshold t")
    axes[1].set_ylabel("FPIR: non-enrolled searches with a hit")
    axes[1].set_title("(b) Open set: more people enrolled, more false alarms", fontsize=11)
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "identification-gallery.png")


if __name__ == "__main__":
    fig_open_closed()
    fig_margin_geometry()
    fig_toy_embeddings()
    fig_histograms()
    fig_roc_det()
    fig_subgroups()
    fig_identification()
