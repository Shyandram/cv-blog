"""Figures for DLCV 11 - Robustness: Distribution Shift, Adversarial Examples and Anomalies.

Run:  .venv/bin/python scripts/figures/dl_11_robustness.py
Writes PNGs to src/assets/deep-learning/dl-11-robustness/ only.
Data: scikit-learn's digits, skimage.data images (camera, moon, coins, text, brick) and synthetic
points; nothing is downloaded and no network weights are used. Each experiment repeats the code
of the matching snippet in the chapter with the same seeds and the same order of random draws, so
the plotted values equal the printed ones. Runs in under a minute on a CPU.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.special import logsumexp, softmax  # noqa: E402
from skimage import data, draw, transform  # noqa: E402
from skimage.util import view_as_windows  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-11-robustness"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.facecolor": "white"})
BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(path).convert("RGB")
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    w, h = Image.open(path).size
    assert w <= 1600, f"{name} is {w} px wide"
    assert path.stat().st_size < 1024 * 1024, f"{name} is larger than 1 MB"
    print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB, {w}x{h} px)")


def load():
    X, y = load_digits(return_X_y=True)
    return X / 16.0, y


# ---------------------------------------------------------------- Figure 1: three problems
def fig_map():
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8))
    a0 = rng.normal([-1.2, 0], 0.45, (60, 2))
    a1 = rng.normal([1.2, 0], 0.45, (60, 2))
    boundary_x = 0.0

    # (a) distribution shift
    ax = axes[0]
    ax.scatter(*a0.T, s=14, c=BLUE, alpha=0.5, label="training, class A")
    ax.scatter(*a1.T, s=14, c=ORANGE, alpha=0.5, label="training, class B")
    s0 = rng.normal([-0.4, 1.3], 0.45, (40, 2))
    s1 = rng.normal([1.9, 1.3], 0.45, (40, 2))
    ax.scatter(*s0.T, s=22, marker="^", facecolors="none", edgecolors=BLUE, label="deployment, class A")
    ax.scatter(*s1.T, s=22, marker="^", facecolors="none", edgecolors=ORANGE, label="deployment, class B")
    ax.axvline(boundary_x, color="k", lw=1.2, ls="--")
    ax.annotate("", xy=(0.9, 1.3), xytext=(-0.6, 0.0), arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))
    ax.set_title("(a) Distribution shift\nsame task, the data moved", fontsize=11)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    # (b) adversarial example
    ax = axes[1]
    ax.scatter(*a0.T, s=14, c=BLUE, alpha=0.5)
    ax.scatter(*a1.T, s=14, c=ORANGE, alpha=0.5)
    ax.axvline(boundary_x, color="k", lw=1.2, ls="--")
    x0 = np.array([-0.35, 0.6])
    eps = 0.5
    ax.add_patch(Rectangle(x0 - eps, 2 * eps, 2 * eps, fill=False, ec=RED, lw=1.5))
    ax.scatter(*x0, s=60, c=BLUE, edgecolors="k", zorder=5)
    xa = x0 + np.array([eps, eps * 0.3])
    ax.scatter(*xa, s=90, marker="*", c=RED, edgecolors="k", zorder=6)
    ax.annotate("", xy=xa, xytext=x0, arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax.text(x0[0] - eps, x0[1] + eps + 0.12, r"$\ell_\infty$ ball of radius $\varepsilon$", color=RED, fontsize=9)
    ax.annotate("adversarial example\n(now classified B)", xy=xa, xytext=(0.6, 2.4), color=RED, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax.set_title("(b) Adversarial example\nsame input, worst-case tiny change", fontsize=11)

    # (c) out-of-distribution input
    ax = axes[2]
    ax.scatter(*a0.T, s=14, c=BLUE, alpha=0.5)
    ax.scatter(*a1.T, s=14, c=ORANGE, alpha=0.5)
    ax.axvline(boundary_x, color="k", lw=1.2, ls="--")
    o = rng.normal([2.6, 2.2], 0.3, (25, 2))
    ax.scatter(*o.T, s=26, marker="x", c=PURPLE, label="new kind of input")
    ax.text(1.3, 2.95, 'classified "B" with\nhigh confidence', color=PURPLE, fontsize=9)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax.set_title("(c) Out-of-distribution input\nnot from any known class", fontsize=11)

    for ax in axes:
        ax.set_xlim(-3, 3.6)
        ax.set_ylim(-1.8, 3.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("feature 1")
        ax.set_ylabel("feature 2")
    fig.tight_layout()
    save(fig, "three-problems.png")


# ---------------------------------------------------------------- Figure 2: corruptions
def fig_shift():
    X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    rng = np.random.default_rng(0)
    corruptions = {
        "noise": lambda X, s: np.clip(X + rng.normal(0, 0.1 * s, X.shape), 0, 1),
        "blur": lambda X, s: gaussian_filter(X.reshape(-1, 8, 8), (0, 0.25 * s, 0.25 * s)).reshape(-1, 64),
        "contrast": lambda X, s: 0.5 + (X - 0.5) * (1 - 0.15 * s),
    }

    def norm(X, ref):
        return (X - ref.mean(0)) / (ref.std(0) + 0.05)

    def fit(X, y):
        return MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=400, random_state=0).fit(norm(X, Xtr), y)

    plain = fit(Xtr, ytr)
    s_rand = rng.integers(1, 6, (len(Xtr), 1))
    augmented = fit(np.vstack([Xtr, corruptions["noise"](Xtr, s_rand)]), np.r_[ytr, ytr])
    res = {}
    for name, corrupt in corruptions.items():
        rows = {"plain": [], "noise-aug": [], "test-norm": []}
        for s in range(6):
            Xs = Xte if s == 0 else corrupt(Xte, s)
            rows["plain"].append(plain.score(norm(Xs, Xtr), yte))
            rows["noise-aug"].append(augmented.score(norm(Xs, Xtr), yte))
            rows["test-norm"].append(plain.score(norm(Xs, Xs), yte))
        res[name] = rows
        for k, v in rows.items():
            print(f"  {name:8s} {k:9s}", " ".join(f"{a:.2f}" for a in v))

    fig = plt.figure(figsize=(10.4, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2.4], hspace=0.12, wspace=0.12)
    img = Xte[3]
    ex_rng = np.random.default_rng(1)
    show = {
        "noise": lambda x, s: np.clip(x + ex_rng.normal(0, 0.1 * s, x.shape), 0, 1),
        "blur": lambda x, s: gaussian_filter(x.reshape(8, 8), 0.25 * s).ravel(),
        "contrast": lambda x, s: 0.5 + (x - 0.5) * (1 - 0.15 * s),
    }
    labels = {"plain": ("plain model", GRAY, "-"), "noise-aug": ("trained with noise augmentation", BLUE, "-"),
              "test-norm": ("plain model + test-batch normalization", GREEN, "--")}
    for j, name in enumerate(corruptions):
        sub = gs[0, j].subgridspec(1, 4, wspace=0.08)
        for i, s in enumerate([0, 1, 3, 5]):
            ax = fig.add_subplot(sub[0, i])
            ax.imshow((img if s == 0 else show[name](img, s)).reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"s = {s}", fontsize=9)
            if i == 0:
                ax.set_ylabel(name, fontsize=10)
        ax = fig.add_subplot(gs[1, j])
        for k, (lab, col, ls) in labels.items():
            ax.plot(range(6), res[name][k], marker="o", color=col, ls=ls, label=lab)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("severity s")
        if j == 0:
            ax.set_ylabel("test accuracy")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{name}", fontsize=10)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, loc="lower left")
    save(fig, "corruption-severity.png")


# ---------------------------------------------------------------- Figure 3: adversarial
def fig_adv():
    X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    rng = np.random.default_rng(0)

    def input_grad(m, X, y):
        (W1, W2), (b1, b2) = m.coefs_, m.intercepts_
        h = np.maximum(X @ W1 + b1, 0)
        z = h @ W2 + b2
        g = np.exp(z - z.max(1, keepdims=True))
        g /= g.sum(1, keepdims=True)
        g[np.arange(len(y)), y] -= 1
        return ((g @ W2.T) * (h > 0)) @ W1.T

    def fgsm(m, X, y, eps):
        return np.clip(X + eps * np.sign(input_grad(m, X, y)), 0, 1)

    def pgd(m, X, y, eps, steps=10, r=None):
        r = r if r is not None else rng
        alpha = 2.5 * eps / steps
        Xa = np.clip(X + r.uniform(-eps, eps, X.shape), 0, 1)
        for _ in range(steps):
            Xa = Xa + alpha * np.sign(input_grad(m, Xa, y))
            Xa = np.clip(np.clip(Xa, X - eps, X + eps), 0, 1)
        return Xa

    def train(adversarial, epochs=50, eps=0.15):
        m = MLPClassifier(hidden_layer_sizes=(128,), random_state=0)
        m.partial_fit(Xtr, ytr, classes=np.arange(10))
        for _ in range(epochs - 1):
            m.partial_fit(pgd(m, Xtr, ytr, eps, steps=5) if adversarial else Xtr, ytr)
        return m

    standard, robust = train(False), train(True)
    epss = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]       # the first five equal the snippet's table
    curves = {k: [] for k in ("sf", "sp", "rf", "rp")}
    for eps in epss:
        acc = [m.score(atk(m, Xte, yte, eps), yte) for m in (standard, robust) for atk in (fgsm, pgd)]
        for k, a in zip(curves, acc):
            curves[k].append(a)
        print(f"  eps={eps:.2f}", " ".join(f"{a:.3f}" for a in acc))

    ex = np.random.default_rng(5)
    i = next(i for i in range(len(Xte))
             if standard.predict(Xte[i:i + 1])[0] == yte[i]
             and standard.predict(pgd(standard, Xte[i:i + 1], yte[i:i + 1], 0.1, r=ex))[0] != yte[i])
    x = Xte[i:i + 1]
    xa = pgd(standard, x, yte[i:i + 1], 0.1, r=np.random.default_rng(6))
    xr = pgd(robust, x, yte[i:i + 1], 0.1, r=np.random.default_rng(6))

    fig = plt.figure(figsize=(10.4, 4.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 3.4], wspace=0.3, hspace=0.35)
    panels = [(x, f"clean digit\nstandard model: {standard.predict(x)[0]}", "gray_r", (0, 1)),
              (xa - x, "PGD perturbation\n(eps = 0.1)", "RdBu_r", (-0.1, 0.1)),
              (xa, f"after attack\nstandard model: {standard.predict(xa)[0]}", "gray_r", (0, 1)),
              (xr, f"attack on robust\nrobust model: {robust.predict(xr)[0]}", "gray_r", (0, 1))]
    for j, (im, title, cmap, (lo, hi)) in enumerate(panels):
        ax = fig.add_subplot(gs[j // 2, j % 2])
        ax.imshow(im.reshape(8, 8), cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[:, 2])
    ax.plot(epss, curves["sf"], "o--", color=GRAY, label="standard, FGSM")
    ax.plot(epss, curves["sp"], "o-", color="k", label="standard, PGD-10")
    ax.plot(epss, curves["rf"], "s--", color=BLUE, label="adversarially trained, FGSM")
    ax.plot(epss, curves["rp"], "s-", color=RED, label="adversarially trained, PGD-10")
    ax.axvline(0.15, color=GRAY, lw=0.8, ls=":")
    ax.text(0.155, 0.97, "eps used in training", fontsize=8, color=GRAY)
    ax.set_xlabel(r"$\ell_\infty$ budget $\varepsilon$ (pixels in [0, 1])")
    ax.set_ylabel("test accuracy under attack")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "adversarial-eps.png")


# ---------------------------------------------------------------- Figure 4: OOD scores
def fig_ood():
    X, y = load()
    known = y < 5
    Xtr, Xid, ytr, yid = train_test_split(X[known], y[known], test_size=0.3, stratify=y[known], random_state=0)
    X_near = X[~known]
    rng = np.random.default_rng(0)
    photos = [transform.resize(im / 255.0, (256, 256)) for im in (data.camera(), data.moon(), data.coins(), data.text())]
    X_far = []
    for _ in range(600):
        im = photos[rng.integers(4)]
        r, c = rng.integers(0, 224, 2)
        p = transform.resize(im[r:r + 32, c:c + 32], (8, 8), anti_aliasing=True)
        X_far.append(((p - p.min()) / (np.ptp(p) + 1e-8)).ravel())
    X_far = np.array(X_far)
    m = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=500, random_state=0).fit(Xtr, ytr)
    (W1, W2), (b1, b2) = m.coefs_, m.intercepts_

    def feat(X):
        return np.maximum(X @ W1 + b1, 0)

    def unit(F):
        return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)

    F = feat(Xtr)
    mu = np.stack([F[ytr == c].mean(0) for c in range(5)])
    cov = sum(np.cov(F[ytr == c].T, bias=True) * np.sum(ytr == c) for c in range(5)) / len(F)
    P = np.linalg.inv(cov + 1e-3 * np.eye(len(cov)))

    def scores(X):
        H = feat(X)
        z = H @ W2 + b2
        d = H[:, None] - mu[None]
        return {"MSP": softmax(z, axis=1).max(1),
                "energy": logsumexp(z, axis=1),
                "Mahalanobis": -np.einsum("ncd,de,nce->nc", d, P, d).min(1),
                "kNN": -np.sort(np.linalg.norm(unit(H)[:, None] - unit(F)[None], axis=2), 1)[:, 9]}

    s_id, s_near, s_far = scores(Xid), scores(X_near), scores(X_far)
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.4))
    xlabels = {"MSP": "max softmax probability", "energy": "negative energy (logsumexp of logits)",
               "Mahalanobis": "-log10(min. squared Mahalanobis dist.)", "kNN": "minus 10th-NN distance"}
    for d in (s_id, s_near, s_far):                   # a monotone change of scale for display only
        d["Mahalanobis"] = -np.log10(-d["Mahalanobis"])
    for ax, k in zip(axes, s_id):
        allv = np.r_[s_id[k], s_near[k], s_far[k]]
        lo, hi = allv.min(), allv.max()
        bins = np.linspace(lo, hi, 40)
        for v, lab, col in [(s_id[k], "ID digits 0-4", BLUE), (s_near[k], "near-OOD digits 5-9", ORANGE),
                            (s_far[k], "far-OOD photo patches", PURPLE)]:
            ax.hist(v, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=col, label=lab)
        t = np.quantile(s_id[k], 0.05)
        ax.axvline(t, color="k", ls="--", lw=1)
        an = [roc_auc_score(np.r_[np.ones(len(Xid)), np.zeros(len(o))], np.r_[s_id[k], o]) for o in (s_near[k], s_far[k])]
        ax.set_title(f"{k}\nAUROC near {an[0]:.3f}, far {an[1]:.3f}", fontsize=10)
        ax.set_xlabel(xlabels[k], fontsize=9)
        ax.set_yticks([])
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("density")
    fig.tight_layout()
    save(fig, "ood-scores.png")


# ---------------------------------------------------------------- Figure 5: anomaly maps
def fig_anomaly():
    rng = np.random.default_rng(0)
    tex = transform.rescale(data.brick() / 255.0, 0.5)
    S, P = 64, 7

    def crop(top):
        r = rng.integers(0, 128 - S) + (0 if top else 128)
        c = rng.integers(0, 256 - S)
        return tex[r:r + S, c:c + S].copy()

    def scratch(im):
        r, c = rng.integers(16, S - 16, 2)
        a = rng.uniform(0, np.pi)
        L = rng.integers(8, 16)
        rr, cc = draw.line(r, c, int(r + L * np.sin(a)), int(c + L * np.cos(a)))
        mask = np.zeros_like(im, bool)
        mask[rr, cc] = True
        im[mask] = np.clip(im[mask] + rng.choice([-0.5, 0.5]), 0, 1)
        return im, mask

    def patches(im, step=1):
        return view_as_windows(np.pad(im, P // 2, mode="reflect"), (P, P), step).reshape(-1, P * P)

    def greedy_coreset(M, n):
        keep, d = [0], np.linalg.norm(M - M[0], axis=1)
        for _ in range(n - 1):
            i = int(d.argmax())
            keep.append(i)
            d = np.minimum(d, np.linalg.norm(M - M[i], axis=1))
        return M[keep]

    bank = np.vstack([patches(crop(top=True), step=2) for _ in range(16)])
    memory = NearestNeighbors(n_neighbors=1).fit(greedy_coreset(bank, len(bank) // 10))
    ims, masks, maps, is_defect, img_score, pix_s, pix_t = [], [], [], [], [], [], []
    for k in range(40):
        im, mask = scratch(crop(top=False)) if k % 2 else (crop(top=False), np.zeros((S, S), bool))
        amap = memory.kneighbors(patches(im))[0].reshape(S, S)
        ims.append(im)
        masks.append(mask)
        maps.append(amap)
        is_defect.append(k % 2)
        img_score.append(amap.max())
        pix_s.append(amap.ravel())
        pix_t.append(mask.ravel())
    auc_i = roc_auc_score(is_defect, img_score)
    auc_p = roc_auc_score(np.concatenate(pix_t), np.concatenate(pix_s))
    print(f"  image AUROC {auc_i:.3f}  pixel AUROC {auc_p:.3f}")

    show = [1, 3, 5, 0]                                 # three defective test images and one normal one
    vmax = np.quantile(np.concatenate(pix_s), 0.999)
    fig = plt.figure(figsize=(10.4, 4.4))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.7], hspace=0.3, wspace=0.18)
    for j, k in enumerate(show):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(ims[k], cmap="gray", vmin=0, vmax=1)
        if masks[k].any():
            ax.contour(masks[k], levels=[0.5], colors=[RED], linewidths=0.8)
        ax.set_title(("defective" if is_defect[k] else "normal") + f" test image\nscore {img_score[k]:.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(maps[k], cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title("anomaly map", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[:, 4])
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.04, pos.y0, pos.width - 0.04, pos.height])
    sc = np.array(img_score)
    lab = np.array(is_defect)
    bins = np.linspace(sc.min(), sc.max(), 16)
    ax.hist(sc[lab == 0], bins=bins, alpha=0.5, color=BLUE, label="normal (20)")
    ax.hist(sc[lab == 1], bins=bins, alpha=0.5, color=RED, label="defective (20)")
    ax.set_xlabel("image score = max patch distance")
    ax.set_ylabel("number of test images")
    ax.set_title(f"image AUROC {auc_i:.3f}\npixel AUROC {auc_p:.3f}", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "anomaly-maps.png")


if __name__ == "__main__":
    fig_map()
    fig_shift()
    fig_adv()
    fig_ood()
    fig_anomaly()
