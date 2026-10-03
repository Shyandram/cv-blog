"""Extra figures for ML 10 (evaluation and deployment).

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/ml_10_evaluation_deployment_extra.py
All numbers come from bundled scikit-learn data or synthetic arrays; CPU only, < 1 min.
Writes only to src/assets/ml/ml-10-evaluation-deployment/.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")      # avoid BLAS oversubscription
from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GroupKFold, KFold, cross_val_score,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "src/assets/ml/ml-10-evaluation-deployment"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, GREY = "#2a6fdb", "#e4572e", "#17a398", "#6b7280"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name, f"{(OUT / name).stat().st_size / 1e3:.0f} kB")


def cv_leakage():
    rng = np.random.default_rng(0)
    n_groups, per_group = 40, 25
    groups = np.repeat(np.arange(n_groups), per_group)
    y = np.repeat(rng.integers(0, 2, n_groups), per_group)
    identity = rng.normal(size=(n_groups, 20))[groups]
    X = identity + 0.3 * y[:, None] + 0.3 * rng.normal(size=(len(y), 20))
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    r = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
    g = cross_val_score(model, X, y, cv=GroupKFold(5), groups=groups)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (s, c, lab) in enumerate([(r, ORANGE, "Random K-fold\n(images mixed)"),
                                     (g, BLUE, "Group K-fold\n(patients separated)")]):
        ax.scatter(np.full(5, i) + np.linspace(-.08, .08, 5), s, color=c, s=40, zorder=3)
        ax.hlines(s.mean(), i - .25, i + .25, color=c, lw=3)
        ax.text(i + .3, s.mean(), f"mean {s.mean():.3f}", va="center", color=c)
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.text(1.45, 0.51, "chance", color=GREY, ha="right", fontsize=10)
    ax.set_xticks([0, 1], ["Random K-fold\n(images mixed)", "Group K-fold\n(patients separated)"])
    ax.set_xlim(-.5, 1.6)
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("Fold accuracy")
    ax.set_title("Same data, same model, different split unit")
    ax.grid(alpha=.25, axis="y")
    save(fig, "cv-leakage.png")


def bootstrap():
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    a = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    b = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")).fit(Xtr, ytr)
    ca, cb = a.predict(Xte) == yte, b.predict(Xte) == yte
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(yte), size=(10_000, len(yte)))
    diff = cb[idx].mean(1) - ca[idx].mean(1)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.hist(diff * 100, bins=(np.arange(-8, 22) - 0.5) / len(yte) * 100, color=BLUE, alpha=.75)
    ax.axvspan(lo * 100, hi * 100, color=GREEN, alpha=.15, label="95% percentile interval")
    ax.axvline(0, color=ORANGE, lw=2, label="no difference")
    ax.axvline((cb.mean() - ca.mean()) * 100, color="k", lw=1.5, ls="--",
               label="observed difference")
    ax.set_xlabel("Accuracy(B) − Accuracy(A), percentage points")
    ax.set_ylabel("Bootstrap resamples")
    ax.set_title("Paired bootstrap of a 540-image test set")
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "bootstrap-ci.png")


def calibration():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Xa, Xte, ya, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
    Xtr, Xcal, ytr, ycal = train_test_split(Xa, ya, test_size=0.3, stratify=ya, random_state=0)
    net = MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-6, max_iter=300,
                        random_state=0).fit(Xtr, ytr)
    logits = lambda X: np.log(np.clip(net.predict_proba(X), 1e-12, 1))

    def nll(T, z, y):
        return -log_softmax(z / T, 1)[np.arange(len(y)), y].mean()

    def rel(prob, y, bins=10):
        conf, ok = prob.max(1), prob.argmax(1) == y
        edges = np.linspace(0, 1, bins + 1)
        xs, ys, ws, e = [], [], [], 0.0
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            m = (conf > lo_) & (conf <= hi_)
            if m.any():
                xs.append(conf[m].mean()); ys.append(ok[m].mean()); ws.append(m.mean())
                e += m.mean() * abs(ok[m].mean() - conf[m].mean())
        return np.array(xs), np.array(ys), e

    T = minimize_scalar(nll, bounds=(.05, 20), method="bounded", args=(logits(Xcal), ycal)).x
    noise = np.random.default_rng(0).normal(size=Xte.shape)
    sigmas = [0.0, 0.15, 0.3, 0.45, 0.6]
    shifted = {s: np.clip(Xte + s * noise, 0, 1) for s in sigmas}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, s in zip(axes[:2], [0.0, 0.3]):
        z = logits(shifted[s])
        ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="perfect calibration")
        for p, c, lab in [(softmax(z, 1), ORANGE, "raw"), (softmax(z / T, 1), BLUE, f"temperature T={T:.2f}")]:
            xs, ys, e = rel(p, yte)
            ax.plot(xs, ys, "o-", color=c, label=f"{lab} (ECE {e:.3f})")
        ax.set_xlabel("Mean confidence in bin")
        ax.set_ylabel("Accuracy in bin")
        ax.set_title(f"Pixel noise σ = {s}")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=.25)
    acc, e_raw, e_ts = [], [], []
    for s in sigmas:
        z = logits(shifted[s])
        acc.append((z.argmax(1) == yte).mean())
        e_raw.append(rel(softmax(z, 1), yte)[2]); e_ts.append(rel(softmax(z / T, 1), yte)[2])
    ax = axes[2]
    ax.plot(sigmas, acc, "s-", color=GREEN, label="accuracy")
    ax.plot(sigmas, e_raw, "o-", color=ORANGE, label="ECE raw")
    ax.plot(sigmas, e_ts, "o--", color=BLUE, label="ECE after scaling")
    ax.set_xlabel("Test-time noise σ (training used σ = 0)")
    ax.set_title("Shift lowers accuracy and calibration")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)
    fig.tight_layout()
    save(fig, "calibration-shift.png")


def quant_error():
    def quantize(x, bits, axis=None):
        qmax = 2**bits - 1
        lo = x.min(axis=axis, keepdims=True) if axis is not None else x.min()
        hi = x.max(axis=axis, keepdims=True) if axis is not None else x.max()
        lo, hi = np.minimum(lo, 0), np.maximum(hi, 0)
        s = (hi - lo) / qmax
        z = np.round(-lo / s)
        return s * (np.clip(np.round(x / s) + z, 0, qmax) - z)
    rng = np.random.default_rng(0)
    W = rng.normal(size=(64, 256)) * np.logspace(-2, 0, 64)[:, None]
    bits = np.arange(2, 9)
    t = [np.abs(quantize(W, b) - W).mean() for b in bits]
    c = [np.abs(quantize(W, b, 1) - W).mean() for b in bits]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.semilogy(bits, t, "o-", color=ORANGE, label="one scale per tensor")
    ax.semilogy(bits, c, "o-", color=BLUE, label="one scale per output channel")
    ax.set_xlabel("Bits per weight")
    ax.set_ylabel("Mean |dequantized − original|")
    ax.set_title("Granularity matters as much as bit width")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25, which="both")
    ax = axes[1]
    x = np.linspace(-1, 3, 400)
    for b, col in [(2, ORANGE), (3, GREEN), (4, BLUE)]:
        ax.plot(x, quantize(x, b), color=col, lw=1.6, label=f"{b}-bit grid")
    ax.plot(x, x, color=GREY, ls="--", lw=1, label="identity")
    ax.set_xlabel("Real value x")
    ax.set_ylabel("Dequantized value")
    ax.set_title("Affine quantization is a staircase")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)
    fig.tight_layout()
    save(fig, "quantization-error.png")


def pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 11); ax.set_ylim(-0.4, 3.4); ax.axis("off")
    boxes = [(0.2, "Train +\nvalidate", "frozen splits,\nseeds, configs"),
             (2.4, "Export", "torch.export /\nONNX graph"),
             (4.6, "Optimize", "quantize, prune,\nfuse, distill"),
             (6.8, "Serve", "ONNX Runtime,\nTensorRT, ..."),
             (9.0, "Monitor", "inputs, confidence,\nlatency, labels")]
    for x, title, sub in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.4), 1.8, 1.3, boxstyle="round,pad=0.05",
                                    fc="#eef3fc", ec=BLUE, lw=1.5))
        ax.text(x + .9, 2.3, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x + .9, 1.75, sub, ha="center", va="center", fontsize=9)
    for x in [2.0, 4.2, 6.4, 8.6]:
        ax.annotate("", xy=(x + .4, 2.05), xytext=(x, 2.05),
                    arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.annotate("", xy=(1.1, 1.3), xytext=(9.9, 1.3),
                arrowprops=dict(arrowstyle="->", lw=1.4, color=ORANGE,
                                connectionstyle="arc3,rad=-0.12"))
    ax.text(5.5, -0.2, "drift or new failure → re-collect, re-label, re-train, re-evaluate",
            ha="center", color=ORANGE, fontsize=10)
    ax.text(5.5, 3.15, "Check accuracy, calibration and latency after EVERY arrow",
            ha="center", fontsize=10, color=GREY)
    save(fig, "deployment-pipeline.png")


def drift():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Xtr, Xlive, ytr, ylive = train_test_split(X, y, test_size=0.5, random_state=0)
    Xtr, Xref, ytr = Xtr[:600], Xtr[600:], ytr[:600]
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    ref = clf.predict_proba(Xref).max(1)
    rng = np.random.default_rng(0)
    out = {}
    for name, noise in [("week 1 (clean)", 0.0), ("week 2 (new camera)", 0.25)]:
        batch = rng.choice(len(Xlive), 200, replace=False)
        Xb = np.clip(Xlive[batch] + noise * rng.normal(size=Xlive[batch].shape), 0, 1)
        out[name] = clf.predict_proba(Xb).max(1)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    grid = np.linspace(0, 1, 201)
    for data, c, lab in [(ref, GREY, "reference window"), (out["week 1 (clean)"], BLUE, "week 1 (clean)"),
                         (out["week 2 (new camera)"], ORANGE, "week 2 (new camera)")]:
        ax.plot(grid, (np.sort(data)[:, None] <= grid).mean(0), color=c, lw=2, label=lab)
    ax.set_xlabel("Top-class confidence")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Label-free drift signal: confidence distribution")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=.25)
    save(fig, "drift-monitor.png")


if __name__ == "__main__":
    cv_leakage()
    bootstrap()
    calibration()
    quant_error()
    pipeline()
    drift()
