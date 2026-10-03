"""Figures for PR 09 (ensembles and validation).

Writes only to src/assets/pattern-recognition/pr-09-ensembles-validation/.
Run: .venv/bin/python scripts/figures/pr_09_ensembles_validation_extra.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer, make_classification, make_moons
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src/assets/pattern-recognition/pr-09-ensembles-validation"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
C = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#6b7280"]


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(path, f"{path.stat().st_size / 1024:.0f} KB")


# ---------------------------------------------------------------- 1. bias-variance
def fig_bias_variance():
    rng = np.random.default_rng(0)
    f = lambda x: np.sin(2 * np.pi * x)
    sigma, n, n_sets = 0.3, 60, 200
    xt = np.linspace(0, 1, 300)[:, None]
    sets = []
    for _ in range(n_sets):
        x = rng.random((n, 1))
        sets.append((x, f(x[:, 0]) + rng.normal(0, sigma, n)))
    depths = list(range(1, 11))
    b2, var = [], []
    preds_by_depth = {}
    for d in depths:
        P = np.array([DecisionTreeRegressor(max_depth=d, random_state=0)
                      .fit(x, y).predict(xt) for x, y in sets])
        preds_by_depth[d] = P
        b2.append(np.mean((P.mean(0) - f(xt[:, 0])) ** 2))
        var.append(np.mean(P.var(0)))
    b2, var = np.array(b2), np.array(var)

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for a, d, title in [(ax[0], 1, "Depth 1: high bias, low variance"),
                        (ax[1], 8, "Depth 8: low bias, high variance")]:
        for P in preds_by_depth[d][:15]:
            a.plot(xt, P, color=C[0], alpha=0.25, lw=1)
        a.plot(xt, preds_by_depth[d].mean(0), color=C[1], lw=2.2,
               label="average over 200 fits")
        a.plot(xt, f(xt), "k--", lw=1.5, label="true function")
        a.set_title(title)
        a.set_xlabel("x"); a.set_ylim(-1.8, 1.8)
        a.legend(loc="lower left", fontsize=9)
    ax[0].set_ylabel("prediction (15 training sets)")
    ax[2].plot(depths, b2, "o-", color=C[1], label="bias$^2$")
    ax[2].plot(depths, var, "s-", color=C[0], label="variance")
    ax[2].plot(depths, b2 + var + sigma ** 2, "^-", color="k",
               label="bias$^2$ + variance + noise")
    ax[2].axhline(sigma ** 2, color=C[5], ls=":", label="noise $\\sigma^2$")
    ax[2].set_xlabel("tree depth"); ax[2].set_ylabel("squared error")
    ax[2].set_title("Decomposition (Monte Carlo)")
    ax[2].set_ylim(0, 0.29)
    ax[2].legend(fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout()
    save(fig, "bias-variance.png")


# ---------------------------------------------------------------- 2. bagging / RF
def fig_bagging():
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    M = np.arange(1, 101)
    for rho, c in zip([0.0, 0.2, 0.5, 0.8], C):
        ax[0].plot(M, rho + (1 - rho) / M, color=c, lw=2, label=f"$\\rho$ = {rho}")
        ax[0].axhline(rho, color=c, ls=":", lw=1)
    ax[0].set_xscale("log"); ax[0].set_ylim(0, 1.05)
    ax[0].set_xlabel("number of averaged predictors $M$")
    ax[0].set_ylabel("Var(average) / $\\sigma^2$")
    ax[0].set_title("Correlation sets a variance floor")
    ax[0].legend()

    X, y = make_classification(n_samples=1500, n_features=20, n_informative=6,
                               n_redundant=4, flip_y=0.05, random_state=7)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=7)
    ns = [1, 2, 5, 10, 20, 50, 100, 200]
    tree_err = np.mean(DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
                       .predict(Xte) != yte)
    res = {"bagging": [], "random forest": [], "random forest (OOB)": []}
    for n in ns:
        bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=n,
                                random_state=0).fit(Xtr, ytr)
        rf = RandomForestClassifier(n_estimators=n, random_state=0,
                                    oob_score=n >= 20).fit(Xtr, ytr)
        res["bagging"].append(np.mean(bag.predict(Xte) != yte))
        res["random forest"].append(np.mean(rf.predict(Xte) != yte))
        res["random forest (OOB)"].append(1 - rf.oob_score_ if n >= 20 else np.nan)
    ax[1].axhline(tree_err, color=C[5], ls="--", label="single deep tree")
    ax[1].plot(ns, res["bagging"], "o-", color=C[0], label="bagging (test)")
    ax[1].plot(ns, res["random forest"], "s-", color=C[2], label="random forest (test)")
    ax[1].plot(ns, res["random forest (OOB)"], "^:", color=C[2],
               label="random forest (OOB estimate)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("number of trees")
    ax[1].set_ylabel("error rate"); ax[1].set_title("Synthetic 20-feature task")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    save(fig, "bagging-forest.png")


# ---------------------------------------------------------------- 3. AdaBoost
def adaboost(X, y, M):
    w = np.full(len(y), 1 / len(y))
    hs, al = [], []
    for _ in range(M):
        h = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y, sample_weight=w)
        p = h.predict(X)
        e = np.clip(np.sum(w * (p != y)) / w.sum(), 1e-12, 1 - 1e-12)
        a = 0.5 * np.log((1 - e) / e)
        w = w * np.exp(-a * y * p); w /= w.sum()
        hs.append(h); al.append(a)
    return hs, al


def fig_adaboost():
    X, y01 = make_classification(n_samples=600, n_features=10, n_informative=4,
                                 flip_y=0.05, random_state=1)
    y = 2 * y01 - 1
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
    M = 300
    hs, al = adaboost(Xtr, ytr, M)
    Ftr = np.cumsum([a * h.predict(Xtr) for h, a in zip(hs, al)], axis=0)
    Fte = np.cumsum([a * h.predict(Xte) for h, a in zip(hs, al)], axis=0)
    r = np.arange(1, M + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax[0].plot(r, np.mean(np.exp(-ytr * Ftr), axis=1), color=C[4], lw=2,
               label="training exponential loss (bound)")
    ax[0].plot(r, np.mean(np.sign(Ftr) != ytr, axis=1), color=C[0], lw=2,
               label="training error")
    ax[0].plot(r, np.mean(np.sign(Fte) != yte, axis=1), color=C[1], lw=2,
               label="test error")
    ax[0].set_xscale("log"); ax[0].set_xlabel("boosting rounds $M$")
    ax[0].set_ylabel("value"); ax[0].set_title("AdaBoost with decision stumps")
    ax[0].legend(fontsize=9)

    Xm, ym01 = make_moons(n_samples=300, noise=0.25, random_state=2)
    ym = 2 * ym01 - 1
    hs2, al2 = adaboost(Xm, ym, 200)
    gx, gy = np.meshgrid(np.linspace(-1.6, 2.6, 300), np.linspace(-1.2, 1.7, 220))
    G = np.c_[gx.ravel(), gy.ravel()]
    F = sum(a * h.predict(G) for h, a in zip(hs2, al2)).reshape(gx.shape)
    ax[1].contourf(gx, gy, F, levels=20, cmap="RdBu", alpha=0.6)
    ax[1].contour(gx, gy, F, levels=[0], colors="k", linewidths=1.5)
    ax[1].scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap="RdBu", edgecolor="k", s=18)
    ax[1].set_title("200 stumps: axis-aligned staircase boundary")
    ax[1].set_xlabel("$x_1$"); ax[1].set_ylabel("$x_2$")
    fig.tight_layout()
    save(fig, "adaboost.png")


# ---------------------------------------------------------------- 4. CV schemes
def fig_cv_schemes():
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_xlim(-3.2, 10.4); ax.set_ylim(-0.5, 15.2); ax.axis("off")
    col = {"train": "#bfdbfe", "valid": "#fca5a5", "test": "#fde68a",
           "inner": "#bbf7d0", "unused": "#e5e7eb"}

    def row(yv, kinds, label, x0=0.0, width=10.0):
        k = len(kinds); w = width / k
        for i, kind in enumerate(kinds):
            ax.add_patch(Rectangle((x0 + i * w, yv), w, 0.75, facecolor=col[kind],
                                   edgecolor="k", lw=0.8))
        ax.text(-0.15, yv + 0.37, label, ha="right", va="center", fontsize=10)

    ax.text(-3.1, 14.6, "k-fold CV (k = 5)", fontsize=12, weight="bold")
    for f in range(5):
        row(13.6 - f * 0.9, ["valid" if i == f else "train" for i in range(5)],
            f"fold {f + 1}")
    ax.text(-3.1, 8.6, "Group k-fold: each cell label = one subject; groups never split",
            fontsize=12, weight="bold")
    groups = [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4]
    w = 10 / len(groups)
    for f in range(2):
        yv = 7.6 - f * 0.9
        for i, g in enumerate(groups):
            kind = "valid" if g in ([0, 3] if f == 0 else [1, 4]) else "train"
            ax.add_patch(Rectangle((i * w, yv), w, 0.75, facecolor=col[kind],
                                   edgecolor="k", lw=0.8))
            ax.text(i * w + w / 2, yv + 0.37, f"S{g}", ha="center", va="center",
                    fontsize=8)
        ax.text(-0.15, yv + 0.37, f"fold {f + 1}", ha="right", va="center")
    ax.text(-3.1, 5.1, "Nested CV: inner loop selects, outer loop evaluates the selection",
            fontsize=12, weight="bold")
    row(4.0, ["train", "train", "train", "train", "valid"], "outer fold 1")
    for j in range(3):
        kinds = ["inner"] * 3
        kinds[j] = "valid"
        row(2.9 - j * 0.85, kinds, f"inner {j + 1}", x0=0.0, width=8.0)
        ax.add_patch(Rectangle((8.0, 2.9 - j * 0.85), 2.0, 0.75,
                               facecolor=col["unused"], edgecolor="k", lw=0.8,
                               hatch="//"))
    handles = [Rectangle((0, 0), 1, 1, facecolor=col[k], edgecolor="k")
               for k in ["train", "valid", "inner"]]
    handles.append(Rectangle((0, 0), 1, 1, facecolor=col["unused"], edgecolor="k",
                             hatch="//"))
    ax.legend(handles, ["fit", "held-out score", "inner fit",
                        "outer held-out (hidden from inner loop)"],
              loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), fontsize=10)
    save(fig, "cv-schemes.png")


# ---------------------------------------------------------------- 5. paired tests
def fig_tests():
    X, y = load_breast_cancer(return_X_y=True)
    A = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    B = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=2)
    J, frac = 30, 0.2
    d = []
    for tr, te in ShuffleSplit(J, test_size=frac, random_state=0).split(X):
        ea = np.mean(A.fit(X[tr], y[tr]).predict(X[te]) != y[te])
        eb = np.mean(B.fit(X[tr], y[tr]).predict(X[te]) != y[te])
        d.append(ea - eb)
    d = np.array(d)
    tq = stats.t.ppf(0.975, J - 1)
    se_n = np.sqrt(d.var(ddof=1) / J)
    se_c = np.sqrt(d.var(ddof=1) * (1 / J + frac / (1 - frac)))
    fig, ax = plt.subplots(figsize=(10, 4))
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, J)
    ax.scatter(d, 2 + jitter, color=C[5], s=22,
               label="per-split difference (30 random 80/20 splits)")
    for yv, se, c, lab in [(1, se_n, C[1], "naive 95% interval"),
                           (0, se_c, C[0], "corrected 95% interval (Nadeau-Bengio)")]:
        ax.errorbar(d.mean(), yv, xerr=tq * se, fmt="o", color=c, capsize=6,
                    lw=2.5, label=lab)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["corrected", "naive", "splits"])
    ax.set_xlabel("error(logistic regression) - error(random forest)")
    ax.set_title("Overlapping training sets make naive intervals too narrow")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(-0.6, 3.2)
    fig.tight_layout()
    save(fig, "paired-differences.png")


if __name__ == "__main__":
    fig_bias_variance()
    fig_bagging()
    fig_adaboost()
    fig_cv_schemes()
    fig_tests()
