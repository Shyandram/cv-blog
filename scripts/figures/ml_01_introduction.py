"""Figures for ML 01 - Introduction to Machine Learning.

Run:  .venv/bin/python scripts/figures/ml_01_introduction.py
Writes only to src/assets/ml/ml-01-introduction/. All data come from
scikit-learn toy datasets or synthetic arrays; nothing is downloaded.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.datasets import load_digits, make_blobs, make_classification, make_regression  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "ml" / "ml-01-introduction"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})
BLUE, ORANGE, GREEN, RED, GRAY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"


def save(fig, name, max_px=1600):
    """Save at 150 dpi, lowering dpi only if the tight-cropped width would exceed max_px."""
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    width = plt.imread(path).shape[1]
    if width > max_px:
        fig.savefig(path, dpi=150 * (max_px - 10) / width, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{name}: {path.stat().st_size / 1024:.0f} KB")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------- 1. rules vs learning
def fig_rules_vs_learning():
    d = load_digits()
    y = d.target
    zeros = d.images[y == 0]
    ones = d.images[y == 1]
    w0 = (zeros.max(axis=1) > 4).sum(axis=1)
    w1 = (ones.max(axis=1) > 4).sum(axis=1)
    # show 6 zeros and 6 ones, choosing the widest ones to make the point visible
    idx1 = np.concatenate([np.argsort(w1)[:3], np.argsort(w1)[-3:]])
    fig = plt.figure(figsize=(10, 3.8))
    gs = fig.add_gridspec(2, 9, width_ratios=[1] * 6 + [0.9, 2.6, 2.6], wspace=0.3)
    for k in range(6):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(zeros[k], cmap="gray_r", vmin=0, vmax=16)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"w={w0[k]}", fontsize=10)
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(ones[idx1[k]], cmap="gray_r", vmin=0, vmax=16)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"w={w1[idx1[k]]}", fontsize=10)
    fig.text(0.02, 0.70, "zeros", rotation=90, fontsize=11, va="center")
    fig.text(0.02, 0.28, "ones", rotation=90, fontsize=11, va="center")
    ax = fig.add_subplot(gs[:, 7:])
    bins = np.arange(1.5, 9.5, 1)
    ax.hist(w0, bins=bins, alpha=0.6, color=BLUE, label="digit 0")
    ax.hist(w1, bins=bins, alpha=0.6, color=ORANGE, label="digit 1")
    ax.axvline(5.5, color="k", ls="--", lw=1.5)
    ax.text(5.6, ax.get_ylim()[1] * 0.55, "rule:\nw <= 5 -> '1'\nw >= 6 -> '0'", ha="left", fontsize=10)
    ax.set_xlabel("w = number of inked columns")
    ax.set_ylabel("number of images")
    ax.legend(loc="upper left")
    ax.set_title("Hand rule accuracy 0.79; learned model 1.00")
    save(fig, "rules-vs-learning.png")


# ---------------------------------------------------------------- 2. regression vs classification
def fig_regression_vs_classification():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    X, y = make_regression(n_samples=60, n_features=1, noise=12.0, random_state=0)
    ax = axes[0]
    ax.scatter(X[:, 0], y, s=18, color=BLUE)
    reg = LinearRegression().fit(X, y)
    xs = np.linspace(X.min(), X.max(), 100)
    ax.plot(xs, reg.predict(xs[:, None]), color=RED, lw=2, label="fitted line")
    ax.set_xlabel("feature x"); ax.set_ylabel("target y (a number)")
    ax.set_title("Regression"); ax.legend()

    Xc, yc = make_classification(n_samples=120, n_features=2, n_redundant=0,
                                 n_clusters_per_class=1, class_sep=1.4, random_state=0)
    ax = axes[1]
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression().fit(Xc, yc)
    ax.scatter(*Xc[yc == 0].T, s=18, color=BLUE, label="class 0")
    ax.scatter(*Xc[yc == 1].T, s=18, color=ORANGE, marker="s", label="class 1")
    xl = np.array(ax.get_xlim())
    w, b = clf.coef_[0], clf.intercept_[0]
    ax.plot(xl, -(w[0] * xl + b) / w[1], "k-", lw=2, label="decision boundary")
    ax.set_xlim(xl)
    ax.set_xlabel("feature x1"); ax.set_ylabel("feature x2")
    ax.set_title("Classification"); ax.legend(fontsize=9, loc="lower right")

    ax = axes[2]
    rng = np.random.default_rng(1)
    x0 = rng.normal(-1.5, 0.7, 30)
    x1 = np.concatenate([rng.normal(1.5, 0.7, 25), rng.uniform(7, 10, 5)])
    x = np.concatenate([x0, x1]); t = np.concatenate([np.zeros(30), np.ones(30)])
    lin = LinearRegression().fit(x[:, None], t)
    lg = LogisticRegression(C=100).fit(x[:, None], t)
    xs = np.linspace(-4, 10.5, 300)
    ax.scatter(x, t, s=18, color=np.where(t == 1, ORANGE, BLUE), zorder=3)
    ax.plot(xs, lin.predict(xs[:, None]), color=GRAY, lw=2, label="linear fit")
    ax.plot(xs, lg.predict_proba(xs[:, None])[:, 1], color=ORANGE, lw=2, label="sigmoid fit")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    xc = (0.5 - lin.intercept_) / lin.coef_[0]
    ax.axvline(xc, color=GRAY, ls="--", lw=1)
    ax.set_ylim(-0.4, 1.6)
    ax.set_xlabel("feature x"); ax.set_ylabel("label / output")
    ax.set_title("Regression on 0/1 labels"); ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save(fig, "regression-vs-classification.png")


# ---------------------------------------------------------------- 3. unsupervised
def fig_unsupervised():
    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(2, 5, width_ratios=[3, 3, 1, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    Xb, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.9, random_state=2)
    km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Xb)
    ax.scatter(*Xb.T, c=km.labels_, cmap="tab10", s=12, vmin=0, vmax=9)
    ax.scatter(*km.cluster_centers_.T, c="k", marker="x", s=120, lw=3)
    ax.set_title("Clustering (k-means, K = 4)")
    ax.set_xticks([]); ax.set_yticks([])

    X, y = load_digits(return_X_y=True)
    Z = PCA(n_components=2).fit_transform(X)
    ax = fig.add_subplot(gs[:, 1])
    sc = ax.scatter(*Z.T, c=y, cmap="tab10", s=6)
    cb = fig.colorbar(sc, ax=ax, ticks=range(10), fraction=0.046)
    cb.set_label("true digit (not used)")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title("Dimensionality reduction (PCA)")

    pca = PCA(n_components=20).fit(X)
    err = ((X - pca.inverse_transform(pca.transform(X))) ** 2).sum(axis=1)
    worst = np.argsort(err)[-6:][::-1]
    for k, i in enumerate(worst):
        ax = fig.add_subplot(gs[k // 3, 2 + k % 3])
        ax.imshow(X[i].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"err {err[i]:.0f}", fontsize=9)
    fig.text(0.86, 0.98, "Anomalies (largest PCA error)", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "unsupervised.png")


# ---------------------------------------------------------------- 4. cost function
def fig_cost_function():
    X, y = make_regression(n_samples=30, n_features=1, noise=15.0, random_state=4)
    x = X[:, 0]
    m = len(x)
    w_best, b_best = np.polyfit(x, y, 1)

    def J(w, b):
        return ((np.multiply.outer(w, x) + np.asarray(b)[..., None] - y) ** 2).sum(-1) / (2 * m)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    ax = axes[0]
    ax.scatter(x, y, color=BLUE, s=18, zorder=3)
    xs = np.linspace(x.min(), x.max(), 50)
    ws = [w_best * 0.3, w_best, w_best * 1.7]
    cols = [GREEN, RED, ORANGE]
    for w, c in zip(ws, cols):
        ax.plot(xs, w * xs + b_best, color=c, lw=2, label=f"w = {w:.0f}")
    for xi, yi in zip(x, y):
        ax.plot([xi, xi], [yi, w_best * xi + b_best], color=RED, lw=0.8, alpha=0.6)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=9)
    ax.set_title("Candidate lines f(x) = wx + b")

    ax = axes[1]
    wgrid = np.linspace(-0.2 * w_best, 2.2 * w_best, 200)
    ax.plot(wgrid, J(wgrid, np.full_like(wgrid, b_best)), "k-", lw=2)
    for w, c in zip(ws, cols):
        ax.plot(w, J(np.array([w]), np.array([b_best]))[0], "o", color=c, ms=9)
    ax.set_xlabel("w  (b fixed)"); ax.set_ylabel("J(w)")
    ax.set_title("Cost as a function of w")

    ax = axes[2]
    W, B = np.meshgrid(np.linspace(w_best - 60, w_best + 60, 200), np.linspace(b_best - 60, b_best + 60, 200))
    Z = J(W, B)
    ax.contour(W, B, Z, levels=np.geomspace(Z.min() * 1.05, Z.max(), 14), cmap="viridis")
    w, b, alpha = w_best - 55, b_best + 50, 0.15
    path = [(w, b)]
    for _ in range(30):
        err = w * x + b - y
        w, b = w - alpha * (err * x).mean(), b - alpha * err.mean()
        path.append((w, b))
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "o-", color=RED, ms=3, lw=1.2)
    ax.plot(w_best, b_best, "k*", ms=14)
    ax.set_xlabel("w"); ax.set_ylabel("b")
    ax.set_title("Contours of J(w, b) + gradient descent")
    fig.tight_layout()
    save(fig, "cost-function.png")


# ---------------------------------------------------------------- 5. learning rate
def fig_learning_rate():
    J = lambda w: (w - 2.0) ** 2 + 1.0  # noqa: E731
    dJ = lambda w: 2 * (w - 2.0)  # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    ws = np.linspace(-3, 7, 300)
    for ax, (alpha, title, n) in zip(axes, [(0.05, "Too small (alpha = 0.05)", 12),
                                           (0.3, "About right (alpha = 0.3)", 8),
                                           (1.05, "Too large (alpha = 1.05)", 8)]):
        ax.plot(ws, J(ws), "k-", lw=1.5)
        w = -1.0
        path = [w]
        for _ in range(n):
            w = w - alpha * dJ(w)
            path.append(w)
        path = np.array(path)
        ax.plot(path, J(path), "o-", color=RED, ms=5, lw=1.2)
        ax.plot(path[0], J(path[0]), "s", color=BLUE, ms=8)
        ax.set_title(title); ax.set_xlabel("w")
        ax.set_xlim(-3, 7); ax.set_ylim(0, 27)
    axes[0].set_ylabel("J(w)")
    fig.tight_layout()
    save(fig, "learning-rate.png")


# ---------------------------------------------------------------- 6. feature scaling
def fig_feature_scaling():
    rng = np.random.default_rng(0)
    m = 100
    x1 = rng.uniform(0, 10, m)
    x2 = rng.uniform(0, 1, m)
    X = np.c_[x1, x2]
    y = 1.0 * x1 + 8.0 * x2 + rng.normal(0, 0.5, m)
    X = X - X.mean(0)
    y = y - y.mean()

    def run(Xm, alpha, steps=40):
        w = np.array([-1.5, -10.0]) if Xm is X else np.array([-3.5, -3.0])
        path = [w.copy()]
        for _ in range(steps):
            w = w - alpha * Xm.T @ (Xm @ w - y) / m
            path.append(w.copy())
        return np.array(path)

    Xs = X / X.std(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, Xm, alpha, title in [(axes[0], X, 0.19, "Raw features (x1 in 0-10, x2 in 0-1)"),
                                 (axes[1], Xs, 0.3, "Standardized features")]:
        wopt = np.linalg.lstsq(Xm, y, rcond=None)[0]
        p = run(Xm, alpha)
        r1 = max(abs(p[:, 0] - wopt[0]).max(), 1.0) * 1.2
        r2 = max(abs(p[:, 1] - wopt[1]).max(), 1.0) * 1.2
        r = max(r1, r2)
        W1, W2 = np.meshgrid(np.linspace(wopt[0] - r, wopt[0] + r, 200), np.linspace(wopt[1] - r, wopt[1] + r, 200))
        E = np.einsum("ij,abj->abi", Xm, np.stack([W1, W2], -1)) - y
        Z = (E ** 2).mean(-1) / 2
        ax.contour(W1, W2, Z, levels=np.geomspace(Z.min() + 1e-3, Z.max(), 18), cmap="viridis")
        ax.plot(p[:, 0], p[:, 1], "o-", color=RED, ms=3, lw=1)
        ax.plot(*wopt, "k*", ms=14)
        ax.set_aspect("equal")
        ax.set_xlabel("w1"); ax.set_ylabel("w2")
        ax.set_title(title + "\n40 gradient-descent steps", fontsize=11)
    fig.tight_layout()
    save(fig, "feature-scaling.png")


# ---------------------------------------------------------------- 7. logistic regression
def fig_logistic():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    z = np.linspace(-8, 8, 300)
    ax = axes[0]
    ax.plot(z, sigmoid(z), color=BLUE, lw=2)
    ax.axhline(0.5, color="k", ls=":", lw=1); ax.axvline(0, color="k", ls=":", lw=1)
    ax.set_xlabel("z = w·x + b"); ax.set_ylabel("g(z)")
    ax.set_title("Sigmoid g(z) = 1 / (1 + exp(-z))")

    p = np.linspace(0.001, 0.999, 300)
    ax = axes[1]
    ax.plot(p, -np.log(p), color=ORANGE, lw=2, label="y = 1:  -log f")
    ax.plot(p, -np.log(1 - p), color=BLUE, lw=2, label="y = 0:  -log(1 - f)")
    ax.set_ylim(0, 5)
    ax.set_xlabel("predicted probability f"); ax.set_ylabel("loss")
    ax.set_title("Log loss (cross-entropy)"); ax.legend()

    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, class_sep=1.5, random_state=3)
    w, b = np.zeros(2), 0.0
    for _ in range(1000):
        pr = sigmoid(X @ w + b)
        w -= 0.5 * X.T @ (pr - y) / len(y)
        b -= 0.5 * (pr - y).mean()
    ax = axes[2]
    x1 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
    x2 = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200)
    G1, G2 = np.meshgrid(x1, x2)
    P = sigmoid(w[0] * G1 + w[1] * G2 + b)
    cf = ax.contourf(G1, G2, P, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.45)
    ax.contour(G1, G2, P, levels=[0.5], colors="k", linewidths=2)
    ax.scatter(*X[y == 0].T, s=14, color=BLUE, edgecolor="k", lw=0.3, label="class 0")
    ax.scatter(*X[y == 1].T, s=14, color=RED, marker="s", edgecolor="k", lw=0.3, label="class 1")
    cb = fig.colorbar(cf, ax=ax, fraction=0.046)
    cb.set_label("P(y = 1 | x)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title("Decision boundary (P = 0.5)"); ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    save(fig, "logistic-regression.png")


# ---------------------------------------------------------------- 8. overfitting
def fig_overfitting():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 25)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, 25)
    X = x[:, None]
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=0)

    def fit(degree, lam=0.0):
        reg = Ridge(alpha=lam) if lam > 0 else LinearRegression()
        return make_pipeline(PolynomialFeatures(degree), StandardScaler(), reg).fit(X_tr, y_tr)

    def mse(model, A, t):
        return ((model.predict(A) - t) ** 2).mean()

    fig = plt.figure(figsize=(14, 6.2))
    gs = fig.add_gridspec(2, 4)
    xs = np.linspace(0, 1, 400)[:, None]
    cases = [(1, 0, "Degree 1: underfit"), (3, 0, "Degree 3: good fit"),
             (15, 0, "Degree 15: overfit"), (15, 1e-2, "Degree 15 + L2 (lambda = 0.01)")]
    for k, (d, lam, title) in enumerate(cases):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        mdl = fit(d, lam)
        ax.plot(xs, np.sin(2 * np.pi * xs), color=GRAY, ls="--", lw=1, label="true curve")
        ax.plot(xs, mdl.predict(xs), color=RED, lw=2, label="model")
        ax.scatter(X_tr, y_tr, color=BLUE, s=22, zorder=3, label="train")
        ax.scatter(X_va, y_va, color=ORANGE, s=30, marker="D", zorder=3, label="validation")
        ax.set_ylim(-2, 2)
        ax.set_title(f"{title}\ntrain MSE {mse(mdl, X_tr, y_tr):.3f}, val MSE {mse(mdl, X_va, y_va):.3g}",
                     fontsize=10)
        if k == 0:
            ax.legend(fontsize=8, loc="lower left")
    ax = fig.add_subplot(gs[:, 2:])
    degs = np.arange(1, 13)
    tr = [mse(fit(d), X_tr, y_tr) for d in degs]
    va = [mse(fit(d), X_va, y_va) for d in degs]
    ax.semilogy(degs, tr, "o-", color=BLUE, label="training error")
    ax.semilogy(degs, va, "D-", color=ORANGE, label="validation error")
    best = degs[int(np.argmin(va))]
    ax.axvline(best, color="k", ls=":", lw=1)
    ax.text(best + 0.2, 0.3, f"best on validation\n(degree {best})", fontsize=10)
    ax.set_xlabel("polynomial degree (model flexibility)")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("Underfitting  <-  ->  Overfitting")
    ax.legend(loc="upper center")
    fig.tight_layout()
    save(fig, "overfitting.png")


if __name__ == "__main__":
    fig_rules_vs_learning()
    fig_regression_vs_classification()
    fig_unsupervised()
    fig_cost_function()
    fig_learning_rate()
    fig_feature_scaling()
    fig_logistic()
    fig_overfitting()
