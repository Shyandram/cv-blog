"""Extra figures for PR 04 (kernels and nonlinear classifiers).

Writes only into src/assets/pattern-recognition/pr-04-kernels-nonlinear-classifiers/.
Run: .venv/bin/python scripts/figures/pr_04_kernels_nonlinear_classifiers_extra.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

OUT = Path(__file__).resolve().parents[2] / "src/assets/pattern-recognition/pr-04-kernels-nonlinear-classifiers"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white"})
C0, C1 = "#2b6cb0", "#c53030"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grid(X, pad=0.4, n=300):
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def draw(ax, model, X, y, title):
    xx, yy, G = grid(X)
    Z = model.predict(G).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=["#dbe8f7", "#f7dede"])
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=0.8)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=10, c=C0)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=10, c=C1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def kernel_lift():
    X, y = make_circles(n_samples=300, factor=0.4, noise=0.07, random_state=0)
    z = (X ** 2).sum(axis=1)
    fig = plt.figure(figsize=(10, 4.2))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=10, c=C0, label="outer ring")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=10, c=C1, label="inner disc")
    ax.set_aspect("equal")
    ax.set_title("Input space: no separating line")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], z[y == 0], s=8, c=C0)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], z[y == 1], s=8, c=C1)
    gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 10), np.linspace(-1.2, 1.2, 10))
    ax.plot_surface(gx, gy, np.full_like(gx, 0.5), alpha=0.25, color="gray")
    ax.set_title("Feature space $(x_1, x_2, r^2=x_1^2+x_2^2)$:\na plane separates")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$r^2$")
    ax.view_init(elev=18, azim=-60)
    save(fig, "kernel-lift.png")


def c_gamma_grid():
    X, y = make_moons(n_samples=200, noise=0.25, random_state=3)
    X = StandardScaler().fit_transform(X)
    Cs, gammas = [0.1, 1, 100], [0.1, 1, 10]
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 9))
    for i, C in enumerate(Cs):
        for j, g in enumerate(gammas):
            m = SVC(kernel="rbf", C=C, gamma=g).fit(X, y)
            ax = axes[i, j]
            draw(ax, m, X, y, f"C={C}, gamma={g}\n{len(m.support_)} support vectors")
            sv = m.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], s=34, facecolors="none", edgecolors="k", linewidths=0.6)
    fig.suptitle("RBF-SVM on noisy moons: rows increase C, columns increase gamma (circles = support vectors)", y=1.0)
    fig.tight_layout()
    save(fig, "svm-c-gamma.png")


def impurity_and_tree():
    p = np.linspace(1e-6, 1 - 1e-6, 400)
    gini = 2 * p * (1 - p)
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    mis = 1 - np.maximum(p, 1 - p)
    X, y = make_moons(n_samples=300, noise=0.25, random_state=1)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6))
    ax = axes[0]
    ax.plot(p, ent / 2, label="entropy / 2", color="#2f855a")
    ax.plot(p, gini, label="Gini", color="#6b46c1")
    ax.plot(p, mis, label="misclassification", color="#dd6b20")
    ax.set_xlabel("fraction of class 1 in the node, $p$")
    ax.set_ylabel("impurity")
    ax.set_title("Two-class impurity measures")
    ax.legend(fontsize=9)
    for ax, depth in zip(axes[1:], [3, None]):
        m = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X, y)
        label = "unlimited" if depth is None else str(depth)
        draw(ax, m, X, y, f"CART tree, max depth {label}\n{m.get_n_leaves()} leaves")
    fig.tight_layout()
    save(fig, "trees-impurity.png")


def rbf_net_vs_mlp():
    X, y = make_moons(n_samples=500, noise=0.18, random_state=2)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2)
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

    class RBFNet:
        def __init__(self, m=12, gamma=1.0):
            self.m, self.gamma = m, gamma

        def phi(self, Z):
            d2 = ((Z[:, None, :] - self.centers[None]) ** 2).sum(-1)
            return np.exp(-self.gamma * d2)

        def fit(self, Z, t):
            self.centers = KMeans(self.m, n_init=10, random_state=0).fit(Z).cluster_centers_
            self.lr = LogisticRegression(C=100, max_iter=2000).fit(self.phi(Z), t)
            return self

        def predict(self, Z):
            return self.lr.predict(self.phi(Z))

        def score(self, Z, t):
            return (self.predict(Z) == t).mean()

    rbf = RBFNet().fit(Ztr, ytr)
    mlp = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=3000, random_state=0).fit(Ztr, ytr)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    draw(axes[0], rbf, Zte, yte, f"RBF network: 12 k-means centres\nheld-out accuracy {rbf.score(Zte, yte):.3f}")
    axes[0].scatter(rbf.centers[:, 0], rbf.centers[:, 1], marker="X", s=90, c="gold", edgecolors="k", label="centres")
    axes[0].legend(loc="lower left", fontsize=9)
    draw(axes[1], mlp, Zte, yte, f"MLP: two hidden layers of 16 ReLUs\nheld-out accuracy {mlp.score(Zte, yte):.3f}")
    fig.tight_layout()
    save(fig, "rbf-network-vs-mlp.png")


if __name__ == "__main__":
    kernel_lift()
    c_gamma_grid()
    impurity_and_tree()
    rbf_net_vs_mlp()
    for f in sorted(OUT.glob("*.png")):
        print(f.name, f.stat().st_size // 1024, "KB")
