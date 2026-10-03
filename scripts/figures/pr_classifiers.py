"""Original decision-theory, KDE and kernel figures for PR 01/02/04.

Run from the repository root:
    MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/pr_classifiers.py

All data are synthetic. The KDE and SVM figures reproduce the chapter seeds,
training splits and settings. No external data or trained models are downloaded.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2] / "src/assets/pattern-recognition"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140})


def save(fig, chapter, name):
    destination = ROOT / chapter
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / name, bbox_inches="tight")
    plt.close(fig)


def risk():
    p = np.linspace(0, 1, 501)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    axes[0].plot(p, p, label="Predict 0: risk = p")
    axes[0].plot(p, 1-p, label="Predict 1: risk = 1-p")
    axes[0].axhline(.2, color="#27804d", label="Reject: risk = 0.2")
    axes[0].axvspan(.2, .8, color="#27804d", alpha=.08)
    axes[0].set(title="Equal error costs + rejection",
                xlabel="Posterior p(class 1 | x)", ylabel="Conditional risk")
    axes[0].legend(fontsize=8)
    axes[1].plot(p, 9*p, label="Predict 0: risk = 9p")
    axes[1].plot(p, 1-p, label="Predict 1: risk = 1-p")
    axes[1].axvline(.1, color="#555", ls="--", label="Threshold 0.1")
    axes[1].set(title="False negative costs 9 times more",
                xlabel="Posterior p(class 1 | x)", ylabel="Conditional risk",
                ylim=(0, 1.2))
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "pr-01-bayes-decisions", "decision-risk.png")


def density():
    rng = np.random.default_rng(4)
    def sample(n):
        means = np.where(rng.random(n) < .6, -1.5, 1.5)
        return (means+rng.normal(0, .4, n)).reshape(-1, 1)
    train, validation = sample(180), sample(100)
    x = np.linspace(-3.5, 3.5, 800)
    truth = .6*norm.pdf(x, -1.5, .4)+.4*norm.pdf(x, 1.5, .4)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
    for ax, h in zip(axes.flat, [.08, .2, .5, 1.2]):
        model = KernelDensity(bandwidth=h).fit(train)
        ax.plot(x, truth, color="#555", ls="--", label="Known synthetic density")
        ax.plot(x, np.exp(model.score_samples(x[:, None])),
                color="#1f5fd6", label="Training KDE")
        ax.plot(train[:, 0], np.full(len(train), -.015), "|",
                color="#d75e3b", alpha=.25)
        score = model.score(validation)/len(validation)
        ax.set_title(f"Bandwidth {h}: validation log density {score:.2f}")
        ax.set_xlabel("Measurement")
        ax.set_ylabel("Density")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "pr-02-density-estimation", "kde-bandwidth.png")


def boundaries():
    X, y = make_moons(n_samples=500, noise=.18, random_state=2)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=.3, stratify=y, random_state=2)
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 220),
                         np.linspace(-1, 1.5, 180))
    grid = np.c_[xx.ravel(), yy.ravel()]
    models = {"Linear SVM": SVC(kernel="linear", C=1),
              "RBF SVM": SVC(kernel="rbf", C=3, gamma=.7),
              "7-neighbor": KNeighborsClassifier(n_neighbors=7)}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharex=True, sharey=True)
    for ax, (name, estimator) in zip(axes, models.items()):
        model = make_pipeline(StandardScaler(), estimator).fit(Xtr, ytr)
        region = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, region, levels=[-.5, .5, 1.5],
                    colors=["#c7dcfa", "#f7d4c5"], alpha=.7)
        ax.contour(xx, yy, region, levels=[.5], colors="#555", linewidths=.8)
        ax.scatter(Xte[:, 0], Xte[:, 1], c=yte, cmap="coolwarm",
                   s=15, edgecolors="white", linewidths=.3)
        ax.set_title(f"{name}\nHeld-out accuracy {model.score(Xte, yte):.3f}")
        ax.set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    fig.tight_layout()
    save(fig, "pr-04-kernels-nonlinear-classifiers", "decision-boundaries.png")


if __name__ == "__main__":
    risk()
    density()
    boundaries()
