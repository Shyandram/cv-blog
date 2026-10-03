"""Evaluation-metric figures for DL 02 · Deep Image Classification.

Run:  .venv/bin/python scripts/figures/dl_02_eval_metrics.py
Writes confusion-matrix.png and roc-pr.png to
src/assets/deep-learning/dl-02-image-classification/ and prints the numbers
quoted in the tutorial text. CPU only, a few seconds.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src/assets/deep-learning/dl-02-image-classification"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})

BLUE, ORANGE, GRAY = "#3b6fb6", "#e08a2c", "#666666"


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p.name}  {p.stat().st_size / 1024:.0f} KB")


def fig_confusion_matrix():
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
    clf = LogisticRegression(max_iter=2000).fit(X_tr[:200], y_tr[:200])  # few labels -> visible errors
    pred = clf.predict(X_te)
    cm = confusion_matrix(y_te, pred, normalize="true")
    print(f"digits (200 train): accuracy {np.mean(pred == y_te):.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            v = cm[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.6 else "black")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("Confusion matrix, normalized by row (recall on the diagonal)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "confusion-matrix.png")


def fig_roc_pr():
    X, y = make_classification(n_samples=20000, n_features=20, n_informative=6, weights=[0.99, 0.01],
                               class_sep=1.0, flip_y=0.0, random_state=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
    models = {"Logistic regression": LogisticRegression(max_iter=2000), "Gaussian naive Bayes": GaussianNB()}
    prev = y_te.mean()
    print(f"positives in test set: {y_te.sum()} of {len(y_te)} ({prev:.3%})")
    print(f"all-negative classifier accuracy: {1 - prev:.3f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for (name, m), c in zip(models.items(), [BLUE, ORANGE]):
        s = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        auc, ap = roc_auc_score(y_te, s), average_precision_score(y_te, s)
        print(f"{name}: ROC-AUC {auc:.3f}  AP {ap:.3f}")
        fpr, tpr, _ = roc_curve(y_te, s)
        prec, rec, _ = precision_recall_curve(y_te, s)
        a1.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {auc:.2f})")
        a2.plot(rec, prec, color=c, lw=2, label=f"{name} (AP {ap:.2f})")
    a1.plot([0, 1], [0, 1], "--", color=GRAY, lw=1, label="random (AUC 0.50)")
    a2.axhline(prev, ls="--", color=GRAY, lw=1, label=f"random (AP = prevalence {prev:.2f})")
    a1.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve",
           xlim=(0, 1), ylim=(0, 1.01))
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve", xlim=(0, 1), ylim=(0, 1.01))
    for a in (a1, a2):
        a.legend(loc="lower right" if a is a1 else "upper right", fontsize=9)
        a.grid(alpha=0.3)
    fig.suptitle("Imbalanced binary task: 1% positives", fontsize=12)
    fig.tight_layout()
    save(fig, "roc-pr.png")


if __name__ == "__main__":
    fig_confusion_matrix()
    fig_roc_pr()
