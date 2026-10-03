"""Figures for PR 10 (recognition capstone).

Writes only to src/assets/pattern-recognition/pr-10-recognition-capstone/.
Everything is computed from scikit-learn's bundled digits dataset with the
same splits and seeds as the tutorial code. Run from the repository root:

    .venv/bin/python scripts/figures/pr_10_recognition_capstone_extra.py
"""
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score,
                                     learning_curve, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", message=".*within_class_std_dev_.*")

OUT = Path(__file__).resolve().parents[2] / "src/assets/pattern-recognition/pr-10-recognition-capstone"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "white",
})


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path, f"{path.stat().st_size / 1024:.0f} KB")


digits = load_digits()
X, y = digits.data, digits.target
idx = np.arange(len(y))
dev, test = train_test_split(idx, test_size=0.25, stratify=y, random_state=0)
# The configuration the tutorial's nested search selects on development data.
chosen = make_pipeline(PCA(30, random_state=0), SVC(C=10, gamma=0.001))


# 1. Pipeline diagram ---------------------------------------------------------
def pipeline_figure():
    steps = [
        ("1  Frame the task", "loss, classes, reject option", "PR 01"),
        ("2  Split & lock test", "sampling unit, stratify", "PR 09"),
        ("3  Baselines", "chance, templates, k-NN", "PR 01, 02, 06"),
        ("4  Features", "scaling, PCA, LDA", "PR 05"),
        ("5  Classifiers", "linear, kernel, ensembles", "PR 03, 04, 09"),
        ("6  Select by CV", "nested CV, paired tests", "PR 09"),
        ("7  Error analysis", "out-of-fold confusions", "PR 07, 09"),
        ("8  Test once", "metrics + counts", "PR 09"),
        ("9  Reject & shift", "coverage, unknowns, stress", "PR 01, 08"),
    ]
    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.axis("off")
    colors = [BLUE, BLUE, AQUA, AQUA, AQUA, VIOLET, VIOLET, ORANGE, ORANGE]
    for i, ((title, sub, ref), c) in enumerate(zip(steps, colors)):
        row, col = divmod(i, 3)
        if row == 1:
            col = 2 - col  # snake layout
        x0, y0 = col + 0.06, 2 - row + 0.12
        box = FancyBboxPatch((x0, y0), 0.88, 0.74,
                             boxstyle="round,pad=0.0,rounding_size=0.06",
                             fc="white", ec=c, lw=2)
        ax.add_patch(box)
        ax.text(x0 + 0.44, y0 + 0.53, title, ha="center", va="center",
                fontsize=11.5, weight="bold", color=INK)
        ax.text(x0 + 0.44, y0 + 0.33, sub, ha="center", va="center",
                fontsize=10, color=MUTED)
        ax.text(x0 + 0.44, y0 + 0.13, ref, ha="center", va="center",
                fontsize=10, color=c, weight="bold")
    arrow = dict(arrowstyle="-|>", color=MUTED, lw=1.5)
    for row in range(3):
        y_mid = 2 - row + 0.49
        for col in range(2):
            if row == 1:
                ax.annotate("", (col + 0.94, y_mid), (col + 1.06, y_mid), arrowprops=arrow)
            else:
                ax.annotate("", (col + 1.06, y_mid), (col + 0.94, y_mid), arrowprops=arrow)
    ax.annotate("", (2.5, 1.86), (2.5, 2.12), arrowprops=arrow)
    ax.annotate("", (0.5, 0.86), (0.5, 1.12), arrowprops=arrow)
    ax.text(1.5, -0.02, "Steps 1-7 use development data only; the locked test set is opened once, "
            "in step 8, and its results never feed back into earlier choices.",
            ha="center", va="top", fontsize=10, color=ORANGE)
    save(fig, "capstone-pipeline.png")


# 2. Data overview -------------------------------------------------------------
def overview_figure():
    fig = plt.figure(figsize=(10.4, 3.8))
    # Left: two example rows and one class-mean row; right: PCA scatter.
    axes = fig.subplots(3, 10, gridspec_kw=dict(left=0.0, right=0.48, wspace=0.08, hspace=0.12))
    for k in range(10):
        ex = dev[y[dev] == k][:2]
        for r in range(2):
            axes[r, k].imshow(X[ex[r]].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
        axes[2, k].imshow(X[dev][y[dev] == k].mean(0).reshape(8, 8), cmap="Blues",
                          vmin=0, vmax=16)
        axes[0, k].set_title(str(k), fontsize=11)
        for r in range(3):
            axes[r, k].set_xticks([])
            axes[r, k].set_yticks([])
    axes[0, 0].set_ylabel("sample", fontsize=10)
    axes[1, 0].set_ylabel("sample", fontsize=10)
    axes[2, 0].set_ylabel("mean", fontsize=10)
    ax = fig.add_axes([0.56, 0.1, 0.42, 0.8])
    Z = PCA(2, random_state=0).fit(X[dev]).transform(X[dev])
    cmap = plt.get_cmap("tab10")
    for k in range(10):
        m = y[dev] == k
        ax.scatter(Z[m, 0], Z[m, 1], s=8, color=cmap(k), alpha=0.7, lw=0)
        cx, cy = np.median(Z[m], axis=0)
        ax.text(cx, cy, str(k), fontsize=13, weight="bold", ha="center", va="center",
                color=INK, bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=cmap(k), lw=1.5))
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Development set, first two principal components", fontsize=11)
    ax.grid(color=GRID, lw=0.6)
    save(fig, "digits-overview.png")


# 3. Model comparison + learning curve ---------------------------------------
def comparison_figure():
    cands = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "naive Bayes": GaussianNB(),
        "class mean": NearestCentroid(),
        "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "random forest": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=1),
        "scaled PCA+SVM": make_pipeline(StandardScaler(), PCA(30, random_state=0),
                                        SVC(C=10, gamma="scale")),
        "3-NN": KNeighborsClassifier(n_neighbors=3),
        "tuned PCA+SVM": chosen,
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
    scores = {n: cross_val_score(m, X[dev], y[dev], cv=cv) for n, m in cands.items()}
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.0),
                               gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.35))
    names = [n for n in cands if n != "majority"]
    for i, n in enumerate(names):
        s = scores[n]
        c = ORANGE if n == "tuned PCA+SVM" else BLUE
        a.plot([s.min(), s.max()], [i, i], color=c, lw=2, alpha=0.5)
        a.scatter(s, np.full(len(s), i), s=14, color=c, alpha=0.6, lw=0)
        a.scatter([s.mean()], [i], s=60, color=c, edgecolor="white", lw=1.5, zorder=3)
    a.set_yticks(range(len(names)))
    a.set_yticklabels(names)
    a.set_xlabel("fold accuracy (5-fold x 2 repeats, development set)")
    a.set_title(f"Candidates (majority-class baseline = {scores['majority'].mean():.2f})",
                fontsize=11)
    a.grid(axis="x", color=GRID, lw=0.6)
    sizes, tr, va = learning_curve(chosen, X[dev], y[dev],
                                   cv=StratifiedKFold(5, shuffle=True, random_state=3),
                                   train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0])
    b.plot(sizes, tr.mean(1), "-o", color=BLUE, lw=2, ms=6, label="training")
    b.plot(sizes, va.mean(1), "-o", color=ORANGE, lw=2, ms=6, label="validation")
    b.fill_between(sizes, va.mean(1) - va.std(1), va.mean(1) + va.std(1),
                   color=ORANGE, alpha=0.15, lw=0)
    b.set_xlabel("training examples")
    b.set_ylabel("accuracy")
    b.set_title("Learning curve of the selected pipeline", fontsize=11)
    b.legend(frameon=False, loc="lower right")
    b.grid(color=GRID, lw=0.6)
    save(fig, "model-comparison.png")


# 4. Out-of-fold error analysis ----------------------------------------------
def error_figure():
    oof = cross_val_predict(chosen, X[dev], y[dev],
                            cv=StratifiedKFold(5, shuffle=True, random_state=3))
    cm = confusion_matrix(y[dev], oof)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    fig = plt.figure(figsize=(10.4, 4.3))
    a = fig.add_axes([0.0, 0.08, 0.42, 0.84])
    im = a.imshow(off, cmap="Oranges", vmin=0)
    for i in range(10):
        for j in range(10):
            if off[i, j]:
                a.text(j, i, off[i, j], ha="center", va="center", fontsize=10, color=INK)
            elif i == j:
                a.text(j, i, cm[i, i], ha="center", va="center", fontsize=8, color=MUTED)
    a.set_xticks(range(10))
    a.set_yticks(range(10))
    a.set_xlabel("predicted")
    a.set_ylabel("true")
    a.set_title(f"Out-of-fold errors: {off.sum()} of {len(dev)}\n(diagonal = correct counts, grey)",
                fontsize=11)
    wrong = np.flatnonzero(oof != y[dev])
    n = len(wrong)
    cols = 6
    rows = int(np.ceil(n / cols))
    for t, w in enumerate(wrong):
        r, c = divmod(t, cols)
        ax = fig.add_axes([0.48 + c * 0.087, 0.92 - (r + 1) * (0.84 / rows), 0.08, 0.84 / rows - 0.04])
        ax.imshow(X[dev[w]].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{y[dev][w]}→{oof[w]}", fontsize=9.5, color=INK, pad=2)
    fig.text(0.74, 0.97, "Every misclassified development image (true→predicted)",
             ha="center", fontsize=11)
    save(fig, "dev-errors.png")


# 5. Robustness ---------------------------------------------------------------
def robustness_figure():
    model = make_pipeline(PCA(30, random_state=0), SVC(C=10, gamma=0.001)).fit(X[dev], y[dev])
    rng = np.random.default_rng(0)

    def shift_right(A, k=1):
        imgs = A.reshape(-1, 8, 8)
        out = np.zeros_like(imgs)
        out[:, :, k:] = imgs[:, :, :-k]
        return out.reshape(len(A), 64)

    conds = {
        "clean": X[test],
        "noise sd=2": np.clip(X[test] + rng.normal(0, 2, X[test].shape), 0, 16),
        "noise sd=4": np.clip(X[test] + rng.normal(0, 4, X[test].shape), 0, 16),
        "contrast x0.5": X[test] * 0.5,
        "shift 1 px": shift_right(X[test], 1),
    }
    accs = {k: (model.predict(v) == y[test]).mean() for k, v in conds.items()}
    print({k: round(v, 3) for k, v in accs.items()})
    fig = plt.figure(figsize=(10.0, 3.6))
    a = fig.add_axes([0.07, 0.15, 0.5, 0.75])
    names = list(accs)
    vals = [accs[n] for n in names]
    bars = a.barh(range(len(names)), vals, color=[BLUE] + [ORANGE] * 4, height=0.6)
    for i, v in enumerate(vals):
        a.text(v - 0.01, i, f"{v:.3f}", ha="right", va="center", color="white", fontsize=10,
               weight="bold")
    a.set_yticks(range(len(names)))
    a.set_yticklabels(names)
    a.invert_yaxis()
    a.set_xlim(0, 1)
    a.set_xlabel("test accuracy (model never retrained)")
    a.grid(axis="x", color=GRID, lw=0.6)
    i0 = test[0]
    for j, (name, A) in enumerate([("clean", conds["clean"]), ("noise sd=4", conds["noise sd=4"]),
                                   ("contrast x0.5", conds["contrast x0.5"]),
                                   ("shift 1 px", conds["shift 1 px"])]):
        ax = fig.add_axes([0.62 + j * 0.095, 0.35, 0.085, 0.4])
        ax.imshow(A[0].reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=9)
    fig.text(0.81, 0.86, f"one test image (label {y[i0]})", ha="center", fontsize=10.5)
    save(fig, "robustness.png")


# 6. Open-set: confidences + coverage-risk ------------------------------------
def openset_figure():
    known = np.flatnonzero(y < 6)
    train, rest = train_test_split(known, test_size=0.4, stratify=y[known], random_state=10)
    valid, ktest = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=10)
    unknown = np.flatnonzero(y >= 8)
    model = make_pipeline(StandardScaler(), PCA(n_components=24, random_state=10),
                          SVC(C=4.0, gamma=0.02, probability=True, random_state=10))
    model.fit(X[train], y[train])
    thr = np.quantile(model.predict_proba(X[valid]).max(1), 0.05)
    pk = model.predict_proba(X[ktest])
    ck, predk = pk.max(1), model.classes_[pk.argmax(1)]
    cu = model.predict_proba(X[unknown]).max(1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 3.8), gridspec_kw=dict(wspace=0.3))
    bins = np.linspace(0, 1, 26)
    a.hist(ck, bins=bins, color=BLUE, alpha=0.75, label=f"known test (0-5), n={len(ck)}")
    a.hist(cu, bins=bins, color=ORANGE, alpha=0.75, label=f"unknown test (8-9), n={len(cu)}")
    a.axvline(thr, color=INK, ls="--", lw=1.2)
    a.text(thr - 0.02, a.get_ylim()[1] * 0.55, f"threshold\n{thr:.2f}", ha="right", va="top",
           fontsize=9.5)
    a.set_xlabel("maximum class probability")
    a.set_ylabel("count")
    a.set_title("Confidence is not a reliable unknown detector", fontsize=11)
    a.legend(frameon=False, loc="upper left", fontsize=9.5)
    order = np.argsort(-ck)
    errs = (predk != y[ktest])[order]
    n_acc = np.arange(1, len(order) + 1)
    risk = np.cumsum(errs) / n_acc
    b.plot(n_acc / len(order), risk, color=BLUE, lw=2)
    cov = (ck >= thr).mean()
    b.axvline(cov, color=INK, ls="--", lw=1.2)
    b.text(cov - 0.02, risk.max() * 0.9, f"coverage at\nthreshold = {cov:.2f}", ha="right",
           va="top", fontsize=9.5)
    b.set_xlabel("coverage (fraction of known test accepted)")
    b.set_ylabel("selective error")
    b.set_title("Coverage-risk curve (known classes only)", fontsize=11)
    b.grid(color=GRID, lw=0.6)
    save(fig, "open-set.png")


if __name__ == "__main__":
    pipeline_figure()
    overview_figure()
    comparison_figure()
    error_figure()
    robustness_figure()
    openset_figure()
