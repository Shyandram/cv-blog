"""Figures for the tutorial ML 02 - Neural Networks (ml-02-neural-networks).

Run:  .venv/bin/python scripts/figures/ml_02_neural_networks.py
Writes PNGs to src/assets/ml/ml-02-neural-networks/.
Uses only synthetic data and scikit-learn toy datasets; CPU only.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import torch
import torch.nn as nn
from scipy.special import erf
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parents[2] / "src/assets/ml/ml-02-neural-networks"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white"})
DPI = 150
C = ["#2a6fdb", "#e4572e", "#17a398", "#f3a712", "#8e5ad6", "#555555"]
torch.set_num_threads(1)  # tiny models: one thread is fastest


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. perceptron vs logistic
def fig_perceptron():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    z = np.linspace(-6, 6, 600)
    ax = axes[0]
    ax.plot(z, (z > 0).astype(float), color=C[0], lw=2.2, label="step 1[z > 0] (perceptron)")
    ax.plot(z, 1 / (1 + np.exp(-z)), color=C[1], lw=2.2, label="sigmoid (logistic regression)")
    ax.set_xlabel("z = w·x + b"); ax.set_ylabel("output")
    ax.set_title("Hard decision vs probability"); ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    lab = np.array([0, 1, 1, 0])
    for k, m, c in [(0, "o", C[0]), (1, "s", C[1])]:
        ax.scatter(*pts[lab == k].T, s=160, marker=m, color=c, label=f"class {k}", zorder=3)
    xs = np.linspace(-0.4, 1.4, 10)
    ax.plot(xs, 1.5 - xs, "--", color=C[5], lw=1.5)
    ax.plot(xs, 0.5 - xs, "--", color=C[5], lw=1.5)
    ax.text(0.95, 0.62, "two lines (a hidden\nlayer) are needed", fontsize=10, color=C[5])
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.4); ax.set_aspect("equal")
    ax.set_title("XOR: no single line separates it")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.legend(loc="lower left", fontsize=10)
    save(fig, "perceptron-vs-logistic.png")


# ---------------------------------------------------------------- 2. activations
def fig_activations():
    x = np.linspace(-4, 4, 800)
    sig = 1 / (1 + np.exp(-x))
    Phi = 0.5 * (1 + erf(x / np.sqrt(2)))
    phi = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    a = 0.1
    funcs = {
        "sigmoid": (sig, sig * (1 - sig)),
        "tanh": (np.tanh(x), 1 - np.tanh(x) ** 2),
        "ReLU": (np.maximum(x, 0), (x > 0).astype(float)),
        "Leaky ReLU (a=0.1)": (np.where(x > 0, x, a * x), np.where(x > 0, 1.0, a)),
        "GELU": (x * Phi, Phi + x * phi),
        "SiLU / Swish": (x * sig, sig * (1 + x * (1 - sig))),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    styles = ["-", "-", "-", "--", "-", "-."]
    for (name, (f, df)), c, ls in zip(funcs.items(), C, styles):
        axes[0].plot(x, f, color=c, lw=2, ls=ls, label=name)
        axes[1].plot(x, df, color=c, lw=2, ls=ls, label=name)
    axes[0].set_ylim(-1.5, 4); axes[0].set_title("Activation f(x)")
    axes[1].set_ylim(-0.3, 1.3); axes[1].set_title("Derivative f'(x)")
    for ax in axes:
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
        ax.grid(alpha=0.3); ax.set_xlabel("x")
    axes[0].legend(fontsize=10, loc="upper left")
    save(fig, "activations.png")


# ---------------------------------------------------------------- helpers for MLPs
def mlp(width, depth, p_drop=0.0):
    layers, d_in = [], 2
    for _ in range(depth):
        layers += [nn.Linear(d_in, width), nn.ReLU()]
        if p_drop > 0:
            layers.append(nn.Dropout(p_drop))
        d_in = width
    layers.append(nn.Linear(d_in, 2))
    return nn.Sequential(*layers)


def t(a, dtype=torch.float32):
    return torch.tensor(a, dtype=dtype)


# ---------------------------------------------------------------- 3. decision boundaries
def fig_boundaries():
    X, y = make_moons(n_samples=400, noise=0.2, random_state=1)
    Xt, yt = t(X), t(y, torch.long)
    widths, depths = [2, 8, 64], [1, 3]
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 300),
                         np.linspace(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3, 200))
    G = t(np.c_[gx.ravel(), gy.ravel()])
    fig, axes = plt.subplots(len(depths), len(widths), figsize=(10.4, 5.6), sharex=True, sharey=True)
    for i, dpt in enumerate(depths):
        for j, w in enumerate(widths):
            torch.manual_seed(0)
            m = mlp(w, dpt)
            opt = torch.optim.Adam(m.parameters(), lr=1e-2)
            for _ in range(1500):
                loss = nn.functional.cross_entropy(m(Xt), yt)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pr = torch.softmax(m(G), 1)[:, 1].numpy().reshape(gx.shape)
                acc = (m(Xt).argmax(1) == yt).float().mean().item()
            ax = axes[i, j]
            ax.contourf(gx, gy, pr, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.55)
            ax.contour(gx, gy, pr, levels=[0.5], colors="k", linewidths=1.5)
            ax.scatter(X[:, 0], X[:, 1], c=np.array(C)[[0, 1]][y], s=8, edgecolors="none")
            ax.set_title(f"{dpt} hidden layer{'s' if dpt > 1 else ''}, width {w}\ntrain acc {acc:.2f}",
                         fontsize=10.5)
            ax.set_xticks([]); ax.set_yticks([])
    save(fig, "moons-width-depth.png")


# ---------------------------------------------------------------- 4. computation graph
def fig_graph():
    fig, ax = plt.subplots(figsize=(10.4, 3.9))
    ax.set_xlim(0, 13.1); ax.set_ylim(0, 4.6); ax.axis("off")
    nodes = [("x", 1.0, "#eeeeee"), ("z1 = W1 x + b1", 3.2, "#dbe7fb"), ("a1 = f(z1)", 5.4, "#dbe7fb"),
             ("z2 = W2 a1 + b2", 7.6, "#fde2d9"), ("p = softmax(z2)", 9.8, "#fde2d9"),
             ("L = -Σ y log p", 12.0, "#fff1cc")]
    yc = 2.8
    for label, xc, col in nodes:
        ax.add_patch(FancyBboxPatch((xc - 0.95, yc - 0.42), 1.9, 0.84, boxstyle="round,pad=0.05",
                                    fc=col, ec="k", lw=1.2))
        ax.text(xc, yc, label, ha="center", va="center", fontsize=10)
    for (_, x0, _), (_, x1, _) in zip(nodes[:-1], nodes[1:]):
        ax.add_patch(FancyArrowPatch((x0 + 1.0, yc + 0.15), (x1 - 1.0, yc + 0.15), arrowstyle="-|>",
                                     mutation_scale=14, color="k", lw=1.4))
    back = [r"$\delta_1=\delta_{a}\odot f'(z_1)$", r"$\delta_{a}=W_2^\top\delta_2$",
            r"$\delta_2=p-y$", r"$\partial L/\partial p$"]
    xs_back = [nodes[i][1] for i in range(1, 6)]
    yb = yc - 0.85
    for k in range(4):
        x1, x0 = xs_back[k], xs_back[k + 1]      # arrow from node x0 back to node x1
        ax.add_patch(FancyArrowPatch((x0 - 0.2, yb), (x1 + 0.2, yb), arrowstyle="-|>",
                                     mutation_scale=14, color=C[1], lw=1.5, ls="--"))
        ax.text((x0 + x1) / 2, yb - 0.5, back[k], ha="center", fontsize=10.5, color=C[1])
    # parameter gradients
    ax.text(3.2, 4.15, r"$\partial L/\partial W_1=\delta_1 x^\top$", ha="center", fontsize=10, color=C[2])
    ax.text(7.6, 4.15, r"$\partial L/\partial W_2=\delta_2 a_1^\top$", ha="center", fontsize=10, color=C[2])
    ax.text(0.2, 0.25, "solid: forward pass (compute values)     dashed: backward pass (chain rule, "
            "right to left)", fontsize=10, color=C[5])
    save(fig, "backprop-graph.png")


# ---------------------------------------------------------------- 5. overfitting curves
def run_curve(Xtr, ytr, Xva, yva, wd=0.0, p_drop=0.0, epochs=600):
    torch.manual_seed(0)
    m = mlp(256, 2, p_drop)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=wd)
    tr, va = [], []
    for _ in range(epochs):
        m.train()
        loss = nn.functional.cross_entropy(m(Xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            tr.append(nn.functional.cross_entropy(m(Xtr), ytr).item())
            va.append(nn.functional.cross_entropy(m(Xva), yva).item())
    return np.array(tr), np.array(va)


def fig_overfit():
    X, y = make_moons(n_samples=2100, noise=0.35, random_state=3)
    Xtr, Xva, ytr, yva = train_test_split(X, y, train_size=100, random_state=0)
    Xtr, Xva, ytr, yva = t(Xtr), t(Xva), t(ytr, torch.long), t(yva, torch.long)
    runs = {"no regularization": {}, "weight decay 5 (AdamW)": {"wd": 5.0},
            "dropout p=0.5": {"p_drop": 0.5}}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for (name, kw), c in zip(runs.items(), C):
        tr, va = run_curve(Xtr, ytr, Xva, yva, **kw)
        axes[0].plot(tr, color=c, lw=1.8, label=name)
        axes[1].plot(va, color=c, lw=1.8, label=name)
        best = int(va.argmin())
        axes[1].plot(best, va[best], "o", color=c, ms=7)
        print(f"  {name}: final train {tr[-1]:.3f}, final val {va[-1]:.3f}, best val {va[best]:.3f} @ {best}")
    axes[0].set_title("Training loss"); axes[1].set_title("Validation loss (dot = minimum)")
    for ax in axes:
        ax.set_xlabel("epoch (full batch)"); ax.set_ylabel("cross-entropy"); ax.grid(alpha=0.3)
    axes[0].set_yscale("log")
    axes[1].set_ylim(0.15, 1.0)
    axes[1].legend(fontsize=10)
    save(fig, "overfitting-regularization.png")


# ---------------------------------------------------------------- 6. normalization axes
def fig_norm():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6))
    B, D = 6, 8
    titles = ["BatchNorm: per feature,\nover the mini-batch", "LayerNorm: per sample,\nover its features",
              "RMSNorm: per sample,\nRMS only (no mean)"]
    for k, ax in enumerate(axes):
        for i in range(B):
            for j in range(D):
                hl = (k == 0 and j == 2) or (k > 0 and i == 3)
                ax.add_patch(Rectangle((j, B - 1 - i), 1, 1, fc=(C[0] if k == 0 else C[1] if k == 1 else C[2])
                                       if hl else "#f2f2f2", ec="k", lw=0.6, alpha=0.85 if hl else 1))
        ax.set_xlim(-0.2, D + 0.2); ax.set_ylim(-1.1, B + 0.2); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(titles[k], fontsize=10.5)
        ax.text(D / 2, -0.7, "features (d) →", ha="center", fontsize=10)
        ax.text(-0.5, B / 2, "samples (N) →", rotation=90, va="center", ha="center", fontsize=10)
    save(fig, "normalization-axes.png")


# ---------------------------------------------------------------- 7. LR schedules
def fig_lr():
    T = 1000
    s = np.arange(T)
    step = 1e-3 * 0.1 ** (s // 400)
    cosine = 0.5 * 1e-3 * (1 + np.cos(np.pi * s / T))
    warm = 100
    wc = np.where(s < warm, 1e-3 * (s + 1) / warm,
                  1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + np.cos(np.pi * (s - warm) / (T - warm))))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(s, step, color=C[0], lw=2, label="step decay (x0.1 every 400 steps)")
    ax.plot(s, cosine, color=C[1], lw=2, label="cosine annealing")
    ax.plot(s, wc, color=C[2], lw=2, ls="--", label="linear warmup + cosine to 10%")
    ax.set_xlabel("training step"); ax.set_ylabel("learning rate"); ax.grid(alpha=0.3)
    ax.legend(fontsize=10); ax.set_title("Common learning-rate schedules (peak 1e-3)")
    save(fig, "lr-schedules.png")


if __name__ == "__main__":
    fig_perceptron()
    fig_activations()
    fig_graph()
    fig_norm()
    fig_lr()
    fig_boundaries()
    fig_overfit()
