"""Extra figures for DL 07 (world models).

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/dl_07_world_models_extra.py
Synthetic data only, CPU, a few seconds. Writes only to src/assets/deep-learning/dl-07-world-models/.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "src/assets/deep-learning/dl-07-world-models"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, GREY, PURPLE = "#2a6fdb", "#e4572e", "#17a398", "#6b7280", "#7b4fd0"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name, f"{(OUT / name).stat().st_size / 1e3:.0f} kB")


def box(ax, x, y, w, h, text, fc="#eef3fc", ec=BLUE, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color="k", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=1.4, color=color, linestyle=ls))


def blur():
    x = np.arange(64)
    frame = lambda c: np.exp(-0.5 * ((x - c) / 2.0) ** 2)
    f1, f2 = frame(20), frame(44)
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.9), sharey=True)
    axes[0].plot(x, f1, color=BLUE, label="future A (p = 0.5)")
    axes[0].plot(x, f2, color=GREEN, label="future B (p = 0.5)")
    axes[0].set_title("Two equally likely futures")
    axes[1].plot(x, (f1 + f2) / 2, color=ORANGE)
    axes[1].set_title("MSE-optimal prediction: mean")
    axes[2].plot(x, f1, color=PURPLE)
    axes[2].set_title("A sample from a generative model")
    for ax in axes:
        ax.set_xlabel("Pixel position")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Brightness")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[1].text(32, 0.8, "two half-bright ghosts\nexpected MSE 0.028", ha="center", fontsize=9)
    axes[2].text(45, 0.8, "sharp and plausible\nexpected MSE 0.055", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "mean-vs-sample.png")


def ensemble():
    rng = np.random.default_rng(0)
    true_reward = lambda a: np.sin(3 * a) * np.exp(-0.3 * a ** 2)
    a_data = rng.uniform(-1, 1, 200)
    r_data = true_reward(a_data) + 0.05 * rng.normal(size=200)
    models = []
    for k in range(5):
        idx = rng.integers(0, 200, 200)
        m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, random_state=k)
        models.append(m.fit(a_data[idx, None], r_data[idx]))
    grid = np.linspace(-3, 3, 601)
    preds = np.stack([m.predict(grid[:, None]) for m in models])
    mean, std = preds.mean(0), preds.std(0)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.axvspan(-1, 1, color=GREY, alpha=.12, label="actions present in the data")
    for i, p in enumerate(preds):
        ax.plot(grid, p, color=BLUE, lw=0.9, alpha=.55, label="5 ensemble members" if i == 0 else None)
    ax.fill_between(grid, mean - 2 * std, mean + 2 * std, color=BLUE, alpha=.12, label="mean ± 2 std")
    ax.plot(grid, true_reward(grid), color="k", lw=2, ls="--", label="true reward (unknown)")
    ax.scatter(a_data, r_data, s=6, color=GREY, alpha=.6)
    a1 = grid[np.argmax(preds[0])]
    a2 = grid[np.argmax(mean - 2 * std)]
    ax.scatter([a1], [preds[0].max()], s=90, marker="*", color=ORANGE, zorder=5,
               label=f"single model's choice (true {true_reward(a1):.2f})")
    ax.scatter([a2], [(mean - 2 * std).max()], s=90, marker="*", color=GREEN, zorder=5,
               label=f"pessimistic choice (true {true_reward(a2):.2f})")
    ax.set_xlabel("Action a")
    ax.set_ylabel("Predicted reward")
    ax.set_title("A planner exploits extrapolation; disagreement reveals it")
    ax.legend(fontsize=8.5, loc="upper center", ncol=2)
    ax.set_ylim(-2.2, 3.2)
    ax.grid(alpha=.25)
    save(fig, "ensemble-exploitation.png")


def three_uses():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    titles = ["Plan online (MPC)", "Learn in imagination", "Search with a value-equivalent model"]
    examples = ["PlaNet, PETS, TD-MPC2", "World Models, Dreamer, IRIS, DIAMOND", "MuZero"]
    for ax, t, e in zip(axes, titles, examples):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        ax.set_title(t, fontsize=12, weight="bold")
        ax.text(5, 0.3, e, ha="center", fontsize=10, color=GREY, style="italic")
    ax = axes[0]
    box(ax, 0.3, 7.2, 3, 1.4, "observe\nstate", fc="#fdf0ec", ec=ORANGE)
    box(ax, 4.2, 7.2, 5.4, 1.4, "sample many action\nsequences", fc="#eef3fc")
    box(ax, 4.2, 4.6, 5.4, 1.6, "roll each through the\nmodel; score cost", fc="#eef3fc")
    box(ax, 4.2, 2.0, 5.4, 1.6, "execute FIRST action\nof the best sequence", fc="#eafaf6", ec=GREEN)
    arrow(ax, 3.3, 7.9, 4.2, 7.9); arrow(ax, 6.9, 7.2, 6.9, 6.2); arrow(ax, 6.9, 4.6, 6.9, 3.6)
    arrow(ax, 4.2, 2.8, 1.8, 2.8); arrow(ax, 1.8, 2.8, 1.8, 7.2)
    ax.text(1.0, 5.0, "replan\nevery\nstep", ha="center", fontsize=9, color=GREY)
    ax = axes[1]
    box(ax, 0.3, 7.2, 4.2, 1.4, "real experience\n(replay buffer)", fc="#fdf0ec", ec=ORANGE)
    box(ax, 5.4, 7.2, 4.3, 1.4, "train world\nmodel", fc="#eef3fc")
    box(ax, 5.4, 4.4, 4.3, 1.8, "imagine thousands\nof latent rollouts", fc="#eef3fc")
    box(ax, 0.3, 4.4, 4.2, 1.8, "train policy + value\non imagined data", fc="#f3eefc", ec=PURPLE)
    box(ax, 0.3, 1.6, 4.2, 1.6, "act in real world\nwith the policy", fc="#eafaf6", ec=GREEN)
    arrow(ax, 4.5, 7.9, 5.4, 7.9); arrow(ax, 7.5, 7.2, 7.5, 6.2); arrow(ax, 5.4, 5.3, 4.5, 5.3)
    arrow(ax, 2.4, 4.4, 2.4, 3.2)
    arrow(ax, 0.05, 2.4, 0.05, 7.9, color=GREY)
    ax.text(-0.25, 5.3, "new data", fontsize=9, color=GREY, rotation=90, va="center")
    ax = axes[2]
    for (cx, cy, lab) in [(5, 8.2, "s⁰"), (2.6, 5.6, "s¹"), (5, 5.6, "s¹"), (7.4, 5.6, "s¹"),
                          (1.6, 3.0, "s²"), (3.6, 3.0, "s²"), (8.4, 3.0, "s²")]:
        ax.add_patch(Circle((cx, cy), 0.62, fc="#eef3fc", ec=BLUE, lw=1.3))
        ax.text(cx, cy, lab, ha="center", va="center", fontsize=10)
    for (x0, y0, x1, y1) in [(5, 7.6, 2.6, 6.2), (5, 7.6, 5, 6.2), (5, 7.6, 7.4, 6.2),
                             (2.6, 5.0, 1.6, 3.6), (2.6, 5.0, 3.6, 3.6), (7.4, 5.0, 8.4, 3.6)]:
        ax.plot([x0, x1], [y0, y1], color=GREY, lw=1.2)
    ax.text(5, 9.3, "h: observation → hidden state s⁰", ha="center", fontsize=9)
    ax.text(5, 1.6, "g: (s, a) → next s, reward    f: s → policy, value\n"
            "s is trained to predict reward, value, policy — not pixels",
            ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "three-uses.png")


def rssm():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(-1.3, 12.6); ax.set_ylim(0, 6.2); ax.axis("off")
    for i, x in enumerate([1.0, 5.0, 9.0]):
        t = ["t−1", "t", "t+1"][i]
        box(ax, x, 3.2, 1.4, 0.9, f"h({t})", fc="#eef3fc", ec=BLUE, fs=11, bold=True)
        box(ax, x, 1.4, 1.4, 0.9, f"z({t})", fc="#f3eefc", ec=PURPLE, fs=11, bold=True)
        box(ax, x + 1.9, 1.4, 1.3, 0.9, f"o({t})", fc="#fdf0ec", ec=ORANGE, fs=10)
        box(ax, x - 0.1, 5.0, 1.6, 0.8, f"r, ô, cont", fc="#eafaf6", ec=GREEN, fs=9)
        arrow(ax, x + 0.7, 3.2, x + 0.7, 2.3, color=PURPLE)
        arrow(ax, x + 1.9, 1.85, x + 1.4, 1.85, color=ORANGE, ls="--")
        arrow(ax, x + 0.7, 4.1, x + 0.7, 5.0, color=GREEN)
        if i < 2:
            arrow(ax, x + 1.4, 3.65, x + 4.0, 3.65)
            ax.annotate("", xy=(x + 4.0, 3.45), xytext=(x + 1.4, 1.85),
                        arrowprops=dict(arrowstyle="->", lw=1.2, color="k"))
            ax.text(x + 2.7, 4.0, f"a({['t−1', 't'][i]})", ha="center", fontsize=10)
    ax.text(6.0, 0.4, "solid purple: prior p(z | h), used when imagining      "
            "dashed orange: posterior q(z | h, o) also sees the image (training)",
            ha="center", fontsize=9.5)
    ax.text(-1.25, 3.65, "deterministic\nmemory", fontsize=9, color=BLUE, ha="left", va="center")
    ax.text(-1.25, 1.85, "stochastic\nstate", fontsize=9, color=PURPLE, ha="left", va="center")
    save(fig, "rssm.png")


if __name__ == "__main__":
    blur()
    ensemble()
    three_uses()
    rssm()
