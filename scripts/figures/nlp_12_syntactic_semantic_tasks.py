"""Original synthetic diagrams for NLP 12: greedy heads vs a decoded tree, and coreference clusters."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-12-syntactic-semantic-tasks"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Heads match the tutorial's graph-based snippet: index 0 is ROOT.
words = ["ROOT", "we", "saw", "robots", "with", "cameras"]
panels = [
    ("Greedy best head per word: with <-> cameras form a cycle, no path to ROOT",
     [2, 0, 2, 5, 4], {4, 5}),
    ("Best-scoring valid tree (all 625 single-root trees enumerated)",
     [2, 0, 2, 5, 3], set()),
]
fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.8))
for ax, (title, heads, bad) in zip(axes, panels):
    ax.set(xlim=(-.5, 5.5), ylim=(-.4, 2.3))
    ax.axis("off")
    for i, w in enumerate(words):
        ax.text(i, 0, w, ha="center", fontsize=12, fontweight="bold" if i == 0 else None)
    for d, h in enumerate(heads, start=1):
        color = "#c0392b" if d in bad else "#2866ae"
        rad = -.45 if h < d else .45
        ax.add_patch(FancyArrowPatch((h, .22), (d, .22), connectionstyle=f"arc3,rad={rad}",
                                     arrowstyle="-|>", mutation_scale=13, color=color, lw=1.8))
    ax.set_title(title, fontsize=11)
fig.suptitle("Independent head choices need a tree decoder (invented arc scores)", fontsize=12)
fig.tight_layout()
save(fig, "greedy-vs-tree.png")

mentions = ["Pump A", "the pump", "it", "Valve B", "the valve"]
key = [[0, 1, 2], [3, 4]]
response = [[0, 1], [2, 3, 4]]
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
colors = ["#2866ae", "#d98c3a"]
for ax, clusters, title in zip(axes, [key, response], ["Key (annotation)", "Response (system)"]):
    ax.set(xlim=(-.6, 4.6), ylim=(-1.0, 1.2))
    ax.axis("off")
    for c, cluster in enumerate(clusters):
        lo, hi = min(cluster), max(cluster)
        ax.add_patch(FancyBboxPatch((lo - .45, -.35), hi - lo + .9, .7,
                                    boxstyle="round,pad=0.02", fc=colors[c], alpha=.18,
                                    ec=colors[c], lw=1.5))
    for i, m in enumerate(mentions):
        ax.text(i, 0, m, ha="center", va="center", fontsize=11)
    ax.set_title(title, fontsize=12)
axes[1].text(2, -.75, "'it' is merged with the valve cluster", ha="center", fontsize=10, color="#c0392b")
fig.suptitle("Same five mentions, two partitions: MUC 0.667, B3 0.733, CEAFe 0.800", fontsize=12)
fig.tight_layout()
save(fig, "coref-clusters.png")
