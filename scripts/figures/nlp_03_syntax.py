"""Two original synthetic structure diagrams for NLP03."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-03-syntax-parsing"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


words = ["we", "saw", "robots", "with", "cameras"]
fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.6))
for ax, attachment, title in zip(axes, [2, 1], ["Cameras modify robots", "Cameras modify seeing"]):
    ax.set(xlim=(-.5, 4.5), ylim=(-.3, 2.1))
    ax.axis("off")
    for i, word in enumerate(words):
        ax.text(i, 0, word, ha="center", fontsize=12)
    edges = [(1, 0, "subject"), (1, 2, "object"), (attachment, 3, "modifier"), (3, 4, "object")]
    for head, dependent, label in edges:
        height = .5 + .23 * abs(head-dependent)
        patch = FancyArrowPatch((head, .2), (dependent, .2),
                                connectionstyle=f"arc3,rad={-.4 if head<dependent else .4}",
                                arrowstyle="-|>", mutation_scale=12, color="#2866ae")
        ax.add_patch(patch)
        ax.text((head+dependent)/2, height, label, ha="center", fontsize=10)
    ax.set_title(title, fontsize=12)
fig.suptitle("Same words, different attachment (illustrative dependencies)")
fig.tight_layout()
save(fig, "attachment.png")

chart = np.zeros((6, 6))
for i in range(5):
    for j in range(i+1, 6):
        chart[i, j] = j-i
fig, ax = plt.subplots(figsize=(7.4, 4.8))
ax.imshow(np.ma.masked_where(chart==0, chart), cmap="Blues", vmin=0, vmax=5)
labels = {(0,1):"NP", (1,2):"V", (2,3):"NP", (3,4):"P", (4,5):"NP",
          (1,3):"VP", (3,5):"PP", (2,5):"NP", (1,5):"VP × 2", (0,3):"S", (0,5):"S × 2"}
for (i, j), label in labels.items():
    ax.text(j, i, label, ha="center", va="center", fontsize=11,
            color="white" if j-i>=4 else "#163b65")
ax.set_xticks(range(6), range(6))
ax.set_yticks(range(6), range(6))
ax.set(xlabel="Exclusive end boundary j", ylabel="Start boundary i",
       title="A chart shares spans across two complete parses")
fig.tight_layout()
save(fig, "chart.png")
