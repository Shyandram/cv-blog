"""Original synthetic diagrams for NLP02; CPU, no downloads."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-02-classification-embeddings"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, filename):
    fig.savefig(OUT / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
matrix = np.array([[1, 0, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]])
axes[0].imshow(matrix, cmap="Blues", vmin=0, vmax=1)
axes[0].set_xticks(range(6), ["refund", "invoice", "payment", "delivery", "parcel", "late"], rotation=45)
axes[0].set_yticks(range(4), ["message A", "message B", "message C", "message D"])
axes[0].set_title("Sparse lexical features")
points = np.array([[-1.2, .5], [-.8, -.2], [1.1, .5], [.8, -.2]])
axes[1].scatter(points[:2, 0], points[:2, 1], s=110, label="billing")
axes[1].scatter(points[2:, 0], points[2:, 1], s=110, label="delivery")
for name, (x, y) in zip("ABCD", points):
    axes[1].annotate(name, (x, y), xytext=(6, 4), textcoords="offset points")
axes[1].axvline(0, color="#666666", ls="--")
axes[1].set(xlim=(-1.7, 1.7), ylim=(-.7, 1.1), xlabel="Illustrative dimension 1",
            ylabel="Illustrative dimension 2", title="Dense document representation")
axes[1].legend()
fig.suptitle("Representations expose different evidence; coordinates are synthetic")
fig.tight_layout()
save(fig, "sparse-dense.png")

fig, ax = plt.subplots(figsize=(9.8, 4.2))
ax.set(xlim=(0, 10), ylim=(0, 4.2))
ax.axis("off")
steps = [("Text + label contract", "What may the model predict?"),
         ("Train-only representation", "Vocabulary and feature weights"),
         ("Classifier", "Scores over permitted labels"),
         ("Validation decision rule", "Threshold, cost, calibration"),
         ("Frozen test evaluation", "Errors, coverage, subgroups")]
for index, (title, sub) in enumerate(steps):
    x = .15 + index * 1.98
    ax.add_patch(FancyBboxPatch((x, 1.1), 1.72, 2.2, boxstyle="round,pad=.04",
                               facecolor="#edf3fb", edgecolor="#2866ae"))
    ax.text(x + .86, 2.7, title.replace(" ", "\n", 1), ha="center", va="center", fontsize=11)
    ax.text(x + .86, 1.65, sub.replace(" ", "\n", 2), ha="center", va="center", fontsize=10)
    if index != len(steps) - 1:
        ax.annotate("", (x + 1.93, 2.15), (x + 1.76, 2.15),
                    arrowprops={"arrowstyle": "->"})
ax.text(5, 3.85, "Classification is a complete decision pipeline", ha="center", fontsize=13)
ax.text(5, .45, "Held-out text must never fit the vocabulary, weights or threshold", ha="center", fontsize=11)
save(fig, "classification-contract.png")
