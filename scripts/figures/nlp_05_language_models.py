"""Causal visibility and temperature from original synthetic examples."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-05-language-models"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, mask, title in zip(axes, [np.tri(5), np.ones((5, 5))], ["Causal self-attention", "Bidirectional self-attention"]):
    ax.imshow(mask, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(5), yticks=range(5), xlabel="Key position", ylabel="Query position", title=title)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, "read" if mask[i, j] else "block", ha="center", va="center", color="white" if mask[i, j] else "#444")
fig.tight_layout()
fig.savefig(OUT / "masks.png", dpi=150)
plt.close(fig)

logits = np.array([2., 1., 0., -1.])
fig, ax = plt.subplots(figsize=(9, 4))
for offset, tau in zip([-.25,0,.25], [.5,1,2]):
    z = logits / tau
    probabilities = np.exp(z-z.max()) / np.exp(z-z.max()).sum()
    ax.bar(np.arange(4)+offset, probabilities, width=.24, label=f"Temperature {tau}")
ax.set(xticks=range(4), xticklabels=["Token A", "Token B", "Token C", "Token D"], ylabel="Sampling probability", ylim=(0,1), title="Same logits; different sampling distributions")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "temperature.png", dpi=150)
plt.close(fig)
