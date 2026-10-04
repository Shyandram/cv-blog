"""NLP 09 figure: top-k and nucleus truncation on synthetic next-token distributions."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-09-language-modeling-transformers"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def nucleus_size(p, top_p):
    return int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), top_p) + 1)


cases = {
    "Peaked context (one obvious next token)": softmax(np.array([5.0, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5])),
    "Flat context (many reasonable next tokens)": softmax(np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])),
}
k, top_p = 3, 0.9
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
for ax, (title, p) in zip(axes, cases.items()):
    n = nucleus_size(p, top_p)
    x = np.arange(len(p))
    colors = ["#1f77b4" if i < min(k, n) else "#ff7f0e" if i < max(k, n) else "#cccccc" for i in x]
    ax.bar(x, p, color=colors)
    ax.axvline(k - 0.5, color="#1f77b4", ls="--", lw=1.5, label=f"top-k cut (k={k})")
    ax.axvline(n - 0.5, color="#d62728", ls=":", lw=2, label=f"nucleus cut (p={top_p}): {n} token{'s' if n > 1 else ''}")
    ax.set(title=title, xlabel="Candidate tokens, sorted by probability", xticks=x,
           xticklabels=[f"t{i+1}" for i in x])
    ax.legend(loc="center right", fontsize=10)
axes[0].set_ylabel("Next-token probability")
fig.suptitle("A fixed k ignores the shape of the distribution; a nucleus adapts its size", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "truncation.png", dpi=150)
plt.close(fig)
print("wrote", OUT / "truncation.png")
