"""Original synthetic NLP07 figures. Writes only its own chapter folder."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-07-discourse-dialogue"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white"})
prior = np.array([.48, .48, .04])
ambiguous = prior * np.array([.6, .5, .2]); ambiguous /= ambiguous.sum()
explicit = ambiguous * np.array([.02, .95, .03]); explicit /= explicit.sum()
fig, ax = plt.subplots(figsize=(10.2, 4.1), layout="constrained")
for i, (values, label, color) in enumerate(zip(
        [prior, ambiguous, explicit], ["Prior", "Ambiguous observation", "Explicit Pump B"],
        ["#769db7", "#8553a8", "#b65332"])):
    ax.bar(np.arange(3) + (i - 1) * .23, values, width=.23, label=label, color=color)
ax.axhline(.9, ls="--", color="#444444", label="Toy action threshold")
ax.set(xticks=range(3), xticklabels=["Pump A", "Pump B", "Unknown"],
       ylabel="Constructed probability", ylim=(0, 1.1), title="Evidence changes the belief state")
ax.legend(loc="upper left")
fig.savefig(OUT / "belief.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(10.2, 4.1), layout="constrained")
turns = np.arange(5)
revision = [0, 1, 2, 2, 3]
ax.step(turns, revision, where="post", lw=2, color="#337d9f", label="Current revision")
ax.scatter(turns, revision, color="#337d9f", s=60)
ax.scatter([3], [2], marker="s", s=150, color="#8553a8", label="Confirmed revision 2")
ax.scatter([4], [2], marker="x", s=150, color="#b65332", label="Old confirmation invalid")
ax.set(xticks=turns, xticklabels=["Initial", "Set Pump A", "Set today", "Confirm", "Correct: Pump B"],
       yticks=range(4), ylabel="State revision", ylim=(-.3, 3.6),
       title="Confirmation belongs to a specific proposal")
ax.legend(loc="upper left")
fig.savefig(OUT / "state.png", dpi=150); plt.close(fig)
print("Generated NLP07 belief and state figures.")
