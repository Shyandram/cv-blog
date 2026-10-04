"""Figures for NLP 05 (recurrent language models); rerun from the project root.

The training curve and temperature figures re-execute the character-level
experiment exactly as printed in the English tutorial, so the plotted numbers
match the quoted output.
"""
import contextlib
import io
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src/assets/nlp/nlp-05-recurrent-neural-networks"
OUT.mkdir(parents=True, exist_ok=True)
MDX = ROOT / "src/content/tutorials/en/nlp/nlp-05-recurrent-neural-networks.mdx"
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})

# ---- Figure 2: an unrolled recurrent language model ----
fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.set(xlim=(-0.6, 10.2), ylim=(-0.4, 4.6))
ax.axis("off")
inputs = ["<s>", "the", "cat", "sat", "down"]
targets = ["the", "cat", "sat", "down", "</s>"]
for i, (x, y) in enumerate(zip(inputs, targets)):
    cx = 0.9 + 2.0 * i
    ax.text(cx, 0.0, x, ha="center", va="center", fontsize=12, family="monospace")
    ax.add_patch(FancyBboxPatch((cx - 0.45, 0.55), 0.9, 0.45, boxstyle="round,pad=.05",
                                fc="#fff6e0", ec="#b07d1a"))
    ax.text(cx, 0.775, "embed", ha="center", va="center", fontsize=10)
    ax.add_patch(FancyBboxPatch((cx - 0.55, 1.55), 1.1, 0.8, boxstyle="round,pad=.05",
                                fc="#eef4ff", ec="#3875cb", lw=1.5))
    ax.text(cx, 1.95, "RNN cell", ha="center", va="center", fontsize=10)
    ax.add_patch(FancyBboxPatch((cx - 0.5, 2.9), 1.0, 0.45, boxstyle="round,pad=.05",
                                fc="#f0f9ee", ec="#4b8741"))
    ax.text(cx, 3.125, "softmax", ha="center", va="center", fontsize=10)
    ax.text(cx, 4.0, y, ha="center", va="center", fontsize=12, family="monospace", color="#4b8741")
    for y0, y1 in [(0.2, 0.55), (1.05, 1.55), (2.4, 2.9), (3.4, 3.8)]:
        ax.add_patch(FancyArrowPatch((cx, y0), (cx, y1), arrowstyle="->", mutation_scale=12, lw=1.2))
    if i < 4:
        ax.add_patch(FancyArrowPatch((cx + 0.6, 1.95), (cx + 1.4, 1.95), arrowstyle="->",
                                     mutation_scale=14, lw=2, color="#3875cb"))
        ax.text(cx + 1.0, 2.12, f"h{i + 1}", ha="center", fontsize=10, color="#3875cb")
ax.text(-0.5, 0.0, "input", ha="left", va="center", fontsize=10, color="#555555")
ax.text(-0.5, 4.0, "target", ha="left", va="center", fontsize=10, color="#555555")
ax.set_title("Same cell at every step; the hidden state carries the prefix forward", pad=8)
fig.tight_layout()
fig.savefig(OUT / "rnn-lm-unrolled.png", dpi=150)
plt.close(fig)

# ---- Re-run the tutorial's experiment ----
text = MDX.read_text(encoding="utf-8")
blocks = re.findall(r"```python\n(.*?)```", text, re.S)
code = next(b for b in blocks if "class CharLSTM" in b)
ns = {"__name__": "nlp05_experiment"}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(code, "nlp05_experiment", "exec"), ns)
printed = buf.getvalue()
assert "test PPL=11.24" in printed and "mean test PPL over 3 seeds: 11.48" in printed, printed
runs, ng_valid = ns["runs"], ns["ng_valid"]

# ---- Figure 4: training curves ----
fig, ax = plt.subplots(figsize=(8.5, 4.8))
colors = ["#3875cb", "#cf5543", "#4b8741"]
for (seed, (_, hist, _)), color in zip(runs.items(), colors):
    ep = [h[0] for h in hist]
    ax.plot(ep, [h[1] for h in hist], color=color, lw=1.2, alpha=0.55, ls=":")
    ax.plot(ep, [h[2] for h in hist], color=color, lw=2, label=f"LSTM seed {seed} (validation)")
ax.axhline(ng_valid, color="black", ls="--", lw=1.4, label=f"interpolated n-gram (validation {ng_valid:.2f})")
ax.plot([], [], color="gray", ls=":", label="LSTM training (dotted)")
ax.set(xlabel="epoch", ylabel="per-character perplexity", ylim=(1.5, 22))
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="upper right")
ax.set_title("Character-level LSTM on the opening of Chapter I")
fig.tight_layout()
fig.savefig(OUT / "char-lm-perplexity.png", dpi=150)
plt.close(fig)

# ---- Figure 3: temperature on the trained model's next-character logits ----
torch = ns["torch"]
model, encode, alphabet = runs[0][0], ns["encode"], ns["ALPHABET"]
model.eval()
with torch.no_grad():
    logits, _ = model(encode("alice ").unsqueeze(0))
z = logits[0, -1]
order = torch.argsort(z, descending=True)[:12]
labels = [repr(alphabet[i])[1:-1].replace(" ", "space") for i in order.tolist()]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)
for ax, tau in zip(axes, (0.5, 1.0, 1.5)):
    q = torch.softmax(z / tau, dim=-1)[order].numpy()
    ax.bar(range(len(order)), q, color="#3875cb")
    ax.set_xticks(range(len(order)), labels, fontsize=10)
    ax.set_title(f"temperature {tau}")
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("sampling probability")
fig.suptitle('Next character after "alice " (12 highest-scoring characters)', y=0.98)
fig.tight_layout()
fig.savefig(OUT / "temperature.png", dpi=150)
plt.close(fig)
print("wrote figures to", OUT)
