"""Figures for NLP 06 (sequence labeling); rerun from the project root.

The results and transition figures re-execute the tagging experiment exactly as
printed in the English tutorial, so the plotted values match the quoted output.
"""
import contextlib
import io
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src/assets/nlp/nlp-06-sequence-labeling"
OUT.mkdir(parents=True, exist_ok=True)
MDX = ROOT / "src/content/tutorials/en/nlp/nlp-06-sequence-labeling.mdx"
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})


def box(ax, x, y, w, h, text, fc, ec, fs=10):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=.04", fc=fc, ec=ec, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, color="black", lw=1.1):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=11, lw=lw, color=color))


# ---- Figure 2: BiLSTM-CRF architecture ----
words = ["Vipo", "repaired", "Valve", "F", "in", "Port", "Suvika"]
tags = ["B-PER", "O", "B-DEV", "I-DEV", "O", "B-LOC", "I-LOC"]
fig, ax = plt.subplots(figsize=(10.5, 6.2))
ax.set(xlim=(-1.4, 13.6), ylim=(-0.6, 7.4))
ax.axis("off")
xs = [0.9 + 1.9 * i for i in range(len(words))]
for i, (x, w, t) in enumerate(zip(xs, words, tags)):
    ax.text(x, -0.2, w, ha="center", va="center", fontsize=11, family="monospace")
    box(ax, x - 0.38, 0.75, 0.68, 0.5, "word\nemb", "#fff6e0", "#b07d1a", 8)
    box(ax, x + 0.38, 0.75, 0.68, 0.5, "char\nBiLSTM", "#fdeceb", "#cf5543", 8)
    arrow(ax, (x - 0.1, 0.0), (x - 0.38, 0.48))
    arrow(ax, (x + 0.1, 0.0), (x + 0.38, 0.48))
    box(ax, x, 2.45, 1.3, 1.0, "BiLSTM\nfwd | bwd", "#eef4ff", "#3875cb", 9)
    arrow(ax, (x - 0.38, 1.02), (x - 0.2, 1.95))
    arrow(ax, (x + 0.38, 1.02), (x + 0.2, 1.95))
    if i < len(words) - 1:
        arrow(ax, (x + 0.67, 2.7), (xs[i + 1] - 0.67, 2.7), "#3875cb", 1.6)
        arrow(ax, (xs[i + 1] - 0.67, 2.2), (x + 0.67, 2.2), "#3875cb", 1.6)
    box(ax, x, 4.15, 1.3, 0.55, "scores u(t, k)", "#f3f0fa", "#6b4fa3", 8)
    arrow(ax, (x, 2.97), (x, 3.86))
    ax.text(x, 6.75, t, ha="center", va="center", fontsize=10, family="monospace", color="#4b8741")
    arrow(ax, (x, 5.75), (x, 6.5))
ax.add_patch(FancyBboxPatch((xs[0] - 0.7, 4.85), xs[-1] - xs[0] + 1.4, 0.85, boxstyle="round,pad=.04",
                            fc="#f0f9ee", ec="#4b8741", lw=1.5))
ax.text((xs[0] + xs[-1]) / 2, 5.27, "CRF layer: transition scores A[j, k] + Viterbi decoding", ha="center",
        va="center", fontsize=11)
for x in xs:
    arrow(ax, (x, 4.43), (x, 4.85))
ax.text(-1.3, -0.2, "words", fontsize=10, color="#555555", va="center")
ax.text(-1.3, 6.75, "tags", fontsize=10, color="#555555", va="center")
ax.set_title("BiLSTM-CRF tagger with character features", pad=6)
fig.tight_layout()
fig.savefig(OUT / "bilstm-crf.png", dpi=150)
plt.close(fig)

# ---- Re-run the tutorial's experiment ----
text = MDX.read_text(encoding="utf-8")
blocks = re.findall(r"```python\n(.*?)```", text, re.S)
code = next(b for b in blocks if "class CRF" in b and "def run(" in b)
ns = {"__name__": "nlp06_experiment"}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(code, "nlp06_experiment", "exec"), ns)
printed = buf.getvalue()
assert "mean 0.929" in printed and "mean 0.828" in printed, printed
results, TAGS = ns["results"], ns["TAGS"]

# ---- Figure 3: span F1 per model and seed ----
fig, ax = plt.subplots(figsize=(8, 4.6))
names = list(results)
colors = ["#9aa5b1", "#3875cb", "#4b8741"]
for i, (name, color) in enumerate(zip(names, colors)):
    f1 = [r[0][2] for r in results[name]]
    ax.bar(i, np.mean(f1), color=color, alpha=0.35, width=0.6)
    ax.scatter([i - 0.12, i, i + 0.12], f1, color=color, s=45, zorder=3)
    ax.text(i, np.mean(f1) + 0.012, f"mean {np.mean(f1):.3f}", ha="center", fontsize=10)
ax.set_xticks(range(len(names)), [n.replace(", ", "\n") for n in names])
ax.set(ylabel="exact span F1 (unseen names)", ylim=(0.7, 1.0))
ax.grid(axis="y", alpha=0.3)
ax.set_title("Three taggers, three seeds each")
fig.tight_layout()
fig.savefig(OUT / "tagger-results.png", dpi=150)
plt.close(fig)

# ---- Figure 4: learned CRF transitions (seed 0), forbidden cells masked ----
crf = results["BiLSTM-CRF, words+chars"][0][3]
A = crf.A.detach().numpy()
banned = crf.ban.numpy() < -1
M = np.ma.array(A, mask=banned)
fig, ax = plt.subplots(figsize=(7.6, 6.4))
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("#d9d9d9")
lim = np.abs(A[~banned]).max()
im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim)
ax.set_xticks(range(len(TAGS)), TAGS, rotation=45, ha="right")
ax.set_yticks(range(len(TAGS)), TAGS)
ax.set(xlabel="next tag k", ylabel="previous tag j")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="learned score A[j, k]")
ax.set_title("CRF transition scores (gray = forbidden by BIO)")
fig.tight_layout()
fig.savefig(OUT / "crf-transitions.png", dpi=150)
plt.close(fig)
print("wrote figures to", OUT)
