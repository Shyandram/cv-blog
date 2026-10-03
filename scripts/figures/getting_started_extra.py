"""Figure for foundations/getting-started: how the six tracks connect.

Run: .venv/bin/python scripts/figures/getting_started_extra.py
Writes only to src/assets/foundations/getting-started/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "src/assets/foundations/getting-started"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11})

# (label, sublabel, x, y, colour)
BOXES = {
    "found": ("0 · Foundations", "start here, textbooks,\nimages as arrays", 0.5, 0.86, "#e8eef7"),
    "dip": ("1 · Image Processing", "DIP 01-13: filters, transforms,\nsegmentation, features", 0.16, 0.55, "#e6f4ea"),
    "pr": ("2 · Pattern Recognition", "PR 01-10: Bayes decisions,\ndensities, classifiers, clustering", 0.5, 0.55, "#fdf0e0"),
    "ml": ("3 · ML/DL", "ML 01-10 (+01A-C math):\nnetworks to foundation models", 0.84, 0.55, "#f3e8f7"),
    "dl": ("4 · DL for Vision", "DL 01-07: detection,\nrestoration, tracking, world models", 0.3, 0.17, "#e3f2f7"),
    "nlp": ("5 · NLP", "NLP 01: text to\nlanguage models", 0.8, 0.17, "#f7e8e8"),
}
EDGES = [
    ("found", "dip"), ("found", "pr"), ("found", "ml"),
    ("dip", "pr"), ("pr", "ml"), ("dip", "dl"), ("ml", "dl"), ("ml", "nlp"),
]
W, H = 0.31, 0.17

fig, ax = plt.subplots(figsize=(10.6, 6.2))
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0, 1)
ax.axis("off")
for key, (title, sub, x, y, colour) in BOXES.items():
    ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=colour, ec="#333333", lw=1.2))
    ax.text(x, y + 0.035, title, ha="center", va="center", weight="bold", fontsize=11.5)
    ax.text(x, y - 0.03, sub, ha="center", va="center", fontsize=10, color="#333333")


def anchor(a, b):
    (_, _, xa, ya, _), (_, _, xb, yb, _) = BOXES[a], BOXES[b]
    if abs(ya - yb) < 1e-6:  # same row: connect side to side
        s = 1 if xb > xa else -1
        return (xa + s * W / 2, ya), (xb - s * W / 2, yb)
    return (xa, ya - H / 2), (xb, yb + H / 2)


for a, b in EDGES:
    p, q = anchor(a, b)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                                 lw=1.3, color="#555555", shrinkA=2, shrinkB=2))
ax.text(0.5, 0.005, "Arrows show which track supplies ideas to another; they are suggestions, not locks.",
        ha="center", va="bottom", fontsize=10, style="italic", color="#444444")
fig.savefig(OUT / "track-map.png", dpi=150, bbox_inches="tight", facecolor="white")
print("wrote", OUT / "track-map.png")
