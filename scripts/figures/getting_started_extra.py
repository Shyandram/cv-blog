"""Eight-track map for foundations/getting-started; synthetic original diagram.
Run: .venv/bin/python scripts/figures/getting_started_extra.py
Writes only src/assets/foundations/getting-started/.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "src/assets/foundations/getting-started"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11})
BOXES = {
    "found": ("0 · Foundations", "entry routes, textbooks,\nimages as arrays", .5, .91, "#e8eef7"),
    "opt": ("6 · Optimization", "OPT 01-08: convex modeling,\nduality, numerical methods", .23, .66, "#f1e8f7"),
    "it": ("7 · Information Theory", "IT 01-08: entropy, coding,\ncapacity, rate distortion", .77, .66, "#fbece4"),
    "dip": ("1 · Image Processing", "DIP 01-13: filters, transforms,\nsegmentation, features", .16, .40, "#e6f4ea"),
    "pr": ("3 · Pattern Recognition", "PR 01-10: decisions,\ndensities, classifiers, clustering", .5, .40, "#fdf0e0"),
    "ml": ("4 · ML/DL", "networks, representations,\ngenerative and foundation models", .84, .40, "#f3e8f7"),
    "dl": ("5 · DL for Vision", "DL 01-07: detection,\nrestoration, tracking, world models", .23, .14, "#e3f2f7"),
    "nlp": ("2 · NLP", "NLP 01-08: syntax, semantics,\nretrieval, dialogue, generation", .77, .14, "#f7e8e8"),
}
EDGES = [("found", "opt"), ("found", "it"), ("opt", "dip"), ("opt", "pr"),
         ("it", "pr"), ("it", "ml"), ("dip", "dl"), ("ml", "dl"), ("ml", "nlp")]
W, H = .31, .14
fig, ax = plt.subplots(figsize=(10.4, 8.0))
ax.set(xlim=(-.02, 1.02), ylim=(0, 1))
ax.axis("off")
for a,b in EDGES:
    xa,ya=BOXES[a][2:4]; xb,yb=BOXES[b][2:4]
    ax.add_patch(FancyArrowPatch((xa,ya-H/2), (xb,yb+H/2), arrowstyle="-|>",
                                mutation_scale=14, lw=1.3, color="#666666", shrinkA=3, shrinkB=3))
for title, sub, x, y, colour in BOXES.values():
    ax.add_patch(FancyBboxPatch((x-W/2,y-H/2),W,H,
                               boxstyle="round,pad=0.01,rounding_size=0.02",
                               fc=colour,ec="#333333",lw=1.2))
    ax.text(x,y+.028,title,ha="center",va="center",weight="bold",fontsize=11)
    ax.text(x,y-.027,sub,ha="center",va="center",fontsize=10,color="#333333")
ax.text(.5,.005,"Selected connections between ideas; each tutorial lists its actual prerequisites.",
        ha="center",va="bottom",fontsize=10,style="italic",color="#444444")
fig.savefig(OUT / "track-map.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", OUT / "track-map.png")
