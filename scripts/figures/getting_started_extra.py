"""Eight-track map for foundations/getting-started; synthetic original diagram.

Layout: Foundations -> the three entry tracks (IP, NLP, PR) -> ML -> DL, with Optimization and
Information Theory drawn underneath as the advanced theory that supports the large topics.

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

W, H = .30, .115
BOXES = {
    "found": ("Foundations", "getting started, images as arrays", .50, .90, "#e8eef7"),
    "ip": ("Image Processing", "IP 01-13: filters, transforms,\nrestoration, segmentation, features", .17, .69, "#e6f4ea"),
    "nlp": ("Natural Language Processing", "NLP 01-12: linguistics, embeddings,\nRNNs, Transformers, BERT", .50, .69, "#f7e8e8"),
    "pr": ("Pattern Recognition", "PR 01-10: decisions, densities,\nclassifiers, clustering", .83, .69, "#fdf0e0"),
    "ml": ("Machine Learning", "ML 01-10: math, networks, CNNs,\nrepresentations, generative models", .68, .45, "#f3e8f7"),
    "dl": ("Deep Learning for CV (DLCV)", "DLCV 01-07: classification, detection,\nrestoration, tracking, world models", .32, .45, "#e3f2f7"),
    "opt": ("Optimization", "OPT 01-08: convexity, duality,\nnumerical methods", .32, .135, "#efe9f8"),
    "it": ("Information Theory", "IT 01-08: entropy, coding,\ncapacity, rate distortion", .68, .135, "#fbece4"),
}
SOLID = [("found", "ip"), ("found", "nlp"), ("found", "pr"),
         ("ip", "dl"), ("pr", "ml"), ("nlp", "ml"), ("nlp", "dl")]

fig, ax = plt.subplots(figsize=(10.4, 8.4))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")


def top(k):
    return BOXES[k][2], BOXES[k][3] + H / 2


def bottom(k):
    return BOXES[k][2], BOXES[k][3] - H / 2


def arrow(p, q, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, shrinkA=3, shrinkB=3, zorder=3, **kw))


for title, sub, x, y, colour in BOXES.values():
    ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=colour, ec="#333333", lw=1.2))
    ax.text(x, y + .024, title, ha="center", va="center", weight="bold", fontsize=10.5)
    ax.text(x, y - .024, sub, ha="center", va="center", fontsize=9.5, color="#333333")

for a, b in SOLID:
    arrow(bottom(a), top(b), lw=1.3, color="#555555")
# ML -> DL, side by side (ML on the right feeds DL on the left).
arrow((BOXES["ml"][2] - W / 2, BOXES["ml"][3]), (BOXES["dl"][2] + W / 2, BOXES["dl"][3]), lw=1.6, color="#555555")

# Theory band: a dashed container around OPT and IT, supporting the large topics above.
band_y0, band_y1 = .055, .245
ax.add_patch(FancyBboxPatch((.07, band_y0), .86, band_y1 - band_y0, boxstyle="round,pad=0.005,rounding_size=0.02",
                            fc="#fafafa", ec="#888888", lw=1.1, ls=(0, (4, 3)), zorder=0.5))
ax.text(.5, band_y1 - .012, "Advanced theory behind the large topics", ha="center", va="top",
        fontsize=10, style="italic", color="#555555")
for target, x in [("dl", .32), ("ml", .68)]:
    arrow((x, band_y1), bottom(target), lw=1.1, color="#888888", ls=(0, (4, 3)))
arrow((.91, band_y1), (.91, BOXES["pr"][3] - H / 2), lw=1.1, color="#888888", ls=(0, (4, 3)))



ax.text(.5, .005, "Solid arrows: main learning path. Dashed: theory used by those topics. "
        "Each tutorial lists its exact prerequisites.", ha="center", va="bottom", fontsize=9.5,
        style="italic", color="#444444")
fig.savefig(OUT / "track-map.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", OUT / "track-map.png")
