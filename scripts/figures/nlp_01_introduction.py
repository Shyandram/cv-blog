"""History timeline for NLP 01 (Introduction to NLP).

Run: MPLCONFIGDIR=<scratch dir> .venv/bin/python scripts/figures/nlp_01_introduction.py
Every event is the publication year of a paper in the NLP 01 reference list.
Writes only src/assets/nlp/nlp-01-introduction/history-timeline.png
(the older nlp_01_introduction_extra.py script owns the other files in that folder).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-01-introduction"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, GREY = "#2a6fdb", "#6b7280"

EVENTS = [  # (year, label, side) side: +1 above the axis, -1 below
    (1966, "ELIZA\n(pattern rules)", 1),
    (1993, "Statistical MT;\nPenn Treebank", -1),
    (2003, "Neural\nprobabilistic LM", 1),
    (2011, "NLP (almost)\nfrom scratch", -1),
    (2013, "word2vec", 1),
    (2017, "Transformer", -1),
    (2018, "ELMo", 1),
    (2019, "BERT", -1),
    (2020, "GPT-3", 1),
    (2022, "InstructGPT", -1),
]
# Approximate eras for orientation only; in practice the approaches overlap.
# Events are placed at equal spacing (the axis is ordinal, not to scale in years).
ERAS = [(0, 0, "rules", "#eef3fc"),
        (1, 1, "statistical", "#eefcf6"),
        (2, 4, "neural networks\n(task-specific)", "#fcf6ee"),
        (5, 9, "pretrained models / LLMs", "#f3eefc")]


def timeline():
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    for a, b, name, color in ERAS:
        ax.axvspan(a - 0.5, b + 0.5, color=color, zorder=0)
        ax.text((a + b) / 2, 1.95, name, ha="center", va="top", fontsize=10, color=GREY)
    ax.plot([-0.5, len(EVENTS) - 0.5], [0, 0], color="k", lw=1.2, zorder=1)
    for i, (year, label, side) in enumerate(EVENTS):
        h = 0.4
        ax.plot([i, i], [0, h * side], color=GREY, lw=1, zorder=2)
        ax.scatter([i], [0], color=BLUE, zorder=3, s=25)
        ax.text(i, (h + 0.08) * side, f"{year}\n{label}" if side > 0 else f"{label}\n{year}",
                ha="center", va="bottom" if side > 0 else "top", fontsize=10)
    ax.set_xlim(-0.5, len(EVENTS) - 0.5)
    ax.set_ylim(-1.6, 2.0)
    ax.axis("off")
    ax.set_title("Milestones cited in NLP 01 (publication years; equal spacing, not to scale; "
                 "era shading is approximate)", fontsize=10)
    fig.savefig(OUT / "history-timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / "history-timeline.png",
          f"{(OUT / 'history-timeline.png').stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    timeline()
