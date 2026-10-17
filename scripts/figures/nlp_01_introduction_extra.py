"""Figures for NLP 01 (introduction).

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/nlp_01_introduction_extra.py
Uses only invented synthetic text (no downloads); CPU, < 1 min.
Writes only to src/assets/nlp/nlp-01-introduction/.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-01-introduction"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, GREY, PURPLE = "#2a6fdb", "#e4572e", "#17a398", "#6b7280", "#7b4fd0"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name, f"{(OUT / name).stat().st_size / 1e3:.0f} kB")


def pipeline():
    fig, ax = plt.subplots(figsize=(10.4, 4.3))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.3); ax.axis("off")
    steps = [("Raw text", "Pump B:\nnot ready."),
             ("Normalize", "Unicode NFC\nkeep case"),
             ("Tokenize", "Pump|B|:\nnot|ready|."),
             ("Token IDs", "[7,12,3,\n8,19,4]"),
             ("Encode", "counts or\nembeddings"),
             ("Predict", "classifier\nor extractor\n→ output")]
    for i, (title, sub) in enumerate(steps):
        x = 0.15 + i * 2.15
        ax.add_patch(FancyBboxPatch((x, 0.9), 1.8, 1.7, boxstyle="round,pad=0.05",
                                    fc="#eef3fc" if i < 4 else "#f3eefc",
                                    ec=BLUE if i < 4 else PURPLE, lw=1.4))
        ax.text(x + 0.9, 2.25, title, ha="center", fontsize=11, weight="bold")
        ax.text(x + 0.9, 1.45, sub, ha="center", va="center", fontsize=10, family="monospace")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 2.15, 1.75), xytext=(x + 1.85, 1.75),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(4.4, 0.35, "text and numerical encoding: keep the fitted rules fixed",
            ha="center", fontsize=10, color=BLUE)
    ax.text(11.0, 0.35, "task predictor", ha="center", fontsize=10, color=PURPLE)
    ax.text(6.5, 3.05, "The token split and IDs illustrate one encoding, not a pretrained tokenizer",
            ha="center", fontsize=10, color=GREY)
    save(fig, "text-pipeline.png")


if __name__ == "__main__":
    pipeline()
