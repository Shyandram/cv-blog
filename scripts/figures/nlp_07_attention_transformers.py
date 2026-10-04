"""Figures for NLP 07 (attention mechanism and Transformers).

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/nlp_07_attention_transformers.py
Uses only hand-built synthetic vectors (no downloads); CPU, a few seconds.
Writes only to src/assets/nlp/nlp-07-attention-transformers/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-07-attention-transformers"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, GREY, PURPLE = "#2a6fdb", "#e4572e", "#17a398", "#6b7280", "#7b4fd0"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name, f"{(OUT / name).stat().st_size / 1e3:.0f} kB")


def toy_attention():
    """Same hand-built example as the tutorial's toy-sentence code block."""
    tokens = ["the", "pump", "stopped", "because", "it", "overheated", "<pad>"]
    X = np.array([[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                  [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=float)
    W_Q = 3.0 * np.array([[0, 1], [1, 0], [1, 0], [0, 0]])
    W_K = np.array([[1, 0], [0.5, 0], [0, 1], [0, 0]])

    def attention(mask):
        Q, K = X @ W_Q, X @ W_K
        s = np.where(mask, Q @ K.T / np.sqrt(K.shape[1]), -np.inf)
        A = np.exp(s - s.max(axis=1, keepdims=True))
        return A / A.sum(axis=1, keepdims=True)

    T = len(tokens)
    real = np.array([t != "<pad>" for t in tokens])
    pad_mask = np.broadcast_to(real, (T, T))
    causal_mask = pad_mask & np.tri(T, dtype=bool)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    for ax, mask, title in [(axes[0], pad_mask, "Padding mask (encoder self-attention)"),
                            (axes[1], causal_mask, "Padding + causal mask (decoder)")]:
        A = attention(mask)
        shown = np.ma.masked_where(~mask, A)
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#eeeeee")
        ax.imshow(shown, cmap=cmap, vmin=0, vmax=1)
        for i in range(T):
            for j in range(T):
                txt = f"{A[i, j]:.2f}" if mask[i, j] else "x"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="white" if mask[i, j] and A[i, j] > 0.5 else ("#444" if mask[i, j] else GREY))
        ax.add_patch(plt.Rectangle((-0.5, 3.5), T, 1, fill=False, ec=ORANGE, lw=2))
        ax.set_xticks(range(T), tokens, rotation=35)
        ax.set_yticks(range(T), tokens)
        ax.set_xlabel("key (token being read)")
        ax.set_ylabel("query (token being updated)")
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    save(fig, "toy-attention.png")


def box(ax, x, y, w, h, text, color, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#333", lw=1.2, alpha=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1, color="#333", lw=1.4, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0))


def encoder_decoder():
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    lb, lo, lg, lp = "#dbe7fb", "#fde3d8", "#d6f2ee", "#ebe2fa"
    # Encoder column
    ax.add_patch(FancyBboxPatch((0.6, 1.5), 3.6, 4.2, boxstyle="round,pad=0.05",
                                fc="none", ec=BLUE, lw=1.5, ls="--"))
    ax.text(2.4, 5.95, "Encoder block  (x N)", ha="center", color=BLUE, fontsize=11, weight="bold")
    box(ax, 1.0, 1.9, 2.8, 0.7, "Self-attention\n(padding mask)", lb)
    box(ax, 1.0, 2.9, 2.8, 0.45, "Add & LayerNorm", "#f3f4f6", 9)
    box(ax, 1.0, 3.65, 2.8, 0.7, "Feed-forward\n(per token)", lg)
    box(ax, 1.0, 4.65, 2.8, 0.45, "Add & LayerNorm", "#f3f4f6", 9)
    box(ax, 1.0, 0.45, 2.8, 0.65, "Source tokens\n+ positions", "#fff7d6")
    arrow(ax, 2.4, 1.1, 2.4, 1.9)
    for y0, y1 in [(2.6, 2.9), (3.35, 3.65), (4.35, 4.65)]:
        arrow(ax, 2.4, y0, 2.4, y1)
    arrow(ax, 2.4, 5.1, 2.4, 6.45)
    ax.text(2.4, 6.6, "memory: one vector per source token", ha="center", fontsize=9.5)
    # Decoder column
    ax.add_patch(FancyBboxPatch((5.9, 1.5), 3.9, 5.1, boxstyle="round,pad=0.05",
                                fc="none", ec=ORANGE, lw=1.5, ls="--"))
    ax.text(7.85, 6.8, "Decoder block  (x N)", ha="center", color=ORANGE, fontsize=11, weight="bold")
    box(ax, 6.3, 1.9, 3.1, 0.7, "Masked self-attention\n(causal + padding)", lo)
    box(ax, 6.3, 2.85, 3.1, 0.4, "Add & LayerNorm", "#f3f4f6", 9)
    box(ax, 6.3, 3.5, 3.1, 0.7, "Cross-attention\nQ: decoder, K,V: memory", lp)
    box(ax, 6.3, 4.45, 3.1, 0.4, "Add & LayerNorm", "#f3f4f6", 9)
    box(ax, 6.3, 5.1, 3.1, 0.6, "Feed-forward", lg)
    box(ax, 6.3, 5.95, 3.1, 0.4, "Add & LayerNorm", "#f3f4f6", 9)
    box(ax, 6.3, 0.45, 3.1, 0.65, "Shifted target tokens\n<bos> y1 ... + positions", "#fff7d6")
    arrow(ax, 7.85, 1.1, 7.85, 1.9)
    for y0, y1 in [(2.6, 2.85), (3.25, 3.5), (4.2, 4.45), (4.85, 5.1), (5.7, 5.95)]:
        arrow(ax, 7.85, y0, 7.85, y1)
    # memory to cross-attention
    ax.annotate("", xy=(6.3, 3.85), xytext=(2.4, 6.45),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(4.75, 4.6, "keys and values\nfrom the encoder", ha="center", color=PURPLE, fontsize=9.5)
    arrow(ax, 7.85, 6.35, 7.85, 6.75)
    ax.text(9.95, 6.55, "linear + softmax\np(y_t | y_<t, x)", ha="left", va="center", fontsize=9.5)
    arrow(ax, 9.4, 6.15, 9.9, 6.45)
    fig.tight_layout()
    save(fig, "encoder-decoder.png")


if __name__ == "__main__":
    toy_attention()
    encoder_decoder()
