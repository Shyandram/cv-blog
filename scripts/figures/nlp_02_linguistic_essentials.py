"""Figures for NLP 02 (Linguistic Essentials).

Run: MPLCONFIGDIR=<scratch dir> .venv/bin/python scripts/figures/nlp_02_linguistic_essentials.py
Corpus: the docstrings of the Python standard library that ships with the project venv
(Python 3.9.7), extracted with `ast` -- real English prose already on disk, no downloads.
Other Python versions ship different docstrings, so the printed numbers can change slightly.
Writes only to src/assets/nlp/nlp-02-linguistic-essentials/.
"""
import ast
import re
import sysconfig
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-02-linguistic-essentials"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREY = "#2a6fdb", "#e4572e", "#6b7280"


def stdlib_docstrings():
    lib = Path(sysconfig.get_paths()["stdlib"])
    texts = []
    for path in sorted(lib.glob("*.py")):          # top-level modules only, sorted for reproducibility
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node)
                if doc:
                    texts.append(doc)
    return texts


def zipf():
    tokens = [w for t in stdlib_docstrings() for w in re.findall(r"[a-z]+", t.lower())]
    counts = Counter(tokens)
    freqs = np.array(sorted(counts.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1)
    # Least-squares fit of log f = log C - s log r over ranks 10..1000 (the middle of the curve).
    lo, hi = 10, 1000
    slope, intercept = np.polyfit(np.log(ranks[lo - 1:hi]), np.log(freqs[lo - 1:hi]), 1)
    s = -slope
    hapax = sum(1 for c in counts.values() if c == 1)
    print(f"tokens={len(tokens)} types={len(counts)} hapax={hapax} ({hapax / len(counts):.1%} of types)")
    print("top 10:", counts.most_common(10))
    print(f"fitted exponent s={s:.3f} over ranks {lo}-{hi}")
    top10_share = freqs[:10].sum() / freqs.sum()
    print(f"top-10 types cover {top10_share:.1%} of tokens")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.loglog(ranks, freqs, ".", ms=3, color=BLUE, label="observed word types")
    fit_r = np.array([1, len(freqs)])
    ax.loglog(fit_r, np.exp(intercept) * fit_r ** slope, "--", color=ORANGE, lw=2,
              label=f"fit on ranks {lo}-{hi}: f ∝ r^-{s:.2f}")
    for word, r in [(counts.most_common()[i][0], i + 1) for i in (0, 9, 99)]:
        ax.annotate(word, (r, counts[word]), textcoords="offset points", xytext=(6, 4), fontsize=10)
    ax.set_xlabel("Frequency rank r (1 = most frequent)")
    ax.set_ylabel("Count f(r)")
    ax.set_title(f"Python 3.9 stdlib docstrings: {len(tokens):,} tokens, {len(counts):,} types",
                 fontsize=11)
    ax.grid(alpha=.25, which="both")
    ax.legend(frameon=False)
    fig.savefig(OUT / "zipf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / "zipf.png", f"{(OUT / 'zipf.png').stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    zipf()
