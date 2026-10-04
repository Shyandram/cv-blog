"""Figures for NLP 01 (introduction).

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/nlp_01_introduction_extra.py
Uses only invented synthetic text (no downloads); CPU, < 1 min.
Writes only to src/assets/nlp/nlp-01-introduction/.
"""
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

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


# ---------- BPE on standard-library docstrings ----------
def synthetic_corpus():
    """Invented compounds, deterministically ordered; no external text."""
    stems = ["cat", "dog", "bird", "fish", "road", "tree", "rain", "sun", "moon", "star",
             "book", "page", "read", "write", "walk", "play", "jump", "train", "test", "learn"]
    prefixes = ["", "re", "un", "pre", "post", "over", "under", "out", "mis", "non"]
    suffixes = ["", "s", "ed", "ing", "er", "ers", "able", "less", "ful", "ness"]
    words = [pre + stem + suf for stem in stems for pre in prefixes for suf in suffixes]
    rng = np.random.default_rng(7)
    rng.shuffle(words)
    return words


def bpe_curve():
    words = synthetic_corpus()
    cut = int(0.8 * len(words))
    train, test = Counter(words[:cut]), Counter(words[cut:])
    vocab = Counter({tuple(w) + ("_",): f for w, f in train.items()})
    chars = sorted({c for w in train for c in w} | {"_"})
    merges = []
    checkpoints = [0, 25, 50, 100, 200, 400, 800, 1600]
    results = []

    def encode_len(word, rank):
        s = list(word) + ["_"]
        while len(s) > 1:
            pairs = [(rank.get((a, b), 1e18), i) for i, (a, b) in enumerate(zip(s, s[1:]))]
            r, i = min(pairs)
            if r == 1e18:
                break
            s[i:i + 2] = [s[i] + s[i + 1]]
        return len(s)

    for step in range(checkpoints[-1] + 1):
        if step in checkpoints:
            rank = {p: k for k, p in enumerate(merges)}
            n_tok = sum(encode_len(w, rank) * f for w, f in test.items())
            results.append((len(chars) + len(merges), n_tok / sum(test.values())))
        pairs = Counter()
        for sym, f in vocab.items():
            for a, b in zip(sym, sym[1:]):
                pairs[a, b] += f
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], p))
        merges.append(best)
        new = Counter()
        for sym, f in vocab.items():
            s, i = [], 0
            while i < len(sym):
                if i + 1 < len(sym) and (sym[i], sym[i + 1]) == best:
                    s.append(sym[i] + sym[i + 1]); i += 2
                else:
                    s.append(sym[i]); i += 1
            new[tuple(s)] += f
        vocab = new
    v, t = zip(*results)
    oov = sum(f for w, f in test.items() if w not in train) / sum(test.values())
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.semilogx(v, t, "o-", color=BLUE)
    for vi, ti in zip(v, t):
        ax.annotate(f"{ti:.2f}", (vi, ti), textcoords="offset points", xytext=(4, 6), fontsize=10)
    ax.axhline(1.0, color=GREY, ls="--", lw=1)
    ax.text(v[0], 1.08, "1 token per word (whole-word vocabulary)", color=GREY, fontsize=10)
    ax.set_xlabel("Vocabulary size (characters + learned merges)")
    ax.set_ylabel("Tokens per held-out word")
    ax.set_title(f"BPE on synthetic compounds: {oov:.1%} of held-out words unseen in training")
    ax.grid(alpha=.25, which="both")
    save(fig, "bpe-vocab-curve.png")
    print("bpe results", results, "oov", oov, "train words", cut)


def embeddings():
    sentences = """the cat drinks milk . the dog drinks water . a cat chases a mouse .
    a dog chases a cat . the kitten drinks milk . the puppy drinks water .
    a kitten chases a ball . a puppy chases a ball . the car needs fuel .
    the truck needs fuel . a car drives on the road . a truck drives on the road .""".split(".")
    tokens = [s.split() for s in sentences if s.strip()]
    vocab = sorted({w for s in tokens for w in s})
    idx = {w: i for i, w in enumerate(vocab)}
    C = np.zeros((len(vocab), len(vocab)))
    for s in tokens:
        for i, w in enumerate(s):
            for j in range(max(0, i - 2), min(len(s), i + 3)):
                if j != i:
                    C[idx[w], idx[s[j]]] += 1
    total = C.sum()
    pw, pc = C.sum(1, keepdims=True) / total, C.sum(0, keepdims=True) / total
    with np.errstate(divide="ignore"):
        ppmi = np.maximum(np.log((C / total) / (pw * pc)), 0)
    U, S, _ = np.linalg.svd(ppmi)
    E = U[:, :8] * S[:8]
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    groups = {"animals": ["cat", "dog", "kitten", "puppy", "mouse"],
              "vehicles": ["car", "truck"], "drinks / fuel": ["milk", "water", "fuel"],
              "verbs": ["drinks", "chases", "needs", "drives"]}
    colors = {"animals": BLUE, "vehicles": ORANGE, "drinks / fuel": GREEN, "verbs": PURPLE}
    shown = [w for g in groups.values() for w in g]
    Es = E[[idx[w] for w in shown]]
    Es = Es - Es.mean(0)
    _, _, Vt = np.linalg.svd(Es, full_matrices=False)
    P = Es @ Vt[:2].T
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    placed = []                                   # merge labels of coinciding points
    for g, ws in groups.items():
        pts = np.array([P[shown.index(w)] for w in ws])
        ax.scatter(pts[:, 0], pts[:, 1], color=colors[g], label=g, s=40)
        for w, pt in zip(ws, pts):
            for item in placed:
                if np.linalg.norm(item[1] - pt) < 0.02:
                    item[0].append(w)
                    break
            else:
                placed.append(([w], pt))
    for names, (x, y) in placed:
        ax.annotate(" ≈ ".join(names), (x, y), textcoords="offset points", xytext=(5, 4), fontsize=10)
    ax.set_title("8-d PPMI-SVD vectors, projected to 2-D")
    ax.legend(fontsize=10, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    ax = axes[1]
    sel = ["cat", "kitten", "dog", "puppy", "car", "truck", "milk", "water"]
    M = E[[idx[w] for w in sel]] @ E[[idx[w] for w in sel]].T
    im = ax.imshow(M, cmap="Blues", vmin=-0.2, vmax=1)
    ax.set_xticks(range(len(sel)), sel, rotation=45)
    ax.set_yticks(range(len(sel)), sel)
    for i in range(len(sel)):
        for j in range(len(sel)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if M[i, j] > 0.6 else "k")
    ax.set_title("Cosine similarity")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save(fig, "ppmi-embeddings.png")


def perplexity_alpha():
    train = ["the cat sleeps", "the dog sleeps", "the cat runs", "the dog runs", "a cat eats",
             "a dog eats", "the bird sings", "a bird sleeps"]
    test = ["the cat eats", "a dog runs", "the bird sleeps", "a robot runs"]
    START, END, UNK = "<s>", "</s>", "<unk>"
    words = {w for s in train for w in s.split()}
    vocab = sorted(words | {END, UNK})
    counts = defaultdict(Counter)
    for s in train:
        t = [START] + [w if w in words else UNK for w in s.split()] + [END]
        for h, w in zip(t, t[1:]):
            counts[h][w] += 1

    def ppl(alpha, sents):
        nll, n = 0.0, 0
        for s in sents:
            t = [START] + [w if w in words else UNK for w in s.split()] + [END]
            for h, w in zip(t, t[1:]):
                row = counts.get(h, {})
                p = (row.get(w, 0) + alpha) / (sum(row.values()) + alpha * len(vocab))
                nll -= math.log(p); n += 1
        return math.exp(nll / n)

    alphas = np.logspace(-3, 1, 60)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.semilogx(alphas, [ppl(a, test) for a in alphas], color=BLUE, label="held-out test sentences")
    ax.semilogx(alphas, [ppl(a, train) for a in alphas], color=ORANGE, ls="--", label="training sentences")
    ax.axhline(len(vocab), color=GREY, ls=":", lw=1)
    ax.text(1.2e-3, len(vocab) + 0.3, f"uniform guess = |V| = {len(vocab)}", color=GREY, fontsize=10)
    best = alphas[np.argmin([ppl(a, test) for a in alphas])]
    ax.axvline(0.2, color=GREEN, lw=1)
    ax.text(0.21, 1.4, "α = 0.2 (code)", color=GREEN, fontsize=10)
    ax.set_xlabel("Additive smoothing α")
    ax.set_ylabel("Bigram perplexity")
    ax.set_title("Too little smoothing punishes unseen pairs; too much flattens")
    ax.legend(fontsize=10, loc="center left")
    ax.set_ylim(0, len(vocab) + 2)
    ax.grid(alpha=.25, which="both")
    save(fig, "bigram-smoothing.png")
    print("best alpha on these 4 test sentences", best)


def attention_weights():
    rng = np.random.default_rng(0)
    T, d = 5, 8
    X = rng.normal(size=(T, d))
    Wq, Wk, Wv = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))

    def attn(causal):
        s = (X @ Wq) @ (X @ Wk).T / np.sqrt(d)
        if causal:
            s = np.where(np.tri(T, dtype=bool), s, -np.inf)
        w = np.exp(s - s.max(1, keepdims=True))
        return w / w.sum(1, keepdims=True)

    toks = ["please", "open", "the", "door", "now"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, causal, title in [(axes[0], False, "Bidirectional (encoder, e.g. BERT)"),
                              (axes[1], True, "Causal (decoder, e.g. GPT)")]:
        w = attn(causal)
        ax.imshow(w, cmap="Blues", vmin=0, vmax=1)
        for i in range(T):
            for j in range(T):
                ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if w[i, j] > 0.6 else "k")
        ax.set_xticks(range(T), toks, rotation=30)
        ax.set_yticks(range(T), toks)
        ax.set_xlabel("key (attended-to token)")
        ax.set_ylabel("query (updated token)")
        ax.set_title(title)
    fig.tight_layout()
    save(fig, "attention-weights.png")


def timeline():
    events = [(2003, "Neural\nprobabilistic LM", 0), (2013, "word2vec", 1), (2014, "GloVe;\nseq2seq", 0),
              (2015, "Attention\nfor NMT", 1), (2016, "BPE\nsubwords", 0), (2017, "Transformer", 1),
              (2018, "ELMo; GPT;\nSentencePiece", 0), (2019, "BERT", 1), (2020, "GPT-3; RAG;\nT5", 0),
              (2021, "CLIP", 1), (2022, "InstructGPT", 0), (2023, "LLaVA", 1)]
    xpos = lambda yr: 0 if yr == 2003 else 1.6 * (yr - 2013) + 1.8   # compress the 2003-2013 gap
    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    ax.plot([-0.6, 0.7], [0, 0], color="k", lw=1.2)
    ax.plot([1.1, xpos(2023) + 0.8], [0, 0], color="k", lw=1.2)
    ax.text(0.9, 0, "//", ha="center", va="center", fontsize=12)
    for year, label, up in events:
        x, y = xpos(year), (1 if up else -1)
        ax.plot([x, x], [0, 0.55 * y], color=GREY, lw=1)
        ax.scatter([x], [0], color=BLUE, zorder=3)
        ax.text(x, 0.65 * y, f"{year}\n{label}" if up else f"{label}\n{year}",
                ha="center", va="bottom" if up else "top", fontsize=10)
    ax.set_ylim(-2.2, 2.2); ax.set_xlim(-1.2, xpos(2023) + 1.0); ax.axis("off")
    ax.set_title("Milestones cited in this tutorial (publication years of the referenced papers)",
                 fontsize=11)
    save(fig, "nlp-timeline.png")


if __name__ == "__main__":
    pipeline()
    bpe_curve()
    embeddings()
    perplexity_alpha()
    attention_weights()
    timeline()
