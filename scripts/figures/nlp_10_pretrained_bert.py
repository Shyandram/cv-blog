"""NLP 10 figures: BERT input packing, and a tiny MLM encoder's training curve and probes.

The training part mirrors the tutorial's tiny BERT code (same corpus, seeds and settings).
"""
import itertools
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-10-pretrained-bert"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})

# ---------------------------------------------------------------- figure 1: packing
tokens = ["[CLS]", "the", "pump", "over", "##heat", "##ed", "[SEP]", "it", "stopped", "[SEP]"]
segments = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
fig, ax = plt.subplots(figsize=(10.5, 3.6))
ax.set_xlim(-3.0, len(tokens) - 0.4); ax.set_ylim(-0.75, 4.0); ax.axis("off")
rows = [("Input pieces", tokens, None), ("Token embedding", [f"E[{t}]" for t in tokens], "#d9e8f5"),
        ("Segment embedding", [f"S[{'A' if s == 0 else 'B'}]" for s in segments], None),
        ("Position embedding", [f"P[{i}]" for i in range(len(tokens))], "#eeeeee")]
for r, (label, cells, color) in enumerate(rows):
    y = 3.4 - r * 1.0
    ax.text(-2.95, y, label, va="center", ha="left", fontsize=10, weight="bold")
    for i, c in enumerate(cells):
        fill = color or ("#ffffff" if r == 0 else ("#fde2c4" if segments[i] == 0 else "#d5efd5"))
        ax.add_patch(plt.Rectangle((i - 0.45, y - 0.32), 0.9, 0.64, facecolor=fill, edgecolor="#555"))
        ax.text(i, y, c, ha="center", va="center", fontsize=8.5 if r else 9.5)
ax.text(len(tokens) / 2 - 1.2, -0.45, "At each position, add the three embeddings in its column, then apply layer normalization",
        ha="center", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "bert-inputs.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- figure 2: tiny MLM run
random.seed(0); torch.manual_seed(0); torch.set_num_threads(1)
frames = {"robot": ["lifts", "moves"], "crane": ["lifts", "eats"],
          "technician": ["repairs", "moves"], "heron": ["eats", "sees"]}
objects = {"lifts": ["steel", "box"], "moves": ["box", "cart"],
           "eats": ["fish", "bread"], "repairs": ["pump", "cart"], "sees": ["fish", "boat"]}
sentences = [f"{d1} {a}{s} {v} {d2} {o}".split()
             for s in frames for v in frames[s] for o in objects[v]
             for d1, d2, a in itertools.product(["the", "a"], ["the", "a"], ["", "old "])]
random.shuffle(sentences)
test, train = sentences[:40], sentences[40:]
special = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
words = sorted({w for s in sentences for w in s})
vocab = special + words
index = {w: i for i, w in enumerate(vocab)}
first_word = len(special)
L = 2 + max(len(s) for s in sentences)


def encode(rows):
    ids = torch.zeros(len(rows), L, dtype=torch.long)
    for r, s in enumerate(rows):
        seq = [index["[CLS]"]] + [index[w] for w in s] + [index["[SEP]"]]
        ids[r, :len(seq)] = torch.tensor(seq)
    return ids


train_ids, test_ids = encode(train), encode(test)


def corrupt(ids, generator):
    real = ids >= first_word
    selected = (torch.rand(ids.shape, generator=generator) < 0.15) & real
    labels = torch.where(selected, ids, torch.full_like(ids, -100))
    u = torch.rand(ids.shape, generator=generator)
    out = ids.clone()
    out[selected & (u < 0.8)] = index["[MASK]"]
    swap = selected & (u >= 0.8) & (u < 0.9)
    out[swap] = torch.randint(first_word, len(vocab), (int(swap.sum()),), generator=generator)
    return out, labels


class TinyBERT(nn.Module):
    def __init__(self, d=32, layers=2):
        super().__init__()
        self.word, self.position = nn.Embedding(len(vocab), d), nn.Embedding(L, d)
        self.segment, self.norm = nn.Embedding(2, d), nn.LayerNorm(d)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(
            d, 4, 4 * d, dropout=0.0, activation="gelu", batch_first=True) for _ in range(layers))
        self.mlm = nn.Linear(d, len(vocab))

    def forward(self, ids):
        pad = ids.eq(index["[PAD]"])
        h = self.norm(self.word(ids) + self.position(torch.arange(ids.shape[1]))
                      + self.segment(torch.zeros_like(ids)))
        states = [h]
        for block in self.blocks:
            h = block(h, src_key_padding_mask=pad)
            states.append(h)
        return states


def heldout_loss(model):
    """Mean loss when each word position of each test sentence is masked in turn."""
    total, n = 0.0, 0
    with torch.no_grad():
        for pos in range(1, L):
            rows = test_ids[:, pos] >= first_word
            if rows.any():
                x = test_ids[rows].clone(); x[:, pos] = index["[MASK]"]
                logits = model.mlm(model(x)[-1][:, pos])
                total += F.cross_entropy(logits, test_ids[rows, pos], reduction="sum").item()
                n += int(rows.sum())
    return total / n


model = TinyBERT()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(1)
steps, train_curve, test_curve = [], [], []
for step in range(600):
    inputs, labels = corrupt(train_ids, g)
    logits = model.mlm(model(inputs)[-1])
    loss = F.cross_entropy(logits.reshape(-1, len(vocab)), labels.reshape(-1))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if step % 20 == 0 or step == 599:
        model.eval()
        steps.append(step); train_curve.append(loss.item()); test_curve.append(heldout_loss(model))
        model.train()
model.eval()

pos = {w: "DET" if w in ("the", "a") else "ADJ" if w == "old" else
       "VERB" if w in objects else "NOUN" for w in words}
control = {w: random.randint(0, 3) for w in words}
crane = [s for s in sentences if "crane" in s]
crane_ids = encode(crane)
sense = np.array([int("eats" in s) for s in crane])


def rows_at(ids, text, layer):
    with torch.no_grad():
        h = model(ids)[layer]
    return np.array([h[r, i + 1].numpy() for r, s in enumerate(text) for i in range(len(s))])


results = []
for layer in range(3):
    Xtr, Xte = rows_at(train_ids, train, layer), rows_at(test_ids, test, layer)
    acc = []
    for table in (pos, control):
        ytr = [table[w] for s in train for w in s]; yte = [table[w] for s in test for w in s]
        acc.append(LogisticRegression(max_iter=3000).fit(Xtr, ytr).score(Xte, yte))
    with torch.no_grad():
        h = model(crane_ids)[layer]
    Xc = np.array([h[r, s.index("crane") + 1].numpy() for r, s in enumerate(crane)])
    acc.append(cross_val_score(LogisticRegression(max_iter=3000), Xc, sense, cv=4).mean())
    results.append(acc)
results = np.array(results)
print("probe accuracies (layer x [POS, control, crane sense]):\n", results.round(2))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
axes[0].plot(steps, train_curve, label="training batch (random masks)", color="#1f77b4")
axes[0].plot(steps, test_curve, label="held-out sentences (each word masked)", color="#d62728")
axes[0].set(xlabel="Optimizer step", ylabel="Masked-token cross-entropy (nats)",
            title="Tiny MLM pretraining on a toy corpus")
axes[0].legend(fontsize=10)
width = 0.26
labels = ["Part of speech", "Control task\n(random label per word)", "Sense of 'crane'"]
for j, (name, color) in enumerate(zip(labels, ["#1f77b4", "#7f7f7f", "#2ca02c"])):
    axes[1].bar(np.arange(3) + (j - 1) * width, results[:, j], width, label=name, color=color)
axes[1].set(xticks=range(3), xticklabels=["Embeddings", "Layer 1", "Layer 2"], ylim=(0, 1.15),
            ylabel="Probe test accuracy", title="Linear probes on frozen representations")
axes[1].legend(fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(OUT / "tiny-bert-probes.png", dpi=150)
plt.close(fig)
print("wrote", OUT)
