"""Figures for NLP 03 (neural networks for NLP). CPU only, synthetic data, no downloads.

Writes to src/assets/nlp/nlp-03-neural-networks-for-nlp/:
  training-curves.png  train/validation loss of the tutorial's three neural classifiers
  textcnn.png          schematic of a 1-D convolutional text classifier
The training code mirrors the experiment printed in the tutorial.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-03-neural-networks-for-nlp"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})

import copy, random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
ASPECTS = ["battery", "screen", "camera", "speaker", "case", "charger"]
POS, NEG = ["good", "great", "solid", "bright"], ["bad", "weak", "noisy", "dull"]

def clause(rng):
    aspect, positive = rng.choice(ASPECTS), rng.random() < 0.5
    negated = rng.random() < 0.4
    words = ["the", aspect, "is"] + ["not"] * negated + [rng.choice(POS if positive else NEG)]
    return words, int(positive != negated), negated

def example(rng):                       # label of "A but B" is the label of B
    words, label, negated = clause(rng)
    if rng.random() < 0.5:
        second, label, _ = clause(rng)
        return " ".join(words + ["but"] + second), label, "contrast"
    return " ".join(words), label, "negated" if negated else "plain"

def make(n, seed):
    rng = random.Random(seed)
    return [example(rng) for _ in range(n)]

train, val, test = make(300, 1), make(300, 2), make(1000, 3)
flip = random.Random(4)                 # 10% training-label noise; val/test stay clean
train = [(t, 1 - y if flip.random() < 0.1 else y, c) for t, y, c in train]
y_test, kind = np.array([r[1] for r in test]), np.array([r[2] for r in test])

def report(name, pred):
    parts = {k: round(float((pred[kind == k] == y_test[kind == k]).mean()), 3)
             for k in ["plain", "negated", "contrast"]}
    print(f"{name:12s} test acc {np.mean(pred == y_test):.3f}", parts)


vocab = ["<pad>", "<unk>"] + sorted({w for r in train for w in r[0].split()})
ids = {w: i for i, w in enumerate(vocab)}

def encode(rows):                       # right-pad every sequence with ID 0
    seqs = [torch.tensor([ids.get(w, 1) for w in r[0].split()]) for r in rows]
    return (nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0),
            torch.tensor([r[1] for r in rows]))

class MeanMLP(nn.Module):               # embedding -> masked mean -> MLP
    def __init__(self, V, D=16, H=32, p=0.3):
        super().__init__()
        self.emb = nn.Embedding(V, D, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Dropout(p), nn.Linear(H, 2))
    def forward(self, x):
        m = (x != 0).unsqueeze(-1).float()
        return self.mlp((self.emb(x) * m).sum(1) / m.sum(1).clamp(min=1))

class TextCNN(nn.Module):               # embedding -> 1-D conv -> masked max -> linear
    def __init__(self, V, D=16, widths=(2, 3, 4), maps=16, p=0.3):
        super().__init__()
        self.emb = nn.Embedding(V, D, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(D, maps, h) for h in widths)
        self.drop, self.out = nn.Dropout(p), nn.Linear(maps * len(widths), 2)
    def forward(self, x):
        e, real = self.emb(x).transpose(1, 2), x != 0         # e: (batch, D, T)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(e))                           # (batch, maps, T-h+1)
            valid = real[:, conv.kernel_size[0] - 1:].unsqueeze(1)  # window ends on a real token
            # ReLU outputs are >= 0, so filling invalid windows with 0 never wins the max,
            # and a text shorter than the filter gets 0 instead of -inf (which would give NaN).
            pooled.append(c.masked_fill(~valid, 0.0).max(dim=2).values)
        return self.out(self.drop(torch.cat(pooled, dim=1)))

def fit(model, weight_decay, epochs=80, batch=32):
    (xt, yt), (xv, yv) = encode(train), encode(val)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    best, history = (float("inf"), None, 0), []
    for epoch in range(1, epochs + 1):
        model.train()
        for b in torch.randperm(len(yt)).split(batch):
            opt.zero_grad()
            F.cross_entropy(model(xt[b]), yt[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            history.append((F.cross_entropy(model(xt), yt).item(),
                            F.cross_entropy(model(xv), yv).item()))
        if history[-1][1] < best[0]:                          # early stopping on val loss
            best = (history[-1][1], copy.deepcopy(model.state_dict()), epoch)
    model.load_state_dict(best[1])
    return history, best[2]


# ---------------------------------------------------------------- training curves
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
for ax, (name, build, wd) in zip(axes, [("mean+MLP", lambda: MeanMLP(len(vocab)), 1e-4),
                                        ("CNN (dropout + weight decay)", lambda: TextCNN(len(vocab)), 1e-4),
                                        ("CNN, no dropout or decay", lambda: TextCNN(len(vocab), p=0.0), 0.0)]):
    torch.manual_seed(0)
    history, best_epoch = fit(build(), wd)
    epochs = range(1, len(history) + 1)
    ax.plot(epochs, [h[0] for h in history], label="train loss", color="#2866ae")
    ax.plot(epochs, [h[1] for h in history], label="validation loss", color="#d9622b")
    ax.axvline(best_epoch, color="#666666", ls="--", lw=1)
    ax.text(best_epoch + 1, 0.95, f"best val\nepoch {best_epoch}", fontsize=10, va="top")
    ax.set(title=name, xlabel="epoch", ylim=(0, 1.0))
    ax.title.set_fontsize(11)
axes[0].set_ylabel("cross-entropy (nats)")
axes[0].legend(loc="lower left", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "training-curves.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- TextCNN schematic
words = ["the", "screen", "is", "not", "bright", "but", "the", "case", "is", "solid"]
fig, ax = plt.subplots(figsize=(10.5, 4.4))
ax.set(xlim=(0, 10.5), ylim=(0, 4.6)); ax.axis("off")
D = 5
for t, w in enumerate(words):                       # embedding matrix: one column per token
    for d in range(D):
        ax.add_patch(Rectangle((0.3 + t * 0.42, 1.2 + d * 0.3), 0.4, 0.28,
                               facecolor="#edf3fb", edgecolor="#2866ae", lw=0.6))
    ax.text(0.5 + t * 0.42, 1.0, w, rotation=60, ha="right", va="top", fontsize=10)
ax.add_patch(Rectangle((0.3 + 3 * 0.42, 1.15), 0.42 * 2, 1.6, fill=False, edgecolor="#d9622b", lw=2))
ax.add_patch(Rectangle((0.3 + 6 * 0.42, 1.12), 0.42 * 3, 1.66, fill=False, edgecolor="#3a8a3a", lw=2))
ax.text(2.4, 3.05, "width-2 window", color="#d9622b", fontsize=10, ha="center")
ax.text(3.9, 3.35, "width-3 window", color="#3a8a3a", fontsize=10, ha="center")
ax.text(2.4, 4.3, "token embeddings X (T x D)", ha="center", fontsize=11)
rng = np.random.default_rng(3)
for row, (color, n, label) in enumerate([("#d9622b", 9, "feature map c (width 2)"),
                                         ("#3a8a3a", 8, "feature map c (width 3)")]):
    y0 = 2.6 - row * 1.1
    vals = rng.uniform(0.1, 1.0, n); vals[3 + row] = 1.0
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((5.6 + i * 0.3, y0), 0.28, 0.4, facecolor=color, alpha=0.25 + 0.7 * v,
                               edgecolor="white"))
    ax.text(5.6, y0 + 0.55, label, fontsize=10)
    ax.add_patch(FancyArrowPatch((5.6 + n * 0.3 + 0.05, y0 + 0.2), (8.75, y0 + 0.2), arrowstyle="->",
                                 mutation_scale=12))
    ax.add_patch(Rectangle((8.8, y0), 0.4, 0.4, facecolor=color, alpha=0.95, edgecolor="black"))
ax.text(9.0, 3.25, "max over\npositions", ha="center", fontsize=10)
ax.add_patch(FancyArrowPatch((4.85, 2.0), (5.5, 2.0), arrowstyle="->", mutation_scale=14))
ax.add_patch(FancyArrowPatch((9.3, 2.0), (9.75, 2.0), arrowstyle="->", mutation_scale=14))
ax.text(10.1, 2.0, "linear\n+ softmax", ha="center", va="center", fontsize=10)
ax.text(5.25, 0.15, "Each filter slides over positions; max pooling keeps its strongest match "
        "and discards where it occurred.", ha="center", fontsize=10)
fig.savefig(OUT / "textcnn.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote training-curves.png and textcnn.png")
