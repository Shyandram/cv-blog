"""Figures for NLP 11: experiment results (re-runs the tutorial's CPU experiment) and span scores."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-11-classification-bert"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})
RESULTS = {}


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---- The tutorial experiment, unchanged except that report() also records results.
import itertools, random
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

torch.manual_seed(0); random.seed(0); torch.set_num_threads(1)
POS, NEG = ['good', 'great', 'fine', 'superb'], ['bad', 'awful', 'poor', 'terrible']
ITEMS, NAMES = ['battery', 'screen', 'camera', 'speaker'], ['mina', 'ravi', 'lena']
VERBS = {'repaired': 'repair', 'cleaned': 'clean', 'tested': 'test'}
OBJS = ['the pump', 'the valve', 'the sensor']
words = set(' '.join(POS + NEG + ITEMS + NAMES + OBJS).split()) | set(VERBS) \
    | set(VERBS.values()) | set('the is not i love hate it someone did'.split())
vocab = ['[PAD]', '[CLS]', '[SEP]', '[MASK]'] + sorted(words)
idx, L, D = {w: i for i, w in enumerate(vocab)}, 16, 32

def batch(examples):                      # each example: (text,) or (text_a, text_b)
    ids, segs = [], []
    for ex in examples:
        toks, seg = ['[CLS]'], [0]
        for s, text in enumerate(ex):
            part = text.split() + ['[SEP]']
            toks += part; seg += [s] * len(part)
        ids.append([idx[t] for t in toks] + [0] * (L - len(toks)))
        segs.append(seg + [0] * (L - len(seg)))
    return torch.tensor(ids), torch.tensor(segs)

class TinyBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos, self.seg = nn.Embedding(len(vocab), D), nn.Embedding(L, D), nn.Embedding(2, D)
        self.norm = nn.LayerNorm(D)
        block = nn.TransformerEncoderLayer(D, 2, 64, dropout=0.0, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(block, 2)
        self.mlm = nn.Linear(D, len(vocab))
    def forward(self, ids, seg):
        h = self.tok(ids) + self.pos(torch.arange(ids.shape[1])) + self.seg(seg)
        return self.encoder(self.norm(h), src_key_padding_mask=ids.eq(0))

# 1) Masked-LM pretraining on unlabeled text (no sentiment or NLI labels).
corpus = [(f'i love it the {i} is {a}',) for i in ITEMS for a in POS] \
    + [(f'i hate it the {i} is {a}',) for i in ITEMS for a in NEG] \
    + [(f'{n} {v} {o}',) for n in NAMES for v in VERBS for o in OBJS]
model = TinyBERT()
ids, seg = batch(corpus)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for step in range(400):
    pick = (torch.rand(ids.shape) < 0.15) & ids.gt(3)    # 15% of real tokens
    logits = model.mlm(model(ids.masked_fill(pick, idx['[MASK]']), seg))
    loss = F.cross_entropy(logits.flatten(0, 1), ids.masked_fill(~pick, -100).flatten())
    opt.zero_grad(); loss.backward(); opt.step()
print('final MLM loss:', round(loss.item(), 3))
pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}

# 2) Fine-tune encoder + [CLS] head end to end.
def finetune(encoder, train, n_classes, epochs=60, lr=1e-3):
    torch.manual_seed(1)
    head = nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Linear(D, n_classes))
    opt = torch.optim.AdamW([*encoder.parameters(), *head.parameters()], lr=lr)
    x, y = batch([ex for ex, _ in train]), torch.tensor([lab for _, lab in train])
    for _ in range(epochs):
        loss = F.cross_entropy(head(encoder(*x)[:, 0]), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda exs: head(encoder(*batch(exs))[:, 0]).argmax(-1).tolist()

def report(name, data, predict):
    gold = [lab for _, lab in data]
    with torch.no_grad():
        pred = predict([ex for ex, _ in data])
    RESULTS[name] = (accuracy_score(gold, pred), f1_score(gold, pred, average="macro"))
    print(f'{name:46s} acc={RESULTS[name][0]:.2f} macro-F1={RESULTS[name][1]:.2f}')
# Task A: sentiment with negation. "superb" and "terrible" never get a label.
def review(item, adj, negated):
    return ((f'the {item} is {"not " * negated}{adj}',), int(adj in POS) ^ negated)
labeled = [a for a in POS + NEG if a not in ('superb', 'terrible')]
train_a = [review(i, a, n) for i in ITEMS for a in labeled for n in (0, 1)]
test_a = [review(i, a, n) for i in ITEMS for a in ('superb', 'terrible') for n in (0, 1)]
report('sentiment, pretrained encoder', test_a, finetune(model, train_a, 2))
report('sentiment, random-init encoder', test_a, finetune(TinyBERT(), train_a, 2))

# Task B: NLI (0 entailment, 1 neutral, 2 contradiction) with a negation shortcut.
def pair(n, v, o, kind):
    p, neg = f'{n} {v} {o}', f'{n} did not {VERBS[v]} {o}'
    other = list(VERBS)[(list(VERBS).index(v) + 1) % 3]
    return {'ent': ((p, f'someone {v} {o}'), 0), 'neu': ((p, f'{n} {other} {o}'), 1),
            'con': ((p, neg), 2),                   # training: "not" only in contradictions
            'neg-ent': ((neg, neg), 0),             # challenge: "not" in an entailment
            'neg-con': ((neg, p), 2)}[kind]         # challenge: contradiction without "not"
triples = list(itertools.product(NAMES, VERBS, OBJS)); random.shuffle(triples)
train_b = [pair(*t, k) for t in triples[:18] for k in ('ent', 'neu', 'con')]
test_b = [pair(*t, k) for t in triples[18:] for k in ('ent', 'neu', 'con')]
challenge = [pair(*t, k) for t in triples[18:] for k in ('neg-ent', 'neu', 'neg-con')]

vec = CountVectorizer(token_pattern=r'\S+').fit([ex[1] for ex, _ in train_b])
hyp_only = LogisticRegression(C=10, max_iter=1000).fit(
    vec.transform([ex[1] for ex, _ in train_b]), [lab for _, lab in train_b])
hyp_predict = lambda exs: hyp_only.predict(vec.transform([ex[1] for ex in exs])).tolist()
model.load_state_dict(pretrained_state)
pair_predict = finetune(model, train_b, 3)
for split, data in [('in-distribution', test_b), ('challenge', challenge)]:
    report(f'NLI {split}, hypothesis-only', data, hyp_predict)
    report(f'NLI {split}, pair encoder', data, pair_predict)
counter = [pair(*t, k) for t in triples[:18] for k in ('neg-ent', 'neg-con')]
model.load_state_dict(pretrained_state)
report('NLI challenge, pair encoder + counterexamples', challenge,
       finetune(model, train_b + counter, 3))

# ---- Figure: grouped bars of macro-F1.
groups = [
    ("Sentiment\n(unseen adjectives)", ["sentiment, pretrained encoder", "sentiment, random-init encoder"],
     ["pretrained", "random init"]),
    ("NLI in-distribution", ["NLI in-distribution, hypothesis-only", "NLI in-distribution, pair encoder"],
     ["hypothesis-only", "pair encoder"]),
    ("NLI challenge", ["NLI challenge, hypothesis-only", "NLI challenge, pair encoder",
                       "NLI challenge, pair encoder + counterexamples"],
     ["hypothesis-only", "pair encoder", "+ counterexamples"]),
]
colors = {"pretrained": "#2866ae", "random init": "#b0b7c3", "hypothesis-only": "#d98c3a",
          "pair encoder": "#2866ae", "+ counterexamples": "#3a8f5c"}
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True)
for ax, (title, keys, labels) in zip(axes, groups):
    vals = [RESULTS[k][1] for k in keys]
    bars = ax.bar(range(len(vals)), vals, color=[colors[l] for l in labels])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(vals)), labels, fontsize=10, rotation=12)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.12)
axes[0].set_ylabel("macro-F1 on held-out examples")
fig.suptitle("Tiny from-scratch encoder: transfer helps, and challenge sets expose shortcuts", fontsize=12)
fig.tight_layout()
save(fig, "experiment-results.png")

# ---- Figure: start + end scores over a short synthetic passage.
tokens = ["Pump", "B", "stopped", "at", "noon", ";", "Pump", "A", "kept", "running"]
start = np.array([2.8, 1.0, -1.0, -1.5, -0.8, -2.0, 1.0, 1.4, -1.2, -1.6])
end = np.array([-1.0, 2.4, -0.6, -1.4, 0.2, -2.0, -0.9, 1.6, -1.0, -0.4])
n, max_len = len(tokens), 3
grid = start[:, None] + end[None, :]
valid = np.array([[i <= j < i + max_len for j in range(n)] for i in range(n)])
fig, ax = plt.subplots(figsize=(7.6, 6.0))
im = ax.imshow(np.ma.masked_where(~valid, grid), cmap="Blues")
ax.imshow(np.ma.masked_where(valid, np.ones_like(grid)), cmap="Greys", vmin=0, vmax=4, alpha=0.6)
i, j = np.unravel_index(np.argmax(np.where(valid, grid, -np.inf)), grid.shape)
ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#c0392b", lw=2.5))
ax.set_xticks(range(n), tokens, rotation=45, ha="right")
ax.set_yticks(range(n), tokens)
ax.set(xlabel="end token j", ylabel="start token i",
       title=f"Span score a_i + b_j (valid: i ≤ j < i + {max_len}); best = '{' '.join(tokens[i:j+1])}'")
fig.colorbar(im, ax=ax, fraction=0.046, label="synthetic score")
fig.tight_layout()
save(fig, "span-scores.png")
