"""Figures for NLP 08 (sequence-to-sequence Transformers).

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/nlp_08_seq2seq_transformers.py
Re-trains the tutorial's tiny synthetic translation model with the same code and seed
(no downloads; CPU, about 20 seconds) and plots its training loss and one cross-attention map.
Writes only to src/assets/nlp/nlp-08-seq2seq-transformers/.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math, random, time
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(0); torch.set_num_threads(1); rng = random.Random(0)
lexicon = {"the": "la", "a": "un", "robot": "robo", "sensor": "senso",
           "pump": "pompa", "valve": "valva", "drone": "drono", "cable": "kablo",
           "red": "rosso", "blue": "blu", "big": "grando", "small": "pico",
           "sees": "vidi", "checks": "kontrolli", "moves": "movi",
           "opens": "aperi", "lifts": "levi"}
DET, NOUN = ["the", "a"], ["robot", "sensor", "pump", "valve", "drone", "cable"]
ADJ, VERB = ["red", "blue", "big", "small"], ["sees", "checks", "moves", "opens", "lifts"]

def noun_phrase():
    adj = [rng.choice(ADJ)] if rng.random() < 0.5 else []
    return rng.choice(DET), adj, rng.choice(NOUN)

def pair():          # source: SVO, adjective before noun; target: SOV, adjective after noun
    (d1, a1, n1), (d2, a2, n2) = noun_phrase(), noun_phrase()
    verb, neg = rng.choice(VERB), rng.random() < 0.3
    src = [d1, *a1, n1] + (["never"] if neg else []) + [verb, d2, *a2, n2]
    np_ = lambda d, a, n: [lexicon[d], lexicon[n]] + [lexicon[x] for x in a]
    tgt = np_(d1, a1, n1) + np_(d2, a2, n2) + ["o", lexicon[verb]] + (["nai"] if neg else [])
    return " ".join(src), " ".join(tgt)

pairs = list(dict(pair() for _ in range(6000)).items())   # unique source sentences
train_pairs, test_pairs = pairs[:3000], pairs[3000:3200]
PAD, BOS, EOS = 0, 1, 2
src_vocab = ["<pad>", "<bos>", "<eos>"] + sorted({w for s, _ in pairs for w in s.split()})
tgt_vocab = ["<pad>", "<bos>", "<eos>"] + sorted({w for _, t in pairs for w in t.split()})
s_id = {w: i for i, w in enumerate(src_vocab)}
t_id = {w: i for i, w in enumerate(tgt_vocab)}

def batch(texts, ids, L):
    rows = [[ids[w] for w in t.split()] + [EOS] for t in texts]
    return torch.tensor([r + [PAD] * (L - len(r)) for r in rows])
S = batch([s for s, _ in train_pairs], s_id, 10)
Y = batch([t for _, t in train_pairs], t_id, 12)
Y_in = torch.cat([torch.full((len(Y), 1), BOS), Y[:, :-1]], dim=1)   # shift right

def sinusoidal(n, d):
    pos = torch.arange(n)[:, None]
    omega = 1.0 / 10000 ** (torch.arange(0, d, 2) / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * omega), torch.cos(pos * omega)
    return pe

class TinyMT(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.d = d
        self.src_emb = nn.Embedding(len(src_vocab), d)
        self.tgt_emb = nn.Embedding(len(tgt_vocab), d)
        self.register_buffer("pe", sinusoidal(32, d))
        enc = nn.TransformerEncoderLayer(d, 4, 128, dropout=0.0, batch_first=True)
        dec = nn.TransformerDecoderLayer(d, 4, 128, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, 2, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec, 2)
        self.out = nn.Linear(d, len(tgt_vocab))
    def embed(self, emb, x):
        return emb(x) * math.sqrt(self.d) + self.pe[: x.shape[1]]
    def encode(self, src):
        return self.encoder(self.embed(self.src_emb, src), src_key_padding_mask=src.eq(PAD))
    def decode(self, memory, src, y_in):
        T = y_in.shape[1]
        causal = torch.ones(T, T, dtype=torch.bool).triu(1)      # True = blocked
        h = self.decoder(self.embed(self.tgt_emb, y_in), memory, tgt_mask=causal,
                         tgt_key_padding_mask=y_in.eq(PAD),
                         memory_key_padding_mask=src.eq(PAD))
        return self.out(h)

@torch.no_grad()
def greedy(src_text, max_len=14):
    src = batch([src_text], s_id, len(src_text.split()) + 1)
    memory, y = model.encode(src), [BOS]
    for _ in range(max_len):
        logp = model.decode(memory, src, torch.tensor([y]))[0, -1].log_softmax(-1)
        y.append(int(logp.argmax()))
        if y[-1] == EOS:
            break
    return [tgt_vocab[i] for i in y[1:] if i != EOS]


OUT = Path(__file__).resolve().parents[2] / "src/assets/nlp/nlp-08-seq2seq-transformers"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})

model = TinyMT()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
for step in range(1, 401):
    idx = torch.randint(0, len(S), (64,))
    logits = model.decode(model.encode(S[idx]), S[idx], Y_in[idx])   # teacher forcing
    loss = F.cross_entropy(logits.reshape(-1, len(tgt_vocab)), Y[idx].reshape(-1),
                           ignore_index=PAD)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
print(f"loss at 100: {losses[99]:.3f}, at 400: {losses[-1]:.3f}")
model.eval()

# Choose a held-out example with adjectives and negation so reordering is visible.
src_text, tgt_text = next((s, t) for s, t in test_pairs
                          if "never" in s and len(set(s.split())) == len(s.split()) == 8)
hyp = greedy(src_text)
print(src_text, "->", " ".join(hyp), "| reference:", tgt_text)

captured = {}
layer = model.decoder.layers[-1]
def grab(module, args, kwargs):
    captured["q"], captured["kv"] = args[0], args[1]
    captured["mask"] = kwargs.get("key_padding_mask")
handle = layer.multihead_attn.register_forward_pre_hook(grab, with_kwargs=True)
src = batch([src_text], s_id, len(src_text.split()) + 1)
y_in = torch.tensor([[BOS] + [t_id[w] for w in hyp]])
with torch.no_grad():
    model.decode(model.encode(src), src, y_in)
    handle.remove()
    _, w = layer.multihead_attn(captured["q"], captured["kv"], captured["kv"],
                                key_padding_mask=captured["mask"], need_weights=True,
                                average_attn_weights=True)
w = w[0].numpy()                       # (target positions, source positions)
rows = hyp + ["<eos>"]
cols = src_text.split() + ["<eos>"]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
ax.plot(range(1, 401), losses, color="#2a6fdb", lw=1.4)
ax.axvline(100, color="#e4572e", ls="--", lw=1.2)
ax.text(108, max(losses) * 0.45, "first evaluation\n(step 100)", color="#e4572e", fontsize=10)
ax.set_yscale("log")
ax.set_xlabel("training step (batch of 64 pairs)")
ax.set_ylabel("teacher-forced cross-entropy (nats)")
ax.set_title("Training loss of the tiny Transformer", fontsize=11)
ax = axes[1]
ax.imshow(w[: len(rows), : len(cols)], cmap="Purples", vmin=0, vmax=max(0.5, w.max()))
ax.set_xticks(range(len(cols)), cols, rotation=35)
ax.set_yticks(range(len(rows)), rows)
ax.set_xlabel("source token (key)")
ax.set_ylabel("token being predicted (query)")
ax.set_title("Last decoder layer cross-attention (head average)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "training-cross-attention.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(OUT / "training-cross-attention.png")
