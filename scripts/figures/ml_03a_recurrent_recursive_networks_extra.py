"""Figures for ML 03A - Recurrent and Recursive Networks.

Run:  .venv/bin/python scripts/figures/ml_03a_recurrent_recursive_networks_extra.py
Writes PNGs only to src/assets/ml/ml-03a-recurrent-recursive-networks/.
Synthetic data and tiny CPU models only (total run time well under a minute).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT = Path(__file__).resolve().parents[2] / "src/assets/ml/ml-03a-recurrent-recursive-networks"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, GOLD, PURPLE, GRAY = ("#2a6fdb", "#e4572e", "#17a398",
                                           "#f3a712", "#8e5ad6", "#555555")
torch.set_num_threads(1)


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name, f"{(OUT / name).stat().st_size / 1024:.0f} KB")


def box(ax, xy, w, h, text, fc, fs=11, ec="k"):
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, color="k", rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=color,
                                 lw=lw, connectionstyle=f"arc3,rad={rad}"))


# ------------------------------------------------------------------ 1. folded vs unrolled
def unrolled():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(-0.5, 11.2); ax.set_ylim(-0.3, 3.4); ax.axis("off")
    # folded
    box(ax, (1, 1.6), 1.0, 0.7, "$h$", "#dfe8f7")
    box(ax, (1, 0.2), 0.8, 0.5, "$x_t$", "#f3f3f3")
    box(ax, (1, 3.0), 0.8, 0.5, "$y_t$", "#d6f2ee")
    arrow(ax, (1, 0.45), (1, 1.25)); arrow(ax, (1, 1.95), (1, 2.75))
    ax.add_patch(FancyArrowPatch((1.5, 1.75), (1.5, 1.45), arrowstyle="-|>", mutation_scale=13,
                                 connectionstyle="arc3,rad=-2.2", lw=1.4, color=ORANGE))
    ax.text(1.95, 1.6, "$W_h$", color=ORANGE, va="center")
    ax.text(1, -0.25, "folded", ha="center", fontsize=10, color=GRAY)
    ax.text(2.85, 1.6, "=", fontsize=22, ha="center", va="center")
    xs = [4.9, 6.7, 8.5, 10.3]
    labels = ["1", "2", "3", "T"]
    box(ax, (3.75, 1.6), 0.55, 0.5, "$h_0$", "white")
    arrow(ax, (4.03, 1.6), (xs[0] - 0.5, 1.6), ORANGE)
    for i, (x, l) in enumerate(zip(xs, labels)):
        box(ax, (x, 1.6), 1.0, 0.7, f"$h_{l}$", "#dfe8f7")
        box(ax, (x, 0.2), 0.8, 0.5, f"$x_{l}$", "#f3f3f3")
        box(ax, (x, 3.0), 0.8, 0.5, f"$y_{l}$", "#d6f2ee")
        arrow(ax, (x, 0.45), (x, 1.25)); arrow(ax, (x, 1.95), (x, 2.75))
        if i < 3:
            arrow(ax, (x + 0.5, 1.6), (xs[i + 1] - 0.5, 1.6), ORANGE)
            ax.text((x + xs[i + 1]) / 2, 1.75, "$W_h$", color=ORANGE, ha="center")
    ax.text(9.4, 1.25, "…", fontsize=16, ha="center")
    ax.text(7.3, -0.25, "unrolled: the same $W_x, W_h, W_y$ at every step", ha="center",
            fontsize=10, color=GRAY)
    save(fig, "rnn-unrolled.png")


# ------------------------------------------------------------------ 2. gradient flow
def gradient_flow():
    torch.manual_seed(0)
    T, H = 80, 32
    x = torch.randn(1, T, 4)
    res = {}
    for name, ctor in [("tanh RNN, small $W_h$ (gain 0.5)", lambda: nn.RNN(4, H, batch_first=True)),
                       ("tanh RNN, large $W_h$ (gain 3)", lambda: nn.RNN(4, H, batch_first=True)),
                       ("LSTM (forget bias +1)", lambda: nn.LSTM(4, H, batch_first=True)),
                       ("GRU (default init)", lambda: nn.GRU(4, H, batch_first=True))]:
        torch.manual_seed(1)
        m = ctor()
        if "RNN" in name:
            gain = 0.5 if "0.5" in name else 3.0
            with torch.no_grad():
                q, _ = torch.linalg.qr(torch.randn(H, H))
                m.weight_hh_l0.copy_(gain * q)
        if "LSTM" in name:
            with torch.no_grad():
                m.bias_hh_l0[H:2 * H] += 1.0
        xi = x.clone().requires_grad_(True)
        out, _ = m(xi)
        out[0, -1].sum().backward()
        res[name] = xi.grad[0].norm(dim=1).numpy()
    lin = {}
    k = np.arange(T)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    for lam, c in [(0.9, BLUE), (1.0, GRAY), (1.1, ORANGE)]:
        axes[0].semilogy(k, lam ** k, color=c, lw=2, label=f"$\\lambda = {lam}$")
    axes[0].set_xlabel("steps back in time $t-s$")
    axes[0].set_ylabel(r"$\lambda^{t-s}$")
    axes[0].set_title("Linear RNN: gradient factor $\\lambda^{t-s}$")
    axes[0].legend(); axes[0].grid(alpha=.3)
    for (name, g), c in zip(res.items(), [BLUE, ORANGE, GREEN, PURPLE]):
        axes[1].semilogy(T - 1 - np.arange(T), g + 1e-30, color=c, lw=2, label=name)
    axes[1].set_xlabel("steps back in time $T-s$")
    axes[1].set_ylabel(r"$\|\partial (\sum h_T) / \partial x_s\|$")
    axes[1].set_ylim(1e-14, 1e3)
    axes[1].set_title("Measured at random initialisation, $T = 80$")
    axes[1].legend(fontsize=9, loc="lower left"); axes[1].grid(alpha=.3)
    save(fig, "gradient-flow.png")
    return res


# ------------------------------------------------------------------ 3. LSTM cell
def lstm_cell():
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    # cell-state highway
    ax.add_patch(FancyArrowPatch((0.3, 4.2), (9.7, 4.2), arrowstyle="-|>", mutation_scale=16,
                                 lw=3, color=GOLD))
    ax.text(0.3, 4.45, "$c_{t-1}$", fontsize=12); ax.text(9.3, 4.45, "$c_t$", fontsize=12)
    ax.add_patch(Circle((3.0, 4.2), 0.25, fc="white", ec="k")); ax.text(3.0, 4.2, "×", ha="center", va="center", fontsize=14)
    ax.add_patch(Circle((5.2, 4.2), 0.25, fc="white", ec="k")); ax.text(5.2, 4.2, "+", ha="center", va="center", fontsize=14)
    # inputs
    ax.add_patch(FancyArrowPatch((0.3, 0.7), (8.0, 0.7), arrowstyle="-", lw=1.5, color=GRAY))
    ax.text(0.3, 0.35, "$[x_t, h_{t-1}]$", fontsize=12)
    gates = [(3.0, "$f_t$\nσ", "#fde2d8"), (4.4, "$i_t$\nσ", "#dfe8f7"),
             (5.9, "$\\tilde c_t$\ntanh", "#d6f2ee"), (7.4, "$o_t$\nσ", "#efe3fb")]
    for x, t, fc in gates:
        box(ax, (x, 2.0), 0.9, 0.85, t, fc, fs=11)
        arrow(ax, (x, 0.7), (x, 1.57), GRAY)
    arrow(ax, (3.0, 2.43), (3.0, 3.95))
    ax.add_patch(Circle((5.2, 3.2), 0.22, fc="white", ec="k")); ax.text(5.2, 3.2, "×", ha="center", va="center", fontsize=13)
    arrow(ax, (4.4, 2.43), (5.05, 3.05)); arrow(ax, (5.9, 2.43), (5.35, 3.05))
    arrow(ax, (5.2, 3.42), (5.2, 3.95))
    # output path
    box(ax, (8.4, 3.3), 0.8, 0.5, "tanh", "#f3f3f3", fs=10)
    arrow(ax, (8.4, 4.2), (8.4, 3.56))
    ax.add_patch(Circle((8.4, 2.3), 0.22, fc="white", ec="k")); ax.text(8.4, 2.3, "×", ha="center", va="center", fontsize=13)
    arrow(ax, (8.4, 3.05), (8.4, 2.53)); arrow(ax, (7.85, 2.0), (8.18, 2.25))
    arrow(ax, (8.62, 2.3), (9.6, 2.3)); ax.text(9.65, 2.3, "$h_t$", va="center", fontsize=12)
    ax.text(5.0, 4.75, "cell state: an additive path (forget × old + input × candidate)",
            ha="center", fontsize=10, color=GRAY)
    save(fig, "lstm-cell.png")


# ------------------------------------------------------------------ 4. encoder-decoder
def encoder_decoder():
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6))
    for ax, attn in zip(axes, [False, True]):
        ax.set_xlim(0, 11); ax.set_ylim(0, 3.4); ax.axis("off")
        enc_x = [0.8, 2.0, 3.2, 4.4]
        dec_x = [6.6, 7.8, 9.0, 10.2]
        for i, x in enumerate(enc_x):
            box(ax, (x, 1.2), 0.8, 0.6, f"$h_{i + 1}$", "#dfe8f7")
            box(ax, (x, 0.25), 0.7, 0.4, f"$x_{i + 1}$", "#f3f3f3", fs=10)
            arrow(ax, (x, 0.45), (x, 0.9))
            if i:
                arrow(ax, (enc_x[i - 1] + 0.4, 1.2), (x - 0.4, 1.2), ORANGE)
        for i, x in enumerate(dec_x):
            box(ax, (x, 1.2), 0.8, 0.6, f"$s_{i + 1}$", "#d6f2ee")
            box(ax, (x, 2.25), 0.7, 0.4, f"$y_{i + 1}$", "#f3f3f3", fs=10)
            arrow(ax, (x, 1.5), (x, 2.05))
            if i:
                arrow(ax, (dec_x[i - 1] + 0.4, 1.2), (x - 0.4, 1.2), GREEN)
        if not attn:
            box(ax, (5.5, 1.2), 0.7, 0.6, "$c$", "#fff1c9")
            arrow(ax, (4.8, 1.2), (5.15, 1.2), ORANGE); arrow(ax, (5.85, 1.2), (6.2, 1.2), GREEN)
            ax.text(5.5, 3.1, "Seq2seq: the whole input squeezed into one vector $c$",
                    ha="center", fontsize=11)
        else:
            arrow(ax, (4.8, 1.2), (6.2, 1.2), GREEN)
            target = dec_x[2]
            ws = [0.1, 0.15, 0.6, 0.15]
            for x, w in zip(enc_x, ws):
                ax.add_patch(FancyArrowPatch((x, 1.5), (target - 0.25, 1.5), arrowstyle="-|>",
                                             mutation_scale=10, lw=0.8 + 6 * w, color=PURPLE,
                                             alpha=0.35 + 0.6 * w, connectionstyle="arc3,rad=-0.1"))
            ax.text(5.5, 3.1, "With attention: step 3 of the decoder reads a weighted mix of all $h_j$",
                    ha="center", fontsize=11)
    save(fig, "encoder-decoder.png")


# ------------------------------------------------------------------ 5. attention on a toy task
class Seq2Seq(nn.Module):
    def __init__(self, vocab=10, d=64):
        super().__init__()
        self.emb = nn.Embedding(vocab + 1, d)        # token `vocab` = start symbol
        self.enc = nn.GRU(d, d, batch_first=True)
        self.dec = nn.GRUCell(2 * d, d)
        self.Wa, self.Ua = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.va = nn.Linear(d, 1, bias=False)
        self.out = nn.Linear(2 * d, vocab)
        self.vocab = vocab

    def forward(self, src, tgt):
        H, h = self.enc(self.emb(src))                # H: B,T,d
        s = h[0]
        prev = torch.full_like(tgt[:, 0], self.vocab)
        logits, attns = [], []
        for t in range(tgt.shape[1]):
            e = self.va(torch.tanh(self.Wa(s)[:, None] + self.Ua(H))).squeeze(-1)
            a = e.softmax(-1)                           # B,T
            ctx = (a[:, :, None] * H).sum(1)
            s = self.dec(torch.cat([self.emb(prev), ctx], -1), s)
            logits.append(self.out(torch.cat([s, ctx], -1)))
            attns.append(a)
            prev = tgt[:, t]                            # teacher forcing
        return torch.stack(logits, 1), torch.stack(attns, 1)


def attention_toy():
    torch.manual_seed(0)
    model = Seq2Seq()
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    L = 10
    for step in range(700):
        src = torch.randint(0, 10, (64, L))
        tgt = src.flip(1)
        logits, _ = model(src, tgt)
        loss = F.cross_entropy(logits.reshape(-1, 10), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        src = torch.randint(0, 10, (512, L)); tgt = src.flip(1)
        logits, att = model(src, tgt)
        acc = (logits.argmax(-1) == tgt).float().mean().item()
    print("toy reversal token accuracy (teacher-forced):", round(acc, 3))
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    im = ax.imshow(att[0].numpy(), cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks(range(L)); ax.set_xticklabels(src[0].tolist())
    ax.set_yticks(range(L)); ax.set_yticklabels(tgt[0].tolist())
    ax.set_xlabel("input sequence (encoder positions)")
    ax.set_ylabel("output sequence (decoder steps)")
    ax.set_title("Learned attention: reverse a sequence")
    fig.colorbar(im, ax=ax, fraction=.046)
    save(fig, "attention-reversal.png")


# ------------------------------------------------------------------ 6. recursive tree
def recursive_tree():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.set_xlim(0, 6); ax.set_ylim(0, 4); ax.axis("off")
    leaves = [(0.7, "a"), (2.1, "very"), (3.5, "good"), (4.9, "movie")]
    for x, w in leaves:
        box(ax, (x, 0.5), 1.0, 0.5, w, "#f3f3f3", fs=10)
    p1 = (2.8, 1.7); p2 = (3.8, 2.7); root = (2.3, 3.6)
    box(ax, p1, 1.3, 0.5, "very good", "#dfe8f7", fs=9)
    box(ax, p2, 1.9, 0.5, "very good movie", "#dfe8f7", fs=9)
    box(ax, root, 2.2, 0.5, "a very good movie", "#d6f2ee", fs=9)
    for a, b in [((2.1, 0.75), p1), ((3.5, 0.75), p1), ((p1[0], 1.95), p2),
                 ((4.9, 0.75), p2), ((0.7, 0.75), root), ((p2[0], 2.95), root)]:
        arrow(ax, a, (b[0], b[1] - 0.25), BLUE)
    ax.set_title("Recursive net on a parse tree: same $W$ at every node", fontsize=11)
    ax = axes[1]
    ax.set_xlim(0, 6); ax.set_ylim(0, 4); ax.axis("off")
    seq = ["a", "very", "good", "movie"]
    for i, w in enumerate(seq):
        x = 0.8 + 1.4 * i
        box(ax, (x, 0.5), 1.0, 0.5, w, "#f3f3f3", fs=10)
        box(ax, (x, 2.0), 0.8, 0.5, f"$h_{i + 1}$", "#dfe8f7")
        arrow(ax, (x, 0.75), (x, 1.75), GRAY)
        if i:
            arrow(ax, (x - 1.0, 2.0), (x - 0.4, 2.0), ORANGE)
    ax.text(3.0, 3.0, "path length from first word to $h_4$: 3 steps", ha="center", fontsize=10, color=GRAY)
    ax.set_title("Recurrent net on the same words: a chain", fontsize=11)
    save(fig, "recursive-vs-recurrent.png")


if __name__ == "__main__":
    unrolled()
    r = gradient_flow()
    for k, v in r.items():
        print(k, "grad at s=0:", f"{v[0]:.2e}", "at s=T-1:", f"{v[-1]:.2e}")
    lstm_cell()
    encoder_decoder()
    attention_toy()
    recursive_tree()
