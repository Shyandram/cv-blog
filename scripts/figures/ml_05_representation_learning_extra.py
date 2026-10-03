"""Extra figures for ML 05 (representation learning).

Trains a tiny SimCLR-style convolutional encoder on scikit-learn's 8x8 digits
(CPU, about one minute) and writes figures to
src/assets/ml/ml-05-representation-learning/.
"""
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

OUT = Path(__file__).resolve().parents[2] / "src/assets/ml/ml-05-representation-learning"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})
BLUE, ORANGE, GREEN, GREY, RED = "#2f6fdf", "#e8743b", "#2a9d6f", "#7a7a7a", "#c0392b"


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ----------------------------------------------------------------- data
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(1)
d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data / 16., d.target, test_size=.3,
                                      random_state=0, stratify=d.target)
Xtr_t = torch.tensor(Xtr, dtype=torch.float32)


def augment(x):
    img = F.pad(x.view(-1, 1, 8, 8), (1, 1, 1, 1))
    i, j = np.random.randint(0, 3, size=2)
    img = img[:, :, i:i + 8, j:j + 8]
    img = img * (0.7 + 0.6 * torch.rand(len(img), 1, 1, 1))
    img = img + 0.1 * torch.randn_like(img)
    return img.reshape(len(x), 64)


def shift(X, dy, dx):
    img = np.zeros((len(X), 10, 10)); img[:, 1:9, 1:9] = X.reshape(-1, 8, 8)
    return img[:, 1 - dy:9 - dy, 1 - dx:9 - dx].reshape(len(X), 64)


rng = np.random.default_rng(0)
moves = rng.integers(-1, 2, size=(len(Xte), 2))
Xte_shift = np.concatenate([shift(x[None], *m) for x, m in zip(Xte, moves)])


def make_backbone():
    return nn.Sequential(nn.Unflatten(1, (1, 8, 8)),
                         nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(2), nn.Flatten())


backbone = make_backbone()
random_backbone = copy.deepcopy(backbone)
head = nn.Sequential(nn.ReLU(), nn.Linear(256, 64))
opt = torch.optim.Adam([*backbone.parameters(), *head.parameters()], lr=1e-3)


def nt_xent(z1, z2, tau=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=-1)
    n = len(z1)
    logits = (z @ z.T / tau).fill_diagonal_(float("-inf"))
    target = (torch.arange(2 * n) + n) % (2 * n)
    return F.cross_entropy(logits, target)


def align_uniform(model_b, model_h):
    with torch.no_grad():
        x = Xtr_t[:512]
        z1 = F.normalize(model_h(model_b(augment(x))), dim=-1)
        z2 = F.normalize(model_h(model_b(augment(x))), dim=-1)
        align = (z1 - z2).square().sum(1).mean().item()
        sq = torch.pdist(z1).square()
        uniform = sq.mul(-2).exp().mean().log().item()
    return align, uniform


hist = []
STEPS = 800
for step in range(STEPS + 1):
    if step % 25 == 0:
        hist.append((step, *align_uniform(backbone, head)))
    if step == STEPS:
        break
    x = Xtr_t[torch.randint(len(Xtr_t), (256,))]
    loss = nt_xent(head(backbone(augment(x))), head(backbone(augment(x))))
    opt.zero_grad(); loss.backward(); opt.step()
print("final loss", loss.item())


def encoder(model):
    return lambda A: model(torch.tensor(A, dtype=torch.float32)).detach().numpy()


pca = PCA(n_components=32, random_state=0).fit(Xtr)
feats = {"Raw pixels": lambda A: A, "PCA (32-d)": pca.transform,
         "Random CNN": encoder(random_backbone), "SimCLR CNN": encoder(backbone)}

# ------------------------------------------------- figure: k-NN accuracy
clean, shifted = [], []
for name, f in feats.items():
    knn = KNeighborsClassifier(5, metric="cosine").fit(f(Xtr), ytr)
    clean.append(knn.score(f(Xte), yte)); shifted.append(knn.score(f(Xte_shift), yte))
    print(name, clean[-1], shifted[-1])
fig, ax = plt.subplots(figsize=(7.5, 3.6))
pos = np.arange(len(feats))
b1 = ax.bar(pos - .2, clean, .4, color=BLUE, label="Clean test digits")
b2 = ax.bar(pos + .2, shifted, .4, color=ORANGE, label="Test digits shifted by up to 1 px")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .01, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=9)
ax.set_xticks(pos, list(feats)); ax.set_ylim(0, 1.08); ax.set_ylabel("5-NN accuracy")
ax.set_title("Same labels, different features: k-NN on 8x8 digits")
ax.legend(loc="lower left", fontsize=9); ax.grid(axis="y", alpha=.3)
save(fig, "digits-knn-shift.png")

# ------------------------------------------ figure: similarity matrix
with torch.no_grad():
    idx = torch.tensor([np.where(ytr == c)[0][0] for c in range(8)])
    x = Xtr_t[idx]
    z = F.normalize(head(backbone(torch.cat([augment(x), augment(x)]))), dim=-1)
    sim = (z @ z.T).numpy()
fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
ax.set_xticks(range(16), [f"{c}a" for c in range(8)] + [f"{c}b" for c in range(8)], fontsize=8, rotation=90)
ax.set_yticks(range(16), [f"{c}a" for c in range(8)] + [f"{c}b" for c in range(8)], fontsize=8)
ax.axhline(7.5, color="white", lw=1); ax.axvline(7.5, color="white", lw=1)
ax.set_title("Cosine similarity of 16 views (8 images x 2)")
fig.colorbar(im, ax=ax, fraction=.046)
save(fig, "similarity-matrix.png")

# ------------------------------------ figure: alignment and uniformity
h = np.array(hist)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
axes[0].plot(h[:, 0], h[:, 1], color=BLUE); axes[0].set_title("Alignment (lower = views agree)")
axes[1].plot(h[:, 0], h[:, 2], color=ORANGE); axes[1].set_title("Uniformity (lower = more spread)")
for a in axes:
    a.set_xlabel("training step"); a.grid(alpha=.3)
save(fig, "alignment-uniformity.png")

# ------------------------------------------ figure: label efficiency
budgets = [1, 2, 5, 10, 20, 50]


def finetune(seed, lp_first):
    """Full fine-tuning from the SimCLR backbone; optionally fit the head first (LP-FT)."""
    torch.manual_seed(seed)
    net = nn.Sequential(copy.deepcopy(backbone), nn.Linear(256, 10))
    X = torch.tensor(Xl, dtype=torch.float32); Y = torch.tensor(yl)
    if lp_first:  # stage 1: train only the linear head on frozen features
        o = torch.optim.Adam(net[1].parameters(), lr=1e-2)
        with torch.no_grad():
            H = net[0](X)
        for _ in range(300):
            loss = F.cross_entropy(net[1](H), Y)
            o.zero_grad(); loss.backward(); o.step()
    o = torch.optim.Adam(net.parameters(), lr=3e-4)
    for _ in range(150):
        loss = F.cross_entropy(net(augment(X)), Y)
        o.zero_grad(); loss.backward(); o.step()
    with torch.no_grad():
        return (net(torch.tensor(Xte_shift, dtype=torch.float32)).argmax(1).numpy() == yte).mean()


curves = {k: [] for k in ["Raw pixels", "PCA (32-d)", "SimCLR CNN (linear probe)", "SimCLR CNN (fine-tuned)", "SimCLR CNN (LP-FT)"]}
for n in budgets:
    accs = {k: [] for k in curves}
    for seed in range(3):
        r = np.random.default_rng(seed)
        sel = np.concatenate([r.choice(np.where(ytr == c)[0], n, replace=False) for c in range(10)])
        Xl, yl = Xtr[sel], ytr[sel]
        for k, f in [("Raw pixels", feats["Raw pixels"]), ("PCA (32-d)", feats["PCA (32-d)"]),
                     ("SimCLR CNN (linear probe)", feats["SimCLR CNN"])]:
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(f(Xl), yl)
            accs[k].append(clf.score(f(Xte_shift), yte))
        accs["SimCLR CNN (fine-tuned)"].append(finetune(seed, False))
        accs["SimCLR CNN (LP-FT)"].append(finetune(seed, True))
    for k in curves:
        curves[k].append(np.mean(accs[k]))
    print(n, {k: round(v[-1], 3) for k, v in curves.items()})
fig, ax = plt.subplots(figsize=(7, 3.8))
for (k, v), c, m in zip(curves.items(), [GREY, GREEN, BLUE, ORANGE, RED], "osD^v"):
    ax.plot(budgets, v, marker=m, color=c, label=k)
ax.set_xscale("log"); ax.set_xticks(budgets, [str(b) for b in budgets])
ax.set_xlabel("labeled examples per class"); ax.set_ylabel("accuracy on shifted test digits")
ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_title("Label efficiency (mean of 3 label draws)")
save(fig, "label-efficiency.png")

# ---------------------------------------------- figure: transfer pipeline
fig, ax = plt.subplots(figsize=(10, 3.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3.4)


def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="black", lw=1, alpha=.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)


def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4))


box(0.1, 2.2, 2.0, .9, "Large unlabeled\ndata", "#dce8fb")
box(2.7, 2.2, 2.2, .9, "Pretext objective\n(contrastive, masked,\nself-distillation)", "#fde3d3")
box(5.5, 2.2, 1.8, .9, "Encoder f\n(weights saved)", "#d7f0e4")
arrow(2.1, 2.65, 2.7, 2.65); arrow(4.9, 2.65, 5.5, 2.65)
box(0.1, 0.3, 2.0, .9, "Small labeled\ntask data", "#dce8fb")
box(3.0, 0.9, 2.6, .7, "k-NN / linear probe\n(f frozen)", "#eeeeee")
box(3.0, 0.0, 2.6, .7, "Fine-tuning\n(f updated)", "#eeeeee")
arrow(2.1, 0.75, 3.0, 1.25); arrow(2.1, 0.75, 3.0, 0.35)
arrow(6.4, 2.2, 5.6, 1.4); arrow(6.4, 2.2, 5.6, 0.55)
box(7.0, 0.3, 2.8, 1.1, "Task predictions\nreport: labels, protocol,\ncompute, data overlap", "#f6f0c8")
arrow(5.6, 1.25, 7.0, 0.95); arrow(5.6, 0.35, 7.0, 0.7)
ax.text(5, 3.3, "Stage 1: pretraining", ha="center", fontsize=11, weight="bold")
ax.text(0.1, 1.6, "Stage 2: transfer", ha="left", fontsize=11, weight="bold")
save(fig, "transfer-pipeline.png")

# ------------------------------------- figure: one-hot vs distributed
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
names = ["red car", "blue car", "red truck", "blue truck"]
onehot = np.eye(4)
dist = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
for a, M, cols, t in [(axes[0], onehot, ["unit 1", "unit 2", "unit 3", "unit 4"], "One-hot (local): one unit per concept"),
                      (axes[1], dist, ["red", "blue", "car", "truck"], "Distributed: concepts share units")]:
    a.imshow(M, cmap="Blues", vmin=0, vmax=1.3)
    a.set_xticks(range(4), cols); a.set_yticks(range(4), names); a.set_title(t, fontsize=10)
    for i in range(4):
        for j in range(4):
            a.text(j, i, int(M[i, j]), ha="center", va="center")
save(fig, "one-hot-vs-distributed.png")
