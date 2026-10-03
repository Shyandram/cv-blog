"""Original teaching figures for ML 03–10; synthetic data, CPU, no downloads.

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/ml_modern_models.py
These are diagrams of mechanisms, not benchmark results or model outputs.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "src/assets/ml"
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN = "#2a6fdb", "#e4572e", "#17a398"


def save(fig, chapter, name):
    folder = OUT / chapter
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(folder / name)


def grids(axes, arrays, titles, cmap="viridis", vmin=None, vmax=None):
    for ax, a, title in zip(axes, arrays, titles):
        ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])


def cnn():
    x = np.zeros((15, 15)); x[4:11, 5:10] = 1
    patch = np.zeros_like(x); patch[6:9, 6:9] = 1
    rf = np.zeros_like(x); rf[4:11, 4:11] = 1
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    grids(axes, [x, patch, rf],
          ["Input: local structure", "One 3 × 3 layer", "Three 3 × 3 layers: 7 × 7"], "Blues")
    save(fig, "ml-03-convolutional-networks", "receptive-fields.png")
    c = np.arange(16, 257, 16)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(c, 9*c*c, label="Dense 3×3: 9C²", color=BLUE)
    ax.plot(c, 9*c+c*c, label="Depthwise + pointwise: 9C + C²", color=ORANGE)
    ax.set(xlabel="Input = output channels C", ylabel="Weights (no bias)",
           title="Spatial filtering and channel mixing can be separated")
    ax.legend(); ax.grid(alpha=.2)
    save(fig, "ml-03-convolutional-networks", "channel-cost.png")


def transformer():
    n = 8
    global_mask = np.ones((n, n))
    causal = np.tril(np.ones((n, n)))
    local = (np.arange(n)[:, None]//4 == np.arange(n)[None, :]//4)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    grids(axes, [global_mask, causal, local],
          ["Bidirectional", "Causal", "Two local windows"], "Blues", 0, 1)
    for ax in axes:
        ax.set_xlabel("Key position"); ax.set_ylabel("Query position")
    save(fig, "ml-04-vision-transformers", "attention-masks.png")
    sizes = np.array([224, 448, 896])
    n = (sizes//16)**2
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(sizes, n*n/(n[0]**2), "o-", label="Global pair scores N²")
    ax.plot(sizes, n*49/(n[0]*49), "o-", label="7×7 window pair scores 49N")
    ax.set(xlabel="Image side (patch size 16)", ylabel="Relative pair-score count",
           title="Doubling both image dimensions multiplies global scores by 16")
    ax.legend(); ax.grid(alpha=.2)
    save(fig, "ml-04-vision-transformers", "attention-cost.png")


def representations():
    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, title in zip(axes, ["Collapse", "Alignment only", "Alignment + spread"]):
        ax.set(xlim=(-2, 2), ylim=(-2, 2), title=title)
        ax.set_aspect("equal"); ax.grid(alpha=.2)
    axes[0].scatter(np.zeros(8), np.zeros(8), s=80, color=BLUE)
    centers = [rng.normal(0, .1, (8, 2)),
               np.c_[np.cos(np.arange(8)*np.pi/4), np.sin(np.arange(8)*np.pi/4)]]
    for ax, pts in zip(axes[1:], centers):
        second = pts + rng.normal(0, .08, pts.shape)
        ax.scatter(*pts.T, color=BLUE, label="View 1")
        ax.scatter(*second.T, marker="x", color=ORANGE, label="View 2")
        for a, b in zip(pts, second):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="gray", lw=.7)
    axes[2].legend(fontsize=9)
    save(fig, "ml-05-representation-learning", "collapse.png")
    image = (np.indices((8, 8)).sum(axis=0) % 2).astype(float)
    visible = rng.random((8, 8)) > .75
    masked = np.where(visible, image, .5)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    grids(axes, [image, masked, visible], ["Target", "Masked input", "Visible positions"],
          "gray", 0, 1)
    save(fig, "ml-05-representation-learning", "masked-prediction.png")


def generative():
    rng = np.random.default_rng(2)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    z = rng.normal(size=(400, 2))
    x = np.c_[2*z[:, 0], .5*z[:, 1]+.7*z[:, 0]]
    for ax, pts, title in zip(axes, [z, x], ["Simple base distribution z", "Invertible map x = Az"]):
        ax.scatter(*pts.T, s=7, alpha=.5, color=BLUE)
        ax.set(title=title, xlim=(-7, 7), ylim=(-4, 4))
        ax.grid(alpha=.2)
    save(fig, "ml-06-generative-models", "flow-change-of-variables.png")
    t = np.linspace(-3, 3, 500)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for mu, sigma, name in [(-1.5, .4, "q(z | image A)"), (1.4, .5, "q(z | image B)"), (0, 1, "Prior p(z)")]:
        ax.plot(t, np.exp(-.5*((t-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi)), label=name)
    ax.set(xlabel="One latent coordinate z", ylabel="Density",
           title="A VAE learns distributions; KL discourages isolated codes")
    ax.legend(); ax.grid(alpha=.2)
    save(fig, "ml-06-generative-models", "vae-latent.png")


def diffusion():
    rng = np.random.default_rng(3)
    x = np.zeros((32, 32)); x[6:26, 11:21] = 1
    eps = rng.normal(size=x.shape)
    ab = [1, .8, .3, .01]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    grids(axes, [np.sqrt(a)*x+np.sqrt(1-a)*eps for a in ab],
          [f"ᾱ = {a}" for a in ab], "gray", -1, 1)
    save(fig, "ml-07-diffusion-flow-matching", "forward-noising.png")
    noise = rng.normal(size=(15, 2))
    angle = np.arange(15)*2*np.pi/15
    data = np.c_[2*np.cos(angle), 2*np.sin(angle)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for a, b in zip(noise, data):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="gray", alpha=.45)
    ax.scatter(*noise.T, color=BLUE, label="Noise z (t=0)")
    ax.scatter(*data.T, color=ORANGE, label="Data x (t=1)")
    ax.set(title="Conditional straight paths; learned marginal paths may curve",
           xlabel="Coordinate 1", ylabel="Coordinate 2")
    ax.legend(fontsize=9); ax.set_aspect("equal")
    save(fig, "ml-07-diffusion-flow-matching", "flow-paths.png")


def autoregression():
    raster = np.arange(16).reshape(4, 4)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, a, title in zip(axes, [raster, np.zeros((2, 2)), np.ones((4, 4))],
                            ["Next pixel: raster order", "Scale 1: 2×2 map", "Scale 2: 4×4 map"]):
        ax.imshow(a, cmap="Blues", vmin=0, vmax=16)
        if ax is axes[0]:
            for (i, j), v in np.ndenumerate(a):
                ax.text(j, i, str(v), ha="center", va="center", fontsize=11)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    save(fig, "ml-08-autoregressive-vision", "generation-order.png")
    temp = np.array([.4, 1, 2])
    logits = np.array([2., 1., 0., -1.])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, t in zip(axes, temp):
        p = np.exp((logits-logits.max())/t); p /= p.sum()
        ax.bar(range(4), p, color=BLUE)
        ax.set(title=f"Temperature {t}", ylim=(0, 1), xlabel="Token", ylabel="Probability")
    save(fig, "ml-08-autoregressive-vision", "temperature.png")


def foundation():
    image = np.array([[1, .1, 0], [.1, 1, 0], [.5, .5, .2]])
    text = np.eye(3)
    image /= np.linalg.norm(image, axis=1, keepdims=True)
    scores = image @ text.T
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(scores, cmap="Blues", vmin=0, vmax=1)
    for (i, j), s in np.ndenumerate(scores):
        ax.text(j, i, f"{s:.2f}", ha="center", va="center")
    ax.set_xticks(range(3), ["Text A", "Text B", "Text C"])
    ax.set_yticks(range(3), ["Image A", "Image B", "Ambiguous image"])
    ax.set_title("Synthetic normalized embeddings: cosine is not confidence")
    fig.colorbar(im, ax=ax)
    save(fig, "ml-09-foundation-models", "similarity.png")
    fig, ax = plt.subplots(figsize=(9, 3))
    labels = ["Frozen features\n+ linear head", "LoRA/adapters\n+ head", "Full fine-tuning"]
    # Illustrative parameter counts: d=1024, K=10, r=8, 12 adapted matrices.
    values = [1024*10, 1024*10+12*8*(1024+1024), 100_000_000]
    ax.barh(labels, values, color=[BLUE, GREEN, ORANGE])
    ax.set_xscale("log")
    ax.set(xlabel="Trainable weights (illustrative architecture)", title="Adaptation budget is a design choice")
    save(fig, "ml-09-foundation-models", "adaptation-budget.png")


def practice():
    p = np.linspace(.05, .95, 10)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[0].plot(p, p, "o-", color=GREEN, label="Illustrative calibrated")
    axes[0].plot(p, .65*p, "o-", color=ORANGE, label="Illustrative overconfident")
    axes[0].set(xlabel="Confidence", ylabel="Observed accuracy", title="Calibration")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=.2)
    latency = np.array([2, 5, 9, 18, 35])
    quality = np.array([.80, .88, .85, .92, .90])
    axes[1].scatter(latency, quality, color=BLUE)
    axes[1].plot(latency[[0, 1, 3]], quality[[0, 1, 3]], color=GREEN, label="Pareto frontier")
    axes[1].set(xlabel="Latency (illustrative ms)", ylabel="Quality (illustrative)",
                title="A slower model can be dominated")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=.2)
    save(fig, "ml-10-evaluation-deployment", "calibration-pareto.png")
    fig, ax = plt.subplots(figsize=(8, 3))
    for row, name in enumerate(["Random frame split", "Split by video"]):
        colors = ([BLUE, ORANGE, BLUE, ORANGE, BLUE, ORANGE] if row == 0
                  else [BLUE, BLUE, BLUE, ORANGE, ORANGE, ORANGE])
        for i, c in enumerate(colors):
            ax.add_patch(plt.Rectangle((i*1.2, row), 1, .6, facecolor=c))
            ax.text(i*1.2+.5, row+.3, f"{'A' if i<3 else 'B'}{i%3+1}", ha="center", va="center", color="white")
        ax.text(-.2, row+.3, name, ha="right", va="center")
    ax.set(xlim=(-3.5, 7.3), ylim=(-.2, 1.9), title="Adjacent frames share information: blue=train, orange=test")
    ax.axis("off")
    save(fig, "ml-10-evaluation-deployment", "group-split.png")


if __name__ == "__main__":
    for fn in [cnn, transformer, representations, generative, diffusion, autoregression, foundation, practice]:
        fn()
