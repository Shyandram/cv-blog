"""Original information-theory teaching figures: analytical and synthetic.

Run: MPLCONFIGDIR=/tmp/cv-blog-mpl .venv/bin/python scripts/figures/information_theory.py
No downloaded assets, trained models or benchmark claims.
"""
from pathlib import Path
import heapq
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2] / "src/assets/information-theory"
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN = "#2a6fdb", "#e4572e", "#17a398"


def h2(p):
    p = np.asarray(p, dtype=float)
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return -q * np.log2(q) - (1 - q) * np.log2(1 - q)


def save(fig, slug, name):
    folder = ROOT / slug
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def entropy():
    p = np.linspace(0, 1, 501)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(p, h2(p), color=BLUE)
    ax.set(xlabel="Probability of symbol 1", ylabel="Entropy (bits / symbol)",
           title="A fair binary source has the largest uncertainty", ylim=(0, 1.08))
    ax.grid(alpha=0.2)
    save(fig, "it-01-entropy-mutual-information", "binary-entropy.png")
    e = np.linspace(0, 0.5, 200)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(e, 1 - h2(e), label="I(X; Y): first noisy copy", color=BLUE)
    ax.plot(e, 1 - h2(e + 0.15 - 2 * e * 0.15),
            label="I(X; Z): another 15% flip", color=ORANGE)
    ax.set(xlabel="First-copy flip probability", ylabel="Mutual information (bits)",
           title="Independent extra corruption cannot restore information", ylim=(0, 1.05))
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    save(fig, "it-01-entropy-mutual-information", "data-processing.png")


def typicality():
    rng = np.random.default_rng(21)
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2), sharey=True)
    p = 0.2
    for ax, n in zip(axes, [20, 200, 2000]):
        counts = rng.binomial(n, p, 5000)
        surprise = (-counts * np.log2(p) - (n - counts) * np.log2(1 - p)) / n
        ax.hist(surprise, bins=25, density=True, color=BLUE, alpha=0.75)
        ax.axvline(h2(p), color=ORANGE)
        ax.set(title=f"Block length n = {n}", xlabel="Bits per symbol", xlim=(0.1, 1.35))
    axes[0].set_ylabel("Density")
    fig.suptitle("Typicality concerns sequence probabilities, not equal symbol counts")
    fig.tight_layout()
    save(fig, "it-02-typicality-entropy-rates", "aep-concentration.png")
    n = np.arange(1, 51)
    rate = h2(0.1)
    joint = 1 + (n - 1) * rate
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(n, joint / n, label="H(X₁…Xₙ) / n")
    ax.axhline(rate, color=ORANGE, linestyle="--", label="Entropy rate H₂(0.1)")
    ax.axhline(1, color="gray", linestyle=":", label="Marginal entropy 1 bit")
    ax.set(xlabel="Block length", ylabel="Bits per symbol",
           title="A persistent binary Markov source has a uniform marginal")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    save(fig, "it-02-typicality-entropy-rates", "markov-rate.png")


def coding():
    probabilities = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
    heap = [(p, i, [i]) for i, p in enumerate(probabilities)]
    heapq.heapify(heap); lengths = np.zeros(5, dtype=int); serial = 5
    while len(heap) > 1:
        a, _, ia = heapq.heappop(heap); b, _, ib = heapq.heappop(heap)
        lengths[ia + ib] += 1
        heapq.heappush(heap, (a + b, serial, ia + ib)); serial += 1
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(5)
    ax.bar(x - 0.18, -np.log2(probabilities), 0.36, label="Ideal −log₂ p")
    ax.bar(x + 0.18, lengths, 0.36, label="Huffman integer length", color=ORANGE)
    ax.set(xticks=x, xticklabels=["A", "B", "C", "D", "E"],
           xlabel="Source symbol", ylabel="Bits", title="Prefix codes allocate short words to frequent symbols")
    ax.legend(fontsize=9)
    save(fig, "it-03-lossless-universal-coding", "huffman-lengths.png")
    rng = np.random.default_rng(23); bits = rng.binomial(1, 0.2, 5000)
    ones, ideal, learned, curve = 0, 0.0, 0.0, []
    for t, bit in enumerate(bits):
        q = (ones + 0.5) / (t + 1)
        learned -= np.log2(q if bit else 1 - q)
        ideal -= np.log2(0.2 if bit else 0.8)
        ones += bit
        curve.append((learned - ideal) / (t + 1))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(np.arange(1, len(bits) + 1), curve, color=BLUE)
    ax.set(xscale="log", xlabel="Symbols already encoded", ylabel="Excess ideal bits / symbol",
           title="A shared adaptive probability model learns an unknown Bernoulli rate")
    ax.axhline(0, color="gray", linewidth=1); ax.grid(alpha=0.2)
    save(fig, "it-03-lossless-universal-coding", "universal-redundancy.png")


def channels():
    p = np.linspace(0, 0.5, 251)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(p, 1 - h2(p), label="Binary symmetric channel: 1 − H₂(p)")
    ax.plot(p, 1 - p, label="Binary erasure channel: 1 − p", color=ORANGE)
    ax.set(xlabel="Flip / erasure probability", ylabel="Capacity (bits / use)",
           title="Knowing which symbols were erased changes the channel")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    save(fig, "it-04-channel-capacity", "binary-capacity.png")
    q = np.linspace(0, 1, 501); a = 0.35
    information = h2(q * (1 - a)) - q * h2(a)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(q, information, color=BLUE)
    idx = information.argmax(); ax.plot(q[idx], information[idx], "o", color=ORANGE)
    ax.set(xlabel="Input probability P(X = 1)", ylabel="I(X; Y) (bits)",
           title="An asymmetric channel need not prefer a uniform input")
    ax.grid(alpha=0.2)
    save(fig, "it-04-channel-capacity", "asymmetric-input.png")


def gaussian():
    snr = np.logspace(-2, 2, 200)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.semilogx(snr, 0.5 * np.log2(1 + snr), color=BLUE)
    ax.set(xlabel="Signal-to-noise power ratio P / N", ylabel="Bits / real channel use",
           title="Power has diminishing returns for the real Gaussian channel")
    ax.grid(alpha=0.2)
    save(fig, "it-05-continuous-gaussian-channels", "gaussian-capacity.png")
    noise = np.array([0.2, 0.8, 1.6, 3.0]); total = 2.0
    low, high = noise.min(), noise.max() + total
    for _ in range(80):
        mid = (low + high) / 2
        if np.maximum(mid - noise, 0).sum() > total: high = mid
        else: low = mid
    power = np.maximum((low + high) / 2 - noise, 0)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(4)
    ax.bar(x, noise, label="Noise power", color="gray", alpha=0.5)
    ax.bar(x, power, bottom=noise, label="Allocated signal power", color=BLUE)
    ax.axhline((low + high) / 2, color=ORANGE, linestyle="--", label="Water level")
    ax.set(xticks=x, xticklabels=["1", "2", "3", "4"], xlabel="Parallel real subchannel",
           ylabel="Power", title="Water-filling can leave noisy subchannels unused")
    ax.legend(fontsize=9)
    save(fig, "it-05-continuous-gaussian-channels", "water-filling.png")


def distortion():
    d = np.linspace(0.001, 0.5, 300)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    axes[0].plot(d, 1 - h2(d))
    axes[0].set(xlabel="Hamming distortion", ylabel="Rate (bits / symbol)",
                title="Fair binary source")
    dg = np.linspace(0.02, 1.1, 300)
    axes[1].plot(dg, np.maximum(0.5 * np.log2(1 / dg), 0), color=ORANGE)
    axes[1].set(xlabel="Squared-error distortion", ylabel="Rate (bits / sample)",
                title="Unit-variance Gaussian source")
    for ax in axes: ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, "it-06-rate-distortion", "rate-distortion.png")
    rng = np.random.default_rng(26); x = rng.normal(size=20000)
    step = np.array([0.1, 0.2, 0.4, 0.8, 1.6])
    rate, error = [], []
    for width in step:
        index = np.rint(x / width).astype(int)
        _, count = np.unique(index, return_counts=True); p = count / len(x)
        rate.append(-(p * np.log2(p)).sum())
        error.append(np.mean((x - index * width)**2))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(error, rate, "o-", label="Synthetic scalar quantizer: marginal index entropy")
    grid = np.linspace(0.001, max(error) * 1.1, 300)
    ax.plot(grid, 0.5 * np.log2(1 / grid), "--", color=ORANGE, label="Gaussian R(D) limit")
    ax.set(xlabel="Measured squared error", ylabel="Bits / sample",
           title="A scalar design sits above the asymptotic rate–distortion limit")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    save(fig, "it-06-rate-distortion", "scalar-quantization.png")


def statistics():
    n = 30; k = np.arange(n + 2)
    false = binom.sf(k - 1, n, 0.3); detection = binom.sf(k - 1, n, 0.6)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(false, detection, "o-", color=BLUE, markersize=3)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="P₀(decide hypothesis 1)", ylabel="P₁(decide hypothesis 1)",
           title="Likelihood-ratio thresholds trade two kinds of error")
    ax.grid(alpha=0.2)
    save(fig, "it-07-statistics-decision-growth", "hypothesis-errors.png")
    p = 0.65; q = np.linspace(0.01, 0.99, 300)
    growth = p * np.log2(2 * q) + (1 - p) * np.log2(2 * (1 - q))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(q, growth, color=BLUE); ax.axvline(p, color=ORANGE, linestyle="--")
    ax.set(xlabel="Fraction allocated to outcome 1", ylabel="Expected log₂ growth / round",
           title="A mathematical two-outcome model: probability mismatch costs KL")
    ax.axhline(0, color="gray", linewidth=1); ax.grid(alpha=0.2)
    save(fig, "it-07-statistics-decision-growth", "growth-divergence.png")


def networks():
    c1, c2, total = 0.5*np.log2(3), 0.5*np.log2(2), 0.5*np.log2(4)
    vertices = np.array([[0,0], [c1,0], [c1,total-c1], [total-c2,c2], [0,c2]])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.fill(vertices[:,0], vertices[:,1], color=BLUE, alpha=0.2)
    ax.plot(np.r_[vertices[:,0],0], np.r_[vertices[:,1],0], color=BLUE)
    ax.set(xlabel="User 1 rate (bits / real use)", ylabel="User 2 rate (bits / real use)",
           title="Gaussian multiple access: individual and sum-rate constraints")
    ax.grid(alpha=0.2)
    save(fig, "it-08-networks-information-learning", "multiple-access.png")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar([0,1,2], [0,0,1], color=[BLUE,BLUE,ORANGE])
    ax.set(xticks=[0,1,2], xticklabels=["I(X₁; Y)", "I(X₂; Y)", "I(X₁,X₂; Y)"],
           ylabel="Bits", ylim=(0,1.15), title="XOR: individually uninformative inputs can matter jointly")
    save(fig, "it-08-networks-information-learning", "synergy-xor.png")


if __name__ == "__main__":
    for function in [entropy, typicality, coding, channels, gaussian, distortion, statistics, networks]:
        function()
        print(function.__name__, "figures written")
