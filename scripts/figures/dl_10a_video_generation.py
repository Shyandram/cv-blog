"""Figures for DLCV 10A - Video Generation.

Run:  .venv/bin/python scripts/figures/dl_10a_video_generation.py
Writes PNGs to src/assets/deep-learning/dl-10a-video-generation/ only.
Data: skimage.data.camera and synthetic arrays only; nothing is downloaded and no network weights
are used. The consistency and Frechet-distance figures recompute the chapter's code examples with
the same seeds. Runs in about ten seconds on a CPU.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import linalg  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage import data, transform  # noqa: E402
from skimage.registration import phase_cross_correlation  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-10a-video-generation"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.facecolor": "white"})

BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"


def save(fig, name, quantize=True):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if quantize:
        im = Image.open(path).convert("RGB")
        im = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        im.save(path, optimize=True)
    w, h = Image.open(path).size
    assert w <= 1600, f"{name} is {w} px wide"
    assert path.stat().st_size < 1024 * 1024, f"{name} is larger than 1 MB"
    print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB, {w}x{h} px)")


def box(ax, x, y, w, h, text, fc, ec="#333333", fs=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)


def arrow(ax, x0, y0, x1, y1, color="#333333", style="-|>", lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw))


def stack(ax, x, y, w, h, n, dx, dy, fc, ec="#333333", highlight_first=None):
    """Draw n offset rectangles (a stack of frames); the first one drawn is the back of the stack."""
    for i in range(n):
        k = n - 1 - i
        color = highlight_first if (highlight_first and k == 0) else fc
        ax.add_patch(Rectangle((x + k * dx, y + k * dy), w, h, fc=color, ec=ec, lw=1.0))


# ------------------------------------------------------------------ Figure 1: video VAE compression
def video_vae():
    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    # Top row: pixels -> encoder -> latent -> patchify -> tokens
    stack(ax, 0.2, 3.0, 1.2, 1.2, 9, 0.09, 0.07, "#dbe9f6")
    ax.text(1.0, 2.85, "video: 17 frames\n256 x 256 x 3", ha="center", va="top", fontsize=10)
    arrow(ax, 2.25, 3.75, 2.85, 3.75)
    box(ax, 2.9, 3.2, 1.6, 1.1, "causal 3D\nVAE encoder\n(time 4x, space 8x)", "#fde2c8", fs=10)
    arrow(ax, 4.55, 3.75, 5.1, 3.75)
    stack(ax, 5.2, 3.35, 0.75, 0.75, 5, 0.14, 0.1, "#d5ecd4", highlight_first="#bfe0f5")
    ax.text(5.85, 2.85, "latent: 5 x 32 x 32\nx 16 channels", ha="center", va="top", fontsize=10)
    arrow(ax, 6.85, 3.75, 7.4, 3.75)
    box(ax, 7.45, 3.2, 1.25, 1.1, "patchify\n2 x 2", "#e7dcf2", fs=10)
    arrow(ax, 8.75, 3.75, 9.2, 3.75)
    for r in range(5):
        for q in range(5):
            ax.add_patch(Rectangle((9.25 + q * 0.2, 3.05 + r * 0.2), 0.18, 0.18, fc="#c9b3e0",
                                   ec="#555555", lw=0.6))
    ax.text(9.75, 2.85, "5 x 16 x 16\n= 1,280 tokens", ha="center", va="top", fontsize=10)
    ax.text(0.2, 5.45, "From pixels to transformer tokens (numbers for this example only)",
            fontsize=11, weight="bold")

    # Bottom row: which frames feed which latent frame (causal grouping)
    y0 = 0.55
    ax.text(0.2, 2.0, "Causal grouping in time: 17 frames (top) become 5 latent frames (bottom); "
            "latent frame k sees only frames up to its own group",
            fontsize=11, weight="bold")
    xs = np.linspace(0.6, 10.0, 17)
    for i, x in enumerate(xs):
        ax.add_patch(Rectangle((x - 0.2, y0 + 0.75), 0.4, 0.4, fc="#dbe9f6", ec="#333333", lw=0.8))
        ax.text(x, y0 + 0.95, str(i), ha="center", va="center", fontsize=9)
    groups = [[0]] + [list(range(1 + 4 * k, 5 + 4 * k)) for k in range(4)]
    colors = ["#bfe0f5", "#d5ecd4", "#d5ecd4", "#d5ecd4", "#d5ecd4"]
    for k, (g, col) in enumerate(zip(groups, colors)):
        cx = np.mean(xs[g])
        ax.add_patch(Rectangle((cx - 0.3, y0 - 0.45), 0.6, 0.45, fc=col, ec="#333333", lw=0.9))
        ax.text(cx, y0 - 0.22, f"z{k}", ha="center", va="center", fontsize=10)
        for i in g:
            arrow(ax, xs[i], y0 + 0.73, cx, y0 + 0.02, color=GRAY, lw=0.9)
    ax.text(10.35, y0 - 0.22, "latent\nframes", fontsize=10, va="center")
    ax.text(xs[0], y0 - 0.65, "first frame alone\n(so a still image is\na 1-frame video)", ha="center",
            va="top", fontsize=9, color="#1a5f8a")
    save(fig, "video-vae.png")


# ------------------------------------------------------------------ Figure 2: factorized vs full attention
def attention():
    fig = plt.figure(figsize=(10.6, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.62, 1.15], hspace=0.3, wspace=0.25)
    T, S = 4, 4  # 4 latent frames of 4 x 4 tokens
    q = (1, 1, 2)  # (frame, row, col) of the query token
    titles = ["Full spatiotemporal attention", "Spatial attention (per frame)",
              "Temporal attention (per position)"]
    rules = [lambda t, r, c: True,
             lambda t, r, c: t == q[0],
             lambda t, r, c: (r, c) == (q[1], q[2])]
    for k in range(3):
        ax = fig.add_subplot(gs[0, k])
        ax.set_xlim(-0.3, T * (S + 1) - 0.7)
        ax.set_ylim(-1.6, S + 0.2)
        ax.set_aspect("equal")
        ax.axis("off")
        n_keys = 0
        for t in range(T):
            for r in range(S):
                for c in range(S):
                    x, y = t * (S + 1) + c, S - 1 - r
                    if (t, r, c) == q:
                        fc = RED
                    elif rules[k](t, r, c):
                        fc = "#8fbce6"
                    else:
                        fc = "#f2f2f2"
                    n_keys += rules[k](t, r, c)
                    ax.add_patch(Rectangle((x, y), 0.92, 0.92, fc=fc, ec="#666666", lw=0.5))
            ax.text(t * (S + 1) + S / 2 - 0.05, -1.0, f"frame {t}", ha="center", fontsize=9)
        ax.set_title(f"{titles[k]}\n{n_keys} keys for the red query", fontsize=10)

    # Cost curve: query-key pairs per layer vs number of latent frames
    ax = fig.add_subplot(gs[1, :])
    s = 30 * 52  # tokens per latent frame for 480 x 832 video, 8x spatial VAE, 2 x 2 patches
    t = np.arange(1, 65)
    full = (t * s) ** 2
    fact = t * s**2 + s * t**2
    ax.semilogy(t, full, color=RED, lw=2, label="full: (T S)^2")
    ax.semilogy(t, fact, color=BLUE, lw=2, label="factorized: T S^2 + S T^2")
    ax.axvline(21, color=GRAY, ls="--", lw=1)
    ax.text(21.6, 3e6, "21 latent frames\n(81 frames, 4x time\ncompression)", fontsize=9, color="#333333")
    ax.set_xlabel("number of latent frames T  (S = 1,560 tokens per latent frame)")
    ax.set_ylabel("query-key pairs per layer")
    ax.set_title("Cost of one attention layer: full grows with the square of the total token count",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right")
    save(fig, "attention-patterns.png")


# ------------------------------------------------------------------ Figure 3: temporal-consistency toy
def consistency_videos():
    """Recompute the chapter's consistency example (same seed and order of random draws)."""
    rng = np.random.default_rng(0)
    T, v = 16, 2
    scene = transform.resize(data.camera() / 255.0, (128, 128), anti_aliasing=True)
    content = np.stack([np.roll(scene, v * t, axis=1) for t in range(T)])

    def detail():
        n = ndi.gaussian_filter(rng.standard_normal(scene.shape), 0.7)
        return 0.04 * n / n.std()

    d0 = detail()
    videos = {
        "independent noise": content + np.stack([detail() for _ in range(T)]),
        "shared noise": content + d0,
        "propagated noise": content + np.stack([np.roll(d0, v * t, axis=1) for t in range(T)]),
    }
    videos["frozen"] = np.repeat(videos["propagated noise"][:1], T, axis=0)
    return content, videos


def consistency_measure(vid):
    c = (slice(None), slice(8, -8), slice(8, -8))
    shifts = [phase_cross_correlation(a, b, upsample_factor=10)[0][1] for a, b in zip(vid[1:], vid[:-1])]
    flicker = np.abs(np.diff(vid, axis=0))[c].mean()
    warped = np.stack([np.roll(f, int(round(s)), axis=1) for f, s in zip(vid[:-1], shifts)])
    warp_err = np.abs(vid[1:] - warped)[c].mean()
    return flicker, warp_err, np.mean(np.abs(shifts))


def consistency():
    content, videos = consistency_videos()
    T = content.shape[0]
    row = 22  # a row through the smooth sky, where invented detail is easy to see
    cols = slice(20, 108)

    def profile(vid):
        # x-t slice after undoing the requested pan: correct motion gives vertical stripes
        comp = np.stack([np.roll(vid[t], -2 * t, axis=1) for t in range(T)])
        return np.repeat(comp[:, row, cols], 3, axis=0)

    fig = plt.figure(figsize=(10.6, 6.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.1], hspace=0.55)
    top = outer[0].subgridspec(1, 5, wspace=0.15)
    bottom = outer[1].subgridspec(1, 3, wspace=0.3)
    panels = [("coarse content\n(no detail)", content)] + list(videos.items())
    for k, (name, vid) in enumerate(panels):
        ax = fig.add_subplot(top[0, k])
        p = profile(vid)
        ax.imshow(p, cmap="gray", vmin=p.min(), vmax=p.max(), aspect="auto")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([0, p.shape[0] - 1])
        ax.set_yticklabels(["t=0", f"t={T - 1}"] if k == 0 else ["", ""])
    fig.text(0.5, 0.505, "Motion-compensated x-t slices (one image row followed over 16 frames, time runs down): "
             "consistent detail forms vertical lines", ha="center", fontsize=10)

    names = list(videos)
    vals = np.array([consistency_measure(videos[n]) for n in names])
    labels = ["frame difference", "warping error\n(own estimated motion)", "estimated motion\n(px per frame)"]
    colors = [ORANGE, RED, GREEN, GRAY]
    for j in range(3):
        ax = fig.add_subplot(bottom[0, j])
        bars = ax.bar(range(4), vals[:, j], color=colors)
        ax.set_xticks(range(4))
        ax.set_xticklabels([n.split()[0] for n in names], fontsize=9, rotation=20)
        ax.set_title(labels[j], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for b, val in zip(bars, vals[:, j]):
            ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.3f}" if j < 2 else f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, vals[:, j].max() * 1.25 + 1e-3)
    save(fig, "consistency-toy.png")


# ------------------------------------------------------------------ Figure 4: Frechet-distance toy
def fvd_toy():
    rng = np.random.default_rng(0)
    T, S = 16, 32
    yy, xx = np.mgrid[:S, :S]

    def real_videos(n):
        vids = np.empty((n, T, S, S))
        for i in range(n):
            p, vel, r = rng.uniform(10, 22, 2), rng.uniform(-0.7, 0.7, 2), rng.uniform(2.5, 4)
            for t in range(T):
                cy, cx = p + vel * t
                vids[i, t] = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2))
        return vids + 0.02 * rng.standard_normal(vids.shape)

    def frame_stats(f):
        gy, gx = np.gradient(f)
        return [f.mean(), f.std(), f.max(), np.abs(gx).mean() + np.abs(gy).mean()]

    def appearance_features(v):
        return np.mean([frame_stats(f) for f in v], axis=0)

    def motion_features(v):
        d = np.abs(np.diff(v, axis=0)).mean(axis=(1, 2))
        return np.concatenate([appearance_features(v), [d.mean(), d.std()]])

    def frechet(a, b):
        mu1, mu2 = a.mean(0), b.mean(0)
        s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
        covmean = linalg.sqrtm(s1 @ s2).real
        return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))

    real, other_real = real_videos(400), real_videos(400)
    blurred = ndi.gaussian_filter(other_real, sigma=(0, 0, 0.7, 0.7))
    shuffled = np.stack([v[rng.permutation(T)] for v in other_real])
    sets = [("second real set", other_real), ("blurred frames", blurred), ("shuffled frames", shuffled)]
    results = {}
    for name, feats in [("appearance features", appearance_features), ("+ motion features", motion_features)]:
        def f(vids, feats=feats):
            return np.stack([feats(v) for v in vids])
        ref = f(real)
        scale = ref.std(0)
        results[name] = [frechet(ref / scale, f(v) / scale) for _, v in sets]

    fig = plt.figure(figsize=(10.6, 4.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0], wspace=0.25, hspace=0.35)
    frames = [0, 3, 6, 9, 12, 15]
    rows = [("real clip", other_real[0]), ("blurred frames", blurred[0]), ("shuffled frames", shuffled[0])]
    for r, (name, vid) in enumerate(rows):
        ax = fig.add_subplot(gs[r, 0])
        strip = np.concatenate([np.pad(vid[t], 1, constant_values=1.0) for t in frames], axis=1)
        ax.imshow(strip, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
        if r == 0:
            ax.set_title("frames 0, 3, 6, 9, 12, 15 of one toy clip", fontsize=10)
    ax = fig.add_subplot(gs[:, 1])
    x = np.arange(3)
    for k, (name, vals) in enumerate(results.items()):
        bars = ax.bar(x + (k - 0.5) * 0.38, np.maximum(vals, 1e-3), 0.38, label=name,
                      color=[BLUE, PURPLE][k])
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, max(val, 1e-3) * 1.2, f"{val:.2f}", ha="center",
                    fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1e5)
    ax.set_xticks(x)
    ax.set_xticklabels([s for s, _ in sets])
    ax.set_ylabel("Frechet distance to the real set")
    ax.set_title("A time-blind feature cannot see shuffled frames", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fvd-toy.png")


# ------------------------------------------------------------------ Figure 5: family timeline
def timeline():
    # (name, first arXiv version as year + (month - 1) / 12, row, label offset in rows)
    rows = ["U-Net video diffusion\n(often inflated from\nan image model)",
            "Latent diffusion or flow\ntransformers (DiT-style)",
            "Token models\n(masked or autoregressive)",
            "Causal / streaming\nvideo diffusion",
            "Control, long video,\ntraining-free tricks"]
    items = [
        ("VDM", 2022.25, 0), ("Make-A-Video", 2022.67, 0), ("Imagen Video", 2022.75, 0),
        ("Video LDM", 2023.25, 0), ("AnimateDiff", 2023.5, 0), ("DynamiCrafter", 2023.75, 0),
        ("SVD", 2023.83, 0), ("Lumiere", 2024.0, 0),
        ("Latte", 2024.0, 1), ("CogVideoX", 2024.58, 1), ("Movie Gen", 2024.75, 1),
        ("Open-Sora Plan", 2024.83, 1), ("HunyuanVideo", 2024.92, 1), ("Open-Sora", 2024.92, 1),
        ("LTX-Video", 2024.92, 1), ("Wan", 2025.17, 1), ("Open-Sora 2.0", 2025.17, 1),
        ("Phenaki", 2022.75, 2), ("MAGVIT", 2022.92, 2), ("MAGVIT-v2", 2023.75, 2),
        ("VideoPoet", 2023.92, 2),
        ("Diffusion Forcing", 2024.5, 3), ("CausVid", 2024.92, 3), ("Self Forcing", 2025.42, 3),
        ("Tune-A-Video", 2022.92, 4), ("Text2Video-Zero", 2023.17, 4), ("NUWA-XL", 2023.17, 4),
        ("DragNUWA", 2023.58, 4), ("FreeNoise", 2023.75, 4), ("MotionCtrl", 2023.92, 4),
        ("StreamingT2V", 2024.17, 4), ("CameraCtrl", 2024.25, 4),
    ]
    colors = [BLUE, ORANGE, GREEN, RED, PURPLE]
    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    n_rows = len(rows)
    for r in range(n_rows):
        y = n_rows - 1 - r
        ax.axhspan(y - 0.45, y + 0.45, color=colors[r], alpha=0.07)
    # hand-placed labels for the crowded transformer row: name -> (label x, vertical offset)
    manual = {"Latte": (2024.0, 0.15), "CogVideoX": (2024.5, 0.15), "Open-Sora Plan": (2024.8, 0.33),
              "HunyuanVideo": (2024.97, 0.15), "Open-Sora 2.0": (2025.33, 0.33), "Movie Gen": (2024.6, -0.2),
              "LTX-Video": (2024.97, -0.2), "Wan": (2025.27, -0.2), "Open-Sora": (2024.82, -0.38)}
    placed = {}
    for name, x, r in sorted(items, key=lambda it: (it[2], it[1])):
        y = n_rows - 1 - r
        ax.plot(x, y, "o", color=colors[r], ms=6)
        if name in manual:
            lx, dy = manual[name]
            ax.plot([x, lx], [y, y + dy * 0.85], color=colors[r], lw=0.7, alpha=0.7)
            ax.text(lx, y + dy, name, fontsize=9, ha="center", va="bottom" if dy > 0 else "top")
            continue
        # stagger labels that are close in time within the same row
        k = placed.get(r, [])
        level = 0
        while any(abs(x - px) < 0.33 and pl == level for px, pl in k):
            level += 1
        k.append((x, level))
        placed[r] = k
        dy = [0.12, -0.2, 0.3, -0.36][level % 4]
        ax.text(x, y + dy, name, fontsize=9, ha="center", va="bottom" if dy > 0 else "top")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rows[::-1], fontsize=10)
    ax.set_xlim(2022.0, 2025.75)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xticks([2022, 2022.5, 2023, 2023.5, 2024, 2024.5, 2025, 2025.5])
    ax.set_xticklabels(["2022", "", "2023", "", "2024", "", "2025", ""])
    ax.set_xlabel("date of the first arXiv version")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Video generators discussed in this chapter, by family (a map, not a ranking)", fontsize=11)
    save(fig, "family-timeline.png")


if __name__ == "__main__":
    video_vae()
    attention()
    consistency()
    fvd_toy()
    timeline()
