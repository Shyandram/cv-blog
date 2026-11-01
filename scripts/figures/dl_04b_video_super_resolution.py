"""Figures for DLCV 04B · Video Super-Resolution (deep-learning/dl-04b-video-super-resolution).

Re-runnable: writes PNGs only to src/assets/deep-learning/dl-04b-video-super-resolution/.
All data are synthetic or come from skimage.data. The toy experiments (multi-frame
least-squares reconstruction, backward warping with Farneback flow, temporal flicker)
are the same as the tested snippets in the tutorial.

Run:  .venv/bin/python scripts/figures/dl_04b_video_super_resolution.py
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from scipy import ndimage as ndi
from skimage import color, data, transform
from skimage.registration import phase_cross_correlation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src/assets/deep-learning/dl-04b-video-super-resolution"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})
BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#2f6db5", "#e07b24", "#3a9a4a", "#c8382f", "#777777", "#7b4fa8"


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if path.stat().st_size > 900 * 1024:                  # keep every PNG well under 1 MB
        from PIL import Image
        Image.open(path).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Toy 1: multi-frame reconstruction (same as the tutorial snippet)
# ---------------------------------------------------------------------------
S, BLUR, NOISE = 4, 1.2, 0.01
HR = color.rgb2gray(data.astronaut())[64:320, 128:384]


def degrade(x, d):
    return ndi.gaussian_filter(ndi.shift(x, d, order=3, mode="reflect"), BLUR)[::S, ::S]


def degrade_T(r, d):
    z = np.zeros((r.shape[0] * S, r.shape[1] * S)); z[::S, ::S] = r
    return ndi.shift(ndi.gaussian_filter(z, BLUR), -np.asarray(d), order=3, mode="reflect")


def upsample(y):
    u, v = np.mgrid[0:y.shape[0] * S, 0:y.shape[1] * S] / S
    return ndi.map_coordinates(y, [u, v], order=3, mode="nearest")


def psnr(x, ref=HR):
    c = (slice(8, -8), slice(8, -8))
    return 10 * np.log10(1 / np.mean((np.clip(x, 0, 1)[c] - ref[c]) ** 2))


def reconstruct(frames, ds, iters=20):
    x = np.mean([ndi.shift(upsample(f), -np.asarray(d), order=3, mode="reflect") for f, d in zip(frames, ds)], 0)
    for _ in range(iters):
        grad = np.mean([degrade_T(degrade(x, d) - f, d) for f, d in zip(frames, ds)], 0)
        x -= S * S * grad
    return x


def make_video(T, seed=0):
    rng = np.random.default_rng(seed)
    shifts = rng.uniform(0, S, size=(T, 2)); shifts[0] = 0
    lr = np.stack([degrade(HR, d) + NOISE * rng.standard_normal((64, 64)) for d in shifts])
    return lr, shifts


LR, SHIFTS = make_video(16)
EST = np.array([-S * phase_cross_correlation(LR[0], f, upsample_factor=100)[0] for f in LR])
BIC = upsample(LR[0])
NAIVE = np.mean([upsample(f) for f in LR], 0)
MULTI_TRUE = reconstruct(LR, SHIFTS)
MULTI_EST = reconstruct(LR, EST)
print(f"toy PSNR: cubic {psnr(BIC):.2f}  naive {psnr(NAIVE):.2f}  16 true {psnr(MULTI_TRUE):.2f}  16 est {psnr(MULTI_EST):.2f}")


# ---------------------------------------------------------------------------
# Figure 1: why sub-pixel motion adds information (1-D)
# ---------------------------------------------------------------------------
def fig_subpixel():
    xs = np.linspace(0, 32, 2000)
    sig = lambda t: 0.55 * np.sin(2 * np.pi * t / 9.0) + 0.35 * np.sin(2 * np.pi * t / 3.1 + 0.6)
    step = 4.0
    pos_a = np.arange(0, 32.01, step)
    pos_b = np.arange(1.0, 32.01, step)
    pos_c = np.arange(2.0, 32.01, step)
    pos_d = np.arange(3.0, 32.01, step)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
    ax = axes[0]
    ax.plot(xs, sig(xs), color=GRAY, lw=1.4, label="scene (fine detail)")
    ax.plot(pos_a, sig(pos_a), "o", color=BLUE, ms=6, label="frame 1 samples")
    ax.plot(pos_b, sig(pos_b), "s", color=ORANGE, ms=6, label="frame 2 (moved 1/4 pixel)")
    ax.plot(pos_c, sig(pos_c), "^", color=GREEN, ms=6, label="frame 3 (moved 2/4 pixel)")
    ax.plot(pos_d, sig(pos_d), "D", color=PURPLE, ms=5, label="frame 4 (moved 3/4 pixel)")
    ax.set_title("(a) Each frame samples the scene at different places")
    ax.set_xlabel("position (fine grid)"); ax.set_yticks([])
    ax.legend(fontsize=9, loc="lower center", ncol=2, framealpha=0.9)
    ax.set_ylim(-2.5, 1.1)
    ax = axes[1]
    from scipy.interpolate import CubicSpline
    one = CubicSpline(pos_a, sig(pos_a))(xs)
    allp = np.sort(np.r_[pos_a, pos_b, pos_c, pos_d])
    four = CubicSpline(allp, sig(allp))(xs)
    ax.plot(xs, sig(xs), color=GRAY, lw=2.2, label="scene")
    ax.plot(xs, one, color=BLUE, lw=1.4, ls="--", label="interpolate frame 1 only")
    ax.plot(xs, four, color=RED, lw=1.4, label="merge 4 aligned frames")
    ax.set_title("(b) Merged samples recover the fine detail")
    ax.set_xlabel("position (fine grid)"); ax.set_yticks([])
    ax.legend(fontsize=9, loc="lower center", ncol=2, framealpha=0.9)
    ax.set_ylim(-2.5, 1.1)
    fig.tight_layout()
    save(fig, "subpixel-sampling.png")


# ---------------------------------------------------------------------------
# Figure 2: the VSR pipeline
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc, fs=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=fc, ec="black", lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, color="black", lw=1.4, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color, lw=lw, linestyle=ls))


def fig_pipeline():
    fig = plt.figure(figsize=(10.2, 3.5))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10.6); ax.set_ylim(0, 3.6); ax.axis("off")
    # LR frames as thumbnails
    for k, t in enumerate([0, 5, 9]):
        ia = fig.add_axes([0.02 + 0.012 * k, 0.52 - 0.14 * k, 0.12, 0.33])
        ia.imshow(LR[t], cmap="gray", vmin=0, vmax=1, interpolation="nearest"); ia.set_xticks([]); ia.set_yticks([])
        for sp in ia.spines.values(): sp.set_color(GRAY)
    ax.text(0.95, 0.12, "LR frames\n$y_{t-k}, \\dots, y_{t+k}$", ha="center", fontsize=10)
    box(ax, 2.3, 1.35, 1.75, 0.95, "1. Alignment\n(flow warp, DCN,\nattention)", "#dce8f7", 10)
    box(ax, 4.6, 1.35, 1.75, 0.95, "2. Fusion /\npropagation", "#fbe3cf", 10)
    box(ax, 6.9, 1.35, 1.75, 0.95, "3. Reconstruction\n+ upsampling\n(pixel shuffle)", "#dff1df", 10)
    arrow(ax, (1.75, 1.82), (2.3, 1.82)); arrow(ax, (4.05, 1.82), (4.6, 1.82)); arrow(ax, (6.35, 1.82), (6.9, 1.82))
    arrow(ax, (8.65, 1.82), (9.0, 1.82))
    ia = fig.add_axes([0.85, 0.27, 0.145, 0.46])
    ia.imshow(np.clip(MULTI_EST, 0, 1), cmap="gray", vmin=0, vmax=1); ia.set_xticks([]); ia.set_yticks([])
    ax.text(9.78, 0.55, "HR frame $\\hat x_t$", ha="center", fontsize=10)
    ax.text(3.18, 2.55, "motion estimation\n+ compensation", ha="center", fontsize=9, color=BLUE)
    ax.text(5.48, 2.55, "combine what each\nframe adds", ha="center", fontsize=9, color=ORANGE)
    ax.text(7.78, 2.55, "add residual detail\nto an upsampled base", ha="center", fontsize=9, color=GREEN)
    save(fig, "vsr-pipeline.png")


# ---------------------------------------------------------------------------
# Figure 3: temporal designs
# ---------------------------------------------------------------------------
def fig_designs():
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0))
    axes = axes.ravel()
    n = 5
    xs = np.arange(n) * 1.0
    def frames(ax, y=0.0, label=True):
        for i, x in enumerate(xs):
            ax.add_patch(Rectangle((x - 0.3, y - 0.25), 0.6, 0.5, fc="#eeeeee", ec="black"))
            if label:
                ax.text(x, y, f"$y_{{{i + 1}}}$", ha="center", va="center", fontsize=10)
    def outs(ax, y=3.0):
        for i, x in enumerate(xs):
            ax.add_patch(Rectangle((x - 0.3, y - 0.25), 0.6, 0.5, fc="#dff1df", ec="black"))
            ax.text(x, y, f"$\\hat x_{{{i + 1}}}$", ha="center", va="center", fontsize=10)
    for ax in axes:
        ax.set_xlim(-0.7, n - 0.3); ax.set_ylim(-0.8, 3.6); ax.axis("off")

    ax = axes[0]; frames(ax); outs(ax)
    ax.add_patch(FancyBboxPatch((0.7, -0.4), 2.6, 0.8, boxstyle="round,pad=0.02", fc="none", ec=BLUE, lw=2, ls="--"))
    ax.add_patch(FancyBboxPatch((1.5, 1.2), 1.0, 0.6, boxstyle="round,pad=0.02", fc="#dce8f7", ec="black"))
    ax.text(2.0, 1.5, "net", ha="center", va="center", fontsize=10)
    for x in (1, 2, 3):
        arrow(ax, (x, 0.28), (2.0, 1.18), color=BLUE, lw=1.1)
    arrow(ax, (2.0, 1.82), (2.0, 2.72), color=BLUE)
    ax.set_title("(a) Sliding window\n(one output from 2k+1 inputs)", fontsize=11)

    ax = axes[1]; frames(ax); outs(ax)
    for x in xs:
        ax.add_patch(Rectangle((x - 0.3, 1.25), 0.6, 0.5, fc="#fbe3cf", ec="black"))
        ax.text(x, 1.5, "$h$", ha="center", va="center", fontsize=10)
        arrow(ax, (x, 0.28), (x, 1.22), lw=1.0); arrow(ax, (x, 1.78), (x, 2.72), lw=1.0)
    for x in xs[:-1]:
        arrow(ax, (x + 0.32, 1.5), (x + 0.68, 1.5), color=ORANGE, lw=1.8)
    ax.set_title("(b) Unidirectional recurrent\n(hidden state flows forward)", fontsize=11)

    ax = axes[2]; frames(ax); outs(ax)
    for x in xs:
        ax.add_patch(Rectangle((x - 0.3, 0.75), 0.6, 0.4, fc="#f3d6f0", ec="black"))
        ax.add_patch(Rectangle((x - 0.3, 1.65), 0.6, 0.4, fc="#fbe3cf", ec="black"))
        ax.text(x, 0.95, "$h^b$", ha="center", va="center", fontsize=9)
        ax.text(x, 1.85, "$h^f$", ha="center", va="center", fontsize=9)
        arrow(ax, (x, 0.28), (x, 0.73), lw=1.0); arrow(ax, (x, 1.17), (x, 1.63), lw=1.0, color=GRAY)
        arrow(ax, (x, 2.07), (x, 2.72), lw=1.0)
    for x in xs[:-1]:
        arrow(ax, (x + 0.32, 1.85), (x + 0.68, 1.85), color=ORANGE, lw=1.8)
        arrow(ax, (x + 0.68, 0.95), (x + 0.32, 0.95), color=PURPLE, lw=1.8)
    ax.set_title("(c) Bidirectional recurrent\n(forward and backward states)", fontsize=11)

    ax = axes[3]; frames(ax); outs(ax)
    rows = [0.75, 1.35, 1.95]
    cols = ["#f3d6f0", "#fbe3cf", "#f3d6f0"]
    for r, cfc in zip(rows, cols):
        for x in xs:
            ax.add_patch(Rectangle((x - 0.25, r - 0.17), 0.5, 0.34, fc=cfc, ec="black"))
    for k, r in enumerate(rows):
        for x in xs[:-1]:
            if k % 2 == 0:
                arrow(ax, (x + 0.72, r), (x + 0.28, r), color=PURPLE, lw=1.4)
            else:
                arrow(ax, (x + 0.28, r), (x + 0.72, r), color=ORANGE, lw=1.4)
    # second-order links (skip one frame) on the middle row
    for x in xs[:-2]:
        ax.add_patch(FancyArrowPatch((x + 0.1, rows[1] + 0.17), (x + 1.9, rows[1] + 0.17),
                                     connectionstyle="arc3,rad=-0.45", arrowstyle="-|>", mutation_scale=10,
                                     color=RED, lw=1.0, linestyle="--"))
    for x in xs:
        arrow(ax, (x, 0.28), (x, 0.56), lw=1.0)
        arrow(ax, (x, 0.94), (x, 1.16), lw=0.9, color=GRAY); arrow(ax, (x, 1.54), (x, 1.76), lw=0.9, color=GRAY)
        arrow(ax, (x, 2.14), (x, 2.72), lw=1.0)
    ax.set_title("(d) Grid, second-order propagation\n(repeated passes; skip links in red)", fontsize=11)
    fig.tight_layout()
    save(fig, "temporal-designs.png")


# ---------------------------------------------------------------------------
# Figure 4: multi-frame toy result + PSNR vs number of frames; ImageCompare pair
# ---------------------------------------------------------------------------
def fig_toy():
    Ts = [1, 2, 4, 8, 16, 32]
    vals = []
    for T in Ts:
        lr, sh = make_video(T)
        vals.append(psnr(reconstruct(lr, sh)))
    print("PSNR vs frames:", dict(zip(Ts, np.round(vals, 2))))
    c = (slice(40, 168), slice(64, 192))
    fig = plt.figure(figsize=(10.4, 6.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.35])
    items = [("Ground truth", HR), (f"1 frame, cubic\n{psnr(BIC):.2f} dB", BIC),
             (f"16 frames averaged,\nno alignment: {psnr(NAIVE):.2f} dB", NAIVE),
             (f"16 frames, estimated\nshifts: {psnr(MULTI_EST):.2f} dB", MULTI_EST)]
    for k, (title, im) in enumerate(items):
        a = fig.add_subplot(gs[k // 2, k % 2])
        a.imshow(np.clip(im[c], 0, 1), cmap="gray", vmin=0, vmax=1)
        a.set_title(title, fontsize=10.5); a.set_xticks([]); a.set_yticks([])
    a = fig.add_subplot(gs[:, 2])
    a.plot(Ts, vals, "o-", color=RED, label="multi-frame, true shifts")
    a.axhline(psnr(BIC), color=BLUE, ls="--", label="1 frame, cubic")
    a.set_xscale("log", base=2); a.set_xticks(Ts); a.set_xticklabels([str(t) for t in Ts])
    a.set_xlabel("number of LR frames"); a.set_ylabel("PSNR (dB)")
    a.set_title("Toy ×4 reconstruction", fontsize=10.5); a.grid(alpha=0.3); a.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save(fig, "multiframe-toy.png")
    # ImageCompare pair (same size): x3 nearest-neighbor enlargement of a crop for visibility
    cc = (slice(56, 152), slice(96, 192))
    for name, im in [("compare-single.png", BIC), ("compare-multi.png", MULTI_EST)]:
        crop = (np.clip(im[cc], 0, 1) * 255).round().astype(np.uint8)
        big = np.kron(crop, np.ones((4, 4), np.uint8))
        plt.imsave(OUT / name, big, cmap="gray", vmin=0, vmax=255)
        print(f"wrote {name} ({(OUT / name).stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Figure 5: backward warping and occlusion
# ---------------------------------------------------------------------------
def bilinear_warp(img, flow):
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    x = np.clip(xx + flow[..., 0], 0, W - 1); y = np.clip(yy + flow[..., 1], 0, H - 1)
    x0 = np.minimum(np.floor(x).astype(int), W - 2); y0 = np.minimum(np.floor(y).astype(int), H - 2)
    ax, ay = x - x0, y - y0
    return ((1 - ay) * ((1 - ax) * img[y0, x0] + ax * img[y0, x0 + 1])
            + ay * ((1 - ax) * img[y0 + 1, x0] + ax * img[y0 + 1, x0 + 1]))


def fig_warp():
    bg = transform.resize(data.camera() / 255.0, (256, 256), anti_aliasing=True)
    obj = transform.resize(data.coins() / 255.0, (60, 60), anti_aliasing=True)
    def frame(t):
        f = ndi.shift(bg, (-2.0 * t, 1.5 * t), order=3, mode="reflect")
        r, c = 90 + 4 * t, 100 + 6 * t
        f[r:r + 60, c:c + 60] = obj
        return f
    I0, I1 = frame(0).astype(np.float32), frame(1).astype(np.float32)
    u8 = lambda a: (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
    est = cv2.calcOpticalFlowFarneback(u8(I0), u8(I1), None, pyr_scale=0.5, levels=4, winsize=21,
                                       iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
    flow = np.zeros((256, 256, 2), np.float32); flow[..., 0], flow[..., 1] = 1.5, -2.0
    flow[90:150, 100:160] = (6.0, 4.0)
    W1 = bilinear_warp(I1, flow)
    m0 = np.zeros((256, 256), np.float32); m0[90:150, 100:160] = 1
    m1 = np.zeros((256, 256), np.float32); m1[94:154, 106:166] = 1
    occl = (bilinear_warp(m1, flow) > 0) & (m0 == 0)
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 7.4))
    axes = axes.ravel()
    axes[0].imshow(I0, cmap="gray", vmin=0, vmax=1); axes[0].set_title("frame $t$ (reference)")
    axes[1].imshow(I1, cmap="gray", vmin=0, vmax=1); axes[1].set_title("frame $t+1$")
    axes[2].imshow(I0, cmap="gray", vmin=0, vmax=1, alpha=0.35)
    st = 16
    yy, xx = np.mgrid[st // 2:256:st, st // 2:256:st]
    axes[2].quiver(xx, yy, est[yy, xx, 0], est[yy, xx, 1], color=RED, angles="xy", scale_units="xy", scale=0.25, width=0.006)
    axes[2].set_title("Farneback flow\n(arrows drawn 4× longer)")
    def err(a, img, title):
        e = np.abs(img - I0)
        a.imshow(e, cmap="magma", vmin=0, vmax=0.5)
        a.set_title(f"{title}\nmean {e[8:-8, 8:-8].mean():.3f}")
    err(axes[3], I1, "|frame $t+1$ − frame $t$|")
    err(axes[4], W1, "after warping, true flow")
    err(axes[5], bilinear_warp(I1, est), "after warping, Farneback")
    ys, xs_ = np.nonzero(occl)
    for a in axes[4:]:
        a.add_patch(Rectangle((xs_.min() - 1, ys.min() - 1), xs_.max() - xs_.min() + 2, ys.max() - ys.min() + 2,
                              fill=False, ec="cyan", lw=1.2, ls="--"))
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    save(fig, "warping-occlusion.png")


# ---------------------------------------------------------------------------
# Figure 6: temporal profiles and flicker
# ---------------------------------------------------------------------------
def fig_flicker():
    rng = np.random.default_rng(0)
    T, v = 12, (1, 2)
    base = transform.resize(data.camera() / 255.0, (256, 256), anti_aliasing=True)
    gt = np.stack([np.roll(base, (v[0] * t, v[1] * t), axis=(0, 1)) for t in range(T)])
    warp_prev = lambda f: np.roll(f, v, axis=(0, 1))
    def fake_detail():
        n = ndi.gaussian_filter(rng.standard_normal(base.shape), 0.8)
        return 0.03 * n / n.std()
    per = gt + np.stack([fake_detail() for _ in range(T)])
    sm, un = per.copy(), per.copy()
    for t in range(1, T):
        sm[t] = 0.3 * per[t] + 0.7 * warp_prev(sm[t - 1])
        un[t] = 0.3 * per[t] + 0.7 * un[t - 1]
    # motion-compensated temporal profile: the same scene row (row 120 of frame 0) followed over time
    row, cols = 120, slice(60, 200)
    def profile(seq):
        return np.stack([np.roll(f, (-v[0] * t, -v[1] * t), axis=(0, 1))[row, cols] for t, f in enumerate(seq)])
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.7))
    for a, (name, seq) in zip(axes, [("ground truth", gt), ("per-frame output", per),
                                     ("smoothing along the motion", sm), ("smoothing, motion ignored", un)]):
        a.imshow(np.repeat(profile(seq), 4, axis=0), cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        a.set_title(name, fontsize=10); a.set_xticks([]); a.set_yticks([])
        a.set_ylabel("time →") if a is axes[0] else None
    fig.tight_layout()
    save(fig, "temporal-profiles.png")


if __name__ == "__main__":
    fig_subpixel()
    fig_pipeline()
    fig_designs()
    fig_toy()
    fig_warp()
    fig_flicker()
