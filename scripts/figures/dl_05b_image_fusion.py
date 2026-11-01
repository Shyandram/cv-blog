"""Figures for DLCV 05B - Image Fusion: Infrared-Visible, Multi-Focus and Multi-Exposure.

Run:  .venv/bin/python scripts/figures/dl_05b_image_fusion.py
Writes PNGs to src/assets/deep-learning/dl-05b-image-fusion/ only.
All inputs come from skimage.data or synthetic arrays; nothing is downloaded.
Everything is NumPy/SciPy/OpenCV arithmetic and runs in a few seconds on a CPU.
The toy data and fusion rules are the same as in the tutorial's code snippets.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.sparse.linalg import cg  # noqa: E402
from skimage import data, img_as_float  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-05b-image-fusion"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white",
                     "axes.titlesize": 10})


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(path).convert("RGB")
    if im.size[0] > 1600:
        im = im.resize((1600, round(im.size[1] * 1600 / im.size[0])), Image.LANCZOS)
    im = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT)   # keeps each PNG well under 1 MB
    im.save(path, optimize=True)
    w, h = im.size
    print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB, {w}x{h})")


def show(ax, img, title=None, cmap="gray", vmin=0, vmax=1):
    if img.ndim == 3:
        ax.imshow(np.clip(img, 0, 1))
    else:
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


# ---------------------------------------------------------------- shared toy data (same as the snippets)
def multifocus_toy():
    rng = np.random.default_rng(0)
    x = img_as_float(data.camera())
    yy, xx = np.mgrid[:512, :512]
    near = (yy - 200) ** 2 + (xx - 250) ** 2 < 150 ** 2
    blurred = ndi.gaussian_filter(x, 3.0)
    A = np.where(near, x, blurred) + 0.01 * rng.standard_normal(x.shape)
    B = np.where(near, blurred, x) + 0.01 * rng.standard_normal(x.shape)
    return x, near, A, B


def focus_measure(img, s=2.0):
    return ndi.gaussian_filter(ndi.laplace(img) ** 2, s)


def laplacian_pyramid(img, levels=5):
    G = [img.astype(np.float32)]
    for _ in range(levels):
        G.append(cv2.pyrDown(G[-1]))
    L = [G[i] - cv2.pyrUp(G[i + 1], dstsize=G[i].shape[1::-1]) for i in range(levels)]
    return L + [G[-1]]


def collapse(L):
    img = L[-1]
    for d in reversed(L[:-1]):
        img = cv2.pyrUp(img, dstsize=d.shape[1::-1]) + d
    return img


def pick_stronger(p, q):
    return np.where(np.abs(p) >= np.abs(q), p, q)


def pyramid_fuse(a, b, levels=5, base="mean"):
    La, Lb = laplacian_pyramid(a, levels), laplacian_pyramid(b, levels)
    fused = [pick_stronger(p, q) for p, q in zip(La[:-1], Lb[:-1])]
    fused.append(np.maximum(La[-1], Lb[-1]) if base == "max" else (La[-1] + Lb[-1]) / 2)
    return collapse(fused).astype(float)


def exposure_stack():
    rgb = img_as_float(data.astronaut()) ** 2.2
    light = 2.0 ** np.linspace(-6, 2, rgb.shape[1])[None, :, None]
    radiance = rgb * light

    def camera(E, t):
        return np.round(np.clip((t * E) ** (1 / 2.2), 0, 1) * 255) / 255

    return [camera(radiance, t) for t in (1 / 8, 1.0, 8.0)]


def mertens_weights(imgs, sigma=0.2, eps=1e-12):
    W = []
    for I in imgs:
        gray = I.mean(axis=2).astype(np.float32)
        C = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        S = I.std(axis=2)
        E = np.prod(np.exp(-(I - 0.5) ** 2 / (2 * sigma ** 2)), axis=2)
        W.append(C * S * E + eps)
    W = np.array(W)
    return W / W.sum(axis=0)


def gauss_pyr(img, n):
    G = [img.astype(np.float32)]
    for _ in range(n):
        G.append(cv2.pyrDown(G[-1]))
    return G


def exposure_fusion(imgs, weights=None):
    n = int(np.log2(min(imgs[0].shape[:2])))
    weights = mertens_weights(imgs) if weights is None else weights
    fused = None
    for I, Wk in zip(imgs, weights):
        level = [g[..., None] * l for g, l in zip(gauss_pyr(Wk, n), laplacian_pyramid(I, n))]
        fused = level if fused is None else [f + l for f, l in zip(fused, level)]
    return np.clip(collapse(fused), 0, 1)


def irvis_toy():
    rng = np.random.default_rng(1)
    scene = img_as_float(data.camera())
    yy, xx = np.mgrid[:512, :512]
    person = ((yy - 330) / 60) ** 2 + ((xx - 420) / 22) ** 2 < 1
    shadow = ndi.gaussian_filter((xx > 340).astype(float), 8)
    vis = scene * (1 - 0.85 * shadow)
    ir = 0.25 + 0.3 * ndi.gaussian_filter(scene, 6)
    ir = np.clip(ir + 0.6 * ndi.gaussian_filter(person.astype(float), 2)
                 + 0.01 * rng.standard_normal(ir.shape), 0, 1)
    return vis, ir, person, rng


def entropy(F, bins=256):
    p = np.histogram(F, bins, (0, 1))[0] / F.size
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def spatial_frequency(F):
    return np.sqrt(np.mean(np.diff(F, axis=1) ** 2) + np.mean(np.diff(F, axis=0) ** 2))


def mutual_info(X, F, bins=64):
    pxy = np.histogram2d(X.ravel(), F.ravel(), bins, [[0, 1], [0, 1]])[0] / X.size
    px, pf = pxy.sum(1, keepdims=True), pxy.sum(0, keepdims=True)
    nz = pxy > 0
    return np.sum(pxy[nz] * np.log2(pxy[nz] / (px @ pf)[nz]))


def q_gradient(A, B, F):
    def grad(I):
        gx, gy = ndi.sobel(I, 1), ndi.sobel(I, 0)
        return np.hypot(gx, gy), np.arctan2(gy, gx)
    gF, aF = grad(F)
    num = den = 0.0
    for X in (A, B):
        gX, aX = grad(X)
        strength = np.minimum(gX, gF) / (np.maximum(gX, gF) + 1e-12)
        d = np.abs(aX - aF) % np.pi
        direction = 1 - np.minimum(d, np.pi - d) / (np.pi / 2)
        num += np.sum(strength * direction * gX)
        den += np.sum(gX)
    return num / den


def fuse_by_loss(a, b, lam=10.0):
    H, W = a.shape
    d = lambda n: sp.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n))  # noqa: E731
    Dx, Dy = sp.kron(sp.eye(H), d(W)).tocsr(), sp.kron(d(H), sp.eye(W)).tocsr()
    gx = [Dx @ s.ravel() for s in (a, b)]
    gy = [Dy @ s.ravel() for s in (a, b)]
    tx = np.where(np.abs(gx[0]) >= np.abs(gx[1]), gx[0], gx[1])
    ty = np.where(np.abs(gy[0]) >= np.abs(gy[1]), gy[0], gy[1])
    target = np.maximum(a, b).ravel()
    M = sp.eye(H * W) + lam * (Dx.T @ Dx + Dy.T @ Dy)
    f, _ = cg(M, target + lam * (Dx.T @ tx + Dy.T @ ty), x0=target, rtol=1e-6, maxiter=2000)
    return np.clip(f.reshape(H, W), 0, 1)


# ---------------------------------------------------------------- Figure 1: the three tasks
def fig_tasks():
    vis, ir, person, _ = irvis_toy()
    x, near, A, B = multifocus_toy()
    w = ndi.gaussian_filter((ndi.median_filter((focus_measure(A) > focus_measure(B)).astype(float), 31) > 0.5)
                            .astype(float), 2.0)
    mf = np.clip(w * A + (1 - w) * B, 0, 1)
    stack = exposure_stack()
    two = [stack[0], stack[2]]
    ef = exposure_fusion(two)
    rows = [("Infrared-visible", vis, "visible", ir, "infrared (thermal)", pyramid_fuse(vis, ir, base="max"),
             "fused"),
            ("Multi-focus", np.clip(A, 0, 1), "focused near", np.clip(B, 0, 1), "focused far", mf,
             "fused (all in focus)"),
            ("Multi-exposure", two[0], "short exposure", two[1], "long exposure", ef, "fused (exposure fusion)")]
    fig, axes = plt.subplots(3, 3, figsize=(9.2, 9.4))
    for r, (task, s1, t1, s2, t2, f, tf) in enumerate(rows):
        show(axes[r, 0], s1, t1)
        show(axes[r, 1], s2, t2)
        show(axes[r, 2], np.clip(f, 0, 1), tf)
        axes[r, 0].set_ylabel(task, fontsize=11)
    fig.suptitle("Three fusion tasks on synthetic data: two sources in, one image out", y=0.995)
    fig.tight_layout()
    save(fig, "fusion-tasks.png")


# ---------------------------------------------------------------- Figure 2: Laplacian-pyramid fusion
def fig_pyramid():
    x, near, A, B = multifocus_toy()
    La, Lb = laplacian_pyramid(A, 5), laplacian_pyramid(B, 5)
    cols = [0, 1, 2, 3, 5]
    names = ["level 0 (finest)", "level 1", "level 2", "level 3", "base (level 5)"]
    fig, axes = plt.subplots(4, 5, figsize=(10.4, 8.6))
    for c, (k, nm) in enumerate(zip(cols, names)):
        a, b = La[k], Lb[k]
        if k < 5:
            s = np.percentile(np.abs(np.concatenate([a.ravel(), b.ravel()])), 99.5)
            show(axes[0, c], a, nm, cmap="RdBu_r", vmin=-s, vmax=s)
            show(axes[1, c], b, None, cmap="RdBu_r", vmin=-s, vmax=s)
            m = (np.abs(a) >= np.abs(b)).astype(float)
            show(axes[2, c], m, None)
            f = pick_stronger(a, b)
            show(axes[3, c], f, None, cmap="RdBu_r", vmin=-s, vmax=s)
        else:
            show(axes[0, c], a, nm)
            show(axes[1, c], b, None)
            axes[2, c].text(0.5, 0.5, "no choice:\naverage\nthe two bases", ha="center", va="center",
                            transform=axes[2, c].transAxes)
            axes[2, c].set_xticks([])
            axes[2, c].set_yticks([])
            show(axes[3, c], (a + b) / 2, None)
    for r, lab in enumerate(["source A\n(sharp center)", "source B\n(sharp border)", "chosen source\n(white = A)",
                             "fused level"]):
        axes[r, 0].set_ylabel(lab, fontsize=10)
    fig.suptitle("Laplacian-pyramid fusion: decompose both sources, keep the stronger detail per coefficient, "
                 "average the base, collapse", y=0.995, fontsize=10)
    fig.tight_layout()
    save(fig, "pyramid-fusion.png")


# ---------------------------------------------------------------- Figure 3: multi-focus decision map
def fig_multifocus():
    x, near, A, B = multifocus_toy()
    fa, fb = focus_measure(A), focus_measure(B)
    raw = fa > fb
    clean = ndi.median_filter(raw.astype(float), 31) > 0.5
    w = ndi.gaussian_filter(clean.astype(float), 2.0)
    fused = np.clip(w * A + (1 - w) * B, 0, 1)
    avg = np.clip((A + B) / 2, 0, 1)
    ratio = np.log10((fa + 1e-6) / (fb + 1e-6))
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 6.0))
    show(axes[0, 0], np.clip(A, 0, 1), "source A (sharp inside disk)")
    show(axes[0, 1], np.clip(B, 0, 1), "source B (sharp outside)")
    show(axes[0, 2], ratio, "log10 focus ratio A / B", cmap="RdBu_r", vmin=-2, vmax=2)
    show(axes[0, 3], raw.astype(float), "raw decision map (white = A)")
    show(axes[1, 0], clean.astype(float), "cleaned map (31 x 31 median)")
    show(axes[1, 1], fused, "fused with the cleaned map")
    show(axes[1, 2], np.abs(avg - x), "|error| of averaging", vmin=0, vmax=0.15, cmap="magma")
    show(axes[1, 3], np.abs(fused - x), "|error| of decision-map fusion", vmin=0, vmax=0.15, cmap="magma")
    fig.tight_layout()
    save(fig, "multifocus-decision-map.png")


# ---------------------------------------------------------------- Figure 4: exposure fusion weights
def fig_exposure():
    stack = exposure_stack()
    W = mertens_weights(stack)
    naive = np.clip(np.sum(W[..., None] * np.array(stack), axis=0), 0, 1)
    R = exposure_fusion(stack)
    ref = cv2.createMergeMertens(1.0, 1.0, 1.0).process(
        [(255 * I[..., ::-1]).round().astype(np.uint8) for I in stack])[..., ::-1]
    fig, axes = plt.subplots(3, 3, figsize=(9.2, 9.4))
    for k, nm in enumerate(["short (t = 1/8)", "middle (t = 1)", "long (t = 8)"]):
        show(axes[0, k], stack[k], nm)
        show(axes[1, k], W[k], f"normalized weight, {nm.split()[0]}", cmap="viridis")
    show(axes[2, 0], naive, "per-pixel weighted blend")
    show(axes[2, 1], R, "pyramid blend (ours)")
    show(axes[2, 2], np.clip(ref, 0, 1), "OpenCV MergeMertens")
    for r, lab in enumerate(["inputs", "weights", "results"]):
        axes[r, 0].set_ylabel(lab, fontsize=11)
    fig.tight_layout()
    save(fig, "exposure-fusion-weights.png")


# ---------------------------------------------------------------- Figure 5: IR-visible toy and its metrics
def fig_irvis():
    vis, ir, person, rng = irvis_toy()
    ring = ndi.binary_dilation(person, iterations=15) & ~person
    cands = {"visible only": vis,
             "average": (vis + ir) / 2,
             "pyramid fusion": np.clip(pyramid_fuse(vis, ir, base="max"), 0, 1),
             "average + noise": np.clip((vis + ir) / 2 + 0.08 * rng.standard_normal(vis.shape), 0, 1),
             "loss minimizer": fuse_by_loss(vis, ir, 10.0)}
    fig = plt.figure(figsize=(11.6, 6.6))
    gs = fig.add_gridspec(2, 7, height_ratios=[1, 0.62])
    show(fig.add_subplot(gs[0, 0]), vis, "visible")
    show(fig.add_subplot(gs[0, 1]), ir, "infrared")
    rows = []
    for c, (nm, F) in enumerate(cands.items()):
        show(fig.add_subplot(gs[0, c + 2]), np.clip(F, 0, 1), nm)
        mi = mutual_info(vis, F) + mutual_info(ir, F)
        tc = F[person].mean() - F[ring].mean()
        tc = 0.0 if abs(tc) < 5e-4 else tc                     # print 0.000, not -0.000
        rows.append([nm, f"{entropy(F):.2f}", f"{spatial_frequency(F):.3f}", f"{F.std():.3f}", f"{mi:.2f}",
                     f"{q_gradient(vis, ir, F):.3f}", f"{tc:.3f}"])
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    cols = ["candidate", "EN (bits)", "SF", "SD", "MI (bits)", "Q_G", "person contrast"]
    tab = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 1.5)
    # highlight the best value in each metric column
    for j in range(1, 7):
        vals = [float(r[j]) for r in rows]
        i = int(np.argmax(vals))
        tab[i + 1, j].set_facecolor("#fde0a6")
    ax.set_title("Higher is 'better' for every column; the shaded cell is each column's winner", fontsize=10)
    fig.tight_layout()
    save(fig, "irvis-metrics.png")


# ---------------------------------------------------------------- Figure 6: failure modes
def fig_failures():
    vis, ir, person, rng = irvis_toy()
    x, near, A, B = multifocus_toy()
    stack = exposure_stack()
    # (a) misregistration: the long exposure is shifted by 6 pixels (camera shake) before fusion
    shifted = [stack[0], stack[1], np.roll(stack[2], 6, axis=1)]
    bad_a, good_a = exposure_fusion(shifted), exposure_fusion(stack)
    crop_a = (slice(20, 220), slice(120, 320))
    # (b) defocus spread: a layered defocus model in which the blurred foreground (the dark silhouette of the
    #     cameraman) spills over the background in the far-focused shot; near the depth edge neither source is sharp
    dark = ndi.binary_opening(x < 0.2, iterations=2)
    lab, n = ndi.label(dark)
    m = ndi.binary_fill_holes(lab == 1 + np.argmax(ndi.sum(dark, lab, range(1, n + 1)))).astype(float)
    A2 = m * x + (1 - m) * ndi.gaussian_filter(x, 5.0)                                   # focused near
    B2 = ndi.gaussian_filter(m * x, 5.0) + (1 - ndi.gaussian_filter(m, 5.0)) * x          # focused far
    clean = ndi.median_filter((focus_measure(A2) > focus_measure(B2)).astype(float), 31) > 0.5
    w = ndi.gaussian_filter(clean.astype(float), 2.0)
    bad_b, good_b = np.clip(w * A2 + (1 - w) * B2, 0, 1), x
    crop_b = (slice(20, 180), slice(110, 270))
    # (c) halos: weights smoothed with a wide Gaussian and blended per pixel
    W = mertens_weights(stack)
    Ws = np.array([ndi.gaussian_filter(Wk, 12) for Wk in W])
    Ws /= Ws.sum(axis=0)
    bad_c = np.clip(np.sum(Ws[..., None] * np.array(stack), axis=0), 0, 1)
    good_c = good_a
    crop_c = (slice(0, 200), slice(130, 330))
    # (d) washed-out thermal target: averaging versus pyramid fusion
    bad_d = (vis + ir) / 2
    good_d = np.clip(pyramid_fuse(vis, ir, base="max"), 0, 1)
    crop_d = (slice(250, 410), slice(340, 500))
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 6.0))
    items = [("misaligned exposure (6 px)", bad_a, "aligned exposures", good_a, crop_a),
             ("defocus spread at a depth edge", bad_b, "all-in-focus truth", good_b, crop_b),
             ("blurred weights, per-pixel blend", bad_c, "pyramid blend", good_c, crop_c),
             ("averaging: faded target", bad_d, "pyramid fusion", good_d, crop_d)]
    for c, (tb, b, tg, g, cr) in enumerate(items):
        show(axes[0, c], np.clip(b[cr], 0, 1), tb)
        show(axes[1, c], np.clip(g[cr], 0, 1), tg)
    axes[0, 0].set_ylabel("failure", fontsize=11)
    axes[1, 0].set_ylabel("reference", fontsize=11)
    fig.tight_layout()
    save(fig, "failure-modes.png")


if __name__ == "__main__":
    fig_tasks()
    fig_pyramid()
    fig_multifocus()
    fig_exposure()
    fig_irvis()
    fig_failures()
