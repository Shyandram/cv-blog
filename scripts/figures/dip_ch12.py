"""Figures for DIP Chapter 12 - Feature Extraction.

Run:  .venv/bin/python scripts/figures/dip_ch12.py
Writes PNGs to src/assets/dip/ch12/ only. All inputs come from skimage.data
or synthetic arrays.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import data, feature, measure, morphology, transform, util
from skimage.color import rgb2gray

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "dip" / "ch12"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
DPI = 150


def save(fig, name):
    path = OUT / name
    # keep the final PNG at most ~1600 px wide
    fig.canvas.draw()
    w_in = fig.get_tightbbox(fig.canvas.get_renderer()).width + 0.2
    fig.savefig(path, dpi=min(DPI, 1590 / w_in), bbox_inches="tight")
    if path.stat().st_size > 900_000:  # photo-heavy figures: palette-quantize
        Image.open(path).convert("RGB").quantize(256, dither=Image.Dither.NONE).save(path, optimize=True)
    plt.close(fig)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def horse_mask():
    """skimage's horse is black on white; return a True-inside mask."""
    m = ~data.horse()
    m = ndi.binary_fill_holes(m)
    lab = measure.label(m)
    largest = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    return lab == largest


def main_contour(mask):
    cs = measure.find_contours(mask.astype(float), 0.5)
    return max(cs, key=len)  # (N, 2) as (row, col)


# ---------------------------------------------------------------- 1. boundary
def chain_code(mask, step):
    """Resample the boundary on a coarse grid and return grid nodes + Freeman 8-code."""
    small = transform.resize(mask.astype(float), (mask.shape[0] // step, mask.shape[1] // step),
                             order=0, anti_aliasing=False) > 0.5
    small = morphology.remove_small_objects(small, 20)
    small = ndi.binary_fill_holes(small)
    lab = measure.label(small)
    small = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    # Moore boundary tracing via OpenCV (CHAIN_APPROX_NONE gives every 8-connected pixel)
    cnts, _ = cv2.findContours(small.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(cnts, key=len)[:, 0, :]  # (x, y)
    dirs = {(1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3, (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7}
    d = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    code = [dirs[(int(dx), int(dy))] for dx, dy in d]
    return small, pts, code


def fig_boundary():
    mask = horse_mask()
    c = main_contour(mask)
    small, pts, code = chain_code(mask, 12)
    poly_fine = measure.approximate_polygon(c, tolerance=2.0)
    poly_coarse = measure.approximate_polygon(c, tolerance=12.0)
    skel = morphology.skeletonize(mask)

    fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
    ax[0].imshow(small, cmap="gray_r", alpha=0.25)
    ax[0].plot(pts[:, 0], pts[:, 1], "-o", ms=2.5, lw=1, color="tab:blue")
    ax[0].set_title(f"8-direction chain code\n(grid 12 px, {len(code)} links)")
    ax[1].plot(c[:, 1], c[:, 0], color="0.7", lw=1, label="boundary")
    ax[1].plot(poly_fine[:, 1], poly_fine[:, 0], "-", color="tab:green", lw=1.2,
               label=f"tol 2 px: {len(poly_fine) - 1} vertices")
    ax[1].plot(poly_coarse[:, 1], poly_coarse[:, 0], "-o", ms=3, color="tab:red", lw=1.5,
               label=f"tol 12 px: {len(poly_coarse) - 1} vertices")
    ax[1].invert_yaxis()
    ax[1].set_aspect("equal")
    ax[1].legend(loc="lower left", fontsize=9)
    ax[1].set_title("Polygonal approximation\n(splitting)")
    ax[2].imshow(mask, cmap="gray_r", alpha=0.25)
    ax[2].imshow(np.ma.masked_where(~ndi.binary_dilation(skel, iterations=1), skel),
                 cmap="autumn", interpolation="nearest")
    ax[2].set_title("Skeleton (medial axis)")
    # signature of the boundary: distance from centroid vs. arc length
    cy, cx = np.array(np.nonzero(mask)).mean(axis=1)
    r = np.hypot(c[:, 0] - cy, c[:, 1] - cx)
    s = np.linspace(0, 1, len(r))
    ax[3].plot(s, r, color="tab:purple")
    ax[3].set_xlabel("normalized arc length")
    ax[3].set_ylabel("distance to centroid (px)")
    ax[3].set_title("Signature r(s)")
    for a in ax[:3]:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    save(fig, "boundary-representations.png")


# --------------------------------------------------------- 2. Fourier descr.
def fig_fourier():
    mask = horse_mask()
    c = main_contour(mask)
    # resample to K equally spaced points along the contour
    K = 512
    seg = np.hypot(*np.diff(c, axis=0).T)
    t = np.concatenate([[0], np.cumsum(seg)])
    tt = np.linspace(0, t[-1], K, endpoint=False)
    z = np.interp(tt, t, c[:, 1]) + 1j * np.interp(tt, t, c[:, 0])
    a = np.fft.fft(z)
    ks = [4, 8, 16, 32, 64, K]
    fig, ax = plt.subplots(1, len(ks), figsize=(15, 2.9))
    for axi, P in zip(ax, ks):
        keep = np.zeros_like(a)
        half = P // 2
        keep[:half] = a[:half]
        if half:
            keep[-half:] = a[-half:]
        zr = np.fft.ifft(keep)
        axi.fill(zr.real, zr.imag, color="tab:blue", alpha=0.25)
        axi.plot(np.r_[zr.real, zr.real[:1]], np.r_[zr.imag, zr.imag[:1]], color="tab:blue", lw=1)
        axi.invert_yaxis(); axi.set_aspect("equal"); axi.set_xticks([]); axi.set_yticks([])
        axi.set_title(f"P = {P}" if P < K else f"all {K}")
    fig.suptitle("Boundary rebuilt from the P lowest-frequency Fourier descriptors", y=1.03)
    save(fig, "fourier-descriptors.png")


# ------------------------------------------------------------------ 3. GLCM
def fig_glcm():
    textures = {"brick": data.brick(), "grass": data.grass(), "gravel": data.gravel()}
    props = ["contrast", "homogeneity", "energy", "correlation"]
    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.5])
    vals = {}
    for i, (name, img) in enumerate(textures.items()):
        patch = img[:256, :256]
        q = (patch // 32).astype(np.uint8)  # 8 gray levels
        P = feature.graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                 levels=8, symmetric=True, normed=True)
        vals[name] = [feature.graycoprops(P, p).mean() for p in props]
        a0 = fig.add_subplot(gs[0, i]); a0.imshow(patch, cmap="gray"); a0.set_title(name)
        a0.axis("off")
        a1 = fig.add_subplot(gs[1, i]); a1.imshow(P[:, :, 0, :].mean(-1), cmap="magma")
        a1.set_title("GLCM (d=1, mean of 4 angles)", fontsize=10)
        a1.set_xlabel("level j"); a1.set_ylabel("level i")
    ab = fig.add_subplot(gs[:, 3])
    x = np.arange(len(props))
    w = 0.26
    base = np.array(list(vals.values()))
    norm = base / base.max(axis=0)  # relative scale per property
    for k, name in enumerate(vals):
        ab.bar(x + (k - 1) * w, norm[k], w, label=name)
    ab.set_xticks(x); ab.set_xticklabels(props, rotation=20)
    ab.set_ylabel("value / max over textures")
    ab.set_title("Haralick-style GLCM features")
    ab.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    save(fig, "glcm-texture.png")
    for k, v in vals.items():
        print(k, np.round(v, 3))


# ------------------------------------------------------------------- 4. PCA
def fig_pca():
    mask = horse_mask()
    rot = transform.rotate(mask.astype(float), 35, resize=True, order=0) > 0.5
    ys, xs = np.nonzero(rot)
    X = np.stack([xs, ys], 1).astype(float)
    m = X.mean(0)
    C = np.cov((X - m).T)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    Y = (X - m) @ evecs  # Hotelling transform
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].imshow(rot, cmap="gray_r", alpha=0.35)
    for v, l, col in zip(evecs.T, evals, ["tab:red", "tab:blue"]):
        L = 2 * np.sqrt(l)
        ax[0].annotate("", xy=m + L * v, xytext=m,
                       arrowprops=dict(arrowstyle="->", lw=2.5, color=col))
    ax[0].set_title("Rotated object + eigenvectors of\npixel-coordinate covariance")
    ax[0].axis("off")
    ax[1].scatter(Y[::7, 0], Y[::7, 1], s=1, color="0.3")
    ax[1].axhline(0, color="tab:blue", lw=1); ax[1].axvline(0, color="tab:red", lw=1)
    ax[1].set_aspect("equal"); ax[1].invert_yaxis()
    ax[1].set_xlabel("1st principal axis"); ax[1].set_ylabel("2nd principal axis")
    ax[1].set_title("After y = A(x - m): centered and axis-aligned")
    fig.tight_layout()
    save(fig, "pca-alignment.png")


# ---------------------------------------------------------------- 5. Harris
def fig_harris():
    img = util.img_as_float(data.camera())[40:440, 60:460]
    R = feature.corner_harris(img, method="k", k=0.05, sigma=1.5)
    b = 8  # ignore the artificial response along the crop border
    R[:b, :] = R[-b:, :] = 0
    R[:, :b] = R[:, -b:] = 0
    pk = feature.corner_peaks(R, min_distance=7, threshold_rel=0.02)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Input (camera)")
    lim = np.percentile(np.abs(R), 99.5)
    im = ax[1].imshow(R, cmap="coolwarm", vmin=-lim, vmax=lim)
    ax[1].set_title("Harris response R\n(red: corner, blue: edge)")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].imshow(img, cmap="gray")
    ax[2].plot(pk[:, 1], pk[:, 0], "o", mfc="none", mec="yellow", ms=6, mew=1.5)
    ax[2].set_title(f"{len(pk)} corners after non-max suppression")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    save(fig, "harris-corners.png")


# ------------------------------------------------------------------ 6. MSER
def fig_mser():
    img = data.page()
    mser = cv2.MSER_create(delta=5, min_area=10, max_area=800, max_variation=0.5)
    regions, boxes = mser.detectRegions(img)
    rgb = np.dstack([img] * 3).astype(float) / 255
    over = rgb.copy()
    rng = np.random.default_rng(0)
    for r in regions:
        over[r[:, 1], r[:, 0]] = rng.uniform(0.1, 0.9, 3)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.2))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Input (page)")
    ax[1].imshow(0.35 * rgb + 0.65 * over); ax[1].set_title(f"{len(regions)} MSER regions (delta = 5, area 10-800 px)")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    save(fig, "mser-regions.png")


# ------------------------------------------------------------- 7. DoG scale
def fig_dog():
    img = util.img_as_float(rgb2gray(data.astronaut()))[30:330, 120:420]
    sig0, kf = 1.6, 2 ** (1 / 3)
    sig = [sig0 * kf ** i for i in range(5)]
    L = [ndi.gaussian_filter(img, s) for s in sig]
    D = [L[i + 1] - L[i] for i in range(4)]
    fig, ax = plt.subplots(2, 4, figsize=(10.5, 5.6))
    for i in range(4):
        ax[0, i].imshow(L[i], cmap="gray"); ax[0, i].set_title(f"L, sigma = {sig[i]:.2f}")
        lim = np.abs(D[i]).max()
        ax[1, i].imshow(D[i], cmap="coolwarm", vmin=-lim, vmax=lim)
        ax[1, i].set_title(f"DoG = L({sig[i + 1]:.2f}) - L({sig[i]:.2f})", fontsize=10)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle("One octave of the SIFT scale space: Gaussians (top) and their differences (bottom)")
    fig.tight_layout()
    save(fig, "sift-scale-space.png")


# ---------------------------------------------------------- 8. SIFT matches
def fig_sift():
    img1 = rgb2gray(data.astronaut())
    tf = transform.AffineTransform(scale=0.7, rotation=np.deg2rad(25), translation=(140, -60))
    img2 = transform.warp(img1, tf.inverse, output_shape=img1.shape)
    img2 = np.clip(0.8 * img2 + 0.1, 0, 1)  # also change brightness / contrast
    det = feature.SIFT()
    det.detect_and_extract(img1); k1, d1 = det.keypoints, det.descriptors
    s1, o1 = det.sigmas, det.orientations
    det.detect_and_extract(img2); k2, d2 = det.keypoints, det.descriptors
    m = feature.match_descriptors(d1, d2, max_ratio=0.6, cross_check=True)
    print("SIFT keypoints:", len(k1), len(k2), "matches:", len(m))
    # check geometric correctness against the known transform
    p1 = k1[m[:, 0]][:, ::-1]; p2 = k2[m[:, 1]][:, ::-1]
    err = np.hypot(*(tf(p1) - p2).T)
    good = err < 3
    print("matches within 3 px of ground truth:", good.sum())
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1, 2]})
    ax[0].imshow(img1, cmap="gray")
    sel = np.argsort(s1)[::-1][:60]
    for (r, c), s, o in zip(k1[sel], s1[sel], o1[sel]):
        ax[0].add_patch(plt.Circle((c, r), 3 * s, fill=False, ec="yellow", lw=1))
        ax[0].plot([c, c + 3 * s * np.cos(o)], [r, r + 3 * s * np.sin(o)], color="yellow", lw=1)
    ax[0].set_title("60 largest-scale SIFT keypoints\n(circle = scale, line = orientation)")
    ax[0].axis("off")
    show = np.arange(len(m))[::max(1, len(m) // 80)]
    feature.plot_matches(ax[1], img1, img2, k1, k2, m[show], only_matches=True)
    ax[1].set_title(f"{len(m)} ratio-test matches ({good.sum()} geometrically correct);\n"
                    "right image rotated 25 deg, scaled 0.7, contrast changed")
    ax[1].axis("off")
    fig.tight_layout()
    save(fig, "sift-matches.png")


if __name__ == "__main__":
    fig_boundary()
    fig_fourier()
    fig_glcm()
    fig_pca()
    fig_harris()
    fig_mser()
    fig_dog()
    fig_sift()
