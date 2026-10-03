"""Figures for DIP Chapter 7 - Wavelet and Other Image Transforms.

Run:  .venv/bin/python scripts/figures/dip_ch07.py
Writes only to src/assets/dip/ch07/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pywt
from matplotlib.patches import Rectangle
from scipy.fft import dct, dctn, idctn
from scipy.linalg import hadamard
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "dip" / "ch07"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})
DPI = 150


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- #
# 1-D transform matrices (rows = basis vectors), all orthonormal
# --------------------------------------------------------------------------- #
def dct_matrix(n):
    return dct(np.eye(n), norm="ortho", axis=0)  # row u = u-th DCT-II basis vector


def sign_changes(row):
    return int(np.sum(np.diff(np.sign(row)) != 0))


def wht_matrix(n):
    """Walsh-Hadamard matrix in sequency order (row k has k sign changes)."""
    h = hadamard(n).astype(float)
    order = np.argsort([sign_changes(r) for r in h])
    return h[order] / np.sqrt(n)


def slant_matrix(n):
    """Recursive slant matrix (Pratt, Chen and Welch), rows reordered by sequency."""
    s = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    m = 2
    while m < n:
        m *= 2
        a = np.sqrt(3 * m * m / (4 * (m * m - 1)))
        b = np.sqrt((m * m - 4) / (4 * (m * m - 1)))
        h = m // 2
        A = np.zeros((m, m))
        A[0, 0], A[0, h] = 1, 1
        A[1, 0], A[1, 1], A[1, h], A[1, h + 1] = a, b, -a, b
        A[h, 1], A[h, h + 1] = 1, -1
        A[h + 1, 0], A[h + 1, 1], A[h + 1, h], A[h + 1, h + 1] = -b, a, b, a
        for k in range(2, h):
            A[k, k], A[k, h + k] = 1, 1
            A[h + k, k], A[h + k, h + k] = 1, -1
        blk = np.zeros((m, m))
        blk[:h, :h] = s
        blk[h:, h:] = s
        s = A @ blk / np.sqrt(2)
    order = np.argsort([sign_changes(r) for r in s], kind="stable")
    return s[order]


def haar_matrix(n):
    """Orthonormal Haar matrix: row 0 constant, then Haar functions by scale."""
    rows = [np.ones(n) / np.sqrt(n)]
    p = 0
    while 2**p < n:
        for q in range(2**p):
            r = np.zeros(n)
            w = n // 2**p
            r[q * w : q * w + w // 2] = 1
            r[q * w + w // 2 : (q + 1) * w] = -1
            rows.append(r / np.linalg.norm(r))
        p += 1
    return np.array(rows)


# sanity checks: orthonormal and sequency-ordered
for name, f in [("dct", dct_matrix), ("wht", wht_matrix), ("slant", slant_matrix), ("haar", haar_matrix)]:
    M = f(8)
    assert np.allclose(M @ M.T, np.eye(8)), name
assert [sign_changes(r) for r in wht_matrix(8)] == list(range(8))
assert [sign_changes(r) for r in slant_matrix(8)] == list(range(8))
assert np.allclose(np.diff(slant_matrix(8)[1]), np.diff(slant_matrix(8)[1])[0])  # linear "slant" row
assert np.allclose(dct_matrix(8)[0], 1 / np.sqrt(8))  # DC row is constant


def basis_grid(M):
    n = M.shape[0]
    pad = 1
    tile = n + pad
    canvas = np.full((n * tile - pad, n * tile - pad), np.nan)
    for u in range(n):
        for v in range(n):
            img = np.outer(M[u], M[v])
            img = img / (np.abs(img).max() + 1e-12)
            canvas[u * tile : u * tile + n, v * tile : v * tile + n] = img
    return canvas


def fig_basis_images():
    fig, axes = plt.subplots(2, 2, figsize=(9, 9.4))
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#e07b39")
    for ax, (title, f) in zip(
        axes.ravel(),
        [("DCT-II", dct_matrix), ("Walsh-Hadamard (sequency order)", wht_matrix),
         ("Slant", slant_matrix), ("Haar", haar_matrix)],
    ):
        ax.imshow(basis_grid(f(8)), cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlabel("v (horizontal index) →")
        ax.set_ylabel("← u (vertical index)")
    fig.suptitle("8×8 basis images  (white = +, black = −, gray = 0; orange = tile gaps)", y=0.995)
    fig.tight_layout()
    save(fig, "basis-images.png")


# --------------------------------------------------------------------------- #
# 2. Time-frequency tiling
# --------------------------------------------------------------------------- #
def fig_tiling():
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.0))
    T, F = 8, 8

    def draw(ax, tiles, title):
        for (t0, f0, w, h) in tiles:
            ax.add_patch(Rectangle((t0, f0), w, h, fill=True, fc="#dbe8f6", ec="#1f4e79", lw=1.2))
        ax.set_xlim(0, T), ax.set_ylim(0, F)
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("time →")
        ax.set_ylabel("frequency →")

    draw(axes[0], [(t, 0, 1, F) for t in range(T)], "Samples (Dirac)")
    draw(axes[1], [(0, f, T, 1) for f in range(F)], "Fourier / DCT")
    draw(axes[2], [(t, f, 2, 2) for t in range(0, T, 2) for f in range(0, F, 2)], "Short-time Fourier")
    tiles = []
    top = F
    for lvl in range(3):
        h = F / 2 ** (lvl + 1)
        w = 2**lvl
        for t in range(0, T, w):
            tiles.append((t, top - h, w, h))
        top -= h
    tiles.append((0, 0, T, top))  # coarsest scaling band
    draw(axes[3], tiles, "Wavelet (dyadic)")
    fig.tight_layout()
    save(fig, "tf-tiling.png")


# --------------------------------------------------------------------------- #
# 3. Scaling and wavelet functions
# --------------------------------------------------------------------------- #
def fig_wavelet_functions():
    fams = ["haar", "db2", "db4"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 4.6), sharey="row")
    for j, name in enumerate(fams):
        w = pywt.Wavelet(name)
        phi, psi, x = w.wavefun(level=8)
        axes[0, j].plot(x, phi, color="#1f4e79", lw=1.6)
        axes[1, j].plot(x, psi, color="#b03a2e", lw=1.6)
        axes[0, j].set_title(f"{name}: scaling function φ")
        axes[1, j].set_title(f"{name}: wavelet ψ")
        for i in range(2):
            axes[i, j].axhline(0, color="gray", lw=0.6)
            axes[i, j].grid(alpha=0.3)
        axes[1, j].set_xlabel("x")
    fig.tight_layout()
    save(fig, "wavelet-functions.png")


# --------------------------------------------------------------------------- #
# 4. Two-level Haar decomposition
# --------------------------------------------------------------------------- #
def norm01(a):
    a = np.abs(a)
    return a / (np.percentile(a, 99.5) + 1e-12)


def fig_haar_two_level():
    img = img_as_float(data.camera())
    cA1, (cH1, cV1, cD1) = pywt.dwt2(img, "haar")
    cA2, (cH2, cV2, cD2) = pywt.dwt2(cA1, "haar")
    n2 = cA2.shape[0]
    n1 = cA1.shape[0]
    mosaic = np.zeros_like(img)
    a = cA2 - cA2.min()
    mosaic[:n2, :n2] = a / a.max()
    mosaic[:n2, n2:n1] = norm01(cH2)
    mosaic[n2:n1, :n2] = norm01(cV2)
    mosaic[n2:n1, n2:n1] = norm01(cD2)
    mosaic[:n1, n1:] = norm01(cH1)
    mosaic[n1:, :n1] = norm01(cV1)
    mosaic[n1:, n1:] = norm01(cD1)
    mosaic = np.clip(mosaic, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input (camera, 512×512)")
    axes[1].imshow(mosaic, cmap="gray")
    axes[1].set_title("2-level Haar DWT  (|detail|, each band rescaled)")
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    lab = dict(color="yellow", fontsize=11, ha="center", va="center", weight="bold")
    axes[1].text(n2 / 2, n2 / 2 + 50, "LL2", **lab)
    axes[1].text(n2 * 1.5, n2 / 2, "H2", **lab)
    axes[1].text(n2 / 2, n2 * 1.5, "V2", **lab)
    axes[1].text(n2 * 1.5, n2 * 1.5, "D2", **lab)
    axes[1].text(n1 * 1.5, n1 / 2, "H1 (horizontal edges)", **lab)
    axes[1].text(n1 / 2, n1 * 1.5, "V1 (vertical edges)", **lab)
    axes[1].text(n1 * 1.5, n1 * 1.5, "D1 (diagonal)", **lab)
    for p in (n1, n2):
        axes[1].axhline(p - 0.5, xmax=1 if p == n1 else 0.5, color="red", lw=1)
        axes[1].axvline(p - 0.5, ymin=0 if p == n1 else 0.5, color="red", lw=1)
    fig.tight_layout()
    save(fig, "haar-2level.png")


# --------------------------------------------------------------------------- #
# 5. Energy compaction: keep 5% of coefficients
# --------------------------------------------------------------------------- #
def keep_top(c, frac):
    t = np.quantile(np.abs(c), 1 - frac)
    return np.where(np.abs(c) >= t, c, 0)


def fig_compaction(frac=0.05):
    img = img_as_float(data.camera())
    n = img.shape[0]
    recs = {}
    # DCT on the whole image
    recs["DCT"] = idctn(keep_top(dctn(img, norm="ortho"), frac), norm="ortho")
    # Walsh-Hadamard on the whole image
    W = wht_matrix(n)
    recs["Walsh-Hadamard"] = W.T @ keep_top(W @ img @ W.T, frac) @ W
    # wavelets
    for name in ["haar", "db4"]:
        coeffs = pywt.wavedec2(img, name, mode="periodization", level=5)
        arr, slices = pywt.coeffs_to_array(coeffs)
        arr = keep_top(arr, frac)
        recs[name] = pywt.waverec2(
            pywt.array_to_coeffs(arr, slices, output_format="wavedec2"), name, mode="periodization"
        )
    fig, axes = plt.subplots(1, 5, figsize=(10.5, 2.9))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    crop = (slice(60, 260), slice(180, 380))
    axes[0].add_patch(Rectangle((180, 60), 200, 200, fill=False, ec="yellow", lw=1.2))
    for ax, (k, r) in zip(axes[1:], recs.items()):
        ax.imshow(np.clip(r, 0, 1)[crop], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{k}\nPSNR {psnr(img, np.clip(r, 0, 1), data_range=1):.1f} dB", fontsize=10)
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"Reconstruction from the largest {int(frac*100)}% of coefficients (crop shown)", y=1.04)
    save(fig, "compaction.png")


# --------------------------------------------------------------------------- #
# 6. Denoising by wavelet thresholding
# --------------------------------------------------------------------------- #
def wavelet_denoise(noisy, wavelet="db4", level=3, mode="soft"):
    coeffs = pywt.wavedec2(noisy, wavelet, mode="periodization", level=level)
    sigma = np.median(np.abs(coeffs[-1][2])) / 0.6745  # MAD of finest diagonal band
    thr = sigma * np.sqrt(2 * np.log(noisy.size))  # universal threshold
    new = [coeffs[0]] + [tuple(pywt.threshold(d, thr, mode=mode) for d in lvl) for lvl in coeffs[1:]]
    return pywt.waverec2(new, wavelet, mode="periodization"), sigma, thr


def fig_denoise():
    rng = np.random.default_rng(7)
    img = img_as_float(data.camera())
    noisy = img + rng.normal(0, 0.1, img.shape)
    hard, s, t = wavelet_denoise(noisy, mode="hard")
    soft, _, _ = wavelet_denoise(noisy, mode="soft")
    print(f"estimated sigma={s:.4f}, threshold={t:.4f}")

    fig = plt.figure(figsize=(10.5, 3.2))
    ax0 = fig.add_subplot(1, 4, 1)
    x = np.linspace(-3, 3, 601)
    ax0.plot(x, pywt.threshold(x, 1, "hard"), label="hard", color="#b03a2e", lw=1.8)
    ax0.plot(x, pywt.threshold(x, 1, "soft"), label="soft", color="#1f4e79", lw=1.8)
    ax0.plot(x, x, color="gray", ls=":", lw=1)
    ax0.set_title("Thresholding rules (T = 1)")
    ax0.set_xlabel("input coefficient")
    ax0.set_ylabel("output")
    ax0.legend(fontsize=10)
    ax0.grid(alpha=0.3)
    crop = (slice(80, 336), slice(160, 416))
    for i, (k, im) in enumerate([("Noisy", noisy), ("Hard threshold", hard), ("Soft threshold", soft)]):
        ax = fig.add_subplot(1, 4, i + 2)
        ax.imshow(np.clip(im, 0, 1)[crop], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{k}\nPSNR {psnr(img, np.clip(im, 0, 1), data_range=1):.1f} dB", fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    save(fig, "denoise.png")

    # identical-size PNGs for the ImageCompare slider
    plt.imsave(OUT / "denoise-noisy.png", np.clip(noisy, 0, 1)[crop], cmap="gray", vmin=0, vmax=1)
    plt.imsave(OUT / "denoise-soft.png", np.clip(soft, 0, 1)[crop], cmap="gray", vmin=0, vmax=1)
    print("wrote denoise-noisy.png, denoise-soft.png")


# --------------------------------------------------------------------------- #
# 7. Edge detection by discarding the approximation band
# --------------------------------------------------------------------------- #
def fig_edges():
    img = img_as_float(data.coins())
    coeffs = pywt.wavedec2(img, "haar", level=2)
    coeffs[0] = np.zeros_like(coeffs[0])
    edges_all = pywt.waverec2(coeffs, "haar")
    coeffs_v = pywt.wavedec2(img, "haar", level=2)
    coeffs_v[0] = np.zeros_like(coeffs_v[0])
    coeffs_v[1:] = [(np.zeros_like(h), v, np.zeros_like(d)) for (h, v, d) in coeffs_v[1:]]
    edges_v = pywt.waverec2(coeffs_v, "haar")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Input (coins)")
    axes[1].imshow(np.abs(edges_all), cmap="gray", vmax=np.percentile(np.abs(edges_all), 99.5))
    axes[1].set_title("LL band set to 0")
    axes[2].imshow(np.abs(edges_v), cmap="gray", vmax=np.percentile(np.abs(edges_v), 99.5))
    axes[2].set_title("Only vertical-detail bands kept")
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    save(fig, "wavelet-edges.png")


if __name__ == "__main__":
    fig_basis_images()
    fig_tiling()
    fig_wavelet_functions()
    fig_haar_two_level()
    fig_compaction()
    fig_denoise()
    fig_edges()
