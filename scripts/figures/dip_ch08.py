"""Figures for DIP Chapter 8 - Image Compression and Watermarking.

Run:  .venv/bin/python scripts/figures/dip_ch08.py
Writes only to src/assets/dip/ch08/. All inputs come from skimage.data or are synthetic.
"""
from pathlib import Path
import heapq
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pywt
from scipy.fft import dctn, idctn
from skimage import data

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "dip" / "ch08"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white"})
DPI = 150

CAM = data.camera().astype(np.uint8)  # 512 x 512


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{name}: {p.stat().st_size / 1024:.0f} KB")


def entropy(x):
    _, c = np.unique(x, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse)


# ---------------------------------------------------------------- 1. entropy
def fig_entropy():
    img = CAM.astype(np.int16)
    diff = img[:, 1:] - img[:, :-1]  # previous-pixel prediction error
    h0, h1 = entropy(img), entropy(diff)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    ax[0].imshow(CAM, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("Input (camera)")
    ax[0].axis("off")
    ax[1].hist(img.ravel(), bins=256, range=(0, 256), color="#3b6ea5")
    ax[1].set_title(f"Pixel values: H = {h0:.2f} bits/pixel")
    ax[1].set_xlabel("gray level")
    ax[1].set_ylabel("count")
    ax[2].hist(diff.ravel(), bins=511, range=(-255.5, 255.5), color="#c0504d")
    ax[2].set_xlim(-60, 60)
    ax[2].set_title(f"Horizontal differences: H = {h1:.2f} bits")
    ax[2].set_xlabel("f(x,y) - f(x,y-1)")
    fig.tight_layout()
    save(fig, "entropy.png")
    print(f"  entropy pixels={h0:.3f}, diffs={h1:.3f}")


# ---------------------------------------------------------------- 2. Huffman tree
SYMS = {"a1": 0.40, "a2": 0.20, "a3": 0.15, "a4": 0.10, "a5": 0.10, "a6": 0.05}


def huffman(probs):
    cnt = itertools.count()
    heap = [(p, next(cnt), {"sym": s, "p": p}) for s, p in probs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        p1, _, n1 = heapq.heappop(heap)
        p2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (p1 + p2, next(cnt), {"p": p1 + p2, "kids": (n2, n1)}))
    root = heap[0][2]
    codes = {}

    def walk(n, c):
        if "sym" in n:
            codes[n["sym"]] = c or "0"
        else:
            walk(n["kids"][0], c + "0")
            walk(n["kids"][1], c + "1")
    walk(root, "")
    return root, codes


def fig_huffman():
    root, codes = huffman(SYMS)
    # layout: leaves get x positions in DFS order, internal nodes centred on children
    pos, xs = {}, itertools.count()

    def place(n, depth):
        if "sym" in n:
            pos[id(n)] = (next(xs), -depth)
        else:
            for k in n["kids"]:
                place(k, depth + 1)
            a, b = (pos[id(k)] for k in n["kids"])
            pos[id(n)] = ((a[0] + b[0]) / 2, -depth)
    place(root, 0)
    fig, ax = plt.subplots(figsize=(9, 5))

    def draw(n):
        x, y = pos[id(n)]
        if "kids" in n:
            for bit, k in zip("01", n["kids"]):
                kx, ky = pos[id(k)]
                ax.plot([x, kx], [y, ky], color="#555", lw=1.5, zorder=1)
                ax.text((x + kx) / 2 + (-0.15 if bit == "0" else 0.15), (y + ky) / 2, bit,
                        color="#c0504d", fontsize=12, ha="center", va="center", fontweight="bold")
                draw(k)
            ax.scatter([x], [y], s=900, color="#dbe7f3", edgecolor="#3b6ea5", zorder=2)
            ax.text(x, y, f"{n['p']:.2f}", ha="center", va="center", fontsize=10, zorder=3)
        else:
            ax.scatter([x], [y], s=1100, marker="s", color="#fdebd3", edgecolor="#c77c11", zorder=2)
            ax.text(x, y + 0.05, f"{n['sym']}\n{n['p']:.2f}", ha="center", va="center", fontsize=10, zorder=3)
            ax.text(x, y - 0.45, codes[n["sym"]], ha="center", va="center", fontsize=11, family="monospace")
    draw(root)
    L = sum(SYMS[s] * len(c) for s, c in codes.items())
    H = -sum(p * np.log2(p) for p in SYMS.values())
    ax.set_title(f"Huffman tree: average length {L:.2f} bits/symbol, entropy {H:.3f} bits/symbol")
    ax.axis("off")
    ax.margins(0.08, 0.15)
    save(fig, "huffman-tree.png")
    print("  codes", codes, f"L={L:.3f} H={H:.4f}")


# ---------------------------------------------------------------- 3. DCT quantization
Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], float)


def qtable(quality):
    s = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip(np.floor((Q50 * s + 50) / 100), 1, 255)


def dct_codec(img, quality):
    """Baseline-JPEG-like luminance path: level shift, 8x8 DCT, quantize, dequantize."""
    Q = qtable(quality)
    f = img.astype(np.float64) - 128
    H, W = f.shape
    blocks = f.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)
    C = dctn(blocks, axes=(2, 3), norm="ortho")
    q = np.round(C / Q)
    rec = idctn(q * Q, axes=(2, 3), norm="ortho")
    rec = rec.transpose(0, 2, 1, 3).reshape(H, W) + 128
    return np.clip(np.round(rec), 0, 255).astype(np.uint8), q


def fig_dct_quality():
    crop = (slice(60, 252), slice(180, 372))  # head and camera of the photographer
    qs = [90, 50, 20, 5]
    fig, ax = plt.subplots(1, 5, figsize=(10.5, 2.9))
    ax[0].imshow(CAM[crop], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax[0].set_title("Original")
    for a, q in zip(ax[1:], qs):
        rec, coef = dct_codec(CAM, q)
        nz = np.count_nonzero(coef) / coef.size
        a.imshow(rec[crop], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        a.set_title(f"Q={q}: PSNR {psnr(CAM, rec):.1f} dB\nnonzero coeffs {100 * nz:.1f}%", fontsize=10)
        print(f"  Q={q}: psnr={psnr(CAM, rec):.2f} nz={100*nz:.2f}%")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    save(fig, "dct-quality.png")


def fig_dct_block():
    y, x = 128, 248
    blk = CAM[y:y + 8, x:x + 8].astype(float)
    C = dctn(blk - 128, norm="ortho")
    Q = qtable(50)
    q = np.round(C / Q)
    rec = np.clip(np.round(idctn(q * Q, norm="ortho") + 128), 0, 255)
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 9.5))
    ax = ax.ravel()
    panels = [(blk, "8x8 block (pixels)", "gray", "{:.0f}"),
              (C, "DCT coefficients", "RdBu_r", "{:.0f}"),
              (q, "Quantized, Q=50 (integers)", "RdBu_r", "{:.0f}"),
              (rec, "Reconstructed block", "gray", "{:.0f}")]
    for a, (m, t, cm, fmt) in zip(ax, panels):
        if cm == "gray":
            a.imshow(m, cmap=cm, vmin=0, vmax=255)
        else:
            v = max(1, np.abs(m).max())
            a.imshow(m, cmap=cm, vmin=-v, vmax=v)
        for i in range(8):
            for j in range(8):
                val = m[i, j]
                col = "white" if (cm == "gray" and val < 110) else "black"
                a.text(j, i, fmt.format(int(np.round(val))), ha="center", va="center", fontsize=10, color=col)
        a.set_title(t)
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    save(fig, "dct-block.png")
    print(f"  block nonzero quantized = {np.count_nonzero(q)} / 64; block psnr={psnr(blk, rec):.2f}")


# ---------------------------------------------------------------- 4. wavelet vs DCT at equal coefficient budget
def keep_top(c, frac):
    t = np.quantile(np.abs(c), 1 - frac)
    return np.where(np.abs(c) >= t, c, 0)


def fig_wavelet_vs_dct(frac=0.02):
    f = CAM.astype(np.float64)
    H, W = f.shape
    blocks = f.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)
    C = dctn(blocks, axes=(2, 3), norm="ortho")
    C = keep_top(C, frac)
    rd = idctn(C, axes=(2, 3), norm="ortho").transpose(0, 2, 1, 3).reshape(H, W)
    rd = np.clip(rd, 0, 255)
    coeffs = pywt.wavedec2(f, "bior4.4", level=5, mode="periodization")
    arr, sl = pywt.coeffs_to_array(coeffs)
    arr = keep_top(arr, frac)
    rw = pywt.waverec2(pywt.array_to_coeffs(arr, sl, output_format="wavedec2"), "bior4.4", mode="periodization")
    rw = np.clip(rw, 0, 255)
    crop = (slice(60, 252), slice(180, 372))
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.9))
    for a, im, t in zip(ax, [f, rd, rw], ["Original",
                                          f"8x8 DCT, top {100*frac:.0f}% coeffs\nPSNR {psnr(f, rd):.1f} dB",
                                          f"Wavelet (bior4.4, 5 levels), top {100*frac:.0f}%\nPSNR {psnr(f, rw):.1f} dB"]):
        a.imshow(im[crop], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        a.set_title(t, fontsize=10.5)
        a.axis("off")
    fig.tight_layout()
    save(fig, "wavelet-vs-dct.png")
    print(f"  equal budget {frac}: dct={psnr(f, rd):.2f}, wavelet={psnr(f, rw):.2f}")


# ---------------------------------------------------------------- 5. bit planes
def fig_bitplanes():
    fig, ax = plt.subplots(2, 4, figsize=(10.5, 5.7))
    for k in range(8):
        a = ax[(7 - k) // 4, (7 - k) % 4]
        a.imshow((CAM >> k) & 1, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        a.set_title(f"bit plane {k}" + (" (MSB)" if k == 7 else " (LSB)" if k == 0 else ""))
        a.axis("off")
    fig.tight_layout()
    save(fig, "bitplanes.png")


# ---------------------------------------------------------------- 6. LSB watermark
def make_mark(shape):
    m = np.zeros(shape, np.uint8)
    cv2.putText(m, "DIP", (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 6.5, 1, 30, cv2.LINE_AA)
    cv2.rectangle(m, (20, 20), (shape[1] - 20, shape[0] - 20), 1, 12)
    return m


def fig_lsb():
    mark = make_mark(CAM.shape)
    wm = (CAM & 0xFE) | mark
    ok, buf = cv2.imencode(".jpg", wm, [cv2.IMWRITE_JPEG_QUALITY, 95])
    jp = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    fig, ax = plt.subplots(1, 4, figsize=(10.5, 3.0))
    items = [(wm, f"Watermarked (PSNR {psnr(CAM, wm):.1f} dB)", 255),
             (CAM & 1, "LSB plane of the original", 1),
             (wm & 1, "LSB plane after embedding", 1),
             (jp & 1, "LSB plane after JPEG Q=95", 1)]
    for a, (im, t, vmax) in zip(ax, items):
        a.imshow(im, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        a.set_title(t, fontsize=10.5)
        a.axis("off")
    fig.tight_layout()
    save(fig, "lsb-watermark.png")
    print(f"  LSB bit agreement after JPEG95: {np.mean((jp & 1) == mark):.3f}")


# ---------------------------------------------------------------- 7. DCT-domain watermark (numbers only, used in text)
def dct_watermark_stats(alpha=12.0, seed=7):
    rng = np.random.default_rng(seed)
    f = CAM.astype(np.float64)
    H, W = f.shape
    nb = (H // 8) * (W // 8)
    bits = rng.integers(0, 2, nb)
    blocks = f.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3).reshape(nb, 8, 8)
    C = dctn(blocks, axes=(1, 2), norm="ortho")
    a, b = (2, 3), (3, 2)          # two mid-frequency positions
    for i, bit in enumerate(bits):
        x, y = C[i][a], C[i][b]
        if bit and x - y < alpha:
            m = (x + y) / 2; C[i][a], C[i][b] = m + alpha / 2, m - alpha / 2
        if not bit and y - x < alpha:
            m = (x + y) / 2; C[i][a], C[i][b] = m - alpha / 2, m + alpha / 2
    rec = idctn(C, axes=(1, 2), norm="ortho").reshape(H // 8, W // 8, 8, 8).transpose(0, 2, 1, 3).reshape(H, W)
    wm = np.clip(np.round(rec), 0, 255).astype(np.uint8)
    out = {"psnr": psnr(CAM, wm)}
    for q in (90, 75, 50):
        ok, buf = cv2.imencode(".jpg", wm, [cv2.IMWRITE_JPEG_QUALITY, q])
        g = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        D = dctn(g.reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3).reshape(nb, 8, 8), axes=(1, 2), norm="ortho")
        est = (D[:, 2, 3] > D[:, 3, 2]).astype(int)
        out[q] = float(np.mean(est != bits))
    print(f"  DCT watermark alpha={alpha}: psnr={out['psnr']:.2f}, BER@90={out[90]:.4f}, @75={out[75]:.4f}, @50={out[50]:.4f}")


# ---------------------------------------------------------------- 8. ImageCompare pair
def compare_pair():
    img = data.astronaut()[:, :, ::-1]  # BGR for OpenCV, 512x512x3
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 4])
    low = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    small = cv2.imencode(".png", img)[1].size
    print(f"  compare: jpeg q4 bytes={buf.size}, png bytes={small}, raw={img.size}, psnr={psnr(img, low):.2f}")
    cv2.imwrite(str(OUT / "compare-original.png"), img)
    cv2.imwrite(str(OUT / "compare-jpeg-q4.png"), low)
    for n in ("compare-original.png", "compare-jpeg-q4.png"):
        print(f"{n}: {(OUT / n).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    fig_entropy()
    fig_huffman()
    fig_dct_quality()
    fig_dct_block()
    fig_wavelet_vs_dct()
    fig_bitplanes()
    fig_lsb()
    dct_watermark_stats(12.0)
    dct_watermark_stats(30.0)
    compare_pair()
