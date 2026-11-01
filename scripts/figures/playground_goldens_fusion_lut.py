"""Golden values for the labs image-fusion (DLCV 05B) and lut-trilinear (DLCV 05A).

Everything is computed here with NumPy, SciPy and scikit-image, independently of the TypeScript, on the exact PNG assets:

* image-fusion: scipy.ndimage.laplace / gaussian_filter, the decision map, the soft blend, PSNR with
  skimage.metrics.peak_signal_noise_ratio(data_range=1), the agreement on textured pixels (the chapter's definition), and the
  detail-contrast ratios quoted in the predict explanation;
* lut-trilinear: the exact transform, a table baked with it, nearest lookup by indexing and trilinear interpolation by
  scipy.ndimage.map_coordinates(order=1), error statistics in 8-bit levels on the gradient and on the astronaut photo.

Run: .venv/bin/python scripts/figures/playground_goldens_fusion_lut.py
Writes only scripts/tests/golden/fusion-lut.json.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.metrics import peak_signal_noise_ratio

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "src/assets/playground"
OUT = ROOT / "scripts/tests/golden/fusion-lut.json"


def load(name, rgb=False):
    return np.asarray(Image.open(ASSETS / name).convert("RGB" if rgb else "L"), float) / 255


# ------------------------------------------------------------------------------------------------ image fusion
T, A, B = load("image-fusion-truth.png"), load("image-fusion-a.png"), load("image-fusion-b.png")
FOCUS = load("image-fusion-focus.png") > 0.5
TEXTURED = ndi.gaussian_filter(np.abs(ndi.laplace(T)), 2) > 0.01


def measure(img, kind, s):
    if kind == "laplacian":
        return ndi.gaussian_filter(ndi.laplace(img) ** 2, s)
    m = ndi.gaussian_filter(img, s)
    return ndi.gaussian_filter(img * img, s) - m * m


def fuse(kind, s, smooth):
    d = measure(A, kind, s) > measure(B, kind, s)
    w = ndi.gaussian_filter(d.astype(float), smooth) if smooth > 0 else d.astype(float)
    f = np.clip(w * A + (1 - w) * B, 0, 1)
    return d, f


def psnr(x, y):
    return float(peak_signal_noise_ratio(x, y, data_range=1))


def detail(img, mask):
    return float(np.sqrt(np.mean((img - ndi.gaussian_filter(img, 2))[mask] ** 2)))


avg = np.clip((A + B) / 2, 0, 1)
fusion = {"psnrAvg": psnr(T, avg), "psnrA": psnr(T, A), "psnrB": psnr(T, B), "texturedFraction": float(TEXTURED.mean()), "cases": []}
for kind in ("laplacian", "variance"):
    for s, smooth in [(2, 0), (0.5, 0), (1, 0), (3, 0), (6, 0), (2, 2), (0.5, 2), (4, 4)]:
        d, f = fuse(kind, s, smooth)
        fusion["cases"].append({"kind": kind, "sigma": s, "smooth": smooth, "psnr": psnr(T, f),
                                "agree": float(np.mean((d == FOCUS)[TEXTURED])), "agreeAll": float(np.mean(d == FOCUS))})
_, f_default = fuse("laplacian", 2, 0)
fusion["detail"] = {"truth": detail(T, FOCUS), "average": detail(avg, FOCUS), "fused": detail(f_default, FOCUS)}
fusion["detail"]["ratioAverage"] = fusion["detail"]["average"] / fusion["detail"]["truth"]
fusion["detail"]["ratioFused"] = fusion["detail"]["fused"] / fusion["detail"]["truth"]
sample = np.arange(30, dtype=float).reshape(5, 6) ** 2
fusion["laplaceSample"] = {"input": sample.ravel().tolist(), "output": ndi.laplace(sample).ravel().tolist()}

# ------------------------------------------------------------------------------------------------ 3D LUT
GAMMA, S_BLEND, SAT, HUE = 0.9, 0.5, 1.15, np.deg2rad(8)
LUMA = np.array([0.299, 0.587, 0.114])
k = np.ones(3) / np.sqrt(3)
K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
ROT = np.eye(3) * np.cos(HUE) + (1 - np.cos(HUE)) * np.outer(k, k) + np.sin(HUE) * K       # Rodrigues rotation about the gray axis
MAT = ROT @ (SAT * np.eye(3) + (1 - SAT) * np.outer(np.ones(3), LUMA))


def curve(t):
    g = np.power(t, GAMMA)
    return (1 - S_BLEND) * g + S_BLEND * (3 * g * g - 2 * g ** 3)


def exact(rgb):
    return np.clip(curve(rgb) @ MAT.T, 0, 1)


def levels(x):
    return np.round(np.clip(x, 0, 1) * 255)


def bake(M):
    g = np.linspace(0, 1, M)
    r, gg, b = np.meshgrid(g, g, g, indexing="ij")
    return exact(np.stack([r, gg, b], -1))


def lookup(lut, rgb, mode):
    M = lut.shape[0]
    p = rgb * (M - 1)
    if mode == "nearest":
        i = np.round(p).astype(int)
        return lut[i[..., 0], i[..., 1], i[..., 2]]
    coords = [p[..., 0], p[..., 1], p[..., 2]]
    return np.stack([ndi.map_coordinates(lut[..., c], coords, order=1) for c in range(3)], -1)


h = w = 256
u, v = np.linspace(0, 1, w)[None, :, None], np.linspace(0, 1, h)[:, None, None]
TL, TR, BL, BR = np.array([.05, .10, .35]), np.array([.10, .55, .60]), np.array([.90, .45, .20]), np.array([.98, .85, .75])
gradient = (1 - v) * ((1 - u) * TL + u * TR) + v * ((1 - u) * BL + u * BR)
photo = load("astronaut-256.png", rgb=True)
lut = {"matrix": MAT.ravel().tolist(), "colours": [], "stats": {}}
for c in ([0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [0.2, 0.6, 0.9], [0.9, 0.3, 0.1], [0.03, 0.02, 0.05]):
    lut["colours"].append({"in": c, "out": exact(np.array(c, float)).tolist()})
for name, img in (("gradient", gradient), ("photo", photo)):
    ex = levels(exact(img))
    lut["stats"][name] = {}
    for M in (2, 3, 5, 9, 17, 33):
        t = bake(M)
        entry = {}
        for mode in ("nearest", "trilinear"):
            e = np.abs(levels(lookup(t, img, mode)) - ex)
            entry[mode] = {"mean": float(e.mean()), "max": float(e.max())}
        lut["stats"][name][str(M)] = entry
rng = np.random.default_rng(1)
pts = rng.random((8, 3))
t5 = bake(5)
lut["trilinearPoints"] = [{"in": p.tolist(), "M": 5, "out": lookup(t5, p[None, :], "trilinear")[0].tolist(), "nearest": lookup(t5, p[None, :], "nearest")[0].tolist()} for p in pts]

OUT.write_text(json.dumps({"fusion": fusion, "lut": lut}, indent=1))
print(OUT)
print("average PSNR", fusion["psnrAvg"], "| default focus-rule PSNR", fusion["cases"][0]["psnr"], "agree", fusion["cases"][0]["agree"])
print("detail ratios", fusion["detail"]["ratioAverage"], fusion["detail"]["ratioFused"])
print({n: {M: (round(s["nearest"]["mean"], 2), round(s["trilinear"]["mean"], 2)) for M, s in d.items()} for n, d in lut["stats"].items()})
