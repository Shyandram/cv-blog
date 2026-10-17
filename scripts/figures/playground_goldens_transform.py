"""Reference numbers for the `transform-coding` lab, computed independently of the TypeScript kit.

The lab runs an orthonormal 8 x 8 block DCT (scipy.fft.dctn, norm="ortho") or a 5-level orthonormal 2-D Haar transform
(PyWavelets, mode="periodization") on src/assets/playground/camera-256.png, rounds every coefficient to a multiple of `step`,
inverts, and measures. scripts/tests/lab-transform.test.ts runs the lab's own TypeScript pipeline on the same PNG and
compares with the JSON written here, so every number quoted in the lab's text is asserted.

Run: .venv/bin/python scripts/figures/playground_goldens_transform.py
Writes only scripts/tests/golden/transform-coding.json.
"""
import json
from pathlib import Path

import numpy as np
import pywt
from PIL import Image
from scipy import ndimage as ndi
from scipy.fft import dctn, idctn

ROOT = Path(__file__).resolve().parents[2]
im = np.asarray(Image.open(ROOT / "src/assets/playground/camera-256.png")).astype(np.float64) / 255
H, W = im.shape
LEVELS = 5
STEPS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
REGIONS = {"sky": (128, 8), "edge": (128, 64), "grass": (168, 152)}
ZOOM = 32


def block_dct(x):
    return dctn(x.reshape(H // 8, 8, W // 8, 8).swapaxes(1, 2), axes=(2, 3), norm="ortho")


def block_idct(c):
    return idctn(c, axes=(2, 3), norm="ortho").swapaxes(1, 2).reshape(H, W)


def haar(x):
    c = pywt.wavedec2(x, "haar", level=LEVELS, mode="periodization")
    return pywt.coeffs_to_array(c)


def ihaar(a, slices):
    c = pywt.array_to_coeffs(a, slices, output_format="wavedec2")
    return pywt.waverec2(c, "haar", mode="periodization")


def psnr(a, b):
    return float(10 * np.log10(1 / np.mean((a - b) ** 2)))


def seam_ratio(x):
    dx = np.abs(np.diff(x, axis=1))   # between column i and i+1
    dy = np.abs(np.diff(x, axis=0))
    ix = np.arange(W - 1) % 8 == 7
    iy = np.arange(H - 1) % 8 == 7
    edge = dx[:, ix].sum() + dy[iy, :].sum()
    n_edge = dx[:, ix].size + dy[iy, :].size
    inner = dx[:, ~ix].sum() + dy[~iy, :].sum()
    n_inner = dx[:, ~ix].size + dy[~iy, :].size
    return float(edge / n_edge / (inner / n_inner))


# Smoothest half = the 32768 pixels with the smallest Sobel gradient magnitude (stable order for ties).
g = np.hypot(ndi.sobel(im, axis=1, mode="nearest"), ndi.sobel(im, axis=0, mode="nearest"))
order = np.argsort(g.ravel(), kind="stable")
busy = np.zeros(H * W, bool)
busy[order[H * W // 2:]] = True
busy = busy.reshape(H, W)

cd = block_dct(im)
ch, slices = haar(im)
rows = []
for kind in ("dct", "haar"):
    for step in STEPS:
        c = cd if kind == "dct" else ch
        q = np.round(c / step) * step
        r = block_idct(q) if kind == "dct" else ihaar(q, slices)
        e = r - im
        row = dict(
            transform=kind, step=step,
            nonzero=float(np.mean(q != 0)),
            psnr=psnr(im, r),
            psnr_clipped=psnr(im, np.clip(r, 0, 1)),
            mae_smooth=float(255 * np.abs(e[~busy]).mean()),
            mae_busy=float(255 * np.abs(e[busy]).mean()),
            seam=seam_ratio(r),
            coef_sq_error=float(np.sum((c - q) ** 2)),
            pixel_sq_error=float(np.sum(e ** 2)),
        )
        for name, (x, y) in REGIONS.items():
            row[f"box_{name}"] = float(255 * np.abs(e[y:y + ZOOM, x:x + ZOOM]).mean())
        rows.append(row)

out = dict(w=W, h=H, levels=LEVELS, original_seam=seam_ratio(im), rows=rows)
dest = ROOT / "scripts/tests/golden/transform-coding.json"
dest.write_text(json.dumps(out, indent=1))
for r in rows:
    print(f"{r['transform']:4s} step {r['step']:<4} nonzero {100 * r['nonzero']:5.2f}%  PSNR {r['psnr']:5.2f} (clipped {r['psnr_clipped']:5.2f})  "
          f"MAE smooth/busy {r['mae_smooth']:.2f}/{r['mae_busy']:.2f}  seam {r['seam']:.2f}  boxes "
          f"{r['box_sky']:.2f}/{r['box_edge']:.2f}/{r['box_grass']:.2f}")
print("original seam ratio", round(out["original_seam"], 3))
print("wrote", dest)
