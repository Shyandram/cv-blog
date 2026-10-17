"""Reference numbers for the IP 05 lab (src/lib/playground/labs/inverse-regularization.ts), computed with NumPy on the exact
asset src/assets/playground/camera-256.png. The Node test scripts/tests/lab-restore.test.ts runs the lab's own TypeScript
pipeline on the same PNG and compares, so every number quoted in the lab's text is checked twice.

The noise is the lab's fixed field: mulberry32(seed 7) with Box-Muller, re-implemented here bit for bit (32-bit integer
arithmetic), stored as float32 like the TypeScript Float32Array.

Run: .venv/bin/python scripts/figures/playground_goldens_restore.py
Writes only scripts/tests/golden/restore.json.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scripts/tests/golden/restore.json"
f = np.asarray(Image.open(ROOT / "src/assets/playground/camera-256.png"), float) / 255
N = f.shape[0]
EPS = 1e-6
SEED = 7


def mulberry32(seed):
    a = seed & 0xFFFFFFFF

    def r():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return r


def normal_field(n, seed):
    r = mulberry32(seed)
    out = np.zeros(n, np.float32)
    for i in range(n):
        out[i] = np.sqrt(-2 * np.log(max(1e-12, r()))) * np.cos(2 * np.pi * r())
    return out


z = normal_field(N * N, SEED).astype(float).reshape(N, N)
fx = np.fft.fftfreq(N)
U, V = np.meshgrid(fx, fx, indexing="ij")
F, Z = np.fft.fft2(f), np.fft.fft2(z)


def psnr(a, b):
    return float(10 * np.log10(1 / np.mean((a - b) ** 2)))


def run(sigma, noise, logk):
    K = 0.0 if logk <= -7 else 10.0 ** logk
    H = np.exp(-2 * np.pi ** 2 * sigma ** 2 * (U ** 2 + V ** 2))
    G = H * F + noise * Z
    if K == 0:
        small = np.abs(H) < EPS
        W = np.where(small, 0.0, 1 / np.where(small, 1, H))
    else:
        W = H / (H ** 2 + K)
    g = np.clip(np.real(np.fft.ifft2(G)), 0, 1)
    r = np.clip(np.real(np.fft.ifft2(W * G)), 0, 1)
    return dict(sigma=sigma, noise=noise, logk=logk, psnr_degraded=psnr(f, g), psnr_restored=psnr(f, r), gain=float(np.abs(W).max()))


grid = [-7 + 0.25 * i for i in range(29)]
out = {"seed": SEED, "noise_head": [float(v) for v in z.ravel()[:8]], "noise_std": float(z.std()), "cases": []}
for sigma, noise, logk in [(2, 0.005, -7), (2, 0.005, -2.75), (2, 0.005, -6), (2, 0.005, 0), (2, 0, -7), (1, 0, -7), (1, 0.001, -7), (1, 0.005, -7), (3, 0.02, -7), (3, 0.02, -2), (2, 0.02, -7)]:
    out["cases"].append(run(sigma, noise, logk))
for key, (sigma, noise) in {"default": (2, 0.005), "noise0": (2, 0.0), "noise02": (2, 0.02), "sigma1": (1, 0.005), "sigma3": (3, 0.005)}.items():
    curve = [run(sigma, noise, lk)["psnr_restored"] for lk in grid]
    best = int(np.argmax(curve))
    out[f"curve_{key}"] = dict(sigma=sigma, noise=noise, psnr=curve, best_logk=grid[best], best_psnr=curve[best], psnr_degraded=run(sigma, noise, -7)["psnr_degraded"])
# Where the plain inverse (K = 0) stops being harmless: the noise level at which restored PSNR falls below the blurred PSNR, sigma 2.
out["crossover_sigma2"] = [dict(noise=n, **{k: v for k, v in run(2, n, -7).items() if k.startswith("psnr")}) for n in (0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3)]
OUT.write_text(json.dumps(out, indent=1))
print("wrote", OUT)
for c in out["cases"]:
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in c.items()})
for k in ("default", "noise0", "noise02", "sigma1", "sigma3"):
    c = out[f"curve_{k}"]
    print(k, "best logK", c["best_logk"], "best", round(c["best_psnr"], 3), "degraded", round(c["psnr_degraded"], 3), "K=0", round(c["psnr"][0], 3))
print(out["crossover_sigma2"])
