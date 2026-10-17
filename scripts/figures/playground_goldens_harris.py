"""Golden values for the Harris corner lab (src/lib/imagekit/harris.ts, src/lib/playground/labs/corner-tensor.ts).

Everything is computed with NumPy and SciPy, independently of the TypeScript, on (a) a small seeded synthetic image and
(b) the exact asset PNG src/assets/playground/camera-256.png. The chapter's own function (DIP 12: Gaussian derivative
sigma_d = 1, Gaussian window sigma_i = 1.5, k = 0.05, scipy's gaussian_filter) is reproduced as `chapter_R` so the lab can be
shown to agree with the text. The lab itself uses a window of W x W pixels with Gaussian weights of sigma = W / 6.

Run: .venv/bin/python scripts/figures/playground_goldens_harris.py
Writes only scripts/tests/golden/harris.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import io

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scripts/tests/golden/harris.json"
K, SIGMA_D = 0.05, 1.0


def flat(a):
    return [float(f"{x:.8g}") for x in np.asarray(a, dtype=float).ravel()]


def gradients(img):
    ix = ndi.gaussian_filter(img, SIGMA_D, order=(0, 1))  # derivative along x (columns)
    iy = ndi.gaussian_filter(img, SIGMA_D, order=(1, 0))  # derivative along y (rows)
    return ix, iy


def window_weights(radius, sigma):
    i = np.arange(-radius, radius + 1)
    g = np.exp(-i * i / (2 * sigma * sigma))
    return g / g.sum()


def tensor_windowed(ix, iy, radius, sigma):
    g = window_weights(radius, sigma)

    def f(a):
        return ndi.correlate1d(ndi.correlate1d(a, g, axis=0, mode="reflect"), g, axis=1, mode="reflect")

    return f(ix * ix), f(iy * iy), f(ix * iy)


def response(xx, yy, xy, k=K):
    return xx * yy - xy ** 2 - k * (xx + yy) ** 2


def eigen(xx, yy, xy):
    tr = xx + yy
    d = np.hypot(xx - yy, 2 * xy)
    return (tr + d) / 2, (tr - d) / 2


def chapter_R(img, sigma_d=1.0, sigma_i=1.5, k=0.05):
    """Verbatim from the DIP 12 chapter."""
    Ix = ndi.gaussian_filter(img, sigma_d, order=(0, 1))
    Iy = ndi.gaussian_filter(img, sigma_d, order=(1, 0))
    Sxx = ndi.gaussian_filter(Ix * Ix, sigma_i)
    Syy = ndi.gaussian_filter(Iy * Iy, sigma_i)
    Sxy = ndi.gaussian_filter(Ix * Iy, sigma_i)
    det, tr = Sxx * Syy - Sxy ** 2, Sxx + Syy
    return det - k * tr ** 2


g = {}

# --- synthetic image: a bright rectangle on a ramp with seeded noise (has flat, edge and corner windows)
rng = np.random.default_rng(20261018)
syn = np.full((40, 48), 0.25) + 0.002 * np.arange(48)[None, :]
syn[10:30, 14:36] = 0.8
syn = np.clip(syn + 0.01 * rng.standard_normal(syn.shape), 0, 1).astype(np.float32).astype(float)
ix, iy = gradients(syn)
xx, yy, xy = tensor_windowed(ix, iy, 4, 9 / 6)
r6 = tensor_windowed(ix, iy, 6, 1.5)
g["synthetic"] = {
    "w": 48, "h": 40, "image": flat(syn), "ix": flat(ix), "iy": flat(iy),
    "xx": flat(xx), "yy": flat(yy), "xy": flat(xy), "R": flat(response(xx, yy, xy)),
    "chapterR": flat(chapter_R(syn)), "radius6": {"xx": flat(r6[0]), "yy": flat(r6[1]), "xy": flat(r6[2])},
}
ev1, ev2 = eigen(xx, yy, xy)
g["synthetic"]["l1"], g["synthetic"]["l2"] = flat(ev1), flat(ev2)

# --- the camera asset, exactly as the browser decodes it (8-bit gray / 255)
cam = io.imread(ROOT / "src/assets/playground/camera-256.png").astype(np.float64) / 255.0
assert cam.shape == (256, 256)
cix, ciy = gradients(cam)
MARKS = {"A": (200, 20), "B": (80, 52), "C": (30, 80), "default": (89, 103), "legL": (122, 232), "legR": (183, 220)}
cam_out = {"marks": {k: list(v) for k, v in MARKS.items()}, "byWindow": {}}
SCALE = 255.0 ** 2  # lab shows eigenvalues in (gray levels per pixel)^2
for win in (5, 9, 13, 25):
    xx, yy, xy = tensor_windowed(cix, ciy, (win - 1) // 2, win / 6)
    R = response(xx, yy, xy)
    l1, l2 = eigen(xx, yy, xy)
    ry, rx = np.unravel_index(R.argmax(), R.shape)
    entry = {"Rmax": float(R.max()), "l1max": float(l1.max() * SCALE), "argmaxR": [int(rx), int(ry)], "points": {}}
    flat_mask = l1 < 0.05 * l1.max()
    big2 = l2 >= 0.05 * l1.max()
    entry["fractions"] = {"flat": float(flat_mask.mean()), "edge": float(((~flat_mask) & (~big2)).mean()), "corner": float(big2.mean())}
    for name, (x, y) in MARKS.items():
        entry["points"][name] = {
            "l1": float(l1[y, x] * SCALE), "l2": float(l2[y, x] * SCALE), "R": float(R[y, x]),
            "Rrel": float(R[y, x] / R.max()), "ratio": float(l2[y, x] / l1[y, x]),
            "angleDeg": float(np.degrees(0.5 * np.arctan2(2 * xy[y, x], xx[y, x] - yy[y, x]))),
        }
    cam_out["byWindow"][str(win)] = entry
# chapter's own function on the camera, for the agreement test
cR = chapter_R(cam)
cam_out["chapterR"] = {"max": float(cR.max()), "argmax": [int(i) for i in np.unravel_index(cR.argmax(), cR.shape)[::-1]],
                       "at": {k: float(cR[y, x]) for k, (x, y) in MARKS.items()}}
cam_out["maxAbsGradient"] = {"ix": float(np.abs(cix).max() * 255), "iy": float(np.abs(ciy).max() * 255)}
g["camera"] = cam_out

OUT.write_text(json.dumps(g))
print("wrote", OUT, f"{OUT.stat().st_size / 1e3:.0f} kB")
print(json.dumps({k: cam_out["byWindow"][k]["points"]["B"] for k in cam_out["byWindow"]}, indent=1))
