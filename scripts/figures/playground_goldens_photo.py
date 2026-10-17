"""Reference numbers for the labs array-dtype, pixel-sampling and color-clipping, computed with NumPy, SciPy, OpenCV and
scikit-image directly on the exact PNG assets in src/assets/playground/ (independent of the TypeScript pipeline).

Every number quoted in those labs' text is asserted in scripts/tests/lab-photo.test.ts against this file, and the TypeScript
pipelines are compared with it on the real assets.

Run: .venv/bin/python scripts/figures/playground_goldens_photo.py
Writes only scripts/tests/golden/photo.json.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import data
from skimage.color import rgb2hsv

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "src/assets/playground"
OUT = ROOT / "scripts/tests/golden/photo.json"
EPS = 1e-6
HUE_MIN = 0.1
SLACK = 1e-6   # a saturation of exactly 0.1 in decimal must not flip with float rounding


def load(name):
    return np.array(Image.open(ASSETS / name))


def psnr255(a, b):
    mse = ((a.astype(float) - b.astype(float)) ** 2).mean()
    return float("inf") if mse == 0 else float(10 * np.log10(255**2 / mse))


# ---------------------------------------------------------------- array-dtype
def dtype_facts():
    cam = load("camera-256.png")
    full = data.camera()
    out = {"n": int(cam.size), "full_n": int(full.size), "full_ge196": int((full >= 196).sum())}
    out["ge196"] = int((cam >= 196).sum())
    wrapped = cam + np.uint8(60)                      # NumPy uint8 wraparound, as in the chapter
    clipped = np.clip(cam.astype(np.int16) + 60, 0, 255).astype(np.uint8)
    out["cv2_add_equals_clip"] = bool(np.array_equal(cv2.add(cam, 60 * np.ones_like(cam)), clipped))
    out["float_clip_equals_clip"] = bool(np.array_equal(np.clip(np.round(cam.astype(np.float32) + 60), 0, 255).astype(np.uint8), clipped))
    out["wrapped_darker"] = int((wrapped < cam).sum())
    d = wrapped.astype(int) - cam.astype(int)
    out["wrapped_delta_values"] = sorted(set(int(v) for v in d[cam >= 196]))
    out["other_delta_values"] = sorted(set(int(v) for v in d[cam < 196]))
    out["mean_in"], out["mean_wrap"], out["mean_clip"] = float(cam.mean()), float(wrapped.mean()), float(clipped.mean())
    out["psnr_wrap_vs_clip"] = psnr255(wrapped, clipped)
    out["brighter_wrap"] = int((wrapped > cam).sum())
    neg = (cam.astype(np.int16) - 100)
    out["lt100"] = int((cam < 100).sum())
    out["mean_wrap_m100"] = float((neg % 256).astype(np.uint8).mean())
    out["mean_clip_m100"] = float(np.clip(neg, 0, 255).mean())
    return out


# ---------------------------------------------------------------- pixel-sampling
def quant(v, bits):
    step = 2 ** (8 - bits)
    return np.floor(v / step) * step + step // 2


def sample(img, s, bits, pre):
    n0, n1 = img.shape
    im = img.astype(float)
    if pre and s > 1:
        im = gaussian_filter(im / 255.0, 0.5 * s, mode="reflect", truncate=4.0) * 255
    im = np.clip(np.round(im), 0, 255)
    sm = quant(im[::s, ::s], bits)
    up = np.repeat(np.repeat(sm, s, 0), s, 1)[:n0, :n1]
    return up, sm


def false_contrast(sm, s, k, c):
    rc = np.pi / (2 * s * k)
    ys, xs = np.mgrid[0 : sm.shape[0], 0 : sm.shape[1]]
    reg = np.hypot(xs * s - c, ys * s - c) > rc
    return float((sm[reg] / 255.0).std()) if reg.any() else None


def sampling_facts():
    cam, zp = load("camera-256.png"), load("zoneplate-256.png")
    c = 127.5
    k = np.pi / (2 * np.hypot(c, c))
    out = {"k": float(k), "alias_radius": {str(s): float(np.pi / (2 * s * k)) for s in range(1, 9)}, "cases": []}
    for s in (1, 2, 3, 4, 8):
        for bits in (8, 4, 1):
            for pre in (0, 1):
                a, _ = sample(cam, s, bits, pre)
                b, sm = sample(zp, s, bits, pre)
                out["cases"].append({"stride": s, "bits": bits, "pre": pre, "psnr_cam": psnr255(cam, a), "psnr_zp": psnr255(zp, b), "false": false_contrast(sm, s, k, c)})
    worse = []
    for s in range(2, 9):
        for bits in range(1, 9):
            for name, img in (("camera", cam), ("zone", zp)):
                if psnr255(img, sample(img, s, bits, 1)[0]) < psnr255(img, sample(img, s, bits, 0)[0]):
                    worse.append([s, bits, name])
    out["blur_lowers_psnr"] = worse
    out["kept_stride4"] = int(np.ceil(256 / 4) ** 2)
    return out


# ---------------------------------------------------------------- color-clipping
def luma(x):
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def apply_gain(x, g, method):
    y = x * g
    m = y.max(-1)
    if method == "clip":
        k = np.ones_like(m)
    elif method == "scale":
        k = np.where(m > 1, 1 / np.maximum(m, 1e-12), 1.0)
    else:
        lum = luma(y)
        k = np.where(lum > 0, (1 + lum / g**2) / (1 + lum), 1.0)
    z = y * k[..., None]
    return np.minimum(z, 1), float((z.max(-1) > 1 + EPS).mean())


def hue_report(x, o, g):
    ov = x.max(-1) * g > 1 + EPS
    h0, h1 = rgb2hsv(x), rgb2hsv(o)
    d0 = (h0[..., 1] >= HUE_MIN - SLACK) & (h0[..., 2] >= HUE_MIN - SLACK)
    d1 = (h1[..., 1] >= HUE_MIN - SLACK) & (h1[..., 2] >= HUE_MIN - SLACK)
    col = ov & d0
    comp = col & d1
    dh = np.abs(h1[..., 0] - h0[..., 0]) * 360
    dh = np.minimum(dh, 360 - dh)
    return {
        "overflow": float(ov.mean()), "coloured": int(col.sum()), "mean_dh": float(dh[comp].mean()) if comp.any() else None,
        "lost": float((col & ~d1).sum() / col.sum()) if col.any() else None, "sat_before": float(h0[..., 1][col].mean()) if col.any() else None, "sat_after": float(h1[..., 1][col].mean()) if col.any() else None,
    }


def color_facts():
    img = load("coffee-256.png").astype(float) / 255
    out = {"shape": list(img.shape), "cases": []}
    for method in ("clip", "scale", "tone"):
        for g in (1.0, 1.5, 2.0, 3.0, 4.0):
            o, clipped = apply_gain(img, g, method)
            out["cases"].append({"method": method, "gain": g, "clipped_after": clipped, **hue_report(img, o, g)})
    orange = np.array([[[200, 100, 40]]], float) / 255
    ex = {}
    for method in ("clip", "scale"):
        o, _ = apply_gain(orange, 2.0, method)
        h0, h1 = rgb2hsv(orange)[0, 0], rgb2hsv(o)[0, 0]
        ex[method] = {"rgb": [float(v) for v in o[0, 0] * 255], "hue_before": float(h0[0] * 360), "hue_after": float(h1[0] * 360), "sat_before": float(h0[1]), "sat_after": float(h1[1])}
    out["orange"] = ex
    rng = np.random.default_rng(7)
    rgb = rng.random((200, 3))
    rgb[:10] = [[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [0.8, 0.4, 0.15]]
    hsv = rgb2hsv(rgb.reshape(1, -1, 3))[0]
    out["hsv"] = [{"rgb": r.tolist(), "hsv": [float(h[0] * 360), float(h[1]), float(h[2])]} for r, h in zip(rgb, hsv)]
    return out


def clean(v):
    """JSON has no inf or nan: an infinite PSNR (identical images) or an empty statistic is written as null."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    return None if isinstance(v, float) and not np.isfinite(v) else v


if __name__ == "__main__":
    facts = {"dtype": dtype_facts(), "sampling": sampling_facts(), "color": color_facts()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(clean(facts), indent=1, allow_nan=False))
    print(OUT, OUT.stat().st_size, "bytes")
