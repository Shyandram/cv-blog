"""Golden values for the in-browser image kit (src/lib/imagekit), computed with NumPy, SciPy and scikit-image.

The TypeScript tests (scripts/tests/imagekit.test.ts) load scripts/tests/golden/imagekit.json and check that the
browser implementations reproduce these numbers, so the values a demo shows match the Python used in the chapters.

Run: .venv/bin/python scripts/figures/playground_goldens.py
Writes only scripts/tests/golden/imagekit.json. Inputs are small seeded arrays; nothing is downloaded.
"""
import json
from pathlib import Path

import numpy as np
import pywt
from scipy import fft, ndimage as ndi
from skimage import filters, metrics

OUT = Path(__file__).resolve().parents[2] / "scripts/tests/golden/imagekit.json"
rng = np.random.default_rng(20261017)


def flat(a):
    return [float(x) for x in np.asarray(a, dtype=float).ravel()]


def smooth_image(h, w, seed):
    """A test image with edges and texture: two blobs, a ramp and noise, scaled to [0, 1]."""
    r = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = 0.3 + 0.4 * x / w + 0.3 * ((x - w * 0.6) ** 2 + (y - h * 0.4) ** 2 < (min(h, w) * 0.25) ** 2) + 0.05 * r.standard_normal((h, w))
    return np.clip(img, 0, 1)


g = {}

# --- FFT: radial low-pass filters on a 32x32 image
img = smooth_image(32, 32, 1)
F = np.fft.fft2(img)
fy = np.fft.fftfreq(32, d=1 / 32)[:, None]
fx = np.fft.fftfreq(32, d=1 / 32)[None, :]
D = np.hypot(fx, fy)
fftcase = {"w": 32, "h": 32, "input": flat(img), "fft_re_sample": flat(F.real.ravel()[:64]), "fft_im_sample": flat(F.imag.ravel()[:64]), "filters": {}}
for name, H in {
    "ideal": (D <= 6).astype(float),
    "butterworth": 1 / (1 + (D / 6) ** 4),
    "gaussian": np.exp(-(D ** 2) / (2 * 6 ** 2)),
}.items():
    fftcase["filters"][name] = {"d0": 6, "order": 2, "output": flat(np.fft.ifft2(F * H).real)}
g["fft"] = fftcase

# --- DCT 8x8 block, and a 16x16 image with uniform quantization
blk = smooth_image(8, 8, 2)
g["dct8"] = {"input": flat(blk), "output": flat(fft.dctn(blk, norm="ortho"))}
img16 = smooth_image(16, 16, 3)
step = 0.2
coef = np.zeros_like(img16)
rec = np.zeros_like(img16)
nz = 0
for i in range(0, 16, 8):
    for j in range(0, 16, 8):
        c = fft.dctn(img16[i:i + 8, j:j + 8], norm="ortho")
        coef[i:i + 8, j:j + 8] = c
        cq = np.round(c / step) * step
        nz += np.count_nonzero(cq)
        rec[i:i + 8, j:j + 8] = fft.idctn(cq, norm="ortho")
g["blockdct"] = {"w": 16, "h": 16, "input": flat(img16), "coef": flat(coef), "step": step, "nonzero": nz / img16.size, "recon": flat(rec)}

# --- Haar, two levels, against PyWavelets
img_h = smooth_image(16, 16, 4)
coeffs = pywt.wavedec2(img_h, "haar", level=2, mode="periodization")
arr, _ = pywt.coeffs_to_array(coeffs)
g["haar"] = {"w": 16, "h": 16, "levels": 2, "input": flat(img_h), "pywt_array": flat(arr)}

# --- Gaussian blur against scipy (default truncate 4, mode reflect), incl. a sigma wide relative to the image
img_b = smooth_image(20, 24, 5)
g["gaussian"] = {"w": 24, "h": 20, "input": flat(img_b), "cases": [{"sigma": s, "output": flat(ndi.gaussian_filter(img_b, s, mode="reflect"))} for s in (0.8, 1.7, 5.17)]}

# --- median 3x3 and Sobel against scipy (edge replication == mode 'nearest')
img_m = smooth_image(18, 22, 6)
g["median3"] = {"w": 22, "h": 18, "input": flat(img_m), "output": flat(ndi.median_filter(img_m, size=3, mode="nearest"))}
g["sobel"] = {"w": 22, "h": 18, "input": flat(img_m), "ix": flat(ndi.sobel(img_m, axis=1, mode="nearest")), "iy": flat(ndi.sobel(img_m, axis=0, mode="nearest"))}

# --- thresholds against scikit-image
bi = np.clip(np.where(rng.random((40, 40)) < 0.4, 0.25, 0.7) + 0.08 * rng.standard_normal((40, 40)), 0, 1)
g["otsu"] = {"input": flat(bi), "threshold": float(filters.threshold_otsu(bi))}
img_t = smooth_image(30, 45, 7)
g["local"] = {"w": 45, "h": 30, "input": flat(img_t), "cases": [{"block": b, "offset": o, "threshold": flat(filters.threshold_local(img_t, b, method="gaussian", offset=o))} for b, o in ((15, 0.0), (35, 0.02))]}

# --- metrics
a, b = smooth_image(12, 12, 8), smooth_image(12, 12, 9)
ma, mb = a > 0.5, b > 0.5
g["metrics"] = {"a": flat(a), "b": flat(b), "psnr": float(metrics.peak_signal_noise_ratio(a, b, data_range=1)), "mse": float(metrics.mean_squared_error(a, b)),
                "iou": float((ma & mb).sum() / (ma | mb).sum()), "mask_a": [int(v) for v in ma.ravel()], "mask_b": [int(v) for v in mb.ravel()]}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(g, separators=(",", ":")))
print(OUT, f"{OUT.stat().st_size / 1e3:.0f} kB")
