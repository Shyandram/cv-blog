"""Golden numbers for the `linear-transform` lab (what does a 2 x 2 matrix do to a picture?).

    .venv/bin/python scripts/figures/playground_goldens_warp.py

Writes only scripts/tests/golden/warp.json:

* `svd`: for a fixed list of 2 x 2 matrices (random and special), numpy.linalg.svd singular values, det, matrix_rank and cond.
* `warp`: a small seeded 9 x 11 float32 image and, for several matrices A, the result of scipy.ndimage.affine_transform
  (order=1, cval=0, mode 'constant' and 'grid-constant') for the forward map q = c + A (p - c) about the image centre c, plus the
  same maps of an all-ones image (the alpha channel). It asserts that skimage.transform.warp(order=1, mode='constant') equals
  scipy's 'grid-constant', so the boundary convention of the TypeScript kit is the one of both libraries.
* `asset`: facts about the real asset src/assets/playground/camera-256.png used by the lab text (default shear, swap = anti-transpose,
  coverage of a 0.5 / 0.9 stretch).
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.transform import AffineTransform, warp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scripts/tests/golden/warp.json"
ASSET = ROOT / "src/assets/playground/camera-256.png"


def rot(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def unit(deg):
    t = np.deg2rad(deg)
    return np.array([np.cos(t), np.sin(t)])


def shear(k, deg):
    u = unit(deg)
    up = np.array([-u[1], u[0]])
    return np.eye(2) + k * np.outer(u, up)


def stretch(k, deg):
    u = unit(deg)
    return np.eye(2) + (k - 1) * np.outer(u, u)


def reflect(k, deg):
    u = unit(deg)
    return k * (2 * np.outer(u, u) - np.eye(2))


def project(k, deg):
    u = unit(deg)
    return k * np.outer(u, u)


def clean(m):
    return np.round(m, 12) + 0.0


def affine_rc(A, shape, mode):
    """Warp an image by q = c + A (p - c) with scipy (x = column, y = row, y down, unit 1 pixel)."""
    h, w = shape
    cx, cy = (w - 1) / 2, (h - 1) / 2
    Ai = np.linalg.inv(A)
    # source column = cx + Ai00 (c' - cx) + Ai01 (r' - cy); source row = cy + Ai10 (c' - cx) + Ai11 (r' - cy)
    M = np.array([[Ai[1, 1], Ai[1, 0]], [Ai[0, 1], Ai[0, 0]]])  # acts on (r', c') -> (row, col)
    c = np.array([cy, cx])
    return M, c - M @ c


def scipy_warp(img, A, mode):
    M, off = affine_rc(A, img.shape, mode)
    return ndi.affine_transform(img.astype(np.float64), M, offset=off, order=1, mode=mode, cval=0.0)


def skimage_warp(img, A):
    h, w = img.shape
    cx, cy = (w - 1) / 2, (h - 1) / 2
    Ai = np.linalg.inv(A)
    T = np.eye(3)
    T[:2, :2] = Ai
    T[:2, 2] = np.array([cx, cy]) - Ai @ np.array([cx, cy])
    return warp(img.astype(np.float64), AffineTransform(matrix=T), order=1, mode="constant", cval=0.0, preserve_range=True)


def svd_cases():
    rng = np.random.default_rng(7)
    mats = [m for m in [
        np.eye(2), 1.5 * rot(30), np.array([[0.0, 1.0], [1.0, 0.0]]), shear(1, 30), stretch(0.5, 0), project(1, 30),
        np.array([[1.0, 2.0], [0.5, 1.0]]), np.zeros((2, 2)), np.diag([-1.0, 1.0]), np.diag([3.0, -2.0]), np.array([[3.0, 0.0], [4.0, 5.0]]),
        np.array([[2.0, 0.6], [0.0, 1.0]]), reflect(1, 45), stretch(-1, 90), shear(2, -60),
    ]]
    mats += [rng.normal(size=(2, 2)) for _ in range(25)]
    out = []
    for A in mats:
        s = np.linalg.svd(A, compute_uv=False)
        full = np.linalg.matrix_rank(A) == 2
        out.append({"A": A.reshape(4).tolist(), "s": s.tolist(), "det": float(np.linalg.det(A)), "rank": int(np.linalg.matrix_rank(A)),
                    "cond": float(np.linalg.cond(A)) if full else None})
    return out


def warp_cases():
    rng = np.random.default_rng(3)
    img = (rng.random((9, 11)) * 0.9 + 0.05).astype(np.float32)
    ones = np.ones_like(img)
    cases = {
        "identity": np.eye(2), "rotate30": rot(30), "rotate90": clean(rot(90)), "mirror": np.diag([-1.0, 1.0]), "swap": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "shear": clean(shear(1, 30)), "stretch": clean(stretch(1.7, 40)), "shrink": 0.5 * np.eye(2), "general": np.array([[0.9, -0.4], [0.3, 1.25]]),
        "reflect": clean(reflect(1.2, 20)), "negstretch": clean(stretch(-0.8, 15)),
    }
    out = []
    for name, A in cases.items():
        rec = {"name": name, "A": A.reshape(4).tolist()}
        for mode, key in (("constant", "constant"), ("grid-constant", "grid")):
            rec[key] = scipy_warp(img, A, mode).tolist()
            rec["alpha_" + key] = scipy_warp(ones, A, mode).tolist()
        sk = skimage_warp(img, A)
        assert np.allclose(sk, np.array(rec["grid"]), atol=1e-6), f"skimage.warp(mode='constant') should equal scipy 'grid-constant' for {name}"
        out.append(rec)
    return {"w": img.shape[1], "h": img.shape[0], "image": img.reshape(-1).tolist(), "cases": out}


def asset_facts():
    img = np.asarray(Image.open(ASSET).convert("L"), dtype=np.float32) / 255
    assert img.shape == (256, 256)
    n = 256
    # Plane unit = 64 pixels, y up, origin at the image centre (127.5, 127.5); pixel space matrix = [[a, -b], [-c, d]].
    def pix(A):
        return np.array([[A[0, 0], -A[0, 1]], [-A[1, 0], A[1, 1]]])

    def run(A, mode="grid-constant"):
        return scipy_warp(img, pix(A), mode)

    swap = run(np.array([[0.0, 1.0], [1.0, 0.0]]))
    anti = img[::-1, ::-1].T  # out[r, c] = img[255 - c, 255 - r]
    assert np.allclose(swap, anti, atol=1e-6), "swap matrix should give the anti-transpose of the photo"
    mirror = run(np.diag([-1.0, 1.0]))
    assert np.allclose(mirror, img[:, ::-1], atol=1e-6)
    rot180 = run(np.diag([-1.0, -1.0]))
    assert np.allclose(rot180, img[::-1, ::-1], atol=1e-6)
    default = clean(shear(1, 30))
    s = np.linalg.svd(default, compute_uv=False)
    half = clean(np.diag([0.5, 0.9]))
    ones = np.ones((n, n))
    cover = float(scipy_warp(ones, pix(half), "constant").sum() / (n * n))
    return {
        "defaultShear": {"A": default.reshape(4).tolist(), "det": float(np.linalg.det(default)), "s": s.tolist(), "cond": float(s[0] / s[1])},
        "coverageHalfNine": {"A": half.reshape(4).tolist(), "fraction": cover, "absDet": abs(float(np.linalg.det(half)))},
        "columnMeanRow127": float(img[:, 127].mean()),
    }


def main():
    data = {"svd": svd_cases(), "warp": warp_cases(), "asset": asset_facts()}
    OUT.write_text(json.dumps(data))
    print("wrote", OUT, f"{OUT.stat().st_size / 1024:.0f} KB")
    print(json.dumps(data["asset"], indent=1))


if __name__ == "__main__":
    main()
