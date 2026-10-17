"""Asset for the IP 09 lab (morphology-shapes): a binary mask of the skimage `coins` image with known clutter.

Clean mask: smooth the photograph (Gaussian, sigma 2), subtract a wide Gaussian background (sigma 50) so the bright
vignette in the top-left corner does not count as foreground, then threshold with Otsu's method. That gives exactly 24
solid coins and no other component. Clutter is then added with a fixed seed, so every problem has a known answer:

  * 40 specks (isolated blobs: 10 each of radius 0, 1, 2 and 3 pixels) in the background,
  * 10 thin scratches in the background (5 horizontal, 5 vertical, 1 or 2 pixels thick, 26 to 40 pixels long),
  * 18 round holes (6 each of radius 1, 2 and 3 pixels) inside the coins.

Licence of the photograph (skimage.data.coins docstring): Brooklyn Museum collection, no known copyright restrictions.

Run: .venv/bin/python scripts/figures/playground_restore_assets.py
Writes only src/assets/playground/coins-mask.png.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import data, filters

OUT = Path(__file__).resolve().parents[2] / "src/assets/playground"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261017


def clean_mask():
    c = data.coins().astype(float) / 255
    d = ndi.gaussian_filter(c, 2) - ndi.gaussian_filter(c, 50)
    return d > filters.threshold_otsu(d)


def build():
    rng = np.random.default_rng(SEED)
    clean = clean_mask()
    h, w = clean.shape
    yy, xx = np.mgrid[0:h, 0:w]
    inside = ndi.distance_transform_edt(clean)      # distance to the background, for coin pixels
    outside = ndi.distance_transform_edt(~clean)    # distance to the nearest coin, for background pixels
    mask = clean.copy()
    taken = np.zeros_like(clean)                    # footprints already used, to keep every piece isolated

    def free(shape_mask, margin):
        grown = ndi.binary_dilation(shape_mask, iterations=margin)
        return not (grown & taken).any()

    def place(shape_mask, margin=6):
        taken[ndi.binary_dilation(shape_mask, iterations=margin)] = True

    holes = []
    cand = np.argwhere(inside >= 10)
    radii = [1, 2, 3] * 6
    rng.shuffle(radii)
    for r in radii:
        for _ in range(10_000):
            y, x = cand[rng.integers(len(cand))]
            disk = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
            if free(disk, 6):
                mask[disk] = False
                place(disk, 6)
                holes.append((int(y), int(x), r))
                break
        else:
            raise RuntimeError("could not place a hole")
    specks = []
    cand = np.argwhere((outside >= 10) & (yy >= 8) & (yy < h - 8) & (xx >= 8) & (xx < w - 8))
    radii = [0, 1, 2, 3] * 10
    rng.shuffle(radii)
    for r in radii:
        for _ in range(10_000):
            y, x = cand[rng.integers(len(cand))]
            disk = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
            if free(disk, 6) and outside[disk].min() >= 8:
                mask[disk] = True
                place(disk, 6)
                specks.append((int(y), int(x), r))
                break
        else:
            raise RuntimeError("could not place a speck")
    scratches = []
    for k in range(10):
        horizontal = k < 5
        length, thick = int(rng.integers(26, 41)), 1 + k % 2
        for _ in range(10_000):
            y, x = cand[rng.integers(len(cand))]
            box = np.zeros_like(clean)
            if horizontal:
                box[y:y + thick, x:x + length] = True
            else:
                box[y:y + length, x:x + thick] = True
            if box.sum() == (box & (yy < h) & (xx < w)).sum() and (box & ((outside >= 8) & (yy >= 6) & (yy < h - 6) & (xx >= 6) & (xx < w - 6))).sum() == box.sum() and free(box, 6):
                mask[box] = True
                place(box, 6)
                scratches.append(("h" if horizontal else "v", int(y), int(x), length, thick))
                break
        else:
            raise RuntimeError("could not place a scratch")
    return clean, mask, holes, specks, scratches


if __name__ == "__main__":
    clean, mask, holes, specks, scratches = build()
    Image.fromarray((mask * 255).astype(np.uint8)).save(OUT / "coins-mask.png", optimize=True)
    st8 = np.ones((3, 3), int)
    print("clean components", ndi.label(clean, st8)[1], "noisy components", ndi.label(mask, st8)[1])
    print("holes", len(holes), "specks", len(specks), "scratches", len(scratches))
    print(OUT / "coins-mask.png", mask.shape, f"{(OUT / 'coins-mask.png').stat().st_size / 1e3:.1f} kB")
