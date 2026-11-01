"""Assets for the Playground lab image-fusion (DLCV 05B): a multi-focus pair with an exactly known answer, cut from skimage.data.

Same recipe as the multi-focus snippet of the chapter, at half the size so it recomputes quickly in a browser:
the camera test image (box-averaged to 256 x 256) is the all-in-focus truth; inside a disk shot A keeps the truth and shot B is
Gaussian-blurred, outside the disk it is the other way round; each shot gets Gaussian noise of standard deviation 0.01 (seed 0),
and is saved as 8-bit PNG. The disk mask is the true decision map (white = take A).

Camera is CC0 by its photographer, Lav Varshney (scikit-image docstring). No downloads.

Run: .venv/bin/python scripts/figures/playground_image_fusion.py
Writes only src/assets/playground/image-fusion-*.png.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import data

OUT = Path(__file__).resolve().parents[2] / "src/assets/playground"
OUT.mkdir(parents=True, exist_ok=True)

SIZE, CENTER, RADIUS, BLUR_SIGMA, NOISE = 256, (100, 125), 75, 2.0, 0.01  # CENTER = (row, column)


def box2(a):
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).mean(axis=(1, 3))


def save(name, a):
    img = Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8))
    img.save(OUT / name, optimize=True)
    print(OUT / name, img.size, f"{(OUT / name).stat().st_size / 1e3:.0f} kB")


def build():
    truth = np.round(box2(data.camera().astype(float))) / 255.0            # 256 x 256, 8-bit levels, in [0, 1]
    yy, xx = np.mgrid[:SIZE, :SIZE]
    near = (yy - CENTER[0]) ** 2 + (xx - CENTER[1]) ** 2 < RADIUS ** 2     # in focus in shot A
    blurred = ndi.gaussian_filter(truth, BLUR_SIGMA)
    rng = np.random.default_rng(0)
    a = np.where(near, truth, blurred) + NOISE * rng.standard_normal(truth.shape)
    b = np.where(near, blurred, truth) + NOISE * rng.standard_normal(truth.shape)
    return truth, near, a, b


if __name__ == "__main__":
    truth, near, a, b = build()
    save("image-fusion-truth.png", truth * 255)
    save("image-fusion-a.png", np.clip(a, 0, 1) * 255)
    save("image-fusion-b.png", np.clip(b, 0, 1) * 255)
    save("image-fusion-focus.png", near * 255)
