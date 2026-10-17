"""Assets for the labs pixel-sampling (zone plate) and color-clipping (coffee).

zoneplate-256.png is SYNTHETIC: I(r) = 0.5 + 0.5 cos(k r^2), where r is the distance in pixels from the image centre
(127.5, 127.5) and k = pi / (2 r_max) with r_max = sqrt(2) * 127.5 (the farthest corner). The local frequency at radius r is
k r / pi cycles per pixel, so it grows linearly from 0 at the centre to exactly 0.5 cycles per pixel (the Nyquist limit of
the 256 x 256 grid) at the corner. Sampling every s-th pixel can therefore show a true pattern only inside the radius
r = pi / (2 s k) and shows false rings beyond it.

coffee-256.png is skimage.data.coffee() (CC0 by the photographer, Rachel Michetti; courtesy of Pikolo Espresso Bar)
reduced from 600 x 400 to 256 x 171 by area averaging.

Run: .venv/bin/python scripts/figures/playground_photo_assets.py
Writes only src/assets/playground/.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import data

OUT = Path(__file__).resolve().parents[2] / "src/assets/playground"
OUT.mkdir(parents=True, exist_ok=True)


def zone_plate(n=256):
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - c, yy - c)
    k = np.pi / (2 * np.hypot(c, c))
    return np.round(255 * (0.5 + 0.5 * np.cos(k * r**2))).astype(np.uint8)


def save(name, a):
    img = Image.fromarray(a)
    img.save(OUT / name, optimize=True)
    print(OUT / name, img.size, f"{(OUT / name).stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    save("zoneplate-256.png", zone_plate())
    coffee = Image.fromarray(data.coffee()).resize((256, 171), Image.BOX)
    save("coffee-256.png", np.array(coffee))
