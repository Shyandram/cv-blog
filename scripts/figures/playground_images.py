"""Small demo images for the image labs in the Playground, cut from skimage.data (no downloads).

Licences (from the scikit-image docstrings): camera is CC0 by its photographer, Lav Varshney; coins comes from the
Brooklyn Museum collection with no known copyright restrictions; astronaut is a NASA photograph in the public domain;
page is a scan shipped with scikit-image.

Run: .venv/bin/python scripts/figures/playground_images.py
Writes only src/assets/playground/.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import data

OUT = Path(__file__).resolve().parents[2] / "src/assets/playground"
OUT.mkdir(parents=True, exist_ok=True)


def box(a, f):
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    a = a[:h, :w].astype(float)
    return a.reshape(h // f, f, w // f, f, *a.shape[2:]).mean(axis=(1, 3))


def save(name, a):
    img = Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8))
    img.save(OUT / name, optimize=True)
    print(OUT / name, img.size, f"{(OUT / name).stat().st_size / 1e3:.0f} kB")


cam = data.camera()
save("camera-256.png", box(cam, 2))                               # 256 x 256 gray
save("camera-crop-128.png", box(cam[40:296, 100:356], 2))         # 128 x 128 gray, strong edges
save("page.png", data.page())                                     # 191 x 384 gray scan, uneven lighting
save("coins.png", data.coins())                                   # 303 x 384 gray
astro = data.astronaut()
save("astronaut-256.png", box(astro, 2))                          # 256 x 256 RGB


# --- A synthetic scanned page with an exactly known text mask, for the thresholding lab.
# Supersampled text gives anti-aliased ink coverage; the truth mask is coverage > 0.5. Lighting is applied in the browser.
from PIL import ImageDraw, ImageFont
import matplotlib

W, H, SS = 384, 192, 4
font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
lines = [
    "Choosing one number",
    "A threshold turns a photograph into a mask: every pixel",
    "darker than the cutoff becomes ink, everything else paper.",
    "It works well while ink and paper stay far apart in",
    "brightness. Uneven light changes that. Where the lamp",
    "is weak, the paper itself becomes as dark as ink was",
    "somewhere else, and one global cutoff has to give up",
    "either the dim side or the bright side of the page.",
    "A local cutoff looks only at a neighborhood and so it",
    "follows the light across the page instead of ignoring it.",
]
canvas = Image.new("L", (W * SS, H * SS), 0)
draw = ImageDraw.Draw(canvas)
y = 6 * SS
for i, line in enumerate(lines):
    size = (15 if i == 0 else 11) * SS
    draw.text((10 * SS, y), line, fill=255, font=ImageFont.truetype(str(font_path), size))
    y += int((22 if i == 0 else 17.5) * SS)
cover = box(np.asarray(canvas, float) / 255, SS)                  # ink coverage in [0, 1]
paper, ink = 0.88, 0.22
doc = paper * (1 - cover) + ink * cover
from scipy import ndimage as ndi
doc = ndi.gaussian_filter(doc, 0.5) + 0.015 * np.random.default_rng(3).standard_normal(doc.shape)
save("doc.png", np.clip(doc, 0, 1) * 255)
save("doc-truth.png", (cover > 0.5) * 255)
