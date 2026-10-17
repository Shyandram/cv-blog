"""Asset for the lab contour-energy (IP 11): one coin of skimage.data.coins() with its neighbours.

coin-crop-144.png is the 144 x 144 window of skimage.data.coins() with rows 121..264 and columns 141..284 (row 121, column 141 is
the top-left pixel). It is centred on one coin (the coin at row 193, column 213 of the full image) and also contains parts of the
eight coins around it, which is what makes the snake's local minima visible.

coins comes from the Brooklyn Museum collection, no known copyright restrictions (skimage.data.coins docstring).

Run: .venv/bin/python scripts/figures/playground_snake_assets.py
Writes only src/assets/playground/coin-crop-144.png.
"""
from pathlib import Path

from PIL import Image
from skimage import data

OUT = Path(__file__).resolve().parents[2] / "src/assets/playground"
OUT.mkdir(parents=True, exist_ok=True)
Y0, X0, SIZE = 121, 141, 144

if __name__ == "__main__":
    crop = data.coins()[Y0:Y0 + SIZE, X0:X0 + SIZE]
    assert crop.shape == (SIZE, SIZE)
    path = OUT / "coin-crop-144.png"
    Image.fromarray(crop).save(path, optimize=True)
    print(path, crop.shape, f"{path.stat().st_size / 1e3:.1f} kB")
