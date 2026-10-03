"""No-reference metric figure for DL 05 - Low-Light Image Enhancement.

Run from the repo root:
    .venv/bin/python scripts/figures/dl_05_nr_metrics.py
Writes nr-metrics.png to src/assets/deep-learning/dl-05-low-light-enhancement/.
Uses only skimage.data.coffee() and the synthetic low-light model of the tutorial.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, data, exposure, img_as_float

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-05-low-light-enhancement"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white"})

rng = np.random.default_rng(0)
clean = img_as_float(data.coffee())


def synth_low_light(x, k=0.05, peak=2000.0, read_std=8e-4, gamma=2.2):
    lin = k * x ** gamma
    lin = rng.poisson(lin * peak) / peak
    lin = lin + rng.normal(0, read_std, lin.shape)
    out = np.clip(lin, 0, 1) ** (1 / gamma)
    return np.round(out * 255) / 255


def loe(ref, out, step=8):
    L, Le = (im.max(axis=2)[::step, ::step].ravel() for im in (ref, out))
    U, Ue = L[:, None] >= L[None, :], Le[:, None] >= Le[None, :]
    return (U ^ Ue).sum(axis=1).mean()


def colorfulness(im):
    R, G, B = (255 * im[..., c] for c in range(3))
    rg, yb = R - G, 0.5 * (R + G) - B
    return np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())


def noise_sigma(im):
    Y = 255 * (0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2])
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    H, W = Y.shape
    return np.sqrt(np.pi / 2) / (6 * (W - 2) * (H - 2)) * np.abs(ndi.convolve(Y, M)[1:-1, 1:-1]).sum()


def main():
    dark = synth_low_light(clean)
    lab = color.rgb2lab(dark)
    L = lab[..., 0] / 100
    he = lab.copy(); he[..., 0] = 100 * exposure.equalize_hist(L)
    cl = lab.copy(); cl[..., 0] = 100 * exposure.equalize_adapthist(L, clip_limit=0.01)
    outs = [
        ("Dark input", dark),
        ("Gain x20 (linear)", np.clip(20 * dark ** 2.2, 0, 1) ** (1 / 2.2)),
        ("Gamma 0.4", dark ** 0.4),
        ("HE on L*", np.clip(color.lab2rgb(he), 0, 1)),
        ("CLAHE on L*", np.clip(color.lab2rgb(cl), 0, 1)),
        ("Clean reference", clean),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
    for ax, (name, im) in zip(axes.ravel(), outs):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(
            f"mean {im.mean():.2f} | LOE {loe(dark, im):.0f}\n"
            f"colorfulness {colorfulness(im):.0f} | noise sigma {noise_sigma(im):.1f}",
            fontsize=10,
        )
    fig.suptitle("No-reference measures vs. the dark input (LOE on an 8-pixel-stride grid, m = 3750)", fontsize=11)
    fig.tight_layout()
    path = OUT / "nr-metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    im = Image.open(path).convert("RGB")
    if im.width > 1600:
        im = im.resize((1600, round(im.height * 1600 / im.width)), Image.LANCZOS)
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG).save(path, optimize=True)
    print(f"nr-metrics.png: {path.stat().st_size / 1024:.0f} KB, {im.size}")


if __name__ == "__main__":
    main()
