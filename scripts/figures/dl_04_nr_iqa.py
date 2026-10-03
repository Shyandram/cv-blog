"""No-reference IQA figure for DL 04 - Deep Image Restoration.

Run:  .venv/bin/python scripts/figures/dl_04_nr_iqa.py
Writes src/assets/deep-learning/dl-04-image-restoration/nr-iqa-mscn.png only.
Input is skimage.data.camera(); nothing is downloaded.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.special import gamma  # noqa: E402
from skimage import data, img_as_float  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "deep-learning" / "dl-04-image-restoration"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white"})


def mscn(img, sigma=7 / 6, C=1.0):
    mu = gaussian_filter(img, sigma)
    sd = np.sqrt(np.abs(gaussian_filter(img * img, sigma) - mu * mu))
    return (img - mu) / (sd + C)


def ggd_fit(x):
    alphas = np.arange(0.2, 10.0, 0.001)
    r_theory = gamma(1 / alphas) * gamma(3 / alphas) / gamma(2 / alphas) ** 2
    r_data = np.mean(x**2) / np.mean(np.abs(x)) ** 2
    return alphas[np.argmin((r_theory - r_data) ** 2)], np.mean(x**2)


def ggd_pdf(t, alpha, var):
    beta = np.sqrt(var * gamma(1 / alpha) / gamma(3 / alpha))
    return alpha / (2 * beta * gamma(1 / alpha)) * np.exp(-((np.abs(t) / beta) ** alpha))


rng = np.random.default_rng(0)
clean = img_as_float(data.camera()) * 255
versions = {
    "Clean": clean,
    "Gaussian blur (sigma = 2)": gaussian_filter(clean, 2),
    "White noise (sigma = 20)": np.clip(clean + rng.normal(0, 20, clean.shape), 0, 255),
    "Over-sharpened (unsharp x2)": np.clip(clean + 2.0 * (clean - gaussian_filter(clean, 2)), 0, 255),
}

fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.4), gridspec_kw={"height_ratios": [1, 0.9]})
t = np.linspace(-3, 3, 400)
for k, (name, img) in enumerate(versions.items()):
    m = mscn(img)
    alpha, var = ggd_fit(m)
    ax = axes[0, k]
    ax.imshow(img[120:376, 150:406], cmap="gray", vmin=0, vmax=255)
    ax.set_title(name, fontsize=10)
    ax.axis("off")
    ax = axes[1, k]
    ax.hist(m.ravel(), bins=121, range=(-3, 3), density=True, color="#9db4d6", label="MSCN histogram")
    ax.plot(t, ggd_pdf(t, alpha, var), color="#c0392b", lw=1.6, label="fitted GGD")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 30)
    ax.set_xlabel("MSCN coefficient")
    ax.set_title(f"GGD shape = {alpha:.2f}\nMSCN variance = {var:.2f}", fontsize=10)
    if k == 0:
        ax.set_ylabel("density (log scale)")
        ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "nr-iqa-mscn.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "nr-iqa-mscn.png")
