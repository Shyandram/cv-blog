"""Figures for DIP Chapter 10 — Image Segmentation I.

Run:  .venv/bin/python scripts/figures/dip_ch10.py
Writes PNGs to src/assets/dip/ch10/ only. All inputs come from skimage.data or
synthetic arrays.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import data, feature, filters, morphology, segmentation, transform, color, util
from skimage.draw import line as draw_line

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "dip" / "ch10"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})
RNG = np.random.default_rng(10)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{path.name}: {path.stat().st_size / 1024:.0f} KB")


def show(ax, img, title, cmap="gray", **kw):
    ax.imshow(img, cmap=cmap, **kw)
    ax.set_title(title)
    ax.axis("off")


camera = util.img_as_float(data.camera())

# 1. Sobel gradients ---------------------------------------------------------
gx = filters.sobel_v(camera)  # derivative across columns (responds to vertical edges)
gy = filters.sobel_h(camera)  # derivative across rows (responds to horizontal edges)
mag = np.hypot(gx, gy)
fig, axes = plt.subplots(1, 4, figsize=(10, 2.9))
show(axes[0], camera, "Input")
show(axes[1], np.abs(gx), "|g_x| (Sobel)")
show(axes[2], np.abs(gy), "|g_y| (Sobel)")
show(axes[3], mag, "Magnitude M(x, y)")
fig.tight_layout()
save(fig, "sobel-gradient.png")

# 2. LoG zero crossings vs Canny --------------------------------------------
def log_zero_crossings(img, sigma, frac=0.04):
    L = ndi.gaussian_laplace(img, sigma)
    thr = frac * np.abs(L).max()
    zc = np.zeros_like(L, dtype=bool)
    # opposite signs across a neighbor pair, with a large enough jump
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        a = L
        b = np.roll(np.roll(L, -dy, axis=0), -dx, axis=1)
        zc |= (a * b < 0) & (np.abs(a - b) > thr)
    return L, zc


L, zc = log_zero_crossings(camera, 3.0)
canny = feature.canny(camera, sigma=2.0, low_threshold=0.05, high_threshold=0.15)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
lim = np.abs(L).max() * 0.5
show(axes[0], L, "LoG response (sigma = 3)", cmap="RdBu_r", vmin=-lim, vmax=lim)
show(axes[1], zc, "LoG zero crossings")
show(axes[2], canny, "Canny (sigma = 2)")
fig.tight_layout()
save(fig, "log-vs-canny.png")

# ImageCompare pair: input vs Canny edges (same size, no axes)
plt.imsave(OUT / "compare-input.png", camera, cmap="gray", vmin=0, vmax=1)
plt.imsave(OUT / "compare-canny.png", ~canny, cmap="gray")
print("compare-input.png, compare-canny.png written")

# 3. Hough transform ----------------------------------------------------------
H, W = 200, 260
img = np.zeros((H, W), bool)
for (r0, c0, r1, c1) in [(20, 10, 180, 240), (190, 20, 40, 250), (100, 5, 100, 255)]:
    rr, cc = draw_line(r0, c0, r1, c1)
    keep = RNG.random(rr.size) > 0.35  # broken lines
    img[rr[keep], cc[keep]] = True
img |= RNG.random((H, W)) > 0.995  # clutter points
thetas = np.deg2rad(np.arange(-90, 90, 0.5))
acc, angles, dists = transform.hough_line(img, theta=thetas)
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), gridspec_kw={"width_ratios": [1, 1.1, 1]})
show(axes[0], ~img, "Edge points (broken + clutter)")
axes[1].imshow(np.log1p(acc), cmap="magma", aspect="auto", origin="lower",
               extent=[np.rad2deg(angles[0]), np.rad2deg(angles[-1]), dists[0], dists[-1]])
axes[1].set_title("Accumulator log(1 + A)")
axes[1].set_xlabel("theta (degrees)")
axes[1].set_ylabel("rho (pixels)")
show(axes[2], ~img, "Detected lines")
for _, ang, dist in zip(*transform.hough_line_peaks(acc, angles, dists, num_peaks=3)):
    axes[1].plot(np.rad2deg(ang), dist, "o", mfc="none", mec="cyan", ms=12, mew=2)
    x = np.array([0, W - 1])
    if abs(np.sin(ang)) > 1e-6:
        y = (dist - x * np.cos(ang)) / np.sin(ang)
        axes[2].plot(x, y, "-", color="tab:red", lw=1.5)
    else:
        axes[2].axvline(dist, color="tab:red", lw=1.5)
axes[2].set_xlim(0, W - 1)
axes[2].set_ylim(H - 1, 0)
fig.tight_layout()
save(fig, "hough.png")

# 4. Otsu ---------------------------------------------------------------------
coins = data.coins()
hist = np.bincount(coins.ravel(), minlength=256).astype(float)
p = hist / hist.sum()
P1 = np.cumsum(p)
m = np.cumsum(np.arange(256) * p)
mG = m[-1]
with np.errstate(divide="ignore", invalid="ignore"):
    sigmaB2 = (mG * P1 - m) ** 2 / (P1 * (1 - P1))
sigmaB2 = np.nan_to_num(sigmaB2)
k_star = int(np.argmax(sigmaB2))
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), gridspec_kw={"width_ratios": [1, 1.3, 1]})
show(axes[0], coins, "Input (coins)")
ax = axes[1]
ax.bar(np.arange(256), p, width=1.0, color="0.55", label="histogram p(i)")
ax.set_xlabel("intensity i")
ax.set_ylabel("p(i)")
ax2 = ax.twinx()
ax2.plot(np.arange(256), sigmaB2, color="tab:blue", lw=1.6, label="between-class variance")
ax2.set_ylabel("sigma_B^2(k)", color="tab:blue")
ax.axvline(k_star, color="tab:red", ls="--", lw=1.5)
ax.set_title(f"Otsu threshold k* = {k_star}")
show(axes[2], coins > k_star, "Mask: f > k*")
fig.tight_layout()
save(fig, "otsu.png")

# 5. Adaptive thresholding on unevenly lit text -----------------------------
page = data.page()
t_otsu = filters.threshold_otsu(page)
t_local = filters.threshold_local(page, block_size=35, offset=10)
t_sauv = filters.threshold_sauvola(page, window_size=25, k=0.2)
fig, axes = plt.subplots(2, 2, figsize=(10, 5.4))
show(axes[0, 0], page, "Input (uneven lighting)")
show(axes[0, 1], page > t_otsu, f"Global Otsu (T = {t_otsu:.0f})")
show(axes[1, 0], page > t_local, "Local mean, 35x35 block")
show(axes[1, 1], page > t_sauv, "Sauvola, 25x25 window")
fig.tight_layout()
save(fig, "adaptive-threshold.png")

# 6. k-means and SLIC ------------------------------------------------------------
astro = util.img_as_float(data.astronaut())[::2, ::2]
lab = color.rgb2lab(astro)
X = lab.reshape(-1, 3)
k = 5
centers = X[RNG.choice(len(X), k, replace=False)]
for _ in range(20):
    d = ((X[:, None, :] - centers[None]) ** 2).sum(-1)
    lbl = d.argmin(1)
    centers = np.array([X[lbl == j].mean(0) if np.any(lbl == j) else centers[j] for j in range(k)])
km = color.lab2rgb(centers[lbl].reshape(lab.shape))
sp = segmentation.slic(astro, n_segments=300, compactness=10, start_label=1)
sp_mean = color.label2rgb(sp, astro, kind="avg", bg_label=0)
fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.3))
show(axes[0], astro, "Input")
show(axes[1], km, "k-means on Lab colour, k = 5")
show(axes[2], segmentation.mark_boundaries(astro, sp, color=(1, 1, 0)), "SLIC, ~300 superpixels")
show(axes[3], sp_mean, "Each superpixel = its mean")
fig.tight_layout()
save(fig, "kmeans-slic.png")

# 7. Watershed on coins -----------------------------------------------------
elev = filters.sobel(coins.astype(float))
markers = np.zeros_like(coins, dtype=np.int32)
markers[coins < 30] = 1   # sure background
markers[coins > 150] = 2  # sure coin
ws = segmentation.watershed(elev, markers)
filled = morphology.remove_small_objects(ndi.binary_fill_holes(ws == 2), 100)
labels, n_obj = ndi.label(filled)
ws_raw = segmentation.watershed(filters.sobel(filters.gaussian(coins.astype(float), 2)))  # no markers: every regional minimum is a basin
fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.9))
show(axes[0], elev, "Elevation (Sobel)")
show(axes[1], color.label2rgb(ws_raw, bg_label=0), f"No markers: {ws_raw.max()} basins")
mk = np.zeros((*coins.shape, 3))
mk[markers == 1] = (0.2, 0.4, 1.0)
mk[markers == 2] = (1.0, 0.6, 0.1)
show(axes[2], mk, "Markers (bg / object)")
show(axes[3], color.label2rgb(labels, coins, bg_label=0, alpha=0.4), f"Marker watershed: {n_obj} regions")
fig.tight_layout()
save(fig, "watershed.png")

# 8. Motion: difference and accumulative difference images -----------------
H, W, T = 120, 200, 8
bg = filters.gaussian(RNG.random((H, W)), 3) * 0.6 + 0.2
frames = []
for t in range(T):
    f = bg.copy()
    r0, c0 = 45, 20 + 18 * t
    f[r0:r0 + 30, c0:c0 + 30] = 0.95
    f = f + RNG.normal(0, 0.02, f.shape)
    frames.append(f)
diff = np.abs(frames[4] - frames[0]) > 0.2
adi_abs = np.zeros((H, W), int)
adi_pos = np.zeros((H, W), int)
for t in range(1, T):
    d = frames[0] - frames[t]
    adi_abs += np.abs(d) > 0.2
    adi_pos += d > 0.2
fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.4))
show(axes[0], frames[0], "Reference frame t = 0")
show(axes[1], diff, "|f(t=4) - f(0)| > T")
show(axes[2], adi_abs, "Absolute ADI", cmap="viridis")
show(axes[3], adi_pos, "Positive ADI (reference object)", cmap="viridis")
fig.tight_layout()
save(fig, "motion-adi.png")
