"""Figures for DIP Chapter 5 - Image Restoration and Reconstruction.

Run:  .venv/bin/python scripts/figures/dip_ch05.py
Writes PNGs to src/assets/dip/ch05/ only. All inputs come from skimage.data
or synthetic arrays; nothing is downloaded.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy import stats  # noqa: E402
from PIL import Image  # noqa: E402
from skimage import data, img_as_float, transform  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "dip" / "ch05"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white"})
rng = np.random.default_rng(5)


def save(fig, name, gray=False):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    # Re-encode with an adaptive 256-colour palette: noisy panels compress badly
    # as RGBA, and the palette keeps every figure well under 1 MB.
    im = Image.open(path).convert("RGB")
    im = im.convert("L") if gray else im.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    im.save(path, optimize=True)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def show(ax, img, title, vmin=0, vmax=1):
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def psnr(ref, x):
    mse = np.mean((np.clip(x, 0, 1) - ref) ** 2)
    return 10 * np.log10(1.0 / mse)


# ---------------------------------------------------------------- helpers
def contraharmonic(g, size, Q):
    g = g + 1e-6
    num = ndi.uniform_filter(g ** (Q + 1), size)
    den = ndi.uniform_filter(g ** Q, size)
    return num / den


def geometric_mean(g, size):
    return np.exp(ndi.uniform_filter(np.log(g + 1e-6), size))


def alpha_trimmed(g, size, d):
    def f(v):
        v = np.sort(v)
        return v[d // 2: len(v) - d // 2].mean()
    return ndi.generic_filter(g, f, size=size, mode="reflect")


def adaptive_median(g, smax=7):
    out = g.copy()
    done = np.zeros(g.shape, bool)
    for s in range(3, smax + 1, 2):
        zmin = ndi.minimum_filter(g, s)
        zmax = ndi.maximum_filter(g, s)
        zmed = ndi.median_filter(g, s)
        level_a = (zmed > zmin) & (zmed < zmax) & ~done
        keep = level_a & (g > zmin) & (g < zmax)
        out[level_a] = np.where(keep, g, zmed)[level_a]
        done |= level_a
    out[~done] = zmed[~done]
    return out


def freq_grid(shape):
    M, N = shape
    u = np.fft.fftfreq(M) * M
    v = np.fft.fftfreq(N) * N
    return np.meshgrid(u, v, indexing="ij")


def motion_otf(shape, a, b, T=1.0):
    U, V = freq_grid(shape)
    s = np.pi * (U * a + V * b)
    H = T * np.sinc(s / np.pi) * np.exp(-1j * s)  # np.sinc(x) = sin(pi x)/(pi x)
    return H


def degrade(f, H, sigma):
    g = np.real(np.fft.ifft2(np.fft.fft2(f) * H))
    return g + sigma * rng.standard_normal(f.shape)


def inverse_filter(g, H, radius=None, eps=1e-6):
    """Direct inverse G/H. Frequencies where |H| < eps (true zeros) are set to 0.
    With `radius`, the result is also windowed by a sharp Butterworth low-pass."""
    G = np.fft.fft2(g)
    small = np.abs(H) < eps
    F = np.where(small, 0, G / np.where(small, 1, H))
    if radius is not None:
        U, V = freq_grid(g.shape)
        F = F / (1 + (np.hypot(U, V) / radius) ** 20)
    return np.real(np.fft.ifft2(F))


def wiener(g, H, K):
    G = np.fft.fft2(g)
    return np.real(np.fft.ifft2(np.conj(H) / (np.abs(H) ** 2 + K) * G))


def cls(g, H, gamma):
    lap = np.zeros(g.shape)
    lap[:3, :3] = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    lap = np.roll(lap, (-1, -1), axis=(0, 1))
    P = np.fft.fft2(lap)
    G = np.fft.fft2(g)
    return np.real(np.fft.ifft2(np.conj(H) / (np.abs(H) ** 2 + gamma * np.abs(P) ** 2) * G))


# ---------------------------------------------------- figure 1: noise PDFs
def fig_noise_pdfs():
    # Synthetic test pattern: three flat gray levels (background, square, disc)
    n = 128
    pat = np.full((n, n), 0.25)
    pat[24:104, 24:104] = 0.5
    yy, xx = np.mgrid[:n, :n]
    pat[(yy - 64) ** 2 + (xx - 64) ** 2 < 24 ** 2] = 0.75

    z = np.linspace(-0.05, 0.6, 600)
    specs = [
        ("Gaussian", stats.norm(0.2, 0.06), lambda s: rng.normal(0.2, 0.06, s) - 0.2),
        ("Rayleigh", stats.rayleigh(0.05, 0.08), lambda s: 0.05 + rng.rayleigh(0.08, s) - (0.05 + 0.08 * np.sqrt(np.pi / 2))),
        ("Erlang (gamma)", stats.gamma(3, scale=0.05), lambda s: rng.gamma(3, 0.05, s) - 0.15),
        ("Exponential", stats.expon(scale=0.1), lambda s: rng.exponential(0.1, s) - 0.1),
        ("Uniform", stats.uniform(0.05, 0.3), lambda s: rng.uniform(0.05, 0.35, s) - 0.2),
    ]
    fig, axes = plt.subplots(3, 6, figsize=(10.6, 5.6),
                             gridspec_kw={"height_ratios": [1, 1.25, 0.9]})
    for k, (name, dist, draw) in enumerate(specs):
        ax = axes[0, k]
        ax.plot(z, dist.pdf(z), color="C0", lw=2)
        ax.set_title(name, fontsize=10)
        ax.set_yticks([])
        ax.set_xticks([0, 0.3, 0.6])
        noisy = np.clip(pat + draw(pat.shape), 0, 1)
        show(axes[1, k], noisy, "")
        axes[2, k].hist(noisy.ravel(), bins=80, range=(0, 1), color="0.3")
        axes[2, k].set_yticks([])
        axes[2, k].set_xticks([0, 0.5, 1])
    # salt-and-pepper
    ax = axes[0, 5]
    ax.bar([0.0, 0.6], [0.5, 0.5], width=0.04, color="C0")
    ax.set_xlim(-0.05, 0.65)
    ax.set_xticks([0, 0.6])
    ax.set_xticklabels(["pepper", "salt"])
    ax.set_yticks([])
    ax.set_title("Salt-and-pepper", fontsize=10)
    sp = pat.copy()
    r = rng.random(pat.shape)
    sp[r < 0.05] = 0
    sp[r > 0.95] = 1
    show(axes[1, 5], sp, "")
    axes[2, 5].hist(sp.ravel(), bins=80, range=(0, 1), color="0.3")
    axes[2, 5].set_yticks([])
    axes[2, 5].set_xticks([0, 0.5, 1])
    axes[0, 0].set_ylabel("PDF p(z)")
    axes[2, 0].set_ylabel("histogram")
    fig.tight_layout()
    save(fig, "noise-pdfs.png")


# ---------------------------------------------- figure 2: filter comparison
def fig_filter_grid():
    f = img_as_float(data.camera())[96:352, 128:384]
    g_gauss = np.clip(f + rng.normal(0, 0.08, f.shape), 1e-3, 1)
    g_pep = f.copy()
    g_pep[rng.random(f.shape) < 0.10] = 0.0
    g_sp = f.copy()
    r = rng.random(f.shape)
    g_sp[r < 0.15] = 0
    g_sp[r > 0.85] = 1  # 30 % impulse noise
    g_mix = np.clip(f + rng.normal(0, 0.05, f.shape), 0, 1)
    r = rng.random(f.shape)
    g_mix[r < 0.05] = 0
    g_mix[r > 0.95] = 1

    panels = [
        (g_gauss, "Gaussian noise, s=0.08"),
        (ndi.uniform_filter(g_gauss, 5), "Arithmetic mean 5x5"),
        (geometric_mean(g_gauss, 5), "Geometric mean 5x5"),
        (g_pep, "Pepper noise, 10%"),
        (contraharmonic(g_pep, 3, 1.5), "Contraharmonic Q=+1.5"),
        (contraharmonic(g_pep, 3, -1.5), "Contraharmonic Q=-1.5 (wrong sign)"),
        (g_sp, "Salt+pepper, 30%"),
        (ndi.median_filter(g_sp, 7), "Median 7x7"),
        (adaptive_median(g_sp, 7), "Adaptive median, Smax=7"),
        (g_mix, "Gaussian + 10% impulse"),
        (ndi.uniform_filter(g_mix, 5), "Arithmetic mean 5x5"),
        (alpha_trimmed(g_mix, 5, 6), "Alpha-trimmed mean 5x5, d=6"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(8.4, 11.6))
    for ax, (img, title) in zip(axes.ravel(), panels):
        show(ax, np.clip(img, 0, 1), f"{title}\nPSNR {psnr(f, img):.1f} dB")
    fig.tight_layout()
    save(fig, "filter-grid.png", gray=True)


# ---------------------------------------------- figure 3: periodic noise
def fig_periodic():
    f = img_as_float(data.moon())[:256, 128:384]
    M, N = f.shape
    yy, xx = np.mgrid[:M, :N]
    noise = 0.25 * np.sin(2 * np.pi * (24 * yy / M + 32 * xx / N))
    g = f + noise
    G = np.fft.fftshift(np.fft.fft2(g))
    # notch reject at the two conjugate spikes (Gaussian notches, centred spectrum)
    U, V = np.mgrid[-M // 2:M // 2, -N // 2:N // 2]
    Hnr = np.ones((M, N))
    for (u0, v0) in [(24, 32), (-24, -32)]:
        Hnr *= 1 - np.exp(-((U - u0) ** 2 + (V - v0) ** 2) / (2 * 3.0 ** 2))
    fhat = np.real(np.fft.ifft2(np.fft.ifftshift(G * Hnr)))
    fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.0))
    show(axes[0], np.clip(g, 0, 1), "Corrupted image")
    axes[1].imshow(np.log1p(np.abs(G)), cmap="magma")
    axes[1].set_title("Spectrum log|G|: two spikes", fontsize=10)
    axes[1].axis("off")
    axes[2].imshow(Hnr, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Notch reject filter H_NR", fontsize=10)
    axes[2].axis("off")
    show(axes[3], np.clip(fhat, 0, 1), f"Restored (PSNR {psnr(f, fhat):.1f} dB)")
    fig.tight_layout()
    save(fig, "periodic-notch.png")


# ------------------------------- figure 4: degradation models (OTFs + PSFs)
def fig_degradation_models():
    f = img_as_float(data.astronaut()[..., 1])[:384, 64:448]
    U, V = freq_grid(f.shape)
    H_turb = np.exp(-0.0025 * (U ** 2 + V ** 2) ** (5 / 6))
    H_mot = motion_otf(f.shape, 0.05, 0.02)
    g_turb = np.real(np.fft.ifft2(np.fft.fft2(f) * H_turb))
    g_mot = np.real(np.fft.ifft2(np.fft.fft2(f) * H_mot))
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.8))
    show(axes[0, 0], f, "Original")
    show(axes[0, 1], g_turb, "Turbulence, k=0.0025")
    show(axes[0, 2], g_mot, "Uniform motion, a=0.05, b=0.02")
    axes[1, 0].axis("off")
    for ax, H, t in [(axes[1, 1], H_turb, "|H| turbulence (centred)"),
                     (axes[1, 2], H_mot, "|H| motion (centred)")]:
        ax.imshow(np.log1p(20 * np.abs(np.fft.fftshift(H))), cmap="viridis")
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    axes[1, 0].text(0.0, 0.55,
                    "Same scene, two physical\nblur models, built directly\nas transfer functions H(u,v).\n\n"
                    "Motion |H| has dark zero\nlines: those frequencies\nare lost completely.",
                    fontsize=10, va="center")
    fig.tight_layout()
    save(fig, "degradation-models.png")


# ------------------------------- figure 5: inverse vs Wiener vs CLS
def fig_inverse_wiener():
    f = img_as_float(data.camera())[::2, ::2]  # 256 x 256
    U, V = freq_grid(f.shape)
    H_t = np.exp(-0.0025 * (U ** 2 + V ** 2) ** (5 / 6))
    g_t = degrade(f, H_t, 0.003)
    H = motion_otf(f.shape, 0.1, 0.1)
    sigma = 0.01
    g = degrade(f, H, sigma)
    rows = [
        (g_t, "Turbulence + noise (s=0.003)"),
        (inverse_filter(g_t, H_t), "Full inverse filter"),
        (inverse_filter(g_t, H_t, radius=70), "Inverse, cut off at D0=70"),
        (wiener(g_t, H_t, 0.001), "Wiener, K=0.001"),
        (g, "Motion blur + noise (s=0.01)"),
        (inverse_filter(g, H), "Full inverse filter"),
        (wiener(g, H, 0.01), "Wiener, K=0.01"),
        (cls(g, H, 0.002), "Constrained LS, gamma=0.002"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(10.6, 6.2))
    for ax, (img, title) in zip(axes.ravel(), rows):
        show(ax, np.clip(img, 0, 1), f"{title}\nPSNR {psnr(f, img):.1f} dB")
    fig.tight_layout(h_pad=1.5)
    save(fig, "inverse-vs-wiener.png", gray=True)

    # Same-size 512 x 512 pair for the ImageCompare slider (a, b halved so the
    # blur length in pixels matches the 256 x 256 figure)
    F = img_as_float(data.camera())
    H2 = motion_otf(F.shape, 0.05, 0.05)
    g2 = degrade(F, H2, sigma)
    for name, img in [("compare-blurred.png", g2), ("compare-wiener.png", wiener(g2, H2, 0.01))]:
        plt.imsave(OUT / name, np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        print(f"wrote {OUT / name}")


# ------------------------------- figure 6: CT - sinogram and FBP
def fig_ct():
    phantom = transform.rescale(data.shepp_logan_phantom(), 0.64, anti_aliasing=True)  # 256 x 256
    theta = np.linspace(0.0, 180.0, 180, endpoint=False)
    sino = transform.radon(phantom, theta=theta)
    bp = transform.iradon(sino, theta=theta, filter_name=None)
    fbp = transform.iradon(sino, theta=theta, filter_name="ramp")
    theta_sparse = np.linspace(0.0, 180.0, 20, endpoint=False)
    fbp_sparse = transform.iradon(transform.radon(phantom, theta=theta_sparse),
                                  theta=theta_sparse, filter_name="ramp")
    fig, axes = plt.subplots(1, 5, figsize=(10.6, 2.5),
                             gridspec_kw={"width_ratios": [1, 1.35, 1, 1, 1]})
    show(axes[0], phantom, "Shepp-Logan phantom")
    axes[1].imshow(sino, cmap="gray", aspect="auto", extent=(0, 180, -sino.shape[0] / 2, sino.shape[0] / 2))
    axes[1].set_title("Sinogram", fontsize=10)
    axes[1].set_xlabel("angle theta (deg)")
    axes[1].set_ylabel("rho")
    show(axes[2], bp / bp.max(), "Plain backprojection")
    show(axes[3], fbp, "Filtered BP, 180 views")
    show(axes[4], fbp_sparse, "Filtered BP, 20 views")
    fig.tight_layout()
    save(fig, "ct-fbp.png")


if __name__ == "__main__":
    fig_noise_pdfs()
    fig_filter_grid()
    fig_periodic()
    fig_degradation_models()
    fig_inverse_wiener()
    fig_ct()
