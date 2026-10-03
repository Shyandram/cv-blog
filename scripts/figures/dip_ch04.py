"""Figures for DIP Chapter 4 - Filtering in the Frequency Domain.

Run:  .venv/bin/python scripts/figures/dip_ch04.py
Writes PNGs to src/assets/dip/ch04/ only. All inputs are skimage.data images
or synthetic arrays.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color, data, img_as_float, transform

OUT = Path(__file__).resolve().parents[2] / "src" / "assets" / "dip" / "ch04"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})
DPI = 150


def save(fig, name, gray=False):
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    if path.stat().st_size > 700 * 1024:  # keep PNGs small: adaptive 256-colour palette
        im = Image.open(path).convert("RGB")
        if gray:
            im.convert("L").save(path, optimize=True)
            return
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def log_spectrum(f):
    F = np.fft.fftshift(np.fft.fft2(f))
    return np.log1p(np.abs(F))


def dist_grid(P, Q):
    u = np.arange(P) - P // 2
    v = np.arange(Q) - Q // 2
    V, U = np.meshgrid(v, u)
    return np.sqrt(U**2 + V**2)


def freq_filter(f, H):
    """Centered filter H (same size as f); no padding (used where noted)."""
    F = np.fft.fftshift(np.fft.fft2(f))
    return np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))


def freq_filter_padded(f, Hfun):
    """Full textbook procedure: pad to 2M x 2N, centre, multiply, invert, crop."""
    M, N = f.shape
    P, Q = 2 * M, 2 * N
    fp = np.pad(f, ((0, P - M), (0, Q - N)), mode="reflect")
    F = np.fft.fftshift(np.fft.fft2(fp))
    H = Hfun(dist_grid(P, Q))
    g = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))
    return g[:M, :N]


camera = img_as_float(data.camera())  # 512 x 512
astro = img_as_float(color.rgb2gray(data.astronaut()))  # 512 x 512


# ---------------------------------------------------------------- 1. spectra
def fig_spectra():
    n = 256
    y, x = np.mgrid[0:n, 0:n]
    rect = np.zeros((n, n))
    rect[118:138, 108:148] = 1.0
    rot = transform.rotate(rect, 30, order=0)
    grating = 0.5 + 0.5 * np.cos(2 * np.pi * (x * 8 / n + y * 4 / n))
    gauss = np.exp(-((x - n / 2) ** 2 + (y - n / 2) ** 2) / (2 * 12.0**2))
    shifted = np.roll(np.roll(rect, 60, axis=0), -70, axis=1)
    pats = [("Rectangle", rect), ("Shifted rectangle", shifted), ("Rotated 30 deg", rot),
            ("Cosine grating", grating), ("Gaussian blob", gauss)]
    fig, ax = plt.subplots(2, 5, figsize=(10.5, 4.6))
    for k, (title, p) in enumerate(pats):
        ax[0, k].imshow(p, cmap="gray")
        ax[0, k].set_title(title, fontsize=10)
        S = log_spectrum(p)
        if title == "Cosine grating":  # three isolated spikes: enlarge them so they are visible
            from scipy.ndimage import maximum_filter
            S = maximum_filter(S, size=7)
        ax[1, k].imshow(S, cmap="magma")
        ax[1, k].set_title("log(1+|F|), centred", fontsize=10)
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    save(fig, "spectra.png")


# ---------------------------------------------------------------- 2. aliasing
def fig_aliasing():
    n = 512
    y, x = np.mgrid[-n // 2:n // 2, -n // 2:n // 2] / n
    zone = 0.5 + 0.5 * np.cos(np.pi * 380 * (x**2 + y**2))  # chirp: frequency grows with radius
    step = 4
    naive = zone[::step, ::step]
    # Gaussian prefilter (sigma ~ step/2) before decimation
    from scipy.ndimage import gaussian_filter
    pre = gaussian_filter(zone, sigma=step / 2.0)[::step, ::step]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.8))
    ax[0].imshow(zone, cmap="gray")
    ax[0].set_title("Zone plate, 512x512")
    ax[1].imshow(naive, cmap="gray", interpolation="nearest")
    ax[1].set_title("Every 4th pixel: aliasing")
    ax[2].imshow(pre, cmap="gray", interpolation="nearest")
    ax[2].set_title("Blur first, then every 4th")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    save(fig, "aliasing.png")


# ---------------------------------------------------------------- 3. phase swap
def fig_phase_swap():
    A = np.fft.fft2(camera)
    B = np.fft.fft2(astro)
    magA_phB = np.real(np.fft.ifft2(np.abs(A) * np.exp(1j * np.angle(B))))
    magB_phA = np.real(np.fft.ifft2(np.abs(B) * np.exp(1j * np.angle(A))))
    phase_only = np.real(np.fft.ifft2(np.exp(1j * np.angle(A))))
    mag_only = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(A))))
    panels = [("Image A (camera)", camera), ("Image B (astronaut)", astro),
              ("|A| with phase of B", magA_phB), ("|B| with phase of A", magB_phA),
              ("Phase of A only", phase_only), ("|A| only (zero phase)", np.log1p(np.abs(mag_only)))]
    fig, ax = plt.subplots(2, 3, figsize=(9.6, 6.6))
    for a, (t, im) in zip(ax.ravel(), panels):
        lo, hi = np.percentile(im, [1, 99])
        a.imshow(np.clip(im, lo, hi), cmap="gray")
        a.set_title(t, fontsize=11)
        a.axis("off")
    fig.tight_layout()
    save(fig, "phase-swap.png", gray=True)


# ---------------------------------------------------------------- 4. lowpass ringing
def fig_lowpass():
    D0 = 40.0
    filters = {
        "Ideal": lambda D: (D <= D0).astype(float),
        "Butterworth n=2": lambda D: 1.0 / (1.0 + (D / D0) ** 4),
        "Gaussian": lambda D: np.exp(-(D**2) / (2 * D0**2)),
    }
    crop = (slice(60, 300), slice(160, 400))
    fig, ax = plt.subplots(3, 3, figsize=(10.2, 10.2))
    for k, (name, Hf) in enumerate(filters.items()):
        D = dist_grid(512, 512)
        H = Hf(D)
        r = np.arange(0, 120)
        ax[0, k].plot(r, Hf(r.astype(float)), lw=2)
        ax[0, k].set_ylim(-0.05, 1.1)
        ax[0, k].set_xlabel("distance D from centre")
        ax[0, k].set_ylabel("H(D)")
        ax[0, k].set_title(f"{name}, D0={D0:.0f}")
        ax[0, k].grid(alpha=0.3)
        h = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(H))))
        line = h[256, 256 - 40:256 + 41]
        ax[1, k].plot(np.arange(-40, 41), line / line.max(), lw=1.5)
        ax[1, k].axhline(0, color="k", lw=0.6)
        ax[1, k].set_title("spatial kernel h(x, 0), normalised")
        ax[1, k].set_xlabel("x (pixels)")
        ax[1, k].grid(alpha=0.3)
        g = freq_filter_padded(camera, Hf)
        ax[2, k].imshow(np.clip(g[crop], 0, 1), cmap="gray", vmin=0, vmax=1)
        ax[2, k].set_title(f"{name} result (crop)")
        ax[2, k].axis("off")
    fig.tight_layout()
    save(fig, "lowpass-ringing.png")


# ---------------------------------------------------------------- 5. highpass family
def fig_highpass():
    D0 = 30.0
    M, N = camera.shape
    ghp = freq_filter_padded(camera, lambda D: 1 - np.exp(-(D**2) / (2 * D0**2)))
    # Laplacian in frequency: H = -4 pi^2 (u^2+v^2)/(P Q-normalised)
    P, Q = 2 * M, 2 * N
    lap = freq_filter_padded(camera, lambda D: -4 * np.pi**2 * (D / P) ** 2)
    lap = lap / np.abs(lap).max()
    sharp = np.clip(camera - lap, 0, 1)
    hfe = freq_filter_padded(camera, lambda D: 0.5 + 1.5 * (1 - np.exp(-(D**2) / (2 * D0**2))))
    panels = [("Gaussian highpass, D0=30", ghp, "sym"),
              ("Laplacian (frequency domain)", lap, "sym"),
              ("f - Laplacian (sharpened)", sharp, "img"),
              ("High-frequency emphasis 0.5+1.5 HP", hfe, "img")]
    fig, ax = plt.subplots(1, 4, figsize=(10.5, 3.2))
    for a, (t, im, mode) in zip(ax, panels):
        if mode == "sym":
            s = np.percentile(np.abs(im), 99)
            a.imshow(im, cmap="gray", vmin=-s, vmax=s)
        else:
            lo, hi = np.percentile(im, [0.5, 99.5])
            a.imshow(np.clip(im, lo, hi), cmap="gray")
        a.set_title(t, fontsize=10)
        a.axis("off")
    fig.tight_layout()
    save(fig, "highpass.png")


# ---------------------------------------------------------------- 6. homomorphic
def fig_homomorphic():
    M, N = camera.shape
    y, x = np.mgrid[0:M, 0:N]
    illum = 0.15 + 0.85 * np.exp(-((x - 0.85 * N) ** 2 + (y - 0.2 * M) ** 2) / (2 * (0.35 * N) ** 2))
    f = np.clip(camera, 0.02, 1) * illum
    z = np.log(f + 1e-3)
    gL, gH, c, D0 = 0.4, 1.6, 1.0, 20.0
    s = freq_filter_padded(z, lambda D: (gH - gL) * (1 - np.exp(-c * D**2 / D0**2)) + gL)
    g = np.exp(s)
    g = (g - g.min()) / (g.max() - g.min())
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.8))
    ax[0].imshow(f, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("Uneven lighting  f = i * r")
    ax[1].imshow(illum, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title("Illumination i (synthetic)")
    lo, hi = np.percentile(g, [0.5, 99.5])
    ax[2].imshow(np.clip(g, lo, hi), cmap="gray")
    ax[2].set_title("Homomorphic filtered")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    save(fig, "homomorphic.png")


# ---------------------------------------------------------------- 7. notch
def fig_notch():
    f = camera
    M, N = f.shape
    y, x = np.mgrid[0:M, 0:N]
    noise = 0.18 * np.cos(2 * np.pi * (40 * x / N + 25 * y / M)) + 0.12 * np.cos(2 * np.pi * (-70 * x / N + 60 * y / M))
    g = f + noise
    G = np.fft.fftshift(np.fft.fft2(g))
    # Gaussian notch reject at each spike (u = row offset, v = column offset) and its mirror
    U, V = np.meshgrid(np.arange(M) - M // 2, np.arange(N) - N // 2, indexing="ij")
    H = np.ones((M, N))
    for (du, dv) in [(25, 40), (60, -70)]:
        for s in (1, -1):
            Dk = np.sqrt((U - s * du) ** 2 + (V - s * dv) ** 2)
            H *= 1 - np.exp(-(Dk**2) / (2 * 4.0**2))
    out = np.real(np.fft.ifft2(np.fft.ifftshift(G * H)))
    fig, ax = plt.subplots(1, 4, figsize=(10.5, 3.0))
    ax[0].imshow(np.clip(g, 0, 1), cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("Image + periodic noise")
    ax[1].imshow(np.log1p(np.abs(G)), cmap="magma")
    ax[1].set_title("Spectrum (spikes circled)")
    for (du, dv) in [(25, 40), (60, -70)]:
        for sgn in (1, -1):
            ax[1].add_patch(plt.Circle((N // 2 + sgn * dv, M // 2 + sgn * du), 14, fill=False, color="cyan", lw=1.5))
    ax[2].imshow(H, cmap="gray")
    ax[2].set_title("Gaussian notch-reject H")
    ax[3].imshow(np.clip(out, 0, 1), cmap="gray", vmin=0, vmax=1)
    ax[3].set_title("Filtered result")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    save(fig, "notch.png")
    # identical-size pair for the ImageCompare slider
    for name, im in [("notch-before.png", g), ("notch-after.png", out)]:
        Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).save(OUT / name, optimize=True)
        print(f"wrote {OUT / name}")


# ---------------------------------------------------------------- 8. wraparound
def fig_wraparound():
    n = 128
    f = np.zeros((n, n))
    f[:, :8] = 1.0  # bright bar touching the left border
    f[40:88, 100:] = 1.0  # block touching the right border
    k = 31
    h = np.zeros((n, n))
    h[:k, :k] = 1.0 / k**2
    h = np.roll(h, (-(k // 2), -(k // 2)), axis=(0, 1))
    circ = np.real(np.fft.ifft2(np.fft.fft2(f) * np.fft.fft2(h)))
    P = 2 * n
    fp = np.zeros((P, P)); fp[:n, :n] = f
    hp = np.zeros((P, P)); hp[:k, :k] = 1.0 / k**2
    lin = np.real(np.fft.ifft2(np.fft.fft2(fp) * np.fft.fft2(hp)))[k // 2:k // 2 + n, k // 2:k // 2 + n]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.7))
    for a, (t, im) in zip(ax, [("Input (objects touch borders)", f),
                               ("31x31 box via DFT, no padding", circ),
                               ("Same, zero-padded to 2M x 2N", lin)]):
        a.imshow(im, cmap="gray", vmin=0, vmax=1)
        a.set_title(t, fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color("tab:red"); sp.set_linewidth(1.5)
    fig.tight_layout()
    save(fig, "wraparound.png")


if __name__ == "__main__":
    fig_spectra()
    fig_aliasing()
    fig_phase_swap()
    fig_lowpass()
    fig_highpass()
    fig_homomorphic()
    fig_notch()
    fig_wraparound()
