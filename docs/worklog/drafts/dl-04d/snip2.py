import numpy as np, cv2
from scipy import ndimage as ndi
from scipy.special import j1
from skimage import data, img_as_float32

def gaussian_kernel(rng, size, smax):
    """Random isotropic or anisotropic Gaussian: k(c) ~ exp(-0.5 c^T S^-1 c), normalized."""
    s1, s2 = rng.uniform(0.2, smax, 2)
    if rng.random() < 0.5:
        s2 = s1                                                     # isotropic half of the time
    th = rng.uniform(0, np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    S = R @ np.diag([s1**2, s2**2]) @ R.T
    ax = np.arange(size) - size // 2
    C = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1)
    k = np.exp(-0.5 * np.einsum("...i,ij,...j", C, np.linalg.inv(S), C))
    return k / k.sum(), f"gauss({s1:.1f},{s2:.1f})"

def sinc_kernel(size, wc):
    """2-D sinc low-pass: k(r) = wc J1(wc r) / (2 pi r), with k(0) = wc^2 / (4 pi)."""
    ax = np.arange(size) - size // 2
    r = np.hypot(*np.meshgrid(ax, ax, indexing="ij"))
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(r == 0, wc**2 / (4 * np.pi), wc * j1(wc * r) / (2 * np.pi * r))
    return k / k.sum()

def jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", (np.clip(img, 0, 1) * 255).round().astype(np.uint8)[..., ::-1],
                           [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255

def one_round(img, rng, log, smax, nmax):
    """blur -> random resize -> noise -> JPEG: one 'classical' degradation."""
    size = int(rng.choice(np.arange(7, 22, 2)))
    if rng.random() < 0.1:
        k, name = sinc_kernel(size, rng.uniform(np.pi / 3, np.pi)), "sinc"
    else:
        k, name = gaussian_kernel(rng, size, smax)
    img = np.stack([ndi.convolve(img[..., c], k, mode="reflect") for c in range(3)], -1)
    f = rng.uniform(0.3, 1.2)                                       # resize factor (illustrative range)
    interp = rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    img = cv2.resize(img, None, fx=f, fy=f, interpolation=interp)
    if rng.random() < 0.5:                                          # Gaussian noise, sigma in 1..nmax (8-bit units)
        sig = rng.uniform(1, nmax) / 255
        shape = img.shape[:2] + ((1,) if rng.random() < 0.4 else (3,))   # gray or color noise
        img = img + sig * rng.standard_normal(shape); noise = f"gauss{sig * 255:.0f}"
    else:                                                           # shot noise with a random photon count
        peak = 10 ** rng.uniform(1.5, 3)
        img = rng.poisson(np.clip(img, 0, 1) * peak) / peak; noise = f"poisson{peak:.0f}"
    q = rng.uniform(30, 95)
    img = jpeg(img, q)
    log.append(f"{name} x{f:.2f} {noise} q{q:.0f}")
    return img

def degrade(hr, rng, scale=4):
    """Second-order pipeline in the spirit of Real-ESRGAN (parameter ranges partly our own)."""
    log = []
    img = one_round(hr, rng, log, smax=3.0, nmax=30)
    if rng.random() < 0.8:
        img = one_round(img, rng, log, smax=1.5, nmax=25)          # second, milder round
    h, w = hr.shape[0] // scale, hr.shape[1] // scale
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)       # bring to the target LR size
    final = [lambda z: jpeg(z, rng.uniform(30, 95)),
             lambda z: np.stack([ndi.convolve(z[..., c], sinc_kernel(15, rng.uniform(np.pi / 3, np.pi)),
                                              mode="reflect") for c in range(3)], -1)]
    if rng.random() < 0.5:
        final = final[::-1]                                         # random order: sinc then JPEG, or the reverse
    for op in final:
        img = op(img)
    return np.clip(img, 0, 1), log

hr = img_as_float32(data.astronaut())                               # (512, 512, 3)
for seed in range(3):
    lr, log = degrade(hr, np.random.default_rng(seed))
    print(f"seed {seed}: LR {lr.shape} | " + " ; ".join(log))
