import numpy as np
from scipy import ndimage as ndi
from skimage import data, img_as_float64
from skimage.color import rgb2gray

hr = img_as_float64(rgb2gray(data.astronaut()))
rng = np.random.default_rng(0)

def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - np.clip(b, 0, 1)) ** 2))

def crispness(img):
    """A crude no-reference score: mean gradient magnitude (higher = 'sharper looking')."""
    return np.mean(np.hypot(ndi.sobel(img, 0), ndi.sobel(img, 1)))

smooth = ndi.gaussian_filter(hr, 1.5)                               # a cautious, blurry restoration
candidates = {
    "smooth (posterior-mean-like)": smooth,
    "unsharp-masked x2": smooth + 2.0 * (smooth - ndi.gaussian_filter(smooth, 2)),
    "smooth + fake fine texture": smooth + 0.04 * rng.standard_normal(hr.shape),
}
for name, img in candidates.items():
    img = np.clip(img, 0, 1)
    print(f"{name:28s} PSNR {psnr(hr, img):6.2f} dB | crispness {crispness(img):.3f}")
print(f"{'ground truth':28s} PSNR    -     | crispness {crispness(hr):.3f}")
