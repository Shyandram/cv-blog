import numpy as np
from scipy import ndimage as ndi
from skimage import data, img_as_float64

x = img_as_float64(data.camera())
s = 4

def lr(img, sigma, shift=0):
    """Blur with a Gaussian kernel centred at `shift` pixels, then keep every s-th pixel."""
    k = ndi.gaussian_filter(img, sigma, mode="wrap")
    return np.roll(k, shift, axis=1)[::s, ::s]

# 1) Shift ambiguity: move the scene one pixel right and the kernel one pixel left.
a = lr(x, 1.5)
b = lr(np.roll(x, 1, axis=1), 1.5, shift=-1)
print("shifted scene + shifted kernel, max |difference|:", f"{np.abs(a - b).max():.1e}")

# 2) Blur ambiguity: a soft scene with a sharp kernel vs a sharp scene with a wide kernel.
soft_scene = ndi.gaussian_filter(x, 1.0, mode="wrap")
c = lr(soft_scene, 1.2)
d = lr(x, np.hypot(1.0, 1.2))                     # Gaussian widths add in quadrature
print("soft scene + sigma 1.2 vs sharp scene + sigma 1.56, max |difference|:", f"{np.abs(c - d).max():.1e}")
