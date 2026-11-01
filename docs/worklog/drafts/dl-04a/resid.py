import numpy as np, cv2
from scipy import ndimage as ndi
from skimage import data, img_as_float64
from skimage.color import rgb2gray

s = 4
hr = img_as_float64(rgb2gray(data.astronaut()))                        # (512, 512), values in [0, 1]
lr = ndi.gaussian_filter(hr, 1.2).reshape(128, s, 128, s).mean(axis=(1, 3))   # the running example
up = cv2.resize(lr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)   # bicubic upsampling
r = hr - up                                                            # what a residual network predicts
print(f"HR image: mean {hr.mean():.3f}  std {hr.std():.3f}")
print(f"residual: mean {r.mean():+.3f}  std {r.std():.3f}  |r| < 0.05 at {np.mean(np.abs(r) < 0.05):.0%} of pixels")
