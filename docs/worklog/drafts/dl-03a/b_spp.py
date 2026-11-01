import numpy as np
from scipy.ndimage import maximum_filter

def spp(x, kernels=(5, 9, 13)):
    """YOLO-style SPP. x: (C, H, W). Stride-1 max pooling with 'same' padding keeps H and W;
    the input and the pooled copies are concatenated along the channel axis."""
    pooled = [maximum_filter(x, size=(1, k, k), mode="constant", cval=-np.inf) for k in kernels]
    return np.concatenate([x] + pooled, axis=0)

rng = np.random.default_rng(0)
x = rng.normal(size=(512, 19, 19))          # e.g. the stride-32 map of a 608 x 608 input
print("SPP:", x.shape, "->", spp(x).shape)

one = np.zeros((1, 19, 19)); one[0, 9, 9] = 1.0   # a single strong activation
y = spp(one)
print("cells reached by one activation per group:", [int((g > 0).sum()) for g in y])
