import numpy as np
from scipy import ndimage as ndi
from scipy.sparse.linalg import LinearOperator, cg
from skimage import data, img_as_float64

s = 2                                                   # scale factor
x = img_as_float64(data.camera())[128:384, 128:384]     # HR test crop (256 x 256)
H, W = x.shape

def blur(z, sigma):                                     # circular Gaussian blur (self-adjoint)
    return ndi.gaussian_filter(z, sigma, mode="wrap")

def A(z, sigma):                                        # degradation: blur, then keep every s-th pixel
    return blur(z, sigma)[::s, ::s]

def At(r, sigma):                                       # its adjoint: zero-fill upsampling, then blur
    up = np.zeros((H, W)); up[::s, ::s] = r
    return blur(up, sigma)

def lap(z):                                             # discrete Laplacian (smoothness prior)
    return ndi.laplace(z, mode="wrap")

def nonblind_sr(y, sigma, lam=2e-3):
    """Solve min_x ||A_sigma x - y||^2 + lam ||grad x||^2 with conjugate gradients."""
    op = LinearOperator((H * W, H * W), dtype=float,
                        matvec=lambda v: (At(A(v.reshape(H, W), sigma), sigma) - lam * lap(v.reshape(H, W))).ravel())
    sol, _ = cg(op, At(y, sigma).ravel(), maxiter=200)
    return sol.reshape(H, W)

def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - np.clip(b, 0, 1)) ** 2))

rng = np.random.default_rng(0)
assumed = [0.6, 1.0, 1.6, 2.2, 2.8]
print("true \\ assumed " + "".join(f"{a:7.1f}" for a in assumed))
for true in [1.0, 1.6, 2.2]:
    y = A(x, true) + 0.003 * rng.standard_normal((H // s, W // s))   # LR observation
    row = [psnr(x, nonblind_sr(y, a)) for a in assumed]
    print(f"sigma = {true:.1f}      " + "".join(f"{p:7.2f}" for p in row))
