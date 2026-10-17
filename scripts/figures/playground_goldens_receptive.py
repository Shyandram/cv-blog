"""Golden values for the receptive-field lab (src/lib/playground/labs/receptive-field.ts).

The receptive field is measured independently of the TypeScript recurrence and interval trace, with SciPy: each layer is a
zero-padded, strided, dilated all-ones correlation (padding = (effective kernel - 1) / 2, as in the lab). The support of one
output unit is found with the adjoint (transpose) of that linear map, and cross-checked by brute force on a 64 x 64 input
(perturb every input pixel and see whether the chosen output moves). All weights are positive, so nothing cancels.
(The repository notes list PyTorch, but the project venv does not have it; autograd would give the same support.)

Run: .venv/bin/python scripts/figures/playground_goldens_receptive.py
Writes only scripts/tests/golden/receptive.json. Nothing is downloaded.
"""
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

OUT = Path(__file__).resolve().parents[2] / "scripts/tests/golden/receptive.json"
N = 256

# (kernel, stride, dilation) per layer: keep in sync with STACKS in receptive-field.ts (the test checks the keys)
STACKS = {
    "plain3": [(3, 1, 1)] * 3,
    "plain4": [(3, 1, 1)] * 4,
    "plain6": [(3, 1, 1)] * 6,
    "stride-first": [(3, 2, 1), (3, 1, 1), (3, 1, 1)],
    "stride-last": [(3, 1, 1), (3, 1, 1), (3, 2, 1)],
    "dilate-first": [(3, 1, 2), (3, 1, 1), (3, 1, 1)],
    "stride-every-2nd": [(3, 1, 1), (3, 2, 1)] * 3,
    "dilated-1248": [(3, 1, 1), (3, 1, 2), (3, 1, 4), (3, 1, 8)],
}


def dilated(k, d):
    kd = np.zeros((k - 1) * d + 1)
    kd[::d] = 1.0
    return kd


def forward(x, layers):
    for k, s, d in layers:
        kd = dilated(k, d)
        y = ndi.correlate1d(ndi.correlate1d(x, kd, axis=0, mode="constant"), kd, axis=1, mode="constant")
        x = y[::s, ::s]
    return x


def grid_sizes(n, layers):
    sizes = [n]
    for k, s, d in layers:
        sizes.append(-(-sizes[-1] // s))
    return sizes


def support(layers, uy, ux, n=N):
    """Input pixels that influence output unit (uy, ux): adjoint of `forward`, run from the output back to the input."""
    sizes = grid_sizes(n, layers)
    g = np.zeros((sizes[-1], sizes[-1]))
    g[uy, ux] = 1.0
    for l in range(len(layers) - 1, -1, -1):
        k, s, d = layers[l]
        kd = dilated(k, d)
        z = np.zeros((sizes[l], sizes[l]))
        z[::s, ::s] = g
        # adjoint of correlation with a centered odd kernel is convolution with the same kernel
        g = ndi.convolve1d(ndi.convolve1d(z, kd, axis=0, mode="constant"), kd, axis=1, mode="constant")
    return g != 0


def extent(layers, uy, ux):
    sup = support(layers, uy, ux)
    cols, rows = np.where(sup.any(0))[0], np.where(sup.any(1))[0]
    return {"x": [int(cols.min()), int(cols.max())], "y": [int(rows.min()), int(rows.max())], "pixels": int(sup.sum())}


def brute_force_support(layers, uy, ux, n=64):
    sup = np.zeros((n, n), bool)
    for y in range(n):
        for x in range(n):
            e = np.zeros((n, n)); e[y, x] = 1.0
            sup[y, x] = forward(e, layers)[uy, ux] != 0
    return sup


res = {}
for name, layers in STACKS.items():
    m = grid_sizes(N, layers)[-1]
    mid = m // 2
    res[name] = {
        "layers": layers, "grid": m, "centerUnit": mid,
        "center": extent(layers, mid, mid), "corner": extent(layers, 0, 0), "last": extent(layers, m - 1, m - 1),
        "neighbor": extent(layers, mid, mid + 1),
    }
    # cross-check the adjoint against brute-force perturbation on a small input (center and a border unit)
    m64 = grid_sizes(64, layers)[-1]
    for uy, ux in [(m64 // 2, m64 // 2), (0, 1)]:
        assert (brute_force_support(layers, uy, ux) == support(layers, uy, ux, 64)).all(), (name, uy, ux)
OUT.write_text(json.dumps(res))
print("wrote", OUT)
for k, v in res.items():
    c = v["center"]
    print(f"{k:18s} grid {v['grid']:3d} center x {c['x']} size {c['x'][1] - c['x'][0] + 1} pixels {c['pixels']} | corner x {v['corner']['x']} y {v['corner']['y']} pixels {v['corner']['pixels']}")
