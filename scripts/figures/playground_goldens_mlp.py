"""Golden numbers for the `mlp-training` lab: an independent NumPy implementation of the same network and training loop.

    .venv/bin/python scripts/figures/playground_goldens_mlp.py

Writes only scripts/tests/golden/mlp.json. The reference below is written in the row convention of the ML 02 NumPy
example (X is N x 2, W1 is 2 x m) and shares nothing with the TypeScript code except the specification:

* network: z1 = X W1 + b1, a1 = tanh(z1), z2 = a1 W2 + b2, p = sigmoid(z2); loss = mean binary cross-entropy;
* initial weights: Xavier normal, std sqrt(2 / (n_in + n_out)), biases zero, from the lab's seeded normal field (a port
  of the site kit's mulberry32 + Box-Muller, rounded to float32 like the kit's Float32Array);
* training: full-batch gradient descent, theta <- theta - lr * grad, loss recorded BEFORE each update (index k = after
  k updates);
* data: scikit-learn make_moons / make_circles as stored in src/lib/playground/labs/data/mlp-data.json
  (made by playground_mlp_data.py).

For each configuration it stores the initial weights, the forward pass (logits of the first five points), the initial
loss and gradient, the training and test loss at several steps, and the accuracies at step 500.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = json.loads((ROOT / "src/lib/playground/labs/data/mlp-data.json").read_text())
OUT = ROOT / "scripts/tests/golden/mlp.json"
SEED = 5
H_MAX = 8
STEPS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
M32 = 0xFFFFFFFF


def mulberry32(seed):
    a = seed & M32

    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & M32
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296

    return nxt


def normal_field(n, seed):
    """Standard normal field: Box-Muller from mulberry32, stored as float32 (as the kit's Float32Array)."""
    r = mulberry32(seed)
    out = np.zeros(n, np.float32)
    for i in range(n):
        u1 = max(1e-12, r())
        u2 = r()
        out[i] = np.float32(np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2))
    return out.astype(np.float64)


def init(m, seed=SEED):
    z = normal_field(3 * H_MAX, seed)
    W1 = np.stack([z[0:2 * m:2], z[1:2 * m:2]]) * np.sqrt(2 / (2 + m))  # 2 x m, column j = unit j
    W2 = z[2 * H_MAX:2 * H_MAX + m] * np.sqrt(2 / (m + 1))             # m
    return [W1, np.zeros(m), W2, 0.0]


def pack(P):
    W1, b1, W2, b2 = P  # the lab stores W1 row-major as m x 2: unit j at [2j, 2j + 1]
    return np.concatenate([W1.T.reshape(-1), b1, W2, [b2]])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, P):
    W1, b1, W2, b2 = P
    A1 = np.tanh(X @ W1 + b1)
    return A1, A1 @ W2 + b2


def bce(z, y):
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def loss_grad(X, y, P):
    W1, b1, W2, b2 = P
    A1, z2 = forward(X, P)
    N = len(y)
    d2 = (sigmoid(z2) - y) / N                      # delta2 = p - y, averaged
    gW2 = A1.T @ d2
    gb2 = d2.sum()
    d1 = np.outer(d2, W2) * (1 - A1 ** 2)           # delta1 = (W2^T delta2) * tanh'(z1)
    return bce(z2, y), [X.T @ d1, d1.sum(0), gW2, gb2]


def get(kind, part):
    d = DATA[kind][part]
    return np.array(d["x"]), np.array(d["y"], float)


def run(kind, m, lr):
    Xtr, ytr = get(kind, "train")
    Xte, yte = get(kind, "test")
    P = init(m)
    out = {"kind": kind, "m": m, "lr": lr, "init": pack(P).tolist()}
    _, g0 = loss_grad(Xtr, ytr, P)
    out["gradInit"] = pack(g0).tolist()
    out["logitsInit"] = forward(Xtr[:5], P)[1].tolist()
    train, test, tracc, teacc, series = {}, {}, {}, {}, []
    for k in range(max(STEPS) + 1):
        if k in STEPS:
            _, z = forward(Xtr, P)
            _, zt = forward(Xte, P)
            train[k] = bce(z, ytr)
            test[k] = bce(zt, yte)
            tracc[k] = float(np.mean((z > 0) == (ytr == 1)))
            teacc[k] = float(np.mean((zt > 0) == (yte == 1)))
            if k == 100:
                out["paramsAt100"] = pack(P).tolist()
        l, g = loss_grad(Xtr, ytr, P)
        series.append(l)
        P = [p - lr * gi for p, gi in zip(P, g)]
    series.append(bce(forward(Xtr, P)[1], ytr))
    # rises[k] = number of updates among the first k at which the training loss went up (a sign of too large a step).
    up = np.cumsum(np.diff(series) > 0)
    out["rises"] = {str(k): int(up[k - 1]) if k else 0 for k in STEPS}
    out.update(trainLoss={str(k): v for k, v in train.items()}, testLoss={str(k): v for k, v in test.items()},
               trainAcc={str(k): v for k, v in tracc.items()}, testAcc={str(k): v for k, v in teacc.items()})
    return out


def seed_study(lr=1.0, steps=2000):
    """Training accuracy after `steps` updates for seeds 1..20, widths 1..3, both sets (initial weights from the lab's generator)."""
    out = {}
    for kind in ("moons", "circles"):
        Xtr, ytr = get(kind, "train")
        for m in (1, 2, 3):
            accs = []
            for seed in range(1, 21):
                P = init(m, seed)
                for _ in range(steps):
                    _, g = loss_grad(Xtr, ytr, P)
                    P = [p - lr * gi for p, gi in zip(P, g)]
                accs.append(float(np.mean((forward(Xtr, P)[1] > 0) == (ytr == 1))))
            out[f"{kind}-{m}"] = accs
    return out


def best_line(X, y, n_angles=20000):
    """Most points one straight line can classify correctly (one class per side), by scanning 20000 directions.

    For each direction the points are sorted by their projection and every threshold between neighbours is tried, in
    both orientations. The lab's TypeScript code uses a different exact method (all lines through two points);
    the two agree on these data (see the test)."""
    best = 0
    for a in np.linspace(0, np.pi, n_angles, endpoint=False):
        order = np.argsort(X @ np.array([np.cos(a), np.sin(a)]), kind="stable")
        ys = y[order]
        zeros_below = np.concatenate([[0], np.cumsum(ys == 0)])
        ones_below = np.concatenate([[0], np.cumsum(ys == 1)])
        up = zeros_below + (ones_below[-1] - ones_below)  # class 1 above the threshold
        best = max(best, int(up.max()), int((len(y) - up).max()))
    return best


def long_run(kind, m, lr, steps):
    """Training accuracy after `steps` updates (beyond the lab's 2000-step limit, for the 'small eta is only slow' remark)."""
    Xtr, ytr = get(kind, "train")
    P = init(m)
    for _ in range(steps):
        _, g = loss_grad(Xtr, ytr, P)
        P = [p - lr * gi for p, gi in zip(P, g)]
    return float(np.mean((forward(Xtr, P)[1] > 0) == (ytr == 1)))


def main():
    raw = normal_field(3 * H_MAX, SEED)
    runs = [run(kind, m, lr) for kind in ("moons", "circles") for m in (1, 2, 3, 8) for lr in (0.1, 1.0, 3.0)]
    # Sensitive cases (large learning rates) are only compared at early steps in the test, see lab-mlp.test.ts.
    runs += [run(kind, 3, lr) for kind in ("moons", "circles") for lr in (10.0, 30.0)]
    OUT.write_text(json.dumps({"seed": SEED, "rawNormals": raw.tolist(), "steps": STEPS, "runs": runs, "seedStudy": seed_study(),
        "bestLine": {k: {part: best_line(*get(k, part)) for part in ("train", "test")} for k in ("moons", "circles")},
        "long": {f"{k}-{lr}": {str(n): long_run(k, 3, lr, n) for n in (2000, 5000)} for k in ("moons", "circles") for lr in (0.1,)},
    }, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(runs)} runs)")


if __name__ == "__main__":
    main()
