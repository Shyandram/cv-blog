"""Golden numbers for the `diffusion-targets` lab (noise, clean-image and velocity targets on the astronaut image).

    .venv/bin/python scripts/figures/playground_goldens_diffusion.py

Independent re-implementation in NumPy / scikit-image of the lab's pipeline, on the exact asset
src/assets/playground/astronaut-256.png:

    x0 = 2 * image - 1                          (RGB, values in [-1, 1], as in the chapter)
    eps, delta = fixed seeded standard-normal fields (a port of the kit's mulberry32 + Box-Muller, float32)
    a = sqrt(abar), b = sqrt(1 - abar),  abar = 1 / (1 + 10^(-snr_dB / 10))
    x_t = a x0 + b eps,  v = a eps - b x0
    predictions: target + s * delta;  x0_hat = (x_t - b eps_hat) / a  |  x0_hat  |  a x_t - b v_hat
    PSNR: skimage peak_signal_noise_ratio(x0 in [0,1], clip(x0_hat, -1, 1) mapped to [0,1], data_range=1)

Writes only scripts/tests/golden/diffusion.json.
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "src/assets/playground/astronaut-256.png"
OUT = ROOT / "scripts/tests/golden/diffusion.json"
SEEDS = {"eps": 7, "err": 8}
SNRS = [-30, -20, -10, 0, 10, 20]
ERRS = [0.0, 0.05, 0.2]


def imul(a: int, b: int) -> int:
    return (a * b) & 0xFFFFFFFF


def mulberry32(seed: int):
    state = [seed & 0xFFFFFFFF]

    def nxt() -> float:
        state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = state[0]
        t = imul(t ^ (t >> 15), t | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return nxt


def normal_field(n: int, seed: int) -> np.ndarray:
    r = mulberry32(seed)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        u = max(1e-12, r())
        out[i] = math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * r())
    return out


def corr(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.corrcoef(p.ravel().astype(np.float64), q.ravel().astype(np.float64))[0, 1])


def main() -> None:
    img = np.asarray(Image.open(ASSET).convert("RGB"), dtype=np.float32) / 255.0
    n = img.size
    x0 = (2 * img - 1).astype(np.float32).ravel()
    eps = normal_field(n, SEEDS["eps"])
    delta = normal_field(n, SEEDS["err"])
    ref01 = img.ravel().astype(np.float64)
    rows = []
    for snr in SNRS:
        abar = 1 / (1 + 10 ** (-snr / 10))
        a, b = math.sqrt(abar), math.sqrt(1 - abar)
        xt = (a * x0 + b * eps).astype(np.float32)
        v = (a * eps - b * x0).astype(np.float32)
        row = {
            "snr": snr, "alpha": abar,
            "corrXtEps": corr(xt, eps), "corrXtX0": corr(xt, x0), "corrVx0": corr(v, x0), "corrVEps": corr(v, eps),
            "runs": [],
        }
        for s in ERRS:
            for kind in ("eps", "x0", "v"):
                target = {"eps": eps, "x0": x0, "v": v}[kind]
                pred = target + np.float32(s) * delta
                if kind == "eps":
                    rec = (xt - b * pred) / a
                elif kind == "x0":
                    rec = pred
                else:
                    rec = a * xt - b * pred
                clipped = np.clip(rec, -1, 1)
                psnr = peak_signal_noise_ratio(ref01, (clipped.astype(np.float64) + 1) / 2, data_range=1)
                err_out = float(np.sqrt(np.mean((pred.astype(np.float64) - target) ** 2)))
                err_x0 = float(np.sqrt(np.mean((rec.astype(np.float64) - x0) ** 2)))
                row["runs"].append({
                    "err": s, "kind": kind, "psnr": float(psnr),
                    "measuredAmplification": (err_x0 / err_out) if err_out > 0 else None,
                    "theoryAmplification": {"eps": b / a, "x0": 1.0, "v": b}[kind],
                })
        rows.append(row)
    payload = {
        "note": "diffusion-targets goldens; see playground_goldens_diffusion.py",
        "epsHead": [float(v) for v in eps[:6]], "epsMean": float(eps.astype(np.float64).mean()), "epsStd": float(eps.astype(np.float64).std()),
        "deltaHead": [float(v) for v in delta[:6]],
        "x0Mean": float(x0.astype(np.float64).mean()), "x0Std": float(x0.astype(np.float64).std()),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print("eps mean/std", payload["epsMean"], payload["epsStd"], " x0 mean/std", payload["x0Mean"], payload["x0Std"])
    for row in rows:
        print(f"snr {row['snr']:4d} dB  abar {row['alpha']:.4f}  corr(xt,eps) {row['corrXtEps']:.4f}  corr(xt,x0) {row['corrXtX0']:.4f}  corr(v,x0) {row['corrVx0']:.4f}  corr(v,eps) {row['corrVEps']:.4f}")
        for r in row["runs"]:
            if r["err"] in (0.05,):
                amp = r["measuredAmplification"]
                print(f"    s={r['err']:.2f} {r['kind']:3s}  PSNR {r['psnr']:6.2f} dB  amp measured {amp:.3f}  theory {r['theoryAmplification']:.3f}")


if __name__ == "__main__":
    main()
