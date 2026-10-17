"""Golden values for the snake kit (src/lib/imagekit/snake.ts) and the lab contour-energy (src/lib/playground/labs/contour-energy.ts).

Everything is computed here with NumPy / SciPy / scikit-image, independently of the TypeScript:

* the semi-implicit scheme of the DIP 11 chapter (`snake_matrix` and `run_snake` of scripts/figures/dip_ch11.py, rewritten from
  scratch in double precision with the same conventions: A + gamma I inverted once, bilinear sampling of np.gradient(e), clipping to
  the image, stop when no point moves more than tol);
* the "true" boundary of the coin: Otsu threshold of the exact asset PNG, holes filled, the component at the image centre, a
  least-squares circle through its boundary pixels;
* a closed-form ring edge field (no image, no Gaussian) on which the TypeScript loop must agree to rounding error;
* the facts quoted in the lab text (they are printed as well, and asserted by scripts/tests/lab-contour.test.ts).

It also checks that the pentadiagonal matrix of the chapter equals the one built inside skimage.segmentation.active_contour (same A),
and prints how the scikit-image algorithm differs (it does NOT take the same steps: Sobel of the already smoothed image, a quadratic
spline for the force, and a tanh cap of one pixel per iteration).

Run: .venv/bin/python scripts/figures/playground_goldens_snake.py
Writes only scripts/tests/golden/snake.json.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scripts/tests/golden/snake.json"
ASSET = ROOT / "src/assets/playground/coin-crop-144.png"

N, GAMMA, KAPPA, TOL, MAX_ITER = 100, 1.0, 4.0, 0.05, 1500
CENTER = 72  # the start circle is centred on pixel (72, 72), the middle of the 144 x 144 crop


def flat(a, digits=10):
    return [float(f"{x:.{digits}g}") for x in np.asarray(a, dtype=float).ravel()]


# ----------------------------------------------------------------------------- the scheme (chapter notation)
def internal_matrix(n, alpha, beta):
    row = np.zeros(n)
    row[[0, 1, 2, -2, -1]] = [2 * alpha + 6 * beta, -alpha - 4 * beta, beta, beta, -alpha - 4 * beta]
    return np.stack([np.roll(row, i) for i in range(n)])


def snake_matrix(n, alpha, beta, gamma):
    return np.linalg.inv(internal_matrix(n, alpha, beta) + gamma * np.eye(n))


def edge_field(img, sigma):
    e = ndi.gaussian_gradient_magnitude(img, sigma)  # sqrt(sum of the squared Gaussian derivatives), mode 'reflect'
    e = e / e.max()
    gy, gx = np.gradient(e)
    return e, gx, gy


def energy(e, x, y, alpha, beta, kappa):
    d1x, d1y = np.roll(x, -1) - x, np.roll(y, -1) - y
    d2x, d2y = np.roll(x, -1) - 2 * x + np.roll(x, 1), np.roll(y, -1) - 2 * y + np.roll(y, 1)
    internal = 0.5 * (alpha * np.sum(d1x ** 2 + d1y ** 2) + beta * np.sum(d2x ** 2 + d2y ** 2))
    external = -kappa * ndi.map_coordinates(e, np.vstack([y, x]), order=1, mode="nearest").sum()
    return internal, external


def run_snake(field, x0, y0, alpha, beta, gamma=GAMMA, kappa=KAPPA, tol=TOL, max_iter=MAX_ITER, snapshots=()):
    e, gx, gy = field
    h, w = e.shape
    x, y = x0.astype(float).copy(), y0.astype(float).copy()
    inv = snake_matrix(len(x), alpha, beta, gamma)
    ints, exts, trace, snaps = [], [], [], {}
    i0, e0 = energy(e, x, y, alpha, beta, kappa)
    ints.append(i0), exts.append(e0)
    move, it = np.inf, 0
    while it < max_iter:
        co = np.vstack([y, x])
        fx = kappa * ndi.map_coordinates(gx, co, order=1, mode="nearest")
        fy = kappa * ndi.map_coordinates(gy, co, order=1, mode="nearest")
        xn = np.clip(inv @ (gamma * x + fx), 0, w - 1)
        yn = np.clip(inv @ (gamma * y + fy), 0, h - 1)
        move = float(np.max(np.hypot(xn - x, yn - y)))
        x, y = xn, yn
        it += 1
        a, b = energy(e, x, y, alpha, beta, kappa)
        ints.append(a), exts.append(b)
        if it in snapshots:
            snaps[it] = (x.copy(), y.copy())
        if move < tol:
            break
    return dict(x=x, y=y, iterations=it, converged=bool(move < tol), last_move=move, internal=np.array(ints), external=np.array(exts), snaps=snaps)


def circle(cx, cy, r, n=N):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return cx + r * np.cos(t), cy + r * np.sin(t)


# ----------------------------------------------------------------------------- the coin's true boundary
def coin_truth(img):
    t = filters.threshold_otsu(img)  # float image: 256 bins between its minimum and maximum, like the kit
    mask = img > t
    filled = ndi.binary_fill_holes(mask)  # background (4-connected) that does not touch the border is a hole
    lab = measure.label(filled, connectivity=2)
    comp = lab == lab[CENTER, CENTER]
    boundary = comp & ~ndi.binary_erosion(comp)  # 4-neighbour erosion, the border counts as outside
    ys, xs = np.nonzero(boundary)
    a = np.column_stack([xs, ys, np.ones_like(xs)]).astype(float)
    sol = np.linalg.lstsq(a, (xs ** 2 + ys ** 2).astype(float), rcond=None)[0]
    cx, cy = sol[0] / 2, sol[1] / 2
    r = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))
    return dict(threshold=float(t), cx=float(cx), cy=float(cy), r=r, area=int(comp.sum()), boundary=int(boundary.sum()), labels=lab, label=int(lab[CENTER, CENTER]), comp=comp)


def mean_distance(x, y, c):
    return float(np.mean(np.abs(np.hypot(x - c["cx"], y - c["cy"]) - c["r"])))


def own_radius(x, y):
    return float(np.mean(np.hypot(x - x.mean(), y - y.mean())))


def verdict(x, y, c):
    d = mean_distance(x, y, c)
    return "found" if d <= 2 else ("inside" if own_radius(x, y) < c["r"] else "outside")


# ----------------------------------------------------------------------------- main
def main():
    img = np.asarray(Image.open(ASSET), dtype=np.uint8)
    assert img.shape == (144, 144)
    img = (img / np.float32(255)).astype(np.float32)  # exactly what the browser and the test helper decode
    img64 = img.astype(np.float64)
    c = coin_truth(img)
    print("coin truth", {k: v for k, v in c.items() if k not in ("labels", "comp")})
    out = dict(asset=ASSET.name, n=N, gamma=GAMMA, kappa=KAPPA, tol=TOL, maxIter=MAX_ITER, center=CENTER,
               truth={k: c[k] for k in ("threshold", "cx", "cy", "r", "area", "boundary")})

    # (1) matrices
    mats = []
    for n, a, b, g in [(12, 0.05, 0.05, 1.0), (12, 0.3, 1.2, 0.7), (100, 0.05, 0.05, 1.0), (100, 1.0, 1.0, 1.0), (60, 0.0, 0.5, 2.0)]:
        m = snake_matrix(n, a, b, g)
        mats.append(dict(n=n, alpha=a, beta=b, gamma=g, inv=flat(m) if n <= 12 else None, firstRow=flat(m[0], 14), trace=float(np.trace(m)), sumAll=float(m.sum())))
    out["matrices"] = mats
    # skimage's A (from its source: A = -alpha * a + beta * b with a, b rolled identity matrices) equals the chapter's A
    n = 30
    eye = np.eye(n)
    a2 = np.roll(eye, -1, axis=0) + np.roll(eye, -1, axis=1) - 2 * eye
    b4 = np.roll(eye, -2, axis=0) + np.roll(eye, -2, axis=1) - 4 * np.roll(eye, -1, axis=0) - 4 * np.roll(eye, -1, axis=1) + 6 * eye
    sk = -0.3 * a2 + 1.2 * b4
    print("max |A_skimage - A_chapter| =", float(np.abs(sk - internal_matrix(n, 0.3, 1.2)).max()))
    out["skimageMatrixDifference"] = float(np.abs(sk - internal_matrix(n, 0.3, 1.2)).max())

    # (2) edge fields on the real crop
    fields = {}
    out["edge"] = {}
    stride = 211
    idx = np.arange(0, 144 * 144, stride)
    for s in (1, 2, 3, 4):
        e, gx, gy = fields[s] = edge_field(img64, s)
        out["edge"][str(s)] = dict(sumE=float(e.sum()), argmax=int(np.argmax(e)), idx=idx.tolist(), e=flat(e.ravel()[idx]), gx=flat(gx.ravel()[idx]), gy=flat(gy.ravel()[idx]))

    # (3) closed-form ring field: the loop must agree to rounding error
    w = h = 100
    yy, xx = np.mgrid[0:h, 0:w]
    ring = np.exp(-(((np.hypot(xx - 49.5, yy - 50.5) - 30) / 4) ** 2))
    rgy, rgx = np.gradient(ring)
    exact = []
    for r0, a, b in [(44, 0.05, 0.05), (44, 0.5, 0.5), (12, 0.1, 0.3)]:
        x0, y0 = circle(49.5, 50.5, r0, 40)
        run = run_snake((ring, rgx, rgy), x0, y0, a, b, tol=1e-6, max_iter=400)
        exact.append(dict(radius=r0, alpha=a, beta=b, n=40, iterations=run["iterations"], converged=run["converged"], x=flat(run["x"], 15), y=flat(run["y"], 15),
                          internal=flat(run["internal"][[0, 1, 5, -1]], 15), external=flat(run["external"][[0, 1, 5, -1]], 15)))
        print("ring field", r0, a, b, run["iterations"], run["converged"])
    out["ring"] = dict(w=w, h=h, cx=49.5, cy=50.5, r=30, width=4, runs=exact)

    # (4) the lab's own runs on the real asset
    configs = [(40, 0.05, 3), (28, 0.05, 3), (40, 1.0, 3), (12, 2.0, 3), (12, 0.05, 2), (12, 0.05, 3), (60, 0.05, 3), (8, 0.05, 3), (40, 1.5, 3), (40, 0.05, 1), (40, 0.05, 6)]
    runs = []
    for radius, smooth, sigma in configs:
        x0, y0 = circle(CENTER, CENTER, radius)
        run = run_snake(fields.setdefault(sigma, edge_field(img64, sigma)), x0, y0, smooth, smooth, snapshots=(10, 30, 100, 300))
        tot = run["internal"] + run["external"]
        rx, ry = circle(c["cx"], c["cy"], c["r"])
        ring_int, ring_ext = energy(fields[sigma][0], rx, ry, smooth, smooth, KAPPA)
        # the mean distance after every step is what the lab plots; keep a few samples of it
        d_trace = None
        rec = dict(radius=radius, smooth=smooth, sigma=sigma, iterations=run["iterations"], converged=run["converged"], lastMove=run["last_move"],
                   x=flat(run["x"], 12), y=flat(run["y"], 12), distance=mean_distance(run["x"], run["y"], c), ownRadius=own_radius(run["x"], run["y"]),
                   verdict=verdict(run["x"], run["y"], c), internalFinal=float(run["internal"][-1]), externalFinal=float(run["external"][-1]), totalFinal=float(tot[-1]),
                   totalStart=float(tot[0]), ringEnergy=float(ring_int + ring_ext), monotone=bool(np.all(np.diff(tot) <= 1e-9)),
                   energySamples=flat(tot[[0, 1, min(5, len(tot) - 1), min(25, len(tot) - 1), -1]]),
                   snapshots={str(k): dict(x=flat(v[0], 12), y=flat(v[1], 12)) for k, v in run["snaps"].items()})
        # distance to the *other* coins: share of final points within 3 px of a boundary pixel of another Otsu component
        others = (c["labels"] > 0) & (c["labels"] != c["label"])
        ob = others & ~ndi.binary_erosion(others)
        oy, ox = np.nonzero(ob)  # exact Euclidean distance from each point to the nearest boundary pixel (centre) of another coin
        near = np.min(np.hypot(run["x"][:, None] - ox[None, :], run["y"][:, None] - oy[None, :]), axis=1)
        rec["nearOtherCoin3px"] = int(np.sum(near <= 3))
        runs.append(rec)
        print(f"r0={radius:3d} smooth={smooth:5} sigma={sigma}: {rec['verdict']:8s} dist={rec['distance']:6.2f} own={rec['ownRadius']:5.1f} it={run['iterations']:4d} conv={run['converged']} "
              f"E0={tot[0]:8.1f} Ef={tot[-1]:8.1f} ring={ring_int + ring_ext:8.1f} monotone={rec['monotone']} nearOther={rec['nearOtherCoin3px']}")
    out["runs"] = runs

    # (5) the capture map quoted (as a whole) in the report: verdict for start radius x smoothness at sigma = 3 and 2
    grid = {}
    for sigma in (2, 3):
        f = fields.setdefault(sigma, edge_field(img64, sigma))
        for smooth in (0.05, 0.2, 0.5, 1.0, 1.5, 2.0):
            row = []
            for radius in range(8, 65, 4):
                x0, y0 = circle(CENTER, CENTER, radius)
                run = run_snake(f, x0, y0, smooth, smooth)
                row.append(f"{radius}:{verdict(run['x'], run['y'], c)[0]}{mean_distance(run['x'], run['y'], c):.1f}")
            grid[f"sigma{sigma}-smooth{smooth}"] = row
            print(f"sigma={sigma} smooth={smooth:4}", " ".join(row))
    out["captureMap"] = grid

    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", OUT, f"{OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
