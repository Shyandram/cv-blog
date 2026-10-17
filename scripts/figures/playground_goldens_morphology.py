"""Golden values for the morphology and labeling kit (src/lib/imagekit/morphology.ts, label.ts) and for the IP 09 lab
(src/lib/playground/labs/morphology-shapes.ts), computed with SciPy and scikit-image.

* Kit goldens: random binary images, every element shape and several sizes, all five operations (scipy.ndimage binary_*
  with an explicit footprint; pixels outside the image are background), the structuring elements themselves
  (skimage.morphology.disk / footprint_rectangle), and skimage.measure.label for 4- and 8-connectivity. The script also
  checks that skimage's own operations agree with SciPy away from the border (they pad differently at the border).
* Lab facts: the numbers quoted in the lab's text, computed on the exact asset src/assets/playground/coins-mask.png.

Run: .venv/bin/python scripts/figures/playground_goldens_morphology.py
Writes only scripts/tests/golden/morphology.json.
"""
import json
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure, morphology

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scripts/tests/golden/morphology.json"
ST8 = np.ones((3, 3), int)


def footprint(shape, r):
    if shape == "disk":
        return morphology.disk(r).astype(bool)
    n = 2 * r + 1
    return {"square": np.ones((n, n), bool), "hline": np.ones((1, n), bool), "vline": np.ones((n, 1), bool)}[shape]


def apply(op, m, fp):
    if op == "erosion":
        return ndi.binary_erosion(m, fp, border_value=0)
    if op == "dilation":
        return ndi.binary_dilation(m, fp, border_value=0)
    if op == "opening":
        return ndi.binary_opening(m, fp, border_value=0)
    if op == "closing":
        return ndi.binary_closing(m, fp, border_value=0)
    return ndi.binary_dilation(m, fp, border_value=0) & ~ndi.binary_erosion(m, fp, border_value=0)


def bits(a):
    return "".join("1" if v else "0" for v in np.asarray(a).ravel())


def random_image(seed, h=30, w=36, sigma=1.4, level=0.15):
    f = ndi.gaussian_filter(np.random.default_rng(seed).standard_normal((h, w)), sigma)
    return f > np.quantile(f, 1 - level - 0.05 * (seed % 3))


SHAPES = ["disk", "square", "hline", "vline"]
OPS = ["erosion", "dilation", "opening", "closing", "gradient"]
SIZES = [1, 2, 3, 5]
g = {"images": [], "elements": {}, "labels": []}
for seed in (1, 2, 3):
    m = random_image(seed)
    case = {"h": m.shape[0], "w": m.shape[1], "mask": bits(m), "ops": {}}
    for shape in SHAPES:
        for r in SIZES:
            fp = footprint(shape, r)
            for op in OPS:
                out = apply(op, m, fp)
                case["ops"][f"{op}|{shape}|{r}"] = bits(out)
                # skimage agrees with SciPy away from the border, where the two pad differently.
                if shape in ("disk", "square") and op in ("erosion", "dilation", "opening", "closing"):
                    sk = {"erosion": morphology.erosion, "dilation": morphology.dilation, "opening": morphology.opening, "closing": morphology.closing}[op](m.astype(np.uint8), fp).astype(bool)
                    inner = (slice(2 * r + 1, -2 * r - 1), slice(2 * r + 1, -2 * r - 1))
                    assert (sk[inner] == out[inner]).all(), (op, shape, r)
    g["images"].append(case)
    for conn in (1, 2):
        lab = measure.label(m, connectivity=conn)
        g["labels"].append({"image": len(g["images"]) - 1, "connectivity": 4 if conn == 1 else 8, "count": int(lab.max()), "labels": [int(v) for v in lab.ravel()]})
# Holes: background components that do not touch the border, with the dual connectivity.
for case in g["images"]:
    m = np.array([c == "1" for c in case["mask"]]).reshape(case["h"], case["w"])
    lab, n = ndi.label(~m)                       # 4-connected background, for an 8-connected foreground
    touching = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
    case["holes8"] = int(sum(1 for k in range(1, n + 1) if k not in touching))
for shape in SHAPES:
    for r in range(1, 13):
        g["elements"][f"{shape}|{r}"] = bits(footprint(shape, r))

# ---- Lab facts on the exact asset.
m = np.asarray(Image.open(ROOT / "src/assets/playground/coins-mask.png")) > 127
h, w = m.shape


def stats(out, small_limit=200):
    lab, n = ndi.label(m, ST8)
    areas = np.bincount(lab.ravel())[1:]
    small_ids = [i + 1 for i in range(n) if areas[i] < small_limit]
    gone = sum(1 for i in small_ids if not out[lab == i].any())
    lo, no = ndi.label(out, ST8)
    hl, nh = ndi.label(~out)
    touching = set(hl[0]) | set(hl[-1]) | set(hl[:, 0]) | set(hl[:, -1])
    coin_px = np.isin(lab, [i + 1 for i in range(n) if areas[i] >= small_limit])      # pixels of the 24 coins in the input
    return dict(area_in=int(m.sum()), area_out=int(out.sum()), area_change_pct=float(100 * (out.sum() - m.sum()) / m.sum()),
                coin_area_in=int(coin_px.sum()), coin_area_out=int((out & coin_px).sum()), coin_change_pct=float(100 * ((out & coin_px).sum() - coin_px.sum()) / coin_px.sum()),
                comps_in=int(n), comps_out=int(no), small_in=len(small_ids), small_gone=int(gone),
                holes_out=int(sum(1 for k in range(1, nh + 1) if k not in touching)))


hl0, nh0 = ndi.label(~m)
touch0 = set(hl0[0]) | set(hl0[-1]) | set(hl0[:, 0]) | set(hl0[:, -1])
lab0, n0 = ndi.label(m, ST8)
areas0 = np.bincount(lab0.ravel())[1:]
lab4, n4 = ndi.label(m)
facts = {"w": w, "h": h, "holes_in": int(sum(1 for k in range(1, nh0 + 1) if k not in touch0)), "comps8": int(n0), "comps4": int(n4),
         "small_areas": sorted(int(a) for a in areas0 if a < 200), "coin_areas": sorted(int(a) for a in areas0 if a >= 200), "cases": {}}
for op, shape, r in [("opening", "disk", 4), ("closing", "disk", 4), ("opening", "hline", 4), ("opening", "vline", 4), ("opening", "square", 4), ("closing", "hline", 4),
                     ("erosion", "disk", 4), ("dilation", "disk", 4), ("gradient", "disk", 4), ("opening", "disk", 1), ("opening", "disk", 3), ("closing", "disk", 3),
                     ("dilation", "disk", 12), ("closing", "disk", 12), ("opening", "disk", 20), ("opening", "disk", 16), ("opening", "disk", 18), ("closing", "disk", 8)]:
    out = apply(op, m, footprint(shape, r))
    facts["cases"][f"{op}|{shape}|{r}"] = stats(out)
# The smallest coin and the first size at which a disk opening makes a coin disappear.
coin_ids = [i + 1 for i in range(n0) if areas0[i] >= 200]
inscribed = []
for i in coin_ids:
    inscribed.append(float(ndi.distance_transform_edt(lab0 == i).max()))
facts["inscribed_radius_min"] = min(inscribed)
facts["inscribed_radius_max"] = max(inscribed)
sweep = {}
for r in range(1, 25):
    sweep[r] = int(ndi.label(ndi.binary_opening(m, footprint("disk", r), border_value=0), ST8)[1])
facts["opening_disk_components"] = sweep
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from playground_restore_assets import build  # noqa: E402

clean, mask, holes, specks, scratches = build()
assert (mask == m).all(), "the asset PNG is out of date: run playground_restore_assets.py"
hc, nhc = ndi.label(~clean)
touch_c = set(hc[0]) | set(hc[-1]) | set(hc[:, 0]) | set(hc[:, -1])
facts["clean"] = {"components": int(ndi.label(clean, ST8)[1]), "holes": int(sum(1 for k in range(1, nhc + 1) if k not in touch_c)), "area": int(clean.sum()),
                  "pixels_differing_from_input": int((clean != m).sum()), "holes_added": len(holes), "specks_added": len(specks), "scratches_added": len(scratches)}
g["lab"] = facts
OUT.write_text(json.dumps(g))
print("wrote", OUT, f"{OUT.stat().st_size / 1e3:.0f} kB")
print("input: components", n0, "holes", facts["holes_in"], "small", len(facts["small_areas"]), "coins", len(facts["coin_areas"]), "area", int(m.sum()))
for k, v in facts["cases"].items():
    print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})
print("inscribed radius of the coins: min", round(facts["inscribed_radius_min"], 2), "max", round(facts["inscribed_radius_max"], 2))
print("opening disk components", sweep)
