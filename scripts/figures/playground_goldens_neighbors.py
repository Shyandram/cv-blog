"""Golden numbers for the `neighbor-geometry` lab (iris k-NN under rescaling).

    .venv/bin/python scripts/figures/playground_goldens_neighbors.py

Writes only scripts/tests/golden/neighbors.json. For every (feature pair, unit of the second feature, k) it stores

* `mine`: leave-one-out predictions of an independent NumPy implementation of the lab's rule (squared distance
  wa*da^2 + wb*db^2 in integer tenths of a centimetre, so equal distances are exact ties; equal distances are ordered
  by the fixed random rank stored in iris.json; a tie between classes goes to the class with the nearest member);
* `sklearn`: leave-one-out predictions of scikit-learn `KNeighborsClassifier(algorithm="brute")` on the same data with
  the second column multiplied by the scale (`StandardScaler` for "z");
* `loose`: 1 where the prediction depends on a tie (the k-th and (k+1)-th nearest distances are equal, or two classes
  tie in the vote), where scikit-learn is free to order equal distances differently.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scripts/tests/golden/neighbors.json"
IRIS = ROOT / "src/lib/playground/labs/data/iris.json"
PAIRS = {"width": (1, 3), "sepal": (0, 1), "petal": (2, 3)}
SCALES = {"m": (1, 100), "dm": (1, 10), "cm": (1, 1), "mm": (10, 1), "tenth": (100, 1), "z": None}
KS = (1, 3, 5, 9, 15, 25)


def weights(a, b, scale):
    if scale == "z":
        return 1.0 / np.var(a), 1.0 / np.var(b)  # population variance in tenths, as StandardScaler
    p, q = SCALES[scale]
    return float(q * q), float(p * p)


def reference(a, b, y, wa, wb, k):
    """Return (predictions, loose flags, nearest-other-sample index) of the lab's rule."""
    n = len(y)
    rank = np.array(json.loads(IRIS.read_text())["tieRank"])
    da = a[:, None] - a[None, :]
    db = b[:, None] - b[None, :]
    d = wa * da * da + wb * db * db
    pred = np.zeros(n, int)
    loose = np.zeros(n, int)
    nn = np.zeros(n, int)
    for i in range(n):
        order = [j for j in np.lexsort((rank, d[i])) if j != i]  # by distance, then by the fixed random rank
        nn[i] = order[0]
        top = order[:k]
        votes = np.bincount(y[top], minlength=3)
        tied = {c for c in range(3) if votes[c] == votes.max()}
        pred[i] = next(int(y[j]) for j in top if int(y[j]) in tied)  # tied classes: the one with the nearest member
        tied_votes = len(tied) > 1
        edge = d[i][order[k - 1]] == d[i][order[k]] if k < n - 1 else False
        loose[i] = int(tied_votes or edge)
    return pred, loose, nn


def main() -> None:
    iris = load_iris()
    y = iris.target
    tenths = np.rint(iris.data * 10).astype(np.int64)
    configs, changed = [], {}
    for pair, (ca, cb) in PAIRS.items():
        a, b = tenths[:, ca].astype(float), tenths[:, cb].astype(float)
        base_nn = None
        changed[pair] = {}
        for scale in SCALES:
            wa, wb = weights(a, b, scale)
            for k in KS:
                pred, loose, _ = reference(a, b, y, wa, wb, k)
                feats = iris.data[:, [ca, cb]].copy()
                if scale == "z":
                    model = KNeighborsClassifier(n_neighbors=k, algorithm="brute")
                    sk = cross_val_predict(model, StandardScaler().fit_transform(feats), y, cv=LeaveOneOut())
                else:
                    p, q = SCALES[scale]
                    feats[:, 1] *= p / q
                    sk = cross_val_predict(KNeighborsClassifier(n_neighbors=k, algorithm="brute"), feats, y, cv=LeaveOneOut())
                configs.append({
                    "pair": pair, "scale": scale, "k": k,
                    "mine": "".join(map(str, pred)), "sklearn": "".join(map(str, sk)), "loose": "".join(map(str, loose)),
                    "mineCorrect": int((pred == y).sum()), "sklearnCorrect": int((sk == y).sum()),
                })
            _, _, nn = reference(a, b, y, wa, wb, 1)
            if scale == "cm":
                base_nn = nn
            changed[pair][scale] = nn.tolist()
        # Replace the stored neighbour lists by counts relative to the as-measured (cm) setting.
        changed[pair] = {s: int((np.array(v) != base_nn).sum()) for s, v in changed[pair].items()}
    # Fraction of tie-free predictions on which scikit-learn and the lab agree.
    free = [(c["mine"][i], c["sklearn"][i]) for c in configs for i in range(150) if c["loose"][i] == "0"]
    disagree_free = sum(1 for m, s in free if m != s)
    max_diff = max(abs(c["mineCorrect"] - c["sklearnCorrect"]) for c in configs)
    max_flips = max(sum(1 for m, s in zip(c["mine"], c["sklearn"]) if m != s) for c in configs)
    payload = {
        "note": "k-NN leave-one-out on iris; see playground_goldens_neighbors.py",
        "labels": "".join(map(str, y)),
        "configs": configs,
        "nearestChanged": changed,
        "tieFreeDisagreements": disagree_free,
        "tieFreePredictions": len(free),
        "maxCorrectDifference": max_diff,
        "maxDifferingPredictions": max_flips,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(configs)} configurations)")
    for c in configs:
        if c["k"] == 5:
            loose = c["loose"].count("1")
            print(f"  {c['pair']:6s} {c['scale']:5s} k=5  lab {c['mineCorrect']:3d}/150  sklearn {c['sklearnCorrect']:3d}/150  tie-dependent samples {loose}")
    print("tie-free predictions:", len(free), "disagreements:", disagree_free)
    print("max |correct difference|:", max_diff, " max differing predictions:", max_flips)


if __name__ == "__main__":
    main()
