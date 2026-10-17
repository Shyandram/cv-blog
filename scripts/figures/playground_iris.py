"""Write the bundled scikit-learn iris data (150 samples, 4 features in cm, 3 classes) as a small JSON file for the
neighbor-geometry lab. No download: `load_iris` ships with scikit-learn.

    .venv/bin/python scripts/figures/playground_iris.py

Writes only src/lib/playground/labs/data/iris.json.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris

TIE_SEED = 20260617
OUT = Path(__file__).resolve().parents[2] / "src/lib/playground/labs/data/iris.json"


def main() -> None:
    d = load_iris()
    # rank[i] = position of sample i in a fixed random order (inverse of the permutation).
    order = np.random.RandomState(TIE_SEED).permutation(len(d.target))
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    payload = {
        "source": "scikit-learn load_iris (Fisher 1936 data; UCI Iris, doi:10.24432/C56C76)",
        "features": ["sepal length", "sepal width", "petal length", "petal width"],
        "unit": "cm",
        "classes": [str(c) for c in d.target_names],
        # Row i: the four measurements of sample i, in centimetres (rounded to 0.1 cm in the source).
        "x": [[round(float(v), 1) for v in row] for row in d.data],
        "y": [int(v) for v in d.target],
        # The file lists the species one after another, so "smaller index wins" would favour setosa, then versicolor.
        # Equal distances are instead ordered by this fixed random rank (a permutation of 0..149, seed below).
        "tieSeed": TIE_SEED,
        "tieRank": [int(v) for v in rank],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(payload['y'])} samples)")


if __name__ == "__main__":
    main()
