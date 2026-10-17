"""Toy datasets for the `mlp-training` lab: scikit-learn `make_moons` and `make_circles`, a training and a held-out test set each.

    .venv/bin/python scripts/figures/playground_mlp_data.py

Writes only src/lib/playground/labs/data/mlp-data.json. Nothing is downloaded: both generators ship with scikit-learn.
Coordinates are rounded to 5 decimals (the golden script and the lab both read the rounded numbers back from this file).
"""
import json
from pathlib import Path

from sklearn.datasets import make_circles, make_moons

OUT = Path(__file__).resolve().parents[2] / "src/lib/playground/labs/data/mlp-data.json"
N = 200  # points per set


def pack(x, y):
    return {"x": [[round(float(a), 5), round(float(b), 5)] for a, b in x], "y": [int(v) for v in y]}


def main() -> None:
    payload = {
        "source": "scikit-learn make_moons(noise=0.2) and make_circles(noise=0.1, factor=0.5); n=200 points per set; "
                  "train uses random_state=0, the held-out test set random_state=1",
        "moons": {"noise": 0.2, "train": pack(*make_moons(n_samples=N, noise=0.2, random_state=0)),
                  "test": pack(*make_moons(n_samples=N, noise=0.2, random_state=1))},
        "circles": {"noise": 0.1, "factor": 0.5, "train": pack(*make_circles(n_samples=N, noise=0.1, factor=0.5, random_state=0)),
                    "test": pack(*make_circles(n_samples=N, noise=0.1, factor=0.5, random_state=1))},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
