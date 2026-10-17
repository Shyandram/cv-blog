"""Golden numbers for the `bagging-trees` lab (CART decision trees and bagging on two-moons).

    .venv/bin/python scripts/figures/playground_goldens_trees.py

Writes only scripts/tests/golden/trees.json. It stores

* the bootstrap samples (first rows of members 0 and 1) produced by a Python port of the kit's mulberry32 stream, to check the
  TypeScript stream bit for bit;
* for several depth limits, single trees (all 300 training rows) and bootstrap trees, the class predictions of
  - `port`: an independent NumPy implementation of the lab's rule (Gini, midpoint thresholds, strict improvement so the first
    candidate wins a tie, feature 0 before feature 1, low thresholds first; ties decided with exact rational arithmetic),
  - `sklearn`: DecisionTreeClassifier(criterion="gini", max_depth=...),
  at the 300 training points, the 1000 test points and an 80 x 56 grid;
* `tied`: whether any node of the tree has two different splits with exactly the same Gini score. scikit-learn visits features
  in a random order and may break such a tie the other way, so only tie-free trees must agree exactly with scikit-learn;
* a node-by-node check of scikit-learn's trees against the port's scoring: every split scikit-learn made has the maximal score
  on its node's samples, and where the maximum is unique it is the very split the port makes;
* the ensemble curves (test error of the majority vote, mean member error, mean pairwise disagreement for M = 1..100) of the
  port, plus the same curves from scikit-learn trees fitted on the same bootstrap samples, for a few depths.
"""
import base64
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from playground_moons import BOOTSTRAP_SEED, DATA_SEED, mulberry32, pick_seed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scripts/tests/golden/trees.json"
DATA = json.loads((ROOT / "src/lib/playground/labs/data/moons.json").read_text())
X = np.array(DATA["train"]["x"])
Y = np.array(DATA["train"]["y"])
XT = np.array(DATA["test"]["x"])
YT = np.array(DATA["test"]["y"])
N = len(Y)
# Same box and grid as the lab (BOX, GRID_W, GRID_H in src/lib/playground/labs/bagging-trees.ts): cell centres, row 0 on top.
BOX = (-2.0, 3.0, -1.5, 2.0)
GRID_W, GRID_H = 80, 56
GX = BOX[0] + (np.arange(GRID_W) + 0.5) * (BOX[1] - BOX[0]) / GRID_W
GY = BOX[3] - (np.arange(GRID_H) + 0.5) * (BOX[3] - BOX[2]) / GRID_H
GRID = np.array([[gx, gy] for gy in GY for gx in GX])
DEPTHS = [1, 2, 3, 4, 6, 8, 30]  # 30 = no limit in practice
EPS = 1e-9


def bootstrap(count):
    rng = mulberry32(BOOTSTRAP_SEED)
    return [[int(rng() * N) for _ in range(N)] for _ in range(count)]


def score_exact(c0l, c1l, c0r, c1r):
    return Fraction(c0l * c0l + c1l * c1l, c0l + c1l) + Fraction(c0r * c0r + c1r * c1r, c0r + c1r)


def candidates(rows):
    """All valid splits of the sample rows: (exact score, feature, threshold). Order: feature 0 first, thresholds ascending."""
    out = []
    y = Y[rows]
    total1 = int(y.sum())
    for f in range(2):
        v = X[rows, f]
        order = np.argsort(v, kind="stable")
        vs, ys = v[order], y[order]
        left1 = np.cumsum(ys)
        for p in range(len(rows) - 1):
            if not vs[p] < vs[p + 1]:
                continue
            nl = p + 1
            l1 = int(left1[p])
            out.append((score_exact(nl - l1, l1, len(rows) - nl - (total1 - l1), total1 - l1), f, (vs[p] + vs[p + 1]) / 2))
    return out


def port_tree(rows, max_depth):
    """Independent builder: returns nested dict nodes. `tied` is set when the best score is reached by 2+ candidates."""
    nodes = []

    def grow(r, depth):
        i = len(nodes)
        ones = int(Y[r].sum())
        node = {"ones": ones, "n": len(r), "f": -1}
        nodes.append(node)
        if depth >= max_depth or ones in (0, len(r)):
            return i
        cands = candidates(r)
        if not cands:
            return i
        best = max(c[0] for c in cands)
        winners = [c for c in cands if c[0] == best]
        _, f, thr = winners[0]  # first in scan order wins
        node.update(f=f, thr=thr, tied=len(winners) > 1, best=best)
        mask = X[r, f] <= thr
        node["l"] = grow([k for k, m in zip(r, mask) if m], depth + 1)
        node["r"] = grow([k for k, m in zip(r, mask) if not m], depth + 1)
        return i

    grow(list(rows), 0)
    return nodes


def port_predict(nodes, pts):
    out = np.zeros(len(pts), dtype=np.uint8)
    for j, (a, b) in enumerate(pts):
        i = 0
        while nodes[i]["f"] >= 0:
            i = nodes[i]["l"] if (a if nodes[i]["f"] == 0 else b) <= nodes[i]["thr"] else nodes[i]["r"]
        out[j] = 1 if nodes[i]["ones"] * 2 > nodes[i]["n"] else 0
    return out


def hexbits(bits):
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes().hex()


def sklearn_nodes_check(clf, rows, nodes):
    """Walk scikit-learn's tree on the bootstrap sample. Returns (internal nodes checked, nodes with a unique best split,
    non-maximal splits, unique-best splits that differ from the port's)."""
    t = clf.tree_
    checked = unique = not_max = differ = 0
    xs = X.astype(np.float32).astype(np.float64)

    def walk(i, r):
        nonlocal checked, unique, not_max, differ
        if t.children_left[i] < 0:
            return
        f, thr = int(t.feature[i]), float(t.threshold[i])
        cands = candidates(r)
        best = max(c[0] for c in cands)
        winners = [c for c in cands if c[0] == best]
        checked += 1
        mask = xs[r, f] <= thr
        nl = int(mask.sum())
        y = Y[r]
        l1, total1 = int(y[mask].sum()), int(y.sum())
        s = score_exact(nl - l1, l1, len(r) - nl - (total1 - l1), total1 - l1)
        if s != best:
            not_max += 1
        if len(winners) == 1:
            unique += 1
            if f != winners[0][1] or abs(thr - winners[0][2]) > 1e-6:
                differ += 1
        walk(int(t.children_left[i]), [k for k, m in zip(r, mask) if m])
        walk(int(t.children_right[i]), [k for k, m in zip(r, mask) if not m])

    walk(0, list(rows))
    return checked, unique, not_max, differ


def case(rows, depth):
    nodes = port_tree(rows, depth)
    clf = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=0).fit(X[rows], Y[rows])
    both = np.vstack([X, XT, GRID])
    mine, sk = port_predict(nodes, both), clf.predict(both.astype(np.float32)).astype(np.uint8)
    n1, n2 = N, N + len(YT)
    checked, unique, not_max, differ = sklearn_nodes_check(clf, rows, nodes)
    return {
        "depth": depth, "tied": any(n.get("tied") for n in nodes), "tiedNodes": sum(1 for n in nodes if n.get("tied")),
        "internal": sum(1 for n in nodes if n["f"] >= 0), "leaves": sum(1 for n in nodes if n["f"] < 0),
        "port": {"train": hexbits(mine[:n1]), "test": hexbits(mine[n1:n2]), "grid": hexbits(mine[n2:])},
        "sklearn": {"train": hexbits(sk[:n1]), "test": hexbits(sk[n1:n2]), "grid": hexbits(sk[n2:])},
        "agree": {"train": int((mine[:n1] == sk[:n1]).sum()), "test": int((mine[n1:n2] == sk[n1:n2]).sum()), "grid": int((mine[n2:] == sk[n2:]).sum())},
        "nodeCheck": {"checked": checked, "unique": unique, "notMaximal": not_max, "differFromPortWhenUnique": differ},
        "skDepth": int(clf.get_depth()), "portLeaves": sum(1 for n in nodes if n["f"] < 0), "skLeaves": int(clf.get_n_leaves()),
    }


def curves(depth, count, use_sklearn):
    samples = bootstrap(count)
    preds = []
    for rows in samples:
        if use_sklearn:
            clf = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=0).fit(X[rows], Y[rows])
            preds.append(clf.predict(XT.astype(np.float32)).astype(int))
        else:
            preds.append(port_predict(port_tree(rows, depth), XT).astype(int))
    preds = np.array(preds)
    bagged, member, dis = [], [], []
    for m in range(1, count + 1):
        votes = preds[:m].sum(0)
        wrong = np.where(2 * votes == m, 0.5, ((2 * votes > m).astype(int) != YT).astype(float))
        bagged.append(float(wrong.mean()))
        member.append(float((preds[:m] != YT).mean()))
        if m >= 2:
            pairs = [(preds[a] != preds[b]).mean() for a in range(m) for b in range(a + 1, m)] if m <= 12 else None
            dis.append(float(np.mean(pairs)) if pairs is not None else float((votes * (m - votes)).sum() / (len(YT) * m * (m - 1) / 2)))
        else:
            dis.append(None)
    def wrong_votes(m):
        ones = preds[:m].sum(0)
        return np.where(YT == 1, m - ones, ones)

    return {"bagged": bagged, "member": member, "disagreement": dis,
            "majorityWrong": {str(m): int((2 * wrong_votes(m) > m).sum()) for m in (10, 50, 100)}}


def main() -> None:
    samples = bootstrap(100)
    cases = []
    for depth in DEPTHS:
        cases.append({"which": "single", **case(list(range(N)), depth)})
        for b in range(4):
            cases.append({"which": f"bootstrap{b}", **case(samples[b], depth)})
    ensembles = {}
    for depth in (1, 2, 4, 5, 8, 14, 30):
        ensembles[str(depth)] = {"port": curves(depth, 100, False), "sklearn": curves(depth, 100, True)}
        a, b = ensembles[str(depth)]["port"]["bagged"], ensembles[str(depth)]["sklearn"]["bagged"]
        print(f"depth {depth:2d}: max |port - sklearn| bagged error over M = {max(abs(x - y) for x, y in zip(a, b)):.4f}")
    free = [c for c in cases if not c["tied"]]
    print(f"{len(cases)} trees, {len(free)} without a tied node; tied nodes in total {sum(c['tiedNodes'] for c in cases)} of {sum(c['internal'] for c in cases)} internal nodes")
    print("tree-free exact agreement (train, test, grid):", all(c['agree']['train'] == N and c['agree']['test'] == len(YT) and c['agree']['grid'] == GRID_W * GRID_H for c in free))
    print("node check: nodes", sum(c['nodeCheck']['checked'] for c in cases), "unique", sum(c['nodeCheck']['unique'] for c in cases),
          "not maximal", sum(c['nodeCheck']['notMaximal'] for c in cases), "differ", sum(c['nodeCheck']['differFromPortWhenUnique'] for c in cases))
    chosen, gains = pick_seed()
    print("seed study: chosen", chosen, "mean gain", float(np.mean(gains)), "gain of the chosen seed", float(gains[chosen]), "wins", int((gains > 0).sum()))
    payload = {
        "seedStudy": {"seeds": 30, "chosen": int(chosen), "dataSeed": DATA_SEED, "gains": [float(g) for g in gains]},
        "note": "CART and bagging goldens; see playground_goldens_trees.py",
        "bootstrapHead": [s[:20] for s in samples[:3]], "bootstrapTail": [s[-5:] for s in samples[:3]], "bootstrapSums": [int(sum(s)) for s in samples],
        "grid": {"box": list(BOX), "w": GRID_W, "h": GRID_H},
        "cases": cases, "ensembles": ensembles,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
