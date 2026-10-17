// CART decision trees and bagging for two-dimensional data, in plain TypeScript (no DOM), for the bagging-trees lab.
import { mulberry32 } from '../../imagekit/rng.ts';

/** A binary classification tree in flat arrays. Node 0 is the root; `feature[i] < 0` marks a leaf. */
export interface Tree {
  feature: Int8Array;
  threshold: Float64Array;
  left: Int32Array;
  right: Int32Array;
  /** Samples of the training set (duplicates counted) that reached the node, and how many of them are class 1. */
  count: Int32Array;
  ones: Int32Array;
  depth: number;
  /** Internal nodes where another split reached the best score as well (see bestSplit). */
  tiedNodes: number;
}
/** Two splits count as tied when their scores differ by less than this (scores are sums of fractions with denominators up to 300, so double rounding is far below it). */
export const TIE_EPS = 1e-11;

export interface BestSplit { feature: number; threshold: number; score: number; tied: boolean }

/**
 * Best axis-aligned split of the samples `idx` by the Gini criterion. For a split into a left part (nL samples, c0L and c1L of
 * each class) and a right part, the weighted child impurity is n - (c0L^2 + c1L^2) / nL - (c0R^2 + c1R^2) / nR (times 1/n), so
 * the best split MAXIMIZES score = (c0L^2 + c1L^2) / nL + (c0R^2 + c1R^2) / nR. Candidate thresholds are midpoints between
 * consecutive distinct values of one feature.
 *
 * Tie rule (deterministic): when two candidates have scores within TIE_EPS, the one found first wins, scanning feature 0 before
 * feature 1 and thresholds from low to high. `tied` reports whether another candidate with a different partition reached the
 * best score. scikit-learn visits the features in a random order, so it may pick the other candidate at such a node.
 */
export function bestSplit(x: ArrayLike<ArrayLike<number>>, y: ArrayLike<number>, idx: number[]): BestSplit | null {
  const n = idx.length;
  let ones = 0;
  for (const i of idx) ones += y[i];
  let best: BestSplit | null = null;
  for (let f = 0; f < 2; f++) {
    const order = idx.slice().sort((a, b) => x[a][f] - x[b][f] || a - b);
    let l1 = 0;
    for (let p = 0; p < n - 1; p++) {
      l1 += y[order[p]];
      const lo = x[order[p]][f], hi = x[order[p + 1]][f];
      if (!(lo < hi)) continue;
      const nl = p + 1, nr = n - nl, r1 = ones - l1;
      const score = (l1 * l1 + (nl - l1) ** 2) / nl + (r1 * r1 + (nr - r1) ** 2) / nr;
      if (!best || score > best.score + TIE_EPS) best = { feature: f, threshold: (lo + hi) / 2, score, tied: false };
      else if (score >= best.score - TIE_EPS) best.tied = true;
    }
  }
  return best;
}

/**
 * Grow a tree on the samples `idx` (row numbers of x and y; repeated numbers are repeated samples, as in a bootstrap sample)
 * with at most `maxDepth` levels of splits. A node is split while it is impure, the depth limit is not reached and a valid split
 * exists (two different values of a feature). A leaf predicts the majority class of its samples; a 50/50 leaf predicts class 0
 * (the lowest class, as numpy.argmax does in scikit-learn).
 */
export function fitTree(x: ArrayLike<ArrayLike<number>>, y: ArrayLike<number>, idx: number[], maxDepth: number): Tree {
  const feature: number[] = [], threshold: number[] = [], left: number[] = [], right: number[] = [], count: number[] = [], ones: number[] = [];
  let deepest = 0, tied = 0;
  const grow = (rows: number[], depth: number): number => {
    const id = feature.length;
    let c1 = 0;
    for (const i of rows) c1 += y[i];
    feature.push(-1); threshold.push(0); left.push(-1); right.push(-1); count.push(rows.length); ones.push(c1);
    if (depth > deepest) deepest = depth;
    if (depth >= maxDepth || c1 === 0 || c1 === rows.length) return id;
    const s = bestSplit(x, y, rows);
    if (!s) return id;
    const lrows: number[] = [], rrows: number[] = [];
    for (const i of rows) (x[i][s.feature] <= s.threshold ? lrows : rrows).push(i);
    feature[id] = s.feature; threshold[id] = s.threshold;
    if (s.tied) tied++;
    left[id] = grow(lrows, depth + 1);
    right[id] = grow(rrows, depth + 1);
    return id;
  };
  grow(idx, 0);
  return {
    feature: Int8Array.from(feature), threshold: Float64Array.from(threshold), left: Int32Array.from(left), right: Int32Array.from(right),
    count: Int32Array.from(count), ones: Int32Array.from(ones), depth: deepest, tiedNodes: tied,
  };
}
/** The leaf reached by the point (a, b). */
export function leafOf(t: Tree, a: number, b: number): number {
  let i = 0;
  while (t.feature[i] >= 0) i = (t.feature[i] === 0 ? a : b) <= t.threshold[i] ? t.left[i] : t.right[i];
  return i;
}
/** Predicted class (0 or 1) of the point (a, b). */
export function predictTree(t: Tree, a: number, b: number): 0 | 1 {
  const l = leafOf(t, a, b);
  return t.ones[l] * 2 > t.count[l] ? 1 : 0;
}
export const leafCount = (t: Tree) => t.feature.reduce((s, f) => s + (f < 0 ? 1 : 0), 0);

/** Fixed bootstrap seed of the lab (same value in scripts/figures/playground_moons.py). */
export const BOOTSTRAP_SEED = 20260618;
/**
 * Bootstrap samples for members 0..count-1 of a training set of n rows: one seeded stream, member b takes the next n draws,
 * row = floor(u * n). The first M members are therefore the same for every M >= them, so growing the ensemble only adds trees.
 */
export function bootstrapSamples(n: number, count: number, seed = BOOTSTRAP_SEED): number[][] {
  const rng = mulberry32(seed), out: number[][] = [];
  for (let b = 0; b < count; b++) {
    const rows: number[] = new Array(n);
    for (let i = 0; i < n; i++) rows[i] = Math.floor(rng() * n);
    out.push(rows);
  }
  return out;
}
/** One tree per bootstrap sample. */
export function fitBagging(x: ArrayLike<ArrayLike<number>>, y: ArrayLike<number>, samples: number[][], maxDepth: number): Tree[] {
  return samples.map(rows => fitTree(x, y, rows, maxDepth));
}

/** Class predictions of every member at every point: `votes[b][j]` is member b's class at point j. */
export function memberPredictions(trees: Tree[], pts: ArrayLike<ArrayLike<number>>): Uint8Array[] {
  return trees.map(t => { const o = new Uint8Array(pts.length); for (let j = 0; j < pts.length; j++) o[j] = predictTree(t, pts[j][0], pts[j][1]); return o; });
}
/** Number of members voting class 1 at every point, using the first m members. */
export function voteCounts(preds: Uint8Array[], m: number): Uint16Array {
  const out = new Uint16Array(preds[0].length);
  for (let b = 0; b < m; b++) { const p = preds[b]; for (let j = 0; j < out.length; j++) out[j] += p[j]; }
  return out;
}
/** Test error of a majority vote of m members: a point with an exact tie (m even) counts as half an error, the expected error of a fair coin. */
export function voteError(votes: ArrayLike<number>, m: number, y: ArrayLike<number>): number {
  let e = 0;
  for (let j = 0; j < y.length; j++) {
    const v2 = 2 * votes[j];
    e += v2 === m ? 0.5 : (v2 > m ? 1 : 0) === y[j] ? 0 : 1;
  }
  return e / y.length;
}
/** Error of one prediction vector. */
export function errorOf(pred: ArrayLike<number>, y: ArrayLike<number>): number {
  let e = 0;
  for (let j = 0; j < y.length; j++) if (pred[j] !== y[j]) e++;
  return e / y.length;
}
/**
 * Mean pairwise disagreement of the first m members: over all m(m-1)/2 pairs of members and all points, the fraction of
 * (pair, point) combinations on which the two members predict different classes. At a point where s of the m members vote
 * class 1, exactly s(m-s) pairs disagree, so no pair loop is needed. Defined for m >= 2 (0 means identical members).
 */
export function disagreement(votes: ArrayLike<number>, m: number): number {
  if (m < 2) return NaN;
  let s = 0;
  for (let j = 0; j < votes.length; j++) s += votes[j] * (m - votes[j]);
  return s / (votes.length * (m * (m - 1) / 2));
}

export interface Curve { m: number[]; bagged: number[]; member: number[]; disagreement: number[] }
/** Everything the lab plots against M, for M = 1..preds.length, using the test labels. */
export function ensembleCurve(preds: Uint8Array[], y: ArrayLike<number>): Curve {
  const out: Curve = { m: [], bagged: [], member: [], disagreement: [] };
  const votes = new Uint16Array(y.length);
  let memberSum = 0;
  for (let b = 0; b < preds.length; b++) {
    const p = preds[b];
    for (let j = 0; j < votes.length; j++) votes[j] += p[j];
    memberSum += errorOf(p, y);
    out.m.push(b + 1); out.bagged.push(voteError(votes, b + 1, y)); out.member.push(memberSum / (b + 1)); out.disagreement.push(disagreement(votes, b + 1));
  }
  return out;
}
