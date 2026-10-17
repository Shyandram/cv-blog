import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  N, LABELS, TIE_RANK, pairColumns, metricFor, nearest, majority, voteTied, leaveOneOut, nearestChanged, decisionMap, boxFor, accuracyCurve, lab as neighborLab, type PairKey,
} from '../../src/lib/playground/labs/neighbor-geometry.ts';

const golden = JSON.parse(readFileSync(new URL('./golden/neighbors.json', import.meta.url), 'utf8'));
const acc = (pair: PairKey, scale: string, k: number) => leaveOneOut(pair, metricFor(pair, scale), k);

test('iris data: 150 flowers, 50 per species, labels equal the golden labels', () => {
  assert.equal(N, 150);
  assert.equal(LABELS.join(''), golden.labels);
  assert.deepEqual([0, 1, 2].map(c => LABELS.filter(v => v === c).length), [50, 50, 50]);
});

test('the lab reproduces an independent NumPy implementation exactly, on every golden configuration', () => {
  assert.equal(golden.configs.length, 108);
  for (const c of golden.configs) {
    const r = acc(c.pair, c.scale, c.k);
    assert.equal(r.pred.join(''), c.mine, `${c.pair} ${c.scale} k=${c.k}`);
    assert.equal(r.correct, c.mineCorrect);
  }
});

test('scikit-learn leave-one-out agrees on every prediction that does not depend on a tie', () => {
  assert.equal(golden.tieFreeDisagreements, 0);
  assert.ok(golden.tieFreePredictions > 8000);
  for (const c of golden.configs) {
    const r = acc(c.pair, c.scale, c.k);
    for (let i = 0; i < N; i++) if (c.loose[i] === '0') assert.equal(String(r.pred[i]), c.sklearn[i], `${c.pair} ${c.scale} k=${c.k} sample ${i}`);
  }
});

test('ties: the lab flags exactly the golden tie-dependent answers; scikit-learn differs by at most 7 flowers (quoted in the note)', () => {
  for (const c of golden.configs) {
    const r = acc(c.pair, c.scale, c.k);
    assert.equal(r.loose.join(''), c.loose, `${c.pair} ${c.scale} k=${c.k}`);
    assert.equal(r.tieDependent, c.loose.split('').filter((v: string) => v === '1').length);
    assert.ok(Math.abs(c.mineCorrect - c.sklearnCorrect) <= 7);
  }
  assert.equal(golden.maxCorrectDifference, 7);
  assert.equal(golden.configs.length, 3 * 6 * 6, '3 pairs x 6 units x 6 values of k');
});

test('tie rules: equal distances follow the tie rank; a tie between classes goes to the class with the nearest member', () => {
  const a = [0, 2, -2, 5], b = [0, 0, 0, 0], m = { wa: 1, wb: 1 };
  assert.deepEqual(nearest(a, b, m, 0, 0, 2, -1, [0, 1, 2, 3]), [0, 1], 'sample 0 is at distance 0; samples 1 and 2 tie at 2, the smaller rank wins');
  assert.deepEqual(nearest(a, b, m, 0, 0, 2, -1, [0, 2, 1, 3]), [0, 2], 'with the ranks swapped the other one wins');
  assert.deepEqual(nearest(a, b, m, 0, 0, 3, -1, [0, 2, 1, 3]), [0, 2, 1]);
  assert.deepEqual(nearest(a, b, m, 0, 0, 2, 0, [0, 1, 2, 3]), [1, 2], 'leave-one-out skips the sample itself');
  assert.equal(majority([1, 1, 2, 2, 0], [2, 0, 1, 3]), 2, 'a 2-2 vote between classes 1 and 2 goes to class 2: its member is nearest');
  assert.equal(majority([1, 1, 2, 2, 0], [0, 2, 1, 3]), 1);
  assert.equal(majority([2, 0, 1], [1, 0, 2]), 0, 'a three-way tie goes to the nearest member');
  assert.equal(voteTied([1, 1, 2, 2, 0], [0, 1, 2, 3]), true);
  assert.equal(voteTied([1, 1, 2, 2, 0], [0, 1, 4]), false);
});

test('the tie rank is a permutation of 0..149, independent of the class order', () => {
  assert.deepEqual([...TIE_RANK].sort((x, y) => x - y), Array.from({ length: 150 }, (_, i) => i));
  const meanRank = [0, 1, 2].map(c => { const r = TIE_RANK.filter((_, i) => LABELS[i] === c); return r.reduce((s, v) => s + v, 0) / r.length; });
  for (const v of meanRank) assert.ok(Math.abs(v - 74.5) < 15, `mean rank per species should be near 74.5, got ${meanRank}`);
});

test('scaling both features by the same factor changes nothing: only the ratio between them matters', () => {
  for (const pair of ['width', 'sepal', 'petal'] as PairKey[]) {
    const m = metricFor(pair, 'mm'), big = { wa: m.wa * 100, wb: m.wb * 100 };
    assert.equal(leaveOneOut(pair, big, 5).pred.join(''), leaveOneOut(pair, m, 5).pred.join(''));
  }
});

test('rescaling is reversible: dividing the scaled column by the scale returns the original numbers', () => {
  const { b } = pairColumns('width');
  assert.deepEqual(b.map(v => (v * 10) / 10), b);
});

test('facts quoted in the lab text (k = 5)', () => {
  // Predict explanation.
  assert.deepEqual([acc('width', 'dm', 5).correct, acc('width', 'cm', 5).correct], [129, 143]);
  assert.deepEqual([acc('sepal', 'cm', 5).correct, acc('sepal', 'tenth', 5).correct], [112, 97]);
  assert.equal((129 / 150).toFixed(3), '0.860'); assert.equal((143 / 150).toFixed(3), '0.953');
  assert.equal((112 / 150).toFixed(3), '0.747'); assert.equal((97 / 150).toFixed(3), '0.647');
  // Multiplying both columns by 10 changes no answer (said in the explanation).
  const w = metricFor('width', 'dm');
  assert.equal(leaveOneOut('width', { wa: w.wa * 100, wb: w.wb * 100 }, 5).pred.join(''), leaveOneOut('width', w, 5).pred.join(''));
  // Default state: sepal width + petal width, petal width in decimetres, against the as-measured centimetres.
  assert.equal(nearestChanged('width', metricFor('width', 'dm'), metricFor('width', 'cm')), golden.nearestChanged.width.dm);
  assert.ok(golden.nearestChanged.width.dm > 30, 'the default should change many nearest neighbours');
});

test('the number of changed nearest neighbours matches the golden count for every pair and unit', () => {
  for (const pair of ['width', 'sepal', 'petal'] as PairKey[])
    for (const scale of ['m', 'dm', 'cm', 'mm', 'tenth', 'z']) assert.equal(nearestChanged(pair, metricFor(pair, scale), metricFor(pair, 'cm')), golden.nearestChanged[pair][scale], `${pair} ${scale}`);
  assert.equal(nearestChanged('width', metricFor('width', 'cm'), metricFor('width', 'cm')), 0);
});

test('the decision map is the k-NN decision: a cell on top of a sample is that sample\'s class for k = 1, and the map changes with the unit', () => {
  const pair: PairKey = 'width', box = boxFor(pair), { a, b } = pairColumns(pair), m = metricFor(pair, 'cm');
  const g = decisionMap(pair, m, 1, box, 64);
  assert.equal(g.length, 64 * 64);
  for (const i of [0, 60, 120]) {
    const gx = Math.floor((a[i] - box.a0) / (box.a1 - box.a0) * 64), gy = Math.floor((box.b1 - b[i]) / (box.b1 - box.b0) * 64);
    const qa = box.a0 + (gx + 0.5) * (box.a1 - box.a0) / 64, qb = box.b1 - (gy + 0.5) * (box.b1 - box.b0) / 64;
    assert.equal(g[gy * 64 + gx], LABELS[nearest(a, b, m, qa, qb, 1)[0]]);
  }
  const other = decisionMap(pair, metricFor(pair, 'dm'), 1, box, 64);
  let diff = 0;
  for (let i = 0; i < g.length; i++) if (g[i] !== other[i]) diff++;
  assert.ok(diff > 100, `the unit should move the boundary, only ${diff} cells changed`);
});

test('the accuracy curve contains the golden accuracies at the decade scales', () => {
  for (const pair of ['width', 'sepal', 'petal'] as PairKey[]) {
    const curve = accuracyCurve(pair, 5);
    for (const [scale, s] of [['m', 0.01], ['dm', 0.1], ['cm', 1], ['mm', 10]] as const) {
      const g = golden.configs.find((c: any) => c.pair === pair && c.scale === scale && c.k === 5);
      assert.equal(curve.find(c => Math.abs(c.s - s) < 1e-12)!.correct, g.mineCorrect);
    }
  }
});

test('the lab definition is consistent: three controls, a valid default, and the default shows a change', () => {
  assert.equal(neighborLab.controls.length, 3);
  assert.equal(neighborLab.id, 'neighbor-geometry');
  assert.ok(neighborLab.predict.options.some(o => o.value === neighborLab.predict.answer));
  const def = Object.fromEntries(neighborLab.controls.map(c => [c.key, c.value]));
  assert.deepEqual(def, { pair: 'width', scale: 'dm', k: 5 });
});
