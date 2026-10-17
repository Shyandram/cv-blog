import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  bestSplit, fitTree, predictTree, leafCount, bootstrapSamples, memberPredictions, voteCounts, voteError, disagreement, ensembleCurve, errorOf, BOOTSTRAP_SEED, TIE_EPS,
} from '../../src/lib/playground/labs/trees.ts';
import {
  lab, runFor, SAMPLES, TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, N, MAX_M, MAX_DEPTH, DEFAULT, BOX, GRID_W, GRID_H, BAYES_ERROR, cellX, cellY, mapVotes, outOfBag, majorityWrong, shade,
} from '../../src/lib/playground/labs/bagging-trees.ts';
import { mulberry32 } from '../../src/lib/imagekit/rng.ts';

const golden = JSON.parse(readFileSync(new URL('./golden/trees.json', import.meta.url), 'utf8'));
const moons = JSON.parse(readFileSync(new URL('../../src/lib/playground/labs/data/moons.json', import.meta.url), 'utf8'));

const unhex = (hex: string, n: number) => { const out: number[] = []; for (let i = 0; i < n; i++) out.push((parseInt(hex.slice(2 * (i >> 3), 2 * (i >> 3) + 2), 16) >> (7 - (i & 7))) & 1); return out; };
const GRID: number[][] = [];
for (let gy = 0; gy < GRID_H; gy++) for (let gx = 0; gx < GRID_W; gx++) GRID.push([cellX(gx), cellY(gy)]);
const pctText = (v: number) => (v * 100).toFixed(2);

test('data: two moons from make_moons, 300 training and 1000 test points, balanced, from different seeds', () => {
  assert.equal(N, 300); assert.equal(TEST_Y.length, 1000);
  assert.equal(TRAIN_Y.filter(v => v === 1).length, 150); assert.equal(TEST_Y.filter(v => v === 1).length, 500);
  assert.equal(moons.dataSeed, 2); assert.equal(moons.testSeed, 1002); assert.equal(moons.noise, 0.3);
  assert.equal(moons.bootstrapSeed, BOOTSTRAP_SEED);
  for (const [a, b] of TRAIN_X) assert.ok(a > BOX.a0 && a < BOX.a1 && b > BOX.b0 && b < BOX.b1, 'every training point lies inside the map window (test points are not drawn)');
  assert.notDeepEqual(TRAIN_X.slice(0, 5), TEST_X.slice(0, 5));
  assert.equal(golden.grid.w, GRID_W); assert.equal(golden.grid.h, GRID_H); assert.deepEqual(golden.grid.box, [BOX.a0, BOX.a1, BOX.b0, BOX.b1]);
});

test('seed rule: the data seed is the one of 30 draws whose bagging gain is closest to the mean (scikit-learn study); bagging won on all 30', () => {
  const s = golden.seedStudy;
  assert.equal(s.seeds, 30); assert.equal(s.gains.length, 30);
  assert.equal(s.chosen, moons.dataSeed);
  const mean = s.gains.reduce((a: number, v: number) => a + v, 0) / 30;
  const closest = s.gains.map((g: number, i: number) => [Math.abs(g - mean), i]).sort((a: number[], b: number[]) => a[0] - b[0])[0][1];
  assert.equal(closest, s.chosen);
  assert.equal(s.gains.filter((g: number) => g > 0).length, 30);
});

test('bootstrap stream: the kit mulberry32 stream reproduces the NumPy port bit for bit; member b takes the next 300 draws', () => {
  assert.equal(SAMPLES.length, MAX_M);
  golden.bootstrapHead.forEach((h: number[], b: number) => { assert.deepEqual(SAMPLES[b].slice(0, 20), h); assert.deepEqual(SAMPLES[b].slice(-5), golden.bootstrapTail[b]); });
  assert.deepEqual(SAMPLES.map(s => s.reduce((a, v) => a + v, 0)), golden.bootstrapSums);
  const r = mulberry32(BOOTSTRAP_SEED), first = Array.from({ length: N }, () => Math.floor(r() * N)), second = Array.from({ length: N }, () => Math.floor(r() * N));
  assert.deepEqual(SAMPLES[0], first); assert.deepEqual(SAMPLES[1], second);
  for (const s of SAMPLES) assert.ok(s.length === N && s.every(i => Number.isInteger(i) && i >= 0 && i < N));
  assert.deepEqual(bootstrapSamples(N, 3), SAMPLES.slice(0, 3), 'the first M members do not depend on how many are asked for');
});

test('CART rules on tiny examples: Gini split, midpoint threshold, majority leaf, depth limit, tie rule', () => {
  // One feature separates the classes perfectly at 0.5 (midpoint of 0 and 1).
  const x = [[0, 5], [0, 1], [1, 4], [1, 2]], y = [0, 0, 1, 1];
  const s = bestSplit(x, y, [0, 1, 2, 3])!;
  assert.deepEqual([s.feature, s.threshold, s.tied], [0, 0.5, false]);
  // Feature 1 separates just as well (y low -> class 0 would need 2.5): feature 0 and 1 tie, feature 0 wins.
  const x2 = [[0, 0], [1, 1], [2, 2], [3, 3]], y2 = [0, 0, 1, 1];
  const t2 = bestSplit(x2, y2, [0, 1, 2, 3])!;
  assert.deepEqual([t2.feature, t2.threshold, t2.tied], [0, 1.5, true], 'identical features tie; the first feature wins and the tie is reported');
  // Impurity decrease equals the chapter formula: parent Gini 0.5, children pure, so the score is the maximum n.
  assert.ok(Math.abs(t2.score - 4) < TIE_EPS);
  const tree = fitTree(x, y, [0, 1, 2, 3], 1);
  assert.equal(tree.depth, 1); assert.equal(leafCount(tree), 2); assert.equal(tree.tiedNodes, 0);
  assert.deepEqual([predictTree(tree, 0, 9), predictTree(tree, 1, -3)], [0, 1]);
  // Depth 0 is a single majority leaf; a 50/50 leaf predicts class 0.
  assert.equal(predictTree(fitTree(x, y, [0, 1, 2, 3], 0), 0, 0), 0);
  assert.equal(predictTree(fitTree([[0, 0], [0, 0]], [0, 1], [0, 1], 5), 0, 0), 0, 'identical points cannot be split; a 1-1 leaf predicts class 0');
  assert.equal(predictTree(fitTree([[0, 0], [0, 0], [0, 0]], [0, 1, 1], [0, 1, 2], 5), 0, 0), 1);
  // Duplicated rows (a bootstrap sample) count with their multiplicity.
  const dup = fitTree(x, y, [0, 0, 0, 2], 1);
  assert.deepEqual([dup.count[0], dup.ones[0]], [4, 1]);
  // A stump on the XOR pattern still splits (zero gain is allowed) and the tree needs depth 2.
  const xr = [[0, 0], [0, 1], [1, 0], [1, 1]], yr = [0, 1, 1, 0];
  assert.equal(fitTree(xr, yr, [0, 1, 2, 3], 5).depth, 2);
});

test('trees match the NumPy port of the same rule on every golden tree, and scikit-learn exactly wherever no node is tied', () => {
  assert.equal(golden.cases.length, 35);
  let free = 0, tiedTrees = 0;
  for (const c of golden.cases) {
    const rows = c.which === 'single' ? Array.from({ length: N }, (_, i) => i) : SAMPLES[Number(c.which.replace('bootstrap', ''))];
    const tree = fitTree(TRAIN_X, TRAIN_Y, rows, c.depth), tag = `${c.which} depth ${c.depth}`;
    const mine = [TRAIN_X, TEST_X, GRID].map(p => p.map(q => predictTree(tree, q[0], q[1])));
    assert.deepEqual(mine[0], unhex(c.port.train, N), `${tag}: training points`);
    assert.deepEqual(mine[1], unhex(c.port.test, 1000), `${tag}: test points`);
    assert.deepEqual(mine[2], unhex(c.port.grid, GRID_W * GRID_H), `${tag}: grid`);
    assert.equal(leafCount(tree), c.portLeaves, tag);
    assert.equal(tree.tiedNodes > 0, c.tied, `${tag}: tie flag`);
    assert.equal(tree.tiedNodes, c.tiedNodes, tag);
    // Every tree: scikit-learn's depth never exceeds the limit and ours matches the port's.
    assert.ok(tree.depth <= c.depth);
    if (!c.tied) {
      free++;
      assert.equal(c.agree.train, N); assert.equal(c.agree.test, 1000); assert.equal(c.agree.grid, GRID_W * GRID_H);
      assert.deepEqual(mine[0], unhex(c.sklearn.train, N), `${tag}: scikit-learn, training`);
      assert.deepEqual(mine[1], unhex(c.sklearn.test, 1000), `${tag}: scikit-learn, test`);
      assert.deepEqual(mine[2], unhex(c.sklearn.grid, GRID_W * GRID_H), `${tag}: scikit-learn, grid`);
      assert.equal(leafCount(tree), c.skLeaves);
    } else tiedTrees++;
  }
  assert.equal(free, 17); assert.equal(tiedTrees, 18);
});

test('node by node, every split scikit-learn makes has the best Gini score on its samples, and equals ours wherever the best is unique', () => {
  const sum = (k: string) => golden.cases.reduce((a: number, c: any) => a + c.nodeCheck[k], 0);
  assert.equal(sum('checked'), 485); assert.equal(sum('unique'), 429);
  assert.equal(sum('notMaximal'), 0); assert.equal(sum('differFromPortWhenUnique'), 0);
  assert.equal(golden.cases.reduce((a: number, c: any) => a + c.tiedNodes, 0), 55);
});

test('ensemble curves equal the NumPy port exactly; against scikit-learn trees on the same bootstrap samples the vote error differs by at most 0.7 points', () => {
  let worst = 0;
  for (const depth of [1, 2, 4, 5, 8, 14, 30]) {
    const g = golden.ensembles[String(depth)], r = runFor(depth), c = r.curve;
    for (let i = 0; i < MAX_M; i++) {
      assert.ok(Math.abs(c.bagged[i] - g.port.bagged[i]) < 1e-12, `depth ${depth} M=${i + 1} bagged`);
      assert.ok(Math.abs(c.member[i] - g.port.member[i]) < 1e-12, `depth ${depth} M=${i + 1} member`);
      if (i >= 1) assert.ok(Math.abs(c.disagreement[i] - g.port.disagreement[i]) < 1e-12, `depth ${depth} M=${i + 1} disagreement`);
      worst = Math.max(worst, Math.abs(c.bagged[i] - g.sklearn.bagged[i]));
    }
    assert.equal(c.m.length, MAX_M);
  }
  assert.ok(worst <= 0.007 + 1e-9 && worst >= 0.0069, `largest difference to scikit-learn trees is ${worst}`);
  // Depths 1 and 2 have no tie that matters: the curves agree exactly.
  for (const depth of [1, 2]) assert.deepEqual(golden.ensembles[String(depth)].port.bagged, golden.ensembles[String(depth)].sklearn.bagged);
});

test('disagreement: the closed form s(m - s) equals the average over all pairs of members, and 0 means identical members', () => {
  const r = runFor(6), preds = r.testPreds;
  for (const m of [2, 3, 10, 25]) {
    let sum = 0, pairs = 0;
    for (let a = 0; a < m; a++) for (let b = a + 1; b < m; b++) { let d = 0; for (let j = 0; j < TEST_Y.length; j++) if (preds[a][j] !== preds[b][j]) d++; sum += d / TEST_Y.length; pairs++; }
    assert.ok(Math.abs(disagreement(voteCounts(preds, m), m) - sum / pairs) < 1e-12, `m=${m}`);
  }
  const same = [new Uint8Array([0, 1, 1]), new Uint8Array([0, 1, 1]), new Uint8Array([0, 1, 1])];
  assert.equal(disagreement(voteCounts(same, 3), 3), 0);
  const opposite = [new Uint8Array([0, 1]), new Uint8Array([1, 0])];
  assert.equal(disagreement(voteCounts(opposite, 2), 2), 1);
  assert.ok(Number.isNaN(disagreement(voteCounts(same, 1), 1)));
});

test('vote error: majority of the first m members; an exact tie counts as half an error; the curve is prefix-consistent', () => {
  const preds = [new Uint8Array([1, 1, 0, 0]), new Uint8Array([1, 0, 0, 1])], y = [1, 1, 0, 1];
  assert.equal(voteError(voteCounts(preds, 1), 1, y), 0.25);
  // Two members: point 1 is a 1-1 tie (half error), point 3 is a 1-1 tie (half), the rest are unanimous and right.
  assert.equal(voteError(voteCounts(preds, 2), 2, y), (0.5 + 0.5) / 4);
  const r = runFor(8);
  for (const m of [1, 2, 7, 50]) assert.equal(r.curve.bagged[m - 1], voteError(voteCounts(r.testPreds, m), m, TEST_Y));
  assert.equal(r.curve.bagged[0], r.curve.member[0], 'with one tree the vote is that tree');
  const small = ensembleCurve(preds, y);
  assert.deepEqual(small.m, [1, 2]); assert.equal(small.bagged[0], errorOf(preds[0], y));
});

test('the map: cells are classified by the same trees, and a bigger ensemble gives smoother (less decided) cells', () => {
  const r = runFor(DEFAULT.depth);
  const one = mapVotes([r.single]);
  assert.equal(one.length, GRID_W * GRID_H);
  for (const [gx, gy] of [[3, 3], [40, 28], [75, 50]]) assert.equal(one[gy * GRID_W + gx], predictTree(r.single, cellX(gx), cellY(gy)));
  const many = mapVotes(r.trees.slice(0, DEFAULT.m));
  assert.ok(many.some(v => v > 0 && v < DEFAULT.m), 'some cells are contested');
  assert.deepEqual(shade(0), [255 + Math.round((0 - 255) * 0.3), 255 + Math.round((114 - 255) * 0.3), 255 + Math.round((178 - 255) * 0.3)]);
  assert.deepEqual(shade(0.5), [255, 255, 255]);
  const oob = outOfBag(SAMPLES[0]);
  assert.equal(oob.length, N);
  for (const i of SAMPLES[0]) assert.equal(oob[i], 0);
  assert.equal(oob.reduce((a, v) => a + v, 0) + new Set(SAMPLES[0]).size, N);
});

test('the best possible classifier makes about 8.5% errors on the distribution make_moons samples (numerical integration)', () => {
  // class 0: (cos t, sin t); class 1: (1 - cos t, 0.5 - sin t); t uniform on [0, pi]; Gaussian noise sd 0.3 per coordinate.
  const s = 0.3, K = 300, c0: number[][] = [], c1: number[][] = [];
  for (let k = 0; k < K; k++) { const t = (k + 0.5) * Math.PI / K; c0.push([Math.cos(t), Math.sin(t)]); c1.push([1 - Math.cos(t), 0.5 - Math.sin(t)]); }
  const dens = (c: number[][], a: number, b: number) => { let d = 0; for (const [u, v] of c) d += Math.exp(-((a - u) ** 2 + (b - v) ** 2) / (2 * s * s)); return d / (K * 2 * Math.PI * s * s); };
  const h = 0.04;
  let err = 0, m0 = 0, m1 = 0;
  for (let a = -2.5 + h / 2; a < 3.5; a += h) for (let b = -2 + h / 2; b < 2.5; b += h) { const p0 = dens(c0, a, b), p1 = dens(c1, a, b); err += Math.min(p0, p1); m0 += p0; m1 += p1; }
  assert.ok(Math.abs(m0 * h * h - 1) < 1e-3 && Math.abs(m1 * h * h - 1) < 1e-3, 'the densities integrate to 1');
  const bayes = err * h * h / 2;
  assert.ok(Math.abs(bayes - BAYES_ERROR) < 5e-4, `Bayes error ${bayes}`);
  assert.equal((BAYES_ERROR * 100).toFixed(1), '8.5');
});

test('facts quoted in the lab text (data seed 2, bootstrap seed 20260618, 1000 test points)', () => {
  const r14 = runFor(MAX_DEPTH), c = r14.curve;
  assert.equal(DEFAULT.depth, 14); assert.equal(DEFAULT.m, 50); assert.equal(DEFAULT.view, 'bagged');
  assert.ok(Math.max(...r14.trees.map(t => t.depth)) <= 14 && Math.max(...r14.trees.map(t => t.depth)) === 14, 'depth 14 is exactly where the deepest of the 101 trees stops');
  assert.equal(runFor(30).curve.bagged.join(), c.bagged.join(), 'nothing changes beyond depth 14: every tree is fully grown');
  assert.equal(r14.single.depth, 11);
  // Challenge and predict explanation at the default depth.
  assert.equal(pctText(c.bagged[0]), '14.60'); assert.equal(pctText(c.bagged[9]), '11.10');
  assert.equal(pctText(c.bagged[49]), '10.25'); assert.equal(pctText(c.bagged[99]), '10.25');
  assert.equal(pctText(r14.singleError), '12.90'); assert.equal(pctText(c.member[49]), '14.73'); assert.equal(pctText(c.disagreement[49]), '13.70');
  assert.equal(majorityWrong(r14.testPreds, 50), 100);
  assert.equal(golden.ensembles['14'].port.majorityWrong['50'], 100);
  // Diminishing returns: the first ten trees buy 3.5 points, trees 11 to 100 buy under 1 point.
  assert.ok(c.bagged[0] - c.bagged[9] > 0.034 && c.bagged[9] - c.bagged[99] < 0.009);
  // Shallow, stable trees: depth 2.
  const r2 = runFor(2);
  assert.equal(pctText(r2.singleError), '10.50'); assert.equal(pctText(r2.curve.bagged[49]), '10.70'); assert.equal(pctText(r2.curve.disagreement[49]), '4.88');
  assert.ok(r2.curve.bagged[49] >= r2.singleError, 'bagging depth-2 trees does not beat one depth-2 tree');
  // Depth 5.
  const r5 = runFor(5);
  assert.equal(pctText(r5.singleError), '9.30'); assert.equal(pctText(r5.curve.bagged[49]), '10.15');
  // The depth-1 stump gains nothing either (not quoted, but part of the story).
  const r1 = runFor(1);
  assert.equal(r1.curve.bagged[49], r1.singleError);
  // Same numbers from the independent NumPy port / scikit-learn study stored in the golden file.
  for (const [d, m, v] of [[14, 49, '10.25'], [14, 99, '10.25'], [14, 9, '11.10'], [14, 0, '14.60'], [2, 49, '10.70'], [5, 49, '10.15']] as const) assert.equal(pctText(golden.ensembles[String(d)].port.bagged[m]), v, `golden depth ${d} M=${m + 1}`);
  assert.equal(pctText(golden.ensembles['14'].port.member[49]), '14.73'); assert.equal(pctText(golden.ensembles['14'].port.disagreement[49]), '13.70');
  // The standard error of a test error of 10% on 1000 points is about 1 point.
  assert.ok(Math.abs(Math.sqrt(0.1 * 0.9 / 1000) - 0.0095) < 1e-4);
  // The gain 12.90 -> 10.25 is a larger move than the quoted noise, and the bagged vote stays above the best possible 8.5%.
  assert.ok(r14.singleError - c.bagged[49] > 0.0095 && c.bagged[99] > BAYES_ERROR);
  // Members are worse than the single tree trained on all the data.
  assert.ok(c.member[49] > r14.singleError);
});

test('the lab definition: three controls with the default that shows the idea, a valid predict question, a pure 2-D pipeline', () => {
  assert.equal(lab.id, 'bagging-trees'); assert.equal(lab.track, 'pattern-recognition');
  assert.equal(lab.controls.length, 3);
  assert.deepEqual(Object.fromEntries(lab.controls.map(c => [c.key, c.value])), { m: 50, depth: 14, view: 'bagged' });
  const m = lab.controls.find(c => c.key === 'm')!, d = lab.controls.find(c => c.key === 'depth')!;
  assert.deepEqual([m.min, m.max, d.min, d.max], [1, 100, 1, 14]);
  assert.ok(lab.predict.options.length >= 2 && lab.predict.options.length <= 4 && lab.predict.options.some(o => o.value === lab.predict.answer));
  assert.equal(lab.predict.answer, 'better');
  assert.deepEqual(lab.related, ['pattern-recognition/pr-09-ensembles-validation']);
  assert.deepEqual(lab.panes.map(p => p.key), ['map', 'curve']);
  assert.ok(lab.panes.every(p => p.kind === 'canvas'));
  for (const s of lab.sources) assert.match(s.url, /^https:\/\//);
  // memberPredictions is the same as predicting point by point.
  const r = runFor(3);
  assert.deepEqual(Array.from(memberPredictions([r.single], TEST_X)[0].slice(0, 20)), TEST_X.slice(0, 20).map(p => predictTree(r.single, p[0], p[1])));
});
