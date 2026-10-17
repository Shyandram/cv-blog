// Tests for the snake kit (src/lib/imagekit/snake.ts) and the lab contour-energy (IP 11).
// Golden values come from scripts/figures/playground_goldens_snake.py: an independent NumPy / SciPy / scikit-image implementation of the
// same semi-implicit scheme. Every number quoted in the lab's text is asserted here by running the lab's own pipeline on the real
// asset (src/assets/playground/coin-crop-144.png).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { readGray, asset } from './helpers/png.ts';
import { edgeField, internalMatrix, snakeMatrix, runSnake, snakeEnergy, circlePoints, fitCircle, numpyGradient, bilinear, invert, type EdgeField } from '../../src/lib/imagekit/snake.ts';
import * as lab from '../../src/lib/playground/labs/contour-energy.ts';

const golden = JSON.parse(readFileSync(new URL('./golden/snake.json', import.meta.url), 'utf8'));
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => { assert.equal(a.length, b.length); let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const rel = (a: number, b: number, tol = 1e-6) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} vs ${b}`);

const image = readGray(asset('coin-crop-144.png'));
const truth = lab.coinTruth(image);
const run = (radius: number, smooth: number, sigma: number) => lab.runFromState(image, truth, { radius, smooth, sigma });
const f1 = (n: number) => String(Number(n.toFixed(1)));

// ------------------------------------------------------------------------------------------- the kit against NumPy

test('the matrix (A + gamma I)^-1 equals numpy.linalg.inv, and the chapter matrix A equals the one inside skimage.active_contour', () => {
  for (const m of golden.matrices) {
    const inv = snakeMatrix(m.n, m.alpha, m.beta, m.gamma);
    if (m.inv) assert.ok(maxDiff(inv, m.inv) < 1e-9, `n=${m.n}`);
    assert.ok(maxDiff(inv.subarray(0, m.n), m.firstRow) < 1e-10, `first row n=${m.n}`);
    let tr = 0, sum = 0;
    for (let i = 0; i < m.n; i++) tr += inv[i * m.n + i];
    for (const v of inv) sum += v;
    rel(tr, m.trace, 1e-10); rel(sum, m.sumAll, 1e-10);
    // circulant: every row is the first row rolled
    for (const i of [1, 7, m.n - 1]) for (let j = 0; j < m.n; j++) assert.ok(Math.abs(inv[i * m.n + j] - inv[((j - i + m.n) % m.n)]) < 1e-12);
  }
  // skimage's source: A = -alpha * a + beta * b with a, b built from rolled identity matrices (written out here in TypeScript)
  const n = 30, alpha = 0.3, beta = 1.2, eye = (i: number, j: number) => (i === j ? 1 : 0);
  const roll0 = (k: number) => (i: number, j: number) => eye((i + k + n) % n, j), roll1 = (k: number) => (i: number, j: number) => eye(i, (j + k + n) % n);
  const A = internalMatrix(n, alpha, beta);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const a = roll0(-1)(i, j) + roll1(-1)(i, j) - 2 * eye(i, j);
    const b = roll0(-2)(i, j) + roll1(-2)(i, j) - 4 * roll0(-1)(i, j) - 4 * roll1(-1)(i, j) + 6 * eye(i, j);
    assert.ok(Math.abs(A[i * n + j] - (-alpha * a + beta * b)) < 1e-12);
  }
  assert.equal(golden.skimageMatrixDifference, 0);
});

test('invert() inverts a random diagonally dominant matrix', () => {
  const n = 9, m = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) m[i] = Math.sin(i * 12.9898) * 0.5;
  for (let i = 0; i < n; i++) m[i * n + i] += 3;
  const inv = invert(m, n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { let s = 0; for (let k = 0; k < n; k++) s += m[i * n + k] * inv[k * n + j]; assert.ok(Math.abs(s - (i === j ? 1 : 0)) < 1e-12); }
});

test('numpyGradient and bilinear follow numpy.gradient and scipy map_coordinates(order=1) conventions', () => {
  const w = 4, h = 3, a = Float64Array.from([0, 1, 4, 9, 2, 3, 8, 15, 5, 7, 12, 21]);
  const { gx, gy } = numpyGradient(a, w, h);
  assert.deepEqual(Array.from(gx.subarray(0, 4)), [1, 2, 4, 5]);
  assert.deepEqual(Array.from(gy.subarray(0, 4)), [2, 2, 4, 6]);
  assert.deepEqual(Array.from(gy.subarray(8, 12)), [3, 4, 4, 6]);
  assert.equal(bilinear(a, w, h, 1.5, 0.5), (1 + 4 + 3 + 8) / 4);
  assert.equal(bilinear(a, w, h, -3, 5), 5, 'clamped to the border');
  assert.equal(bilinear(a, w, h, 3, 2), 21, 'the last pixel itself');
});

test('edge map and its gradient on the real crop match SciPy (gaussian_gradient_magnitude, numpy.gradient)', () => {
  for (const s of [1, 2, 3, 4]) {
    const g = golden.edge[String(s)], f = edgeField(image.data, image.w, image.h, s);
    let sum = 0;
    for (const v of f.e) sum += v;
    rel(sum, g.sumE, 1e-5);
    const pick = (a: Float64Array) => Float64Array.from(g.idx as number[], i => a[i]);
    assert.ok(maxDiff(pick(f.e), g.e) < 2e-6, `e sigma ${s}`);
    assert.ok(maxDiff(pick(f.gx), g.gx) < 2e-6, `gx sigma ${s}`);
    assert.ok(maxDiff(pick(f.gy), g.gy) < 2e-6, `gy sigma ${s}`);
    assert.equal(f.e.indexOf(Math.max(...f.e)), g.argmax);
  }
});

test('the snake loop agrees with the NumPy loop to rounding error on a closed-form ring field (no image involved)', () => {
  const r = golden.ring, w = r.w, h = r.h, e = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) e[y * w + x] = Math.exp(-(((Math.hypot(x - r.cx, y - r.cy) - r.r) / r.width) ** 2));
  const field: EdgeField = { w, h, e, ...numpyGradient(e, w, h) };
  for (const g of r.runs) {
    const c = circlePoints(r.cx, r.cy, g.radius, g.n);
    const res = runSnake(field, c.x, c.y, { alpha: g.alpha, beta: g.beta, gamma: 1, kappa: 4, tol: 1e-6, maxIter: 400 });
    assert.equal(res.iterations, g.iterations); assert.equal(res.converged, g.converged);
    assert.ok(maxDiff(res.x, g.x) < 1e-9 && maxDiff(res.y, g.y) < 1e-9, `positions ${g.radius} ${g.alpha}`);
    const at = [0, 1, 5, res.iterations];
    assert.ok(maxDiff(at.map(i => res.internal[i]), g.internal) < 1e-8);
    assert.ok(maxDiff(at.map(i => res.external[i]), g.external) < 1e-8);
  }
});

test('fitCircle recovers the centre and radius of an exact circle', () => {
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < 50; i++) { const t = i * 0.1; xs.push(12.5 + 7 * Math.cos(t)); ys.push(-3 + 7 * Math.sin(t)); }
  const c = fitCircle(xs, ys);
  assert.ok(Math.abs(c.cx - 12.5) < 1e-9 && Math.abs(c.cy + 3) < 1e-9 && Math.abs(c.r - 7) < 1e-9);
});

// ------------------------------------------------------------------------------------------- the lab against NumPy

test('the coin boundary (Otsu mask, filled, component at the centre, fitted circle) matches the NumPy / scikit-image recipe', () => {
  const g = golden.truth;
  assert.equal(truth.area, g.area); assert.equal(truth.boundary, g.boundary);
  rel(truth.threshold, g.threshold, 1e-6);
  assert.ok(Math.abs(truth.cx - g.cx) < 1e-7 && Math.abs(truth.cy - g.cy) < 1e-7 && Math.abs(truth.r - g.r) < 1e-7);
  assert.equal(image.w, 144); assert.equal(image.h, 144);
});

test('the lab reproduces the NumPy snake on the real asset: iterations, convergence, verdict, distance, energies, final points', () => {
  let worst = 0;
  for (const g of golden.runs) {
    const r = run(g.radius, g.smooth, g.sigma);
    const tag = `r0=${g.radius} smooth=${g.smooth} sigma=${g.sigma}`;
    assert.equal(r.result.iterations, g.iterations, tag); assert.equal(r.result.converged, g.converged, tag);
    assert.equal(r.verdict, g.verdict, tag);
    worst = Math.max(worst, maxDiff(r.result.x, g.x), maxDiff(r.result.y, g.y));
    assert.ok(Math.abs(r.distance - g.distance) < 1e-3, tag);
    assert.ok(Math.abs(r.ownRadius - g.ownRadius) < 1e-3, tag);
    assert.ok(Math.abs(r.energy[r.energy.length - 1] - g.totalFinal) < 1e-2, tag);
    assert.ok(Math.abs(r.energy[0] - g.totalStart) < 1e-4, tag);
    assert.ok(Math.abs(r.ringEnergy - g.ringEnergy) < 1e-4, tag);
    assert.equal(r.nearNeighbours, g.nearOtherCoin3px, tag);
    for (const [k, snap] of Object.entries(g.snapshots as Record<string, { x: number[]; y: number[] }>)) {
      const s = r.snapshots.find(q => q.iteration === Number(k))!;
      assert.ok(s && maxDiff(s.x, snap.x) < 1e-2 && maxDiff(s.y, snap.y) < 1e-2, `${tag} snapshot ${k}`);
    }
    assert.equal(r.snapshots.length, Object.keys(g.snapshots).length, tag);
    assert.equal(r.distanceTrace.length, r.result.iterations + 1);
    if (g.monotone) for (let i = 1; i < r.energy.length; i++) assert.ok(r.energy[i] <= r.energy[i - 1] + 1e-9, `${tag} energy rises at ${i}`);
  }
  console.log('largest final-point difference to NumPy (pixels):', worst);
  assert.ok(worst < 1e-3, `final points differ by ${worst}`);
});

// ------------------------------------------------------------------------------------------- every number in the text

test('facts: coin and constants quoted in the note and controls', () => {
  assert.equal(f1(truth.r), '23.1', 'the coin radius is about 23');
  assert.equal(lab.N_POINTS, 100); assert.equal(lab.TOL, 0.05); assert.equal(lab.MAX_ITER, 1500); assert.equal(lab.GAMMA, 1); assert.equal(lab.KAPPA, 4); assert.equal(lab.FOUND_PX, 2);
  assert.equal(lab.NEAR_PX, 3); assert.equal(lab.CENTER, 72);
  assert.deepEqual(lab.SNAPSHOTS, [10, 30, 100, 300]);
  assert.equal(truth.others.length > 100, true);
  // "a snake sitting on the rim is still about half a pixel from it"
  for (const [r0, s] of [[28, 0.05], [24, 0.05], [40, 1.0]] as const) { const d = run(r0, s, 3).distance; assert.ok(d > 0.35 && d < 0.6, `${d}`); }
  // The chapter's defaults for the figure code are 0.05 / 0.05 / 1 / kappa 4; the lab's default smoothness is the chapter's 0.05.
  assert.equal(lab.DEFAULTS.smooth, 0.05); assert.equal(lab.DEFAULTS.radius, 40); assert.equal(lab.DEFAULTS.sigma, 3);
  const c = lab.lab.controls;
  assert.equal(c.length, 3);
  assert.deepEqual(c.map(x => [x.key, x.value, x.min, x.max]), [['radius', 40, 8, 64], ['smooth', 0.05, 0.05, 3], ['sigma', 3, 1, 6]]);
});

test('facts: the default state (radius 40, smoothness 0.05, sigma 3) and the predict explanation', () => {
  const d = run(40, 0.05, 3), found = run(28, 0.05, 3);
  assert.equal(d.verdict, 'outside');
  assert.equal(d.result.converged, true);
  assert.equal(d.result.iterations, 871);
  assert.equal(f1(d.distance), '17.7');
  assert.equal(d.nearNeighbours, 74);
  assert.equal(f1(d.energy[d.energy.length - 1]), '-266.2');
  assert.equal(f1(d.ringEnergy), '-283.4');
  assert.equal(found.verdict, 'found');
  assert.equal(f1(found.energy[found.energy.length - 1]), '-298.9');
  assert.ok(d.energy[d.energy.length - 1] > d.ringEnergy && found.energy[found.energy.length - 1] < found.ringEnergy);
  assert.equal(lab.lab.predict.answer, 'neighbours');
  const ex = lab.lab.predict.explain;
  for (const s of ['74 of its 100', '17.7 px', '871 iterations', '−266.2', '−283.4', '−298.9', 'radius 28']) assert.ok(ex.en.includes(s), s);
  for (const s of ['74', '17.7', '871', '−266.2', '−283.4', '−298.9', '半徑 28']) assert.ok(ex['zh-tw'].includes(s), s);
  assert.ok(lab.lab.predict.question.en.includes('radius 40') && lab.lab.predict.question.en.includes('about 23'));
  assert.ok(lab.lab.predict.options.some(o => o.value === lab.lab.predict.answer));
});

test('facts: the numbers of the challenge text', () => {
  const rows: [number, number, number, string, string][] = [
    [40, 0.05, 3, 'outside', '17.7'],
    [28, 0.05, 3, 'found', '0.5'],
    [40, 1.0, 3, 'found', '0.4'],
    [12, 2.0, 3, 'inside', '17.4'],
    [12, 0.05, 3, 'found', '1.1'],
    [12, 0.05, 2, 'inside', '10.2'],
  ];
  for (const [r0, s, sig, verdict, dist] of rows) {
    const r = run(r0, s, sig);
    assert.equal(r.verdict, verdict, `${r0} ${s} ${sig}`); assert.equal(f1(r.distance), dist, `${r0} ${s} ${sig}`);
  }
  const ch = lab.lab.challenge;
  for (const s of ['radius 40', 'about 23', '17.7 px', 'radius to 28', '0.5 px', 'smoothness to 1.0', '0.4 px', 'smoothness 2.0', 'radius 12', '17.4 px', 'smoothness 0.05', 'σ = 3', '1.1 px', 'σ = 2', '10.2 px']) assert.ok(ch.en.includes(s), s);
  for (const s of ['半徑為 40', '約為 23', '17.7', '降到 28', '0.5', '提高到 1.0', '0.4', '平滑度 2.0', '半徑 12', '17.4', '平滑度 0.05', 'σ = 3', '1.1', 'σ = 2', '10.2']) assert.ok(ch['zh-tw'].includes(s), s);
  // (2) "a stiffer, more contracting snake reaches the coin from farther away": found from every start radius from 20 to 44 at smoothness 1.0,
  // and from none of 36..44 at the default smoothness
  for (let r0 = 20; r0 <= 44; r0 += 4) assert.equal(run(r0, 1.0, 3).verdict, 'found', `smooth 1.0 radius ${r0}`);
  for (let r0 = 36; r0 <= 44; r0 += 4) assert.notEqual(run(r0, 0.05, 3).verdict, 'found', `smooth 0.05 radius ${r0}`);
});

test('the slider extremes run, and sigma = 1 does not converge within the budget (stated in the description)', () => {
  // slider extremes run without trouble and give finite numbers
  for (const radius of [8, 64]) for (const smooth of [0.05, 3]) for (const sigma of [1, 6]) {
    const r = run(radius, smooth, sigma);
    assert.ok(Number.isFinite(r.distance) && Number.isFinite(r.energy[r.energy.length - 1]) && r.result.iterations >= 1 && r.result.iterations <= 1500, `${radius} ${smooth} ${sigma}`);
  }
  // sigma = 1 at the default start does not converge within the budget (the contour keeps jittering on the coin's relief)
  const j = run(40, 0.05, 1);
  assert.equal(j.result.converged, false); assert.equal(j.result.iterations, 1500);
});

test('descriptions: three verdicts and the not-converged suffix, both languages, no NaN', () => {
  const cases = [run(40, 0.05, 3), run(28, 0.05, 3), run(12, 2.0, 3), run(40, 0.05, 1)];
  assert.deepEqual(cases.map(c => c.verdict), ['outside', 'found', 'inside', 'outside']);
  for (const c of cases) for (const lang of ['en', 'zh-tw'] as const) {
    const t = lab.describe(lang, c, truth);
    assert.ok(!/NaN|undefined|Infinity/.test(t) && t.length > 60, t);
  }
  assert.ok(lab.describe('en', cases[0], truth).includes('74 of its 100 points lie within 3 px'));
  assert.ok(lab.describe('en', cases[0], truth).includes('17.7 px'));
  assert.ok(lab.describe('en', cases[3], truth).includes('did not converge'));
  assert.ok(lab.describe('zh-tw', cases[3], truth).includes('沒有收斂'));
  assert.ok(!lab.describe('en', cases[0], truth).includes('did not converge'));
  assert.equal(lab.verdictLabel('zh-tw', 'inside'), '停在硬幣裡面');
});

test('the default state shows the idea: a converged snake that did not find the coin', () => {
  const d = run(lab.DEFAULTS.radius, lab.DEFAULTS.smooth, lab.DEFAULTS.sigma);
  assert.equal(d.verdict, 'outside'); assert.ok(d.result.converged);
  const ok = run(28, lab.DEFAULTS.smooth, lab.DEFAULTS.sigma);
  assert.ok(ok.distance < 1 && d.distance > 10, 'one control change (radius 40 to 28) moves the distance from 17.7 to 0.5');
});

test('the energy never rises along the default and the captured run, and the first entries are the start contour', () => {
  for (const r of [run(40, 0.05, 3), run(28, 0.05, 3)]) {
    const c = circlePoints(lab.CENTER, lab.CENTER, r.params.radius, lab.N_POINTS), f = lab.fieldFor(image, r.params.sigma);
    const e0 = snakeEnergy(f, c.x, c.y, r.params.smooth, r.params.smooth, lab.KAPPA);
    rel(r.energy[0], e0.internal + e0.external, 1e-12);
    for (let i = 1; i < r.energy.length; i++) assert.ok(r.energy[i] <= r.energy[i - 1] + 1e-9);
  }
});
