// Facts for the `tracking-association` lab (DLCV 06): the kit (Hungarian, IoU, Kalman, colour embedding) and the lab's own
// pipeline compared with scripts/tests/golden/tracking.json (SciPy, NumPy; scripts/figures/playground_goldens_tracking.py).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mulberry32, normalField } from '../../src/lib/imagekit/rng.ts';
import { hungarian, assignmentCost, iouBox, KalmanBox, colourEmbedding, countSwitches, mota, type Box } from '../../src/lib/imagekit/tracking.ts';
import {
  lab, getScene, truthBoxes, detectionsFor, runLab, frameCosts, describe, DEFAULTS, JITTER_SEED, T, K, MEET, BOX_W, BOX_H, W, H,
} from '../../src/lib/playground/labs/tracking-association.ts';

const G = JSON.parse(readFileSync(new URL('./golden/tracking.json', import.meta.url), 'utf8'));
const asBox = (a: number[]): Box => ({ cx: a[0], cy: a[1], w: a[2], h: a[3] });
const near = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);
const scene = getScene();

// ------------------------------------------------------------------------------------------------ kit

test('Hungarian algorithm equals scipy.optimize.linear_sum_assignment on 120 random matrices (square and rectangular)', () => {
  assert.equal(G.hungarian.length, 120);
  for (const c of G.hungarian) {
    const got = hungarian(c.cost);
    assert.deepEqual(got, c.rowToCol, JSON.stringify(c.cost));
    near(assignmentCost(c.cost, got), c.total, 1e-9, 'total cost');
  }
});

test('Hungarian algorithm: optimal against brute force on small matrices, with ties and negative entries', () => {
  const perms = (n: number): number[][] => (n === 0 ? [[]] : perms(n - 1).flatMap(p => Array.from({ length: n }, (_, i) => [...p.slice(0, i), n - 1, ...p.slice(i)])));
  let seed = 1; const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  for (let rep = 0; rep < 60; rep++) {
    const n = 2 + (rep % 4), cost = Array.from({ length: n }, () => Array.from({ length: n }, () => Math.round(rnd() * 6) - 2)); // integers: many ties
    const best = Math.min(...perms(n).map(p => p.reduce((s, j, i) => s + cost[i][j], 0)));
    const got = hungarian(cost);
    assert.equal(new Set(got).size, n);
    assert.equal(assignmentCost(cost, got), best, JSON.stringify(cost));
  }
  assert.deepEqual(hungarian([[0.3, 0.1], [0.9, 0.8], [0.05, 0.6]]), [1, -1, 0], 'more rows than columns leaves the dearest row out');
  assert.deepEqual(hungarian([]), []);
});

test('IoU matches the NumPy implementation of the chapter', () => {
  for (const c of G.iou) near(iouBox(asBox(c.a), asBox(c.b)), c.iou, 1e-12, 'iou');
  assert.equal(iouBox({ cx: 0, cy: 0, w: 4, h: 4 }, { cx: 10, cy: 0, w: 4, h: 4 }), 0);
  near(iouBox({ cx: 0, cy: 0, w: 4, h: 4 }, { cx: 0, cy: 0, w: 4, h: 4 }), 1, 1e-8, 'identical boxes');
});

test('Kalman filter equals the chapter KalmanBox (NumPy) step by step', () => {
  const kf = new KalmanBox(asBox(G.kalman.z[0]));
  G.kalman.steps.forEach((s: { pred: number[]; x: number[]; P: number[] }, t: number) => {
    const p = kf.predict();
    [p.cx, p.cy, p.w, p.h].forEach((v, i) => near(v, s.pred[i], 1e-9, `pred ${t}.${i}`));
    kf.update(asBox(G.kalman.z[t + 1]));
    kf.x.forEach((v, i) => near(v, s.x[i], 1e-9, `x ${t}.${i}`));
    kf.P.forEach((row, i) => near(row[i], s.P[i], 1e-9, `P ${t}.${i}`));
  });
});

test('the seeded generators are the ones the NumPy port reproduces', () => {
  const r = mulberry32(12345);
  G.rng.mulberry.forEach((v: number) => near(r(), v, 1e-15, 'mulberry'));
  const n = normalField(6, 12345);
  G.rng.normal.forEach((v: number, i: number) => near(n[i], v, 1e-6, 'normal'));
});

// ------------------------------------------------------------------------------------------------ scene and pipeline

test('scene: frame size, true boxes, rendered pixels and colour embeddings equal the NumPy mirror', () => {
  assert.deepEqual([W, H, T, K], [160, 128, 30, 3]);
  const truth = truthBoxes(), fr = [0, 14, 15, 29];
  fr.forEach((f, a) => truth[f].forEach((b, k) => [b.cx, b.cy, b.w, b.h].forEach((v, i) => near(v, G.scene.boxes[a * 3 + k][i], 1e-9, `box ${f}.${k}.${i}`))));
  [0, 15, 29].forEach((f, a) => { let s = 0; for (const v of scene.frames[f].data) s += v; near(s / G.scene.pixelSums[a], 1, 1e-9, `pixel sum frame ${f}`); });
  const e = colourEmbedding(scene.frames[15], truth[15][0]);
  e.forEach((v, i) => near(v, G.scene.embedding[i], 1e-9, `embedding ${i}`));
  near(Math.hypot(...e), 1, 1e-12, 'unit length');
});

test('pipeline: identities, switches, mean IoU and MOTA equal the NumPy tracker on a grid of lambda and sigma', () => {
  assert.equal(G.pipelineSeed, JITTER_SEED);
  assert.equal(G.grid.length, 15);
  for (const g of G.grid) {
    const r = runLab(scene, g.lam, g.sigma);
    assert.deepEqual(r.run.identity, g.ids, `ids at lambda ${g.lam}, sigma ${g.sigma}`);
    assert.equal(r.switches, g.switches);
    assert.equal(r.firstSwitch, g.firstSwitchFrame ?? -1);
    assert.equal(r.wrongFrames, g.wrongObjectFrames);
    near(r.meanIou, g.meanIou, 1e-6, 'mean IoU');
    near(r.mota, g.mota, 1e-12, 'MOTA');
  }
});

test('cost matrices of single frames equal the NumPy ones; the Hungarian choice is the same', () => {
  const r = runLab(scene, 1, 2.5);
  for (const [f, c] of Object.entries(G.frameCosts) as [string, { C: number[][]; Ci: number[][]; Ca: number[][]; cols: number[]; truth: number[] }][]) {
    const fc = frameCosts(r, Number(f));
    c.C.forEach((row, k) => row.forEach((v, j) => { near(fc.total[k][j], v, 1e-9, `C ${f}.${k}.${j}`); near(fc.iou[k][j], c.Ci[k][j], 1e-9, `Ci ${f}`); near(fc.app[k][j], c.Ca[k][j], 1e-9, `Ca ${f}`); }));
    assert.deepEqual(fc.chosen, c.cols);
    assert.deepEqual(r.detections[Number(f)].map(d => d.truth), c.truth);
  }
});

// ------------------------------------------------------------------------------------------------ facts quoted by the lab

const run = (lambda: number, sigma: number, seed = JITTER_SEED) => runLab(scene, lambda, sigma, seed);
const swapRate = (lambda: number, sigma: number) => { let n = 0; for (let s = 0; s < 200; s++) if (run(lambda, sigma, s).switches > 0) n++; return n; };

test('defaults: motion only, jitter 2.5, frame 15 shows the swap; the swap is the first switch and costs less than keeping identities', () => {
  assert.deepEqual(Object.fromEntries(lab.controls.map(c => [c.key, c.value])), { lambda: DEFAULTS.lambda, sigma: DEFAULTS.sigma, frame: DEFAULTS.frame });
  const r = run(1, 2.5), c = frameCosts(r, DEFAULTS.frame);
  assert.equal(r.switches, 2);
  assert.equal(r.firstSwitch, MEET + 1);
  assert.equal(DEFAULTS.frame, r.firstSwitch);
  assert.ok(c.switched && c.chosenCost < c.keepCost);
  assert.deepEqual(c.same, [false, false, true], 'tracks 1 and 2 swap, track 3 stays');
  assert.equal(r.wrongFrames, 2 * (T - r.firstSwitch), 'two objects carry the wrong id from frame 15 to 29');
  assert.equal(r.wrongFrames, 30);
  assert.equal(T * K, 90);
  assert.equal(r.meanIou.toFixed(3), '0.582');
  assert.equal(r.mota.toFixed(3), '0.978');
  assert.equal(run(0.5, 2.5).switches, 0);
  assert.equal(run(0.5, 2.5).meanIou.toFixed(3), '0.850');
  assert.equal(run(0, 2.5).switches, 0);
});

test('the swap is caused by the turn, not by noise: with no jitter, overlap alone swaps and any colour weight does not', () => {
  const r = run(1, 0);
  assert.equal(r.switches, 2);
  assert.equal(r.firstSwitch, MEET + 1);
  for (const l of [0.75, 0.5, 0.25, 0]) assert.equal(run(l, 0).switches, 0, `lambda ${l}`);
});

test('smallest colour weight that removes the swap at the default jitter: lambda 0.9 still swaps, lambda 0.8 does not', () => {
  assert.equal(run(0.9, 2.5).switches, 2);
  assert.equal(run(0.8, 2.5).switches, 0);
});

test('more jitter breaks the colour cue too: sigma 5 gives 6 switches with overlap only and 4 with lambda 0.5', () => {
  assert.equal(run(1, 5).switches, 6);
  assert.equal(run(0.5, 5).switches, 4);
});

test('over 200 other jitter draws: overlap only switches in 164, lambda 0.5 in 12 and colour only in 6 (sigma 2.5); 165 and 90 at sigma 5', () => {
  for (const [name, lam, sigma] of [['motion', 1, 2.5], ['mixed', 0.5, 2.5], ['appearance', 0, 2.5], ['motionQuiet', 1, 0], ['motionLoud', 1, 5], ['mixedLoud', 0.5, 5]] as const) {
    assert.equal(swapRate(lam, sigma), G.robust[name].withSwitch, name);
  }
  assert.deepEqual([G.robust.motion.withSwitch, G.robust.mixed.withSwitch, G.robust.appearance.withSwitch, G.robust.motionLoud.withSwitch, G.robust.mixedLoud.withSwitch], [164, 12, 6, 165, 90]);
  assert.equal(G.robust.motionQuiet.withSwitch, 200);
});

test('predict: the two walkers have identical box sizes, so size is no cue; overlap alone swaps, colour does not', () => {
  const truth = truthBoxes();
  for (const row of truth) { assert.equal(row[0].w, BOX_W); assert.equal(row[1].w, BOX_W); assert.equal(row[0].h, BOX_H); assert.equal(row[1].h, BOX_H); }
  assert.equal(lab.predict.answer, 'colour');
  assert.ok(lab.predict.options.some(o => o.value === lab.predict.answer));
  assert.ok(lab.predict.options.length >= 2 && lab.predict.options.length <= 4);
});

test('the two walkers stand at the same x at frame 14 and are 8 pixels apart vertically (the premise of the text)', () => {
  const t = truthBoxes()[MEET];
  assert.equal(t[0].cx, t[1].cx);
  assert.equal(t[1].cy - t[0].cy, 8);
  near(iouBox(t[0], t[1]), 960 / 1440, 1e-6, 'IoU of the two boxes at the meeting frame');
  // object 2 never touches the others
  for (const row of truthBoxes()) { assert.equal(iouBox(row[2], row[0]), 0); assert.equal(iouBox(row[2], row[1]), 0); }
});

test('metrics: MOTA as in the chapter with FN = FP = 0; switches counted per object against the previous frame', () => {
  assert.equal(mota(0, 0, 2, 90), 1 - 2 / 90);
  assert.deepEqual(countSwitches([[0, 1, 2], [1, 0, 2], [1, 0, 2]]), { switches: 2, firstFrame: 1, wrongFrames: 4, total: 9 });
  assert.deepEqual(countSwitches([[0, 1], [0, 1]]), { switches: 0, firstFrame: -1, wrongFrames: 0, total: 4 });
});

test('detections are the same for every lambda (the slider never re-rolls) and scale linearly with sigma', () => {
  const a = detectionsFor(scene, 2.5), b = detectionsFor(scene, 5), t = truthBoxes();
  for (let f = 0; f < T; f++) a[f].forEach((d, j) => {
    const k = d.truth, dx = d.box.cx - t[f][k].cx, dx2 = b[f][j].box.cx - t[f][k].cx;
    near(dx2, 2 * dx, 1e-9, 'jitter doubles with sigma');
    assert.equal(b[f][j].truth, k);
  });
});

// ------------------------------------------------------------------------------------------------ text

test('text: every number quoted in the challenge, note and predict explanation is asserted here and appears in both languages', () => {
  const both = (field: 'challenge' | 'note') => (s: string) => assert.ok(lab[field].en.includes(s) && lab[field]['zh-tw'].includes(s), `${field} mentions ${s}`);
  ['15', '0.582', '0.850', '0.9', '0.8', '0.5', '0.47', '0.61'].forEach(both('challenge'));
  ['0.03', '0.022', '164', '12', '165', '90', '160', '128', '30', '40', '4'].forEach(both('note'));
  // 0.47 and 0.61: colour costs of track #1 at frame 13 with sigma 5
  const r5 = run(0.5, 5), c13 = frameCosts(r5, 13), other = r5.detections[13].findIndex(d => d.truth === 1), own = r5.detections[13].findIndex(d => d.truth === 0);
  assert.equal(c13.app[0][other].toFixed(2), '0.47');
  assert.equal(c13.app[0][own].toFixed(2), '0.61');
  assert.ok(c13.app[0][other] < c13.app[0][own], 'colour prefers the wrong walker at that frame');
  assert.equal(r5.switches, 4);
  // explain: 0.36, 0.50, 0.02, 0.19 at the default frame, track #2 (blue)
  const r1 = run(1, 2.5), c15 = frameCosts(r1, 15), orange = r1.detections[15].findIndex(d => d.truth === 0), blue = r1.detections[15].findIndex(d => d.truth === 1);
  assert.equal(c15.iou[1][orange].toFixed(2), '0.36');
  assert.equal(c15.iou[1][blue].toFixed(2), '0.50');
  assert.equal(c15.app[1][blue].toFixed(2), '0.02');
  assert.equal(c15.app[1][orange].toFixed(2), '0.19');
  for (const s of ['0.36', '0.50', '0.02', '0.19', '30 × 40']) assert.ok(lab.predict.explain.en.includes(s) && lab.predict.explain['zh-tw'].includes(s), `explain mentions ${s}`);
  // note: geometry and MOTA
  assert.equal((1 - mota(0, 0, 2, 90)).toFixed(3), '0.022');
  assert.deepEqual([W, H, T, MEET, BOX_W, BOX_H], [160, 128, 30, 14, 30, 40]);
  assert.equal(truthBoxes()[1][0].cx - truthBoxes()[0][0].cx, 4, 'four pixels per frame');
  assert.equal(JITTER_SEED, 16);
  assert.ok(lab.note.en.includes('seed 16') && lab.note['zh-tw'].includes('種子 16'));
});

test('describe: reads the swap at the default frame and a quiet frame in both languages', () => {
  const r = run(1, 2.5), c = frameCosts(r, 15), q = frameCosts(r, 5);
  const en = describe('en', r, c, 1), zh = describe('zh-tw', r, c, 1);
  assert.match(en, /cheapest matching/); assert.match(zh, /匈牙利演算法選擇了交換/);
  assert.match(describe('en', r, q, 1), /every track takes the detection/);
  assert.match(describe('zh-tw', run(0.5, 2.5), frameCosts(run(0.5, 2.5), 15), 0.5), /沒有任何身分切換/);
  assert.ok(!/NaN|undefined/.test(en + zh));
});
