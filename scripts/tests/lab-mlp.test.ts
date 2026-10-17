import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { normalField } from '../../src/lib/imagekit/rng.ts';
import {
  SETS, SEED, MAX_STEPS, TEST_STRIDE, LEARNING_RATES, DEFAULTS, BEST_LINE, paramCount, initParams, logits, lossAndGrad, score, newTrajectory, extend,
  testScore, paramsAt, risesUpTo, viewOf, describe, bestLineCorrect, collinearTriples, logitGrid, lab as mlpLab,
} from '../../src/lib/playground/labs/mlp.ts';

const golden = JSON.parse(readFileSync(new URL('./golden/mlp.json', import.meta.url), 'utf8'));
const near = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (diff ${Math.abs(a - b)})`);

test('the seeded normal numbers equal the NumPy port of the kit generator', () => {
  const z = normalField(24, golden.seed);
  assert.equal(golden.seed, SEED);
  golden.rawNormals.forEach((v: number, i: number) => near(z[i], v, 1e-12, `normal ${i}`));
});

test('data: 200 training and 200 test points per set, balanced classes', () => {
  for (const k of ['moons', 'circles'] as const) {
    for (const part of ['train', 'test'] as const) {
      const p = SETS[k][part];
      assert.equal(p.n, 200);
      assert.equal(p.y.reduce((s, v) => s + v, 0), 100);
    }
  }
});

test('initial weights, forward pass, loss and gradient match NumPy with identical weights', () => {
  for (const r of golden.runs) {
    const m = r.m, tr = SETS[r.kind as 'moons'].train, p = initParams(m);
    assert.equal(p.length, paramCount(m));
    r.init.forEach((v: number, i: number) => near(p[i], v, 1e-12, `init ${r.kind} m=${m} [${i}]`));
    const z = logits(p, m, tr);
    r.logitsInit.forEach((v: number, i: number) => near(z[i], v, 1e-12, `logit ${i}`));
    const g = new Float64Array(p.length), s = lossAndGrad(p, m, tr, g);
    near(s.loss, r.trainLoss['0'], 1e-12, `initial loss ${r.kind} m=${m}`);
    r.gradInit.forEach((v: number, i: number) => near(g[i], v, 1e-12, `grad ${r.kind} m=${m} [${i}]`));
  }
});

test('the backpropagated gradient equals a central finite difference in every coordinate', () => {
  for (const kind of ['moons', 'circles'] as const) {
    for (const m of [1, 3, 8]) {
      const tr = SETS[kind].train;
      // A trained-ish point, not only the initial weights: 30 steps of descent first, so the weights are not symmetric/zero-bias.
      const t = extend(newTrajectory(kind, m, 1), 30), p = Float64Array.from(paramsAt(t, 30));
      const g = new Float64Array(p.length);
      lossAndGrad(p, m, tr, g);
      const eps = 1e-6;
      for (let i = 0; i < p.length; i++) {
        const q = Float64Array.from(p); q[i] += eps; const up = score(q, m, tr).loss;
        q[i] -= 2 * eps; const dn = score(q, m, tr).loss;
        near(g[i], (up - dn) / (2 * eps), 1e-8, `${kind} m=${m} parameter ${i}`);
      }
    }
  }
});

// Gradient descent with a step that is too large is a chaotic map: once the loss goes up and down, a 1e-16 difference in
// tanh or exp between two libraries grows to 1e-3 after a few hundred steps. So the comparison is exact (1e-6) at every
// step the golden run reaches without ever raising the training loss, and for the first 20 steps otherwise.
const stable = (r: any, k: number) => r.rises[String(k)] === 0;

test('the full training trajectory matches NumPy: train and test loss at steps 1, 10, 100, 500 and 2000 (tolerance 1e-6)', () => {
  let exact500 = 0, exact2000 = 0, total = 0;
  for (const r of golden.runs) {
    const t = extend(newTrajectory(r.kind, r.m, r.lr), 2000);
    for (const k of golden.steps) {
      if (!(stable(r, k) || k <= 20)) continue;
      near(t.trainLoss[k], r.trainLoss[k], 1e-6, `train loss ${r.kind} m=${r.m} lr=${r.lr} step ${k}`);
      near(testScore(t, k).loss, r.testLoss[k], 1e-6, `test loss ${r.kind} m=${r.m} lr=${r.lr} step ${k}`);
      near(t.trainAcc[k], r.trainAcc[String(k)], 1e-12, `train accuracy ${r.kind} m=${r.m} lr=${r.lr} step ${k}`);
      near(testScore(t, k).accuracy, r.testAcc[String(k)], 1e-12, `test accuracy ${r.kind} m=${r.m} lr=${r.lr} step ${k}`);
      if (k === 500) exact500++;
      if (k === 2000) exact2000++;
    }
    if (stable(r, 100)) paramsAt(t, 100).forEach((v, i) => near(v, r.paramsAt100[i], 1e-6, `parameter ${i} after 100 steps`));
    total++;
  }
  assert.equal(total, 28);
  assert.ok(exact500 >= 16 && exact2000 >= 16, `most golden runs must be compared in full: ${exact500} at 500, ${exact2000} at 2000`);
});

test('the sampled test-loss curve is the held-out loss of the stored weights', () => {
  const t = extend(newTrajectory('moons', 3, 1), 200);
  for (const k of [0, 4, 8, 100, 200]) near(t.testLoss[k / TEST_STRIDE], testScore(t, k).loss, 1e-15, `test loss at ${k}`);
});

test('extending a cached trajectory gives the same numbers as one long run, and steps beyond the maximum are ignored', () => {
  const a = extend(extend(newTrajectory('circles', 3, 0.3), 120), 600), b = extend(newTrajectory('circles', 3, 0.3), 600);
  for (const k of [1, 10, 100, 119, 120, 121, 500, 600]) assert.equal(a.trainLoss[k], b.trainLoss[k]);
  assert.equal(extend(newTrajectory('moons', 2, 1), 99999).done, MAX_STEPS);
});

// ---------------------------------------------------------------------------------------------- facts quoted in the lab text

const run = (set: 'moons' | 'circles', m: number, lr: number, steps = 2000, seed = SEED) => extend(newTrajectory(set, m, lr, seed), steps);
const pc = (a: number) => (100 * a).toFixed(1);

test('best straight line: 175 of 200 training moons and 133 of 200 training circle points (exact search = NumPy 20000-direction scan)', () => {
  for (const k of ['moons', 'circles'] as const) {
    assert.equal(collinearTriples(SETS[k].train), 0);
    assert.equal(collinearTriples(SETS[k].test), 0);
    assert.equal(bestLineCorrect(SETS[k].train), golden.bestLine[k].train, `${k} train`);
    assert.equal(bestLineCorrect(SETS[k].test), golden.bestLine[k].test, `${k} test`);
  }
  assert.deepEqual([bestLineCorrect(SETS.moons.train), bestLineCorrect(SETS.circles.train)], [175, 133]);
  assert.deepEqual(BEST_LINE, { moons: { train: 175 }, circles: { train: 133 } });
  assert.deepEqual([(100 * 175 / 200).toFixed(1), (100 * 133 / 200).toFixed(1)], ['87.5', '66.5']);
});

test('one hidden unit is a straight line: its training accuracy never beats the best line, for every learning rate and every 10th step', () => {
  for (const [set, best] of [['moons', 175], ['circles', 133]] as const) {
    let top = 0;
    for (const lr of LEARNING_RATES) {
      const t = run(set, 1, lr);
      for (let k = 0; k <= 2000; k += 10) top = Math.max(top, Math.round(t.trainAcc[k] * 200));
    }
    assert.ok(top <= best, `${set}: a one-unit network got ${top} right, above the best line ${best}`);
  }
});

test('default state (m = 3, eta = 1, 1000 steps): both curved boundaries are learned, no loss rises', () => {
  assert.deepEqual(DEFAULTS, { m: 3, lr: 1, steps: 1000 });
  const a = viewOf(run('moons', 3, 1), 1000), b = viewOf(run('circles', 3, 1), 1000);
  assert.deepEqual([pc(a.trainAcc), pc(a.testAcc), a.trainLoss.toFixed(3), a.rises], ['97.5', '95.5', '0.071', 0]);
  assert.deepEqual([pc(b.trainAcc), pc(b.testAcc), b.trainLoss.toFixed(3), b.rises], ['97.5', '98.5', '0.081', 0]);
});

test('scrubbing the steps at the default: moons 85.0% at 100 steps (loss 0.302, a plateau), 94.5% at 300, 98.0% at 500; circles 78.0% at 100', () => {
  const t = run('moons', 3, 1), c = run('circles', 3, 1);
  assert.deepEqual([100, 300, 500].map(k => pc(t.trainAcc[k])), ['85.0', '94.5', '98.0']);
  assert.equal(t.trainLoss[100].toFixed(3), '0.302');
  assert.ok(Math.abs(t.trainLoss[100] - t.trainLoss[60]) < 0.02, 'the moons loss sits on a plateau between step 60 and 100');
  assert.equal(pc(c.trainAcc[100]), '78.0');
});

test('fewer hidden units (eta = 0.3, 1000 steps): moons 86.0% and 85.0% with 2 and 1 units, circles 77.5% and 61.5%', () => {
  assert.deepEqual([2, 1].map(m => pc(run('moons', m, 0.3).trainAcc[1000])), ['86.0', '85.0']);
  assert.deepEqual([2, 1].map(m => pc(run('circles', m, 0.3).trainAcc[1000])), ['77.5', '61.5']);
  for (const m of [1, 2]) for (const set of ['moons', 'circles'] as const) assert.equal(risesUpTo(run(set, m, 0.3), 1000), 0, 'eta = 0.3 does not overshoot');
});

test('a small learning rate is slow, not wrong: moons 84.5%, 85.0%, 94.5% at eta 0.03, 0.1, 0.3; eta = 0.1 reaches 98.0% after 5000 steps (NumPy agrees)', () => {
  assert.deepEqual([0.03, 0.1, 0.3].map(lr => pc(run('moons', 3, lr).trainAcc[1000])), ['84.5', '85.0', '94.5']);
  const long = extend(newTrajectory('moons', 3, 0.1, SEED, 5000), 5000);
  assert.equal(pc(long.trainAcc[5000]), '98.0');
  assert.equal(long.trainAcc[5000], golden.long['moons-0.1']['5000']);
  assert.equal(long.trainAcc[2000], golden.long['moons-0.1']['2000']);
  assert.equal(risesUpTo(long, 5000), 0);
});

test('too large a learning rate: the loss rises on many steps (eta 3, 10, 30), and at 30 it does not settle; NumPy agrees', () => {
  for (const lr of [3, 10, 30]) {
    const mo = run('moons', 3, lr), ci = run('circles', 3, lr);
    for (const k of [mo, ci]) assert.ok(risesUpTo(k, 1000) > 50, `eta = ${lr}: ${risesUpTo(k, 1000)} rises`);
    for (const kind of ['moons', 'circles']) {
      const g = golden.runs.find((r: any) => r.kind === kind && r.m === 3 && r.lr === lr);
      assert.ok(g.rises['1000'] > 50, `NumPy: ${kind} eta ${lr}`);
    }
  }
  for (const kind of ['moons', 'circles'] as const) {
    const t = run(kind, 3, 30), g = golden.runs.find((r: any) => r.kind === kind && r.m === 3 && r.lr === 30);
    for (const k of [1000, 2000]) { assert.ok(t.trainLoss[k] > 1, `${kind} eta 30 step ${k}`); assert.ok(g.trainLoss[String(k)] > 1, `NumPy ${kind} eta 30 step ${k}`); }
  }
  // Up to eta = 1 the loss falls on every step for three units.
  for (const lr of [0.03, 0.1, 0.3, 1]) for (const set of ['moons', 'circles'] as const) assert.equal(risesUpTo(run(set, 3, lr), 2000), 0, `${set} eta ${lr}`);
});

test('chaos: at eta = 30 the lab and NumPy part ways within 1000 steps (so large-eta curves may differ between engines); at eta = 1 they agree', () => {
  const g30 = golden.runs.find((r: any) => r.kind === 'moons' && r.m === 3 && r.lr === 30), t30 = run('moons', 3, 30);
  assert.ok(Math.abs(t30.trainLoss[1000] - g30.trainLoss['1000']) > 0.1);
  const g1 = golden.runs.find((r: any) => r.kind === 'moons' && r.m === 3 && r.lr === 1), t1 = run('moons', 3, 1);
  assert.ok(Math.abs(t1.trainLoss[1000] - g1.trainLoss['1000']) < 1e-9);
});

test('twenty seeds (eta = 1, 2000 steps): 3 units reach 95% on both sets every time, 1 unit never, 2 units at most once on the moons and never on the circles', () => {
  const count = (set: 'moons' | 'circles', m: number) => Array.from({ length: 20 }, (_, i) => run(set, m, 1, 2000, i + 1).trainAcc[2000]).filter(a => a >= 0.95).length;
  assert.equal(count('moons', 3), 20); assert.equal(count('circles', 3), 20);
  assert.equal(count('moons', 1), 0); assert.equal(count('circles', 1), 0);
  assert.ok(count('moons', 2) <= 1); assert.equal(count('circles', 2), 0);
  const py = (key: string) => golden.seedStudy[key].filter((a: number) => a >= 0.95).length;
  assert.deepEqual(['moons-3', 'circles-3', 'moons-1', 'circles-1', 'circles-2'].map(py), [20, 20, 0, 0, 0]);
  assert.ok(py('moons-2') <= 1);
});

test('the decision-map grid holds the logit at each cell centre, row 0 at the top', () => {
  const p = paramsAt(run('moons', 3, 1, 300), 300), box = { x0: -1, x1: 2, y0: -1, y1: 1.5 }, z = logitGrid(p, 3, box, 6, 5);
  assert.equal(z.length, 30);
  const at = (c: number, r: number) => ({ n: 1, x: Float64Array.from([box.x0 + (c + 0.5) * 3 / 6, box.y1 - (r + 0.5) * 2.5 / 5]), y: Float64Array.from([0]) });
  near(z[0], logits(p, 3, at(0, 0))[0], 1e-12, 'top-left cell');
  near(z[2 * 6 + 4], logits(p, 3, at(4, 2))[0], 1e-12, 'cell (4, 2)');
});

test('the text: default summary quotes the verified numbers in both languages; m = 1 mentions the best line; step 0 explains itself', () => {
  const a = viewOf(run('moons', 3, 1), 1000), b = viewOf(run('circles', 3, 1), 1000);
  for (const lang of ['en', 'zh-tw'] as const) {
    const text = describe(lang, 3, 1, 1000, a, b);
    for (const v of ['97.5%', '95.5%', '98.5%']) assert.ok(text.includes(v), `${lang}: ${v}`);
    const one = describe(lang, 1, 0.3, 1000, viewOf(run('moons', 1, 0.3), 1000), viewOf(run('circles', 1, 0.3), 1000));
    assert.ok(one.includes('175') && one.includes('133'), `${lang}: best line numbers`);
    assert.ok(describe(lang, 3, 1, 0, viewOf(run('moons', 3, 1), 0), viewOf(run('circles', 3, 1), 0)).length > 20);
  }
  const over = describe('en', 3, 30, 1000, viewOf(run('moons', 3, 30), 1000), viewOf(run('circles', 3, 30), 1000));
  assert.ok(over.includes('too large'));
});

test('the lab definition is consistent: three controls, valid defaults, a verified predict answer, bilingual text, canvas panes only', () => {
  assert.equal(mlpLab.id, 'mlp-training');
  assert.equal(mlpLab.controls.length, 3);
  const def = Object.fromEntries(mlpLab.controls.map(c => [c.key, c.value]));
  assert.deepEqual(def, { m: 3, lr: '1', steps: 1000 });
  assert.deepEqual(mlpLab.controls[1].options!.map(o => o.value), LEARNING_RATES.map(String));
  assert.ok(mlpLab.predict.options.length >= 2 && mlpLab.predict.options.length <= 4);
  assert.ok(mlpLab.predict.options.some(o => o.value === mlpLab.predict.answer));
  assert.equal(mlpLab.predict.answer, 'line');
  assert.ok(mlpLab.panes.every(p => p.kind === 'canvas'));
  for (const text of [mlpLab.title, mlpLab.description, mlpLab.challenge, mlpLab.note, mlpLab.predict.question, mlpLab.predict.explain]) assert.ok(text.en.length > 10 && text['zh-tw'].length > 10);
  // Every percentage in the challenge and in the predict explanation is one of the facts asserted above, in both languages.
  const nums = (t: string) => [...t.matchAll(/\d+(?:\.\d+)?%/g)].map(m => m[0]);
  assert.deepEqual(nums(mlpLab.challenge.en), nums(mlpLab.challenge['zh-tw']));
  assert.deepEqual(nums(mlpLab.predict.explain.en), nums(mlpLab.predict.explain['zh-tw']));
  assert.deepEqual(nums(mlpLab.challenge.en), ['85.0%', '94.5%', '98.0%', '86.0%', '85.0%', '77.5%', '61.5%', '84.5%', '85.0%', '94.5%', '98.0%']);
  assert.deepEqual(nums(mlpLab.predict.explain.en), ['87.5%', '97.5%', '95.5%', '66.5%']);
});

test('speed: the widest network on both sets for 2000 steps takes well under two seconds, cold, in Node', () => {
  const t0 = performance.now();
  run('moons', 8, 1.0, 2000, 99); run('circles', 8, 1.0, 2000, 99);
  assert.ok(performance.now() - t0 < 2000);
});
