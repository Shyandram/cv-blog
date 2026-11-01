// Facts for the `adversarial-budget` lab (DLCV 11): the data, the logistic regression, the input gradient, FGSM and random signs,
// and every number quoted in the lab's text.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  lab, SIDES, SIGMA, Z0, DEFAULTS, EPS_MAX, stripePattern, makeDataset, trainLogistic, sigmoid, logit, lossGradient, fgsmDirection,
  randomDirection, perturb, bundleFor, attackResult, accuracyCurve, halfAccuracyLevel, exampleIndex, medianMargin, describe,
} from '../../src/lib/playground/labs/adversarial-budget.ts';

const near = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);
const loss = (m: ReturnType<typeof trainLogistic>, x: ArrayLike<number>, y: number) => { const z = logit(m, x); return y ? Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)); };

// ------------------------------------------------------------------------------------------------ data

test('the stripe pattern has unit root-mean-square, so the amplitude per pixel is s and the class gap is 2·Z0·σ at every size', () => {
  for (const n of SIDES) {
    const a = stripePattern(n), ds = makeDataset(n, 4, 3);
    near(Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length), 1, 1e-12, 'rms');
    near(ds.s * Math.sqrt(n * n), Z0 * SIGMA, 1e-12, 'total amplitude s·√d');
    near(2 * ds.s * Math.sqrt(n * n), 6 * SIGMA, 1e-12, 'distance between the class means');
  }
});

test('the datasets are deterministic, balanced, inside [0, 1] and differ between train and test seeds', () => {
  const a = makeDataset(16, 20, 11), b = makeDataset(16, 20, 11), c = makeDataset(16, 20, 12);
  assert.deepEqual(Array.from(a.X), Array.from(b.X));
  assert.notDeepEqual(Array.from(a.X.slice(0, 50)), Array.from(c.X.slice(0, 50)));
  assert.equal(a.y.reduce((s, v) => s + v, 0), 10);
  assert.ok(a.X.every(v => v >= 0 && v <= 1));
});

// ------------------------------------------------------------------------------------------------ model and attacks

test('sigmoid is stable for large logits', () => {
  assert.equal(sigmoid(0), 0.5);
  assert.ok(sigmoid(800) === 1 && sigmoid(-800) === 0 && Number.isFinite(sigmoid(-800)));
  near(sigmoid(2) + sigmoid(-2), 1, 1e-15, 'symmetry');
});

test('the input gradient (p − y)·w equals a central finite difference of the cross-entropy loss', () => {
  const b = bundleFor(8), k = 5, x = Float64Array.from(b.test.X.subarray(k * 64, (k + 1) * 64)), y = b.test.y[k], g = lossGradient(b.model, x, y);
  for (const i of [0, 7, 20, 33, 63]) {
    const h = 1e-6, xp = Float64Array.from(x), xm = Float64Array.from(x);
    xp[i] += h; xm[i] -= h;
    near(g[i], (loss(b.model, xp, y) - loss(b.model, xm, y)) / (2 * h), 1e-7, `d loss / d x_${i}`);
  }
});

test('FGSM direction: the gradient sign pushes the logit toward the wrong class (−sign(w) for class A, +sign(w) for class B); zero weights give zero', () => {
  const b = bundleFor(8);
  for (let k = 0; k < 6; k++) {
    const dir = fgsmDirection(b.model, b.test.X, b.test.y[k], k * 64), sgn = b.test.y[k] ? -1 : 1; // class A: raise the loss by lowering the logit
    for (let i = 0; i < 64; i++) assert.equal(dir[i], sgn * Math.sign(b.model.w[i]));
  }
  const zero = { n: 2, d: 4, w: new Float64Array(4), b: 0 };
  assert.deepEqual(Array.from(fgsmDirection(zero, new Float64Array(4).fill(0.5), 1)), [0, 0, 0, 0]);
  // a very confident prediction still gets a direction (no underflow of p − y)
  const sure = { n: 2, d: 2, w: Float64Array.of(50, -50), b: 0 };
  assert.deepEqual(Array.from(fgsmDirection(sure, Float64Array.of(1, 0), 1)), [-1, 1]);
});

test('FGSM raises the loss, and without clipping it moves the logit by exactly ε·‖w‖₁', () => {
  const b = bundleFor(16), d = b.d, eps = 3 / 255;
  for (let k = 0; k < 5; k++) {
    const x = new Float64Array(d).fill(0.5), y = b.test.y[k]; // mid-gray: no clipping at 0 or 1
    const dir = fgsmDirection(b.model, x, y), adv = perturb(x, 0, dir, eps), shift = (y ? 1 : -1) * (logit(b.model, x) - logit(b.model, adv));
    near(shift, eps * b.l1, 1e-9, 'logit shift');
    assert.ok(loss(b.model, adv, y) > loss(b.model, x, y));
  }
});

test('perturb clips to [0, 1] and uses ε in 0..1 pixel units', () => {
  const out = perturb(Float64Array.of(0, 0.5, 1), 0, Int8Array.of(-1, 1, 1), 0.1);
  assert.deepEqual(Array.from(out), [0, 0.6, 1]);
});

test('random directions are fixed per sample, ±1, and balanced', () => {
  const a = randomDirection(1024, 3), b = randomDirection(1024, 3), c = randomDirection(1024, 4);
  assert.deepEqual(Array.from(a), Array.from(b));
  assert.notDeepEqual(Array.from(a), Array.from(c));
  assert.ok(a.every(v => v === 1 || v === -1));
  assert.ok(Math.abs(a.reduce((s, v) => s + v, 0)) < 130, 'about as many + as −');
});

test('training: the classifier fits the training set and generalises at every size', () => {
  for (const n of SIDES) {
    const b = bundleFor(n);
    let tr = 0, te = 0;
    for (let k = 0; k < b.train.count; k++) if ((logit(b.model, b.train.X, k * b.d) > 0 ? 1 : 0) === b.train.y[k]) tr++;
    for (let k = 0; k < b.test.count; k++) if ((b.z0[k] > 0 ? 1 : 0) === b.test.y[k]) te++;
    assert.equal(tr, b.train.count, `train accuracy ${n}`);
    assert.ok(te / b.test.count >= 0.99, `test accuracy ${n}: ${te / b.test.count}`);
  }
});

// ------------------------------------------------------------------------------------------------ the idea

test('measured FGSM shift matches the linear prediction ε·‖w‖₁ within clipping, and random shifts have rms ≈ ε·‖w‖₂ and mean ≈ 0', () => {
  for (const n of SIDES) {
    const b = bundleFor(n);
    for (const e of [4, 8, 16]) {
      const f = attackResult(b, e, 'fgsm'), r = attackResult(b, e, 'random'), lin = (e / 255) * b.l1;
      assert.ok(f.meanShift <= lin + 1e-9 && f.meanShift >= 0.97 * lin, `n ${n} eps ${e}: ${f.meanShift} vs ${lin}`);
      near(r.rmsShift, (e / 255) * b.l2, 0.06 * (e / 255) * b.l2, `random rms n ${n} eps ${e}`);
      assert.ok(Math.abs(r.meanShift) < 0.1 * r.rmsShift + 0.03, `random mean n ${n} eps ${e}`);
    }
  }
});

test('with the classes a fixed distance apart, the FGSM budget that halves accuracy shrinks like 1/√d', () => {
  const half = SIDES.map(n => halfAccuracyLevel(accuracyCurve(bundleFor(n), 'fgsm')));
  assert.equal(half[0], null, '8 × 8 never reaches 50% by ε = 16');
  near(half[1]!, 10.49, 0.01, '16 × 16');
  near(half[2]!, 4.74, 0.01, '32 × 32');
  near(half[1]! / half[2]!, 2.21, 0.02, 'ratio of the two budgets (√4 = 2 in theory)');
  // monotone curves, and random signs stay within 1 point of the clean accuracy at every ε
  for (const n of SIDES) {
    const b = bundleFor(n), f = accuracyCurve(b, 'fgsm'), r = accuracyCurve(b, 'random');
    for (let e = 1; e <= EPS_MAX; e++) assert.ok(f[e] <= f[e - 1] + 1e-12, `FGSM accuracy non-increasing at ${e} (n ${n})`);
    for (let e = 0; e <= EPS_MAX; e++) assert.ok(Math.abs(r[e] - r[0]) <= 0.01 + 1e-12, `random accuracy flat at ${e} (n ${n})`);
  }
  assert.equal(halfAccuracyLevel(Float64Array.of(1, 0.6, 0.4, 0.4)), 1 + (0.6 - 0.5) / (0.6 - 0.4));
});

// ------------------------------------------------------------------------------------------------ facts quoted by the lab

const pct1 = (v: number) => (100 * v).toFixed(1) + '%';
const both = (field: 'challenge' | 'note' | 'explain') => (s: string) => {
  const text = field === 'explain' ? lab.predict.explain : lab[field];
  assert.ok(text.en.includes(s) && text['zh-tw'].includes(s), `${field} mentions ${s}`);
};

test('defaults: 32 × 32, ε = 8, FGSM; accuracy collapses under FGSM and not under random signs', () => {
  assert.deepEqual(Object.fromEntries(lab.controls.map(c => [c.key, String(c.value)])), { eps: '8', dim: '1024', attack: 'fgsm' });
  assert.equal(DEFAULTS.eps, 8);
  const b = bundleFor(32), f = attackResult(b, 8, 'fgsm'), r = attackResult(b, 8, 'random');
  assert.equal(pct1(f.clean), '99.3%');
  assert.equal(pct1(f.acc), '2.7%');
  assert.equal(pct1(r.acc), '99.3%');
  assert.equal(f.meanShift.toFixed(1), '5.8');
  assert.equal(medianMargin(b).toFixed(1), '3.5');
  assert.ok(f.meanShift > medianMargin(b));
  assert.equal(r.rmsShift.toFixed(1), '0.2');
  assert.equal(((8 / 255) * b.l1).toFixed(1), '5.8');
});

test('the shown example is a correctly classified class-A image that FGSM flips at the default ε and random signs do not', () => {
  const b = bundleFor(32), k = exampleIndex(b);
  assert.equal(b.test.y[k], 1);
  assert.ok(b.z0[k] > 0);
  const fg = perturb(b.test.X, k * b.d, b.fgsm.subarray(k * b.d, (k + 1) * b.d), 8 / 255), rd = perturb(b.test.X, k * b.d, b.rand.subarray(k * b.d, (k + 1) * b.d), 8 / 255);
  assert.ok(logit(b.model, fg) < 0, 'FGSM flips the example');
  assert.ok(logit(b.model, rd) > 0, 'random signs do not');
  for (let i = 0; i < b.d; i++) assert.ok(Math.abs(fg[i] - b.test.X[k * b.d + i]) <= 8 / 255 + 1e-12);
});

test('text: the numbers in the challenge are what the code produces, and appear in both languages', () => {
  const at = (n: number, e: number) => attackResult(bundleFor(n), e, 'fgsm').acc;
  assert.equal(pct1(at(16, 8)), '76.3%');
  assert.equal(pct1(at(8, 8)), '94.7%');
  assert.equal(pct1(at(8, 16)), '70.7%');
  const first = (n: number) => { const c = accuracyCurve(bundleFor(n), 'fgsm'); return c.findIndex(v => v < 0.5); };
  assert.equal(first(32), 5);
  assert.equal(first(16), 11);
  assert.equal(first(8), -1);
  ['99.3%', '2.7%', '+5.8', '3.5', '76.3%', '94.7%', 'ε = 5', 'ε = 11', 'ε = 16', '70.7%', '16 × 16', '8 × 8', '32 × 32'].forEach(both('challenge'));
});

test('text: the numbers in the note and the explanation are what the code produces', () => {
  const l = (n: number) => bundleFor(n);
  assert.deepEqual(SIDES.map(n => l(n).l2.toFixed(1)), ['7.5', '7.5', '6.9']);
  assert.deepEqual(SIDES.map(n => l(n).l1.toFixed(1)), ['54.4', '102.8', '186.2']);
  assert.equal(N(), 300);
  ['7.5', '6.9', '54.4', '102.8', '186.2', '0.2', '51', '0.001', '300', '50'].forEach(both('note'));
  assert.ok(lab.note.en.includes('seeds 11 and 12') && lab.note['zh-tw'].includes('種子 11 與 12'));
  ['2.7%', '99.3%', '5.8', '3.5', '0.2', '1,024', '8'].forEach(both('explain'));
  assert.equal(lab.predict.answer, 'fgsm');
  assert.ok(lab.predict.options.some(o => o.value === lab.predict.answer) && lab.predict.options.length <= 4);
});
function N() { return bundleFor(8).test.count; }

test('text: no Simplified-only wording and no placeholders; describe reads in both languages', () => {
  const b = bundleFor(32), f = attackResult(b, 8, 'fgsm'), r = attackResult(b, 8, 'random');
  const en = describe('en', b, 8, f, r), zh = describe('zh-tw', b, 8, f, r);
  assert.match(en, /2\.7%/); assert.match(en, /99\.3%/); assert.match(zh, /2\.7%/); assert.match(zh, /1,024/);
  assert.ok(!/NaN|undefined|TBD/.test(en + zh + JSON.stringify(lab)));
  assert.ok(!/[数据图网络视频质样优误]/.test(JSON.stringify(lab)), 'Simplified characters');
  const zero = describe('en', b, 0, attackResult(b, 0, 'fgsm'), attackResult(b, 0, 'random'));
  assert.match(zero, /ε = 0/);
});
