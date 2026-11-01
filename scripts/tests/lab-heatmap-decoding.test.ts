// Facts for the `heatmap-decoding` lab (DLCV 06A): the windowed heatmap equals a full map, the three decoders behave as the chapter says
// (argmax error 0.383 x stride, quarter offset about half, Taylor step exact on a clean Gaussian), and the default state shows the idea.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  lab, keypoints, heatmapWindow, decode, peak, sweep, mean, windowRadius, STRIDES, DECODERS, DEFAULTS, INPUT, MARGIN, N_KEYPOINTS, SQUARE_MEAN_DIST, EXAMPLE, type Decoder, type Pt,
} from '../../src/lib/playground/labs/heatmap-decoding.ts';

const near = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);

/** The chapter's NumPy decoders on a full (noise-free) heatmap of (INPUT / s)^2 cells. */
function fullMap(pt: Pt, s: number, sigma: number) {
  const n = INPUT / s, h = new Float64Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) h[y * n + x] = Math.max(1e-6, Math.exp(-((x - pt.x / s) ** 2 + (y - pt.y / s) ** 2) / (2 * sigma * sigma)));
  return { h, n };
}
function fullDecode({ h, n }: { h: Float64Array; n: number }, s: number, method: Decoder): Pt {
  let best = -Infinity, bx = 0, by = 0;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if (h[y * n + x] > best) { best = h[y * n + x]; bx = x; by = y; }
  const v = (x: number, y: number) => h[y * n + x];
  let px = bx, py = by;
  if (method === 'quarter') { px += 0.25 * Math.sign(v(bx + 1, by) - v(bx - 1, by)); py += 0.25 * Math.sign(v(bx, by + 1) - v(bx, by - 1)); }
  if (method === 'taylor') {
    const L = (x: number, y: number) => Math.log(Math.max(v(x, y), 1e-10));
    const gx = (L(bx + 1, by) - L(bx - 1, by)) / 2, gy = (L(bx, by + 1) - L(bx, by - 1)) / 2;
    const hxx = L(bx + 1, by) - 2 * L(bx, by) + L(bx - 1, by), hyy = L(bx, by + 1) - 2 * L(bx, by) + L(bx, by - 1);
    const hxy = (L(bx + 1, by + 1) - L(bx - 1, by + 1) - L(bx + 1, by - 1) + L(bx - 1, by - 1)) / 4, det = hxx * hyy - hxy * hxy;
    if (det > 0 && hxx < 0) { px += -(hyy * gx - hxy * gy) / det; py += -(-hxy * gx + hxx * gy) / det; }
  }
  return { x: px * s, y: py * s };
}

test('keypoints are reproducible, uniform in the margin-inset square, and 500 of them', () => {
  const a = keypoints(), b = keypoints();
  assert.equal(a.length, N_KEYPOINTS);
  assert.deepEqual(a, b);
  for (const p of a) assert.ok(p.x >= MARGIN && p.x <= INPUT - MARGIN && p.y >= MARGIN && p.y <= INPUT - MARGIN);
  near(mean(a.map(p => p.x)), INPUT / 2, 8, 'mean x');
});

test('the windowed heatmap decodes exactly like the full noise-free map (all three decoders, strides 1 to 8, sigma 1 to 4)', () => {
  const pts = keypoints(25);
  for (const sigma of [1, 2, 4]) for (const s of STRIDES) for (const pt of pts) {
    const w = heatmapWindow(pt, s, sigma, 0, 0), f = fullMap(pt, s, sigma);
    for (const d of DECODERS) { const a = decode(w, s, d), b = fullDecode(f, s, d); near(a.x, b.x, 1e-9, `${d} x s${s} sigma${sigma}`); near(a.y, b.y, 1e-9, `${d} y`); }
  }
});

test('outside the window the Gaussian is below exp(-2) = 0.135, far under the peak (at least 0.77) and the noise (0.01)', () => {
  for (let sigma = 1; sigma <= 4; sigma += 0.5) {
    const R = windowRadius(sigma), nearest = R + 1 - 0.5; // a cell just outside the window is at least R + 0.5 cells from the true point
    assert.ok(Math.exp(-(nearest * nearest) / (2 * sigma * sigma)) < Math.exp(-2) + 1e-12, `sigma ${sigma}`);
    assert.ok(R >= 4, 'the zoomed grid of 9 x 9 cells fits in the window');
  }
  assert.ok(Math.exp(-0.5 / (2 * 1 * 1)) > 0.77, 'at sigma 1 the brightest cell is at least 0.77: half a cell on each axis is at most 0.5 squared distance');
});

test('noise-free argmax error is 0.383 x stride, the quarter offset about halves it and the Taylor step is exact', () => {
  const pts = keypoints();
  for (const s of STRIDES) {
    const e = { argmax: 0, quarter: 0, taylor: 0 }, worst = { argmax: 0, taylor: 0 };
    for (const pt of pts) {
      const w = heatmapWindow(pt, s, 2, 0, 0);
      for (const d of DECODERS) { const p = decode(w, s, d), err = Math.hypot(p.x - pt.x, p.y - pt.y); e[d] += err / pts.length; if (d !== 'quarter') worst[d] = Math.max(worst[d], err); }
    }
    near(e.argmax, SQUARE_MEAN_DIST * s, 0.03 * s, `argmax at stride ${s}`);
    assert.ok(worst.argmax <= (s * Math.SQRT2) / 2 + 1e-9, 'a nearest-grid-point error never exceeds half the cell diagonal');
    assert.ok(e.quarter / e.argmax > 0.4 && e.quarter / e.argmax < 0.6, `quarter / argmax = ${e.quarter / e.argmax} at stride ${s}`);
    assert.ok(worst.taylor < 1e-6, `Taylor on a clean Gaussian: worst error ${worst.taylor} at stride ${s}`);
  }
});

test('the exact SQUARE_MEAN_DIST is (sqrt 2 + ln(1 + sqrt 2)) / 6 = 0.3826', () => near(SQUARE_MEAN_DIST, 0.3826, 5e-5, 'constant'));

test('with noise the sweep is deterministic and reproduces the chapter’s run (same toy, other random numbers)', () => {
  const a = sweep(2), b = sweep(2);
  for (const s of STRIDES) for (const d of DECODERS) assert.deepEqual(a[s][d], b[s][d]);
  // chapter: argmax 0.39 0.80 1.54 3.20; quarter 0.20 0.40 0.78 1.59; taylor 0.05 0.11 0.21 0.43
  const chapter: Record<Decoder, number[]> = { argmax: [0.39, 0.8, 1.54, 3.2], quarter: [0.2, 0.4, 0.78, 1.59], taylor: [0.05, 0.11, 0.21, 0.43] };
  for (const d of DECODERS) STRIDES.forEach((s, i) => near(mean(a[s][d]), chapter[d][i], 0.04 * s + 0.01, `${d} stride ${s}`));
});

test('default state: argmax error doubles with the stride and the Taylor step removes most of it', () => {
  assert.equal(DEFAULTS.decoder, 'argmax'); assert.equal(DEFAULTS.stride, 4); assert.equal(DEFAULTS.sigma, 2);
  const sw = sweep(DEFAULTS.sigma), am = STRIDES.map(s => mean(sw[s].argmax)), qu = STRIDES.map(s => mean(sw[s].quarter)), ta = STRIDES.map(s => mean(sw[s].taylor));
  for (let i = 1; i < 4; i++) { near(am[i] / am[i - 1], 2, 0.15, 'doubling'); assert.ok(qu[i] > qu[i - 1] && ta[i] > ta[i - 1]); }
  STRIDES.forEach((s, i) => { near(am[i], 0.383 * s, 0.05 * s, `argmax ${s}`); assert.ok(qu[i] / am[i] > 0.45 && qu[i] / am[i] < 0.6, 'quarter about half'); assert.ok(ta[i] / am[i] < 0.2, 'Taylor removes over 80%'); });
  // numbers quoted in the lab text
  const r2 = (x: number) => x.toFixed(2);
  assert.deepEqual(am.map(r2), ['0.37', '0.78', '1.58', '3.12']);
  assert.deepEqual(ta.map(r2), ['0.05', '0.11', '0.21', '0.41']);
  assert.equal(r2(qu[2]), '0.79');
  assert.equal(r2(mean(sweep(4)[8].taylor)), '1.58');
});

test('a wider blob makes the Taylor step worse under the same noise (the lab’s challenge)', () => {
  const t = (sigma: number) => mean(sweep(sigma)[8].taylor);
  assert.ok(t(4) > 2 * t(2) && t(2) > 2 * t(1), `${t(1)} ${t(2)} ${t(4)}`);
});

test('the example keypoint is visibly missed by argmax and found by the Taylor step at the default stride', () => {
  const w = heatmapWindow(EXAMPLE, 4, 2, 0.01, 78), err = (d: Decoder) => { const p = decode(w, 4, d); return Math.hypot(p.x - EXAMPLE.x, p.y - EXAMPLE.y); };
  assert.ok(err('argmax') > 1.2 && err('taylor') < 0.3, `${err('argmax')} ${err('taylor')}`);
  const pk = peak(w); assert.ok(pk.i >= 3 && pk.i < w.side - 3 && pk.j >= 3 && pk.j < w.side - 3);
});

test('metadata: three controls in range, predict answer verified by the numbers, bilingual text, three canvases, a source per method', () => {
  assert.equal(lab.controls.length, 3);
  for (const c of lab.controls) if (c.kind === 'range') assert.ok(c.min! <= Number(c.value) && Number(c.value) <= c.max!);
  assert.deepEqual(lab.controls[0].options!.map(o => o.value), STRIDES.map(String));
  assert.deepEqual(lab.controls[1].options!.map(o => o.value), [...DECODERS]);
  assert.ok(lab.predict.options.some(o => o.value === lab.predict.answer));
  const e = mean(sweep(2)[4].argmax); // the predict question: stride 4, almost perfect heatmap, brightest cell
  assert.equal(lab.predict.answer, e < 0.5 ? 'zero' : e < 2.5 ? 'one-half' : 'four');
  assert.equal(lab.panes.length, 3);
  assert.equal(lab.related[0], 'deep-learning/dl-06a-pose-keypoints');
  assert.ok(lab.sources.length >= 4);
  for (const f of [lab.title, lab.description, lab.challenge, lab.note, lab.predict.question, lab.predict.explain]) assert.ok(f.en.length > 20 && f['zh-tw'].length > 10);
});
