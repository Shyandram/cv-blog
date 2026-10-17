// Tests for the two labs by helper `geometry`: corner-tensor (IP 12) and receptive-field (DLCV 01 / ML 03).
// Every number quoted in the labs' texts is asserted here by running the lab's own pipeline on the real asset
// (src/assets/playground/camera-256.png); goldens come from scripts/figures/playground_goldens_harris.py and
// playground_goldens_receptive.py (NumPy and SciPy, independent of the TypeScript).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { readGray, asset } from './helpers/png.ts';
import { gaussianGradient, structureTensorMaps, harrisResponse, windowWeights, eigenSym2, harrisRatioBoundary } from '../../src/lib/imagekit/harris.ts';
import * as ct from '../../src/lib/playground/labs/corner-tensor.ts';
import * as rf from '../../src/lib/playground/labs/receptive-field.ts';

const golden = (name: string) => JSON.parse(readFileSync(new URL(`./golden/${name}.json`, import.meta.url), 'utf8'));
/** Every element within max(abs, rel * |expected|) of the expected value. */
const close = (a: ArrayLike<number>, b: ArrayLike<number>, rel = 1e-4, abs = 1e-9) => {
  assert.equal(a.length, b.length);
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]) / Math.max(abs, rel * Math.abs(b[i])));
  assert.ok(worst <= 1, `max error ratio ${worst}`);
};
const maxAbs = (a: ArrayLike<number>) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m; };

const cam = readGray(asset('camera-256.png'));
const grads = ct.gradientsOf(cam);
const hg = golden('harris'), hc = hg.camera;
const field = (win: number) => ct.harrisField(grads, win);
const rel = (a: number, b: number, tol = 1e-3) => assert.ok(Math.abs(a - b) <= tol * Math.max(1e-12, Math.abs(b)), `${a} vs ${b}`);

// ------------------------------------------------------------------------------------------- Harris kit vs SciPy

test('Gaussian derivatives, windowed tensor and Harris response match SciPy on a seeded synthetic image', () => {
  const s = hg.synthetic, img = Float32Array.from(s.image), w = s.w, h = s.h;
  const { ix, iy } = gaussianGradient(img, w, h, 1);
  close(ix, s.ix, 1e-4, 1e-6); close(iy, s.iy, 1e-4, 1e-6);
  const m = structureTensorMaps(ix, iy, w, h, windowWeights(4, 9 / 6));
  close(m.xx, s.xx, 2e-4, 1e-8); close(m.yy, s.yy, 2e-4, 1e-8); close(m.xy, s.xy, 2e-4, 1e-8);
  close(harrisResponse(m.xx, m.yy, m.xy, 0.05), s.R, 5e-4, 1e-10);
  // The chapter's own function (scipy gaussian_filter, sigma_i = 1.5, truncated at 4 sigma = radius 6) is the same pipeline with radius 6.
  const m6 = structureTensorMaps(ix, iy, w, h, windowWeights(6, 1.5));
  close(m6.xx, s.radius6.xx, 2e-4, 1e-8); close(m6.xy, s.radius6.xy, 2e-4, 1e-8);
  close(harrisResponse(m6.xx, m6.yy, m6.xy, 0.05), s.chapterR, 5e-4, 1e-10);
  const e = eigenSym2(m.xx[500], m.yy[500], m.xy[500]);
  rel(e.l1, s.l1[500], 1e-3); assert.ok(Math.abs(e.l2 - s.l2[500]) < 1e-7);
});

test('R is positive exactly when lambda2 / lambda1 exceeds the root t = 0.0557 of k t^2 + (2k - 1) t + k', () => {
  const t = harrisRatioBoundary(0.05);
  assert.ok(Math.abs(t - 0.0557) < 1e-4, `t = ${t}`);
  for (const [l1, l2] of [[1, 0.05], [1, 0.06], [4, 0.2], [4, 0.23], [9, 5]]) assert.equal(l1 * l2 - 0.05 * (l1 + l2) ** 2 > 0, l2 / l1 > t);
});

// ------------------------------------------------------------------------------------------- corner-tensor lab facts

test('the lab pipeline reproduces the NumPy and SciPy numbers on the real camera asset at four window sizes', () => {
  for (const win of [5, 9, 13, 25]) {
    const f = field(win), g = hc.byWindow[String(win)];
    rel(f.rmax, g.Rmax, 2e-3); rel(f.l1max * 255 * 255, g.l1max, 2e-3);
    const best = f.R.indexOf(f.rmax);
    assert.deepEqual([best % 256, Math.floor(best / 256)], g.argmaxR);
    for (const [name, v] of Object.entries(hc.marks) as [string, number[]][]) {
      const p = ct.probeAt(grads, f, v[0], v[1]), q = g.points[name];
      rel(p.l1, q.l1, 2e-3); assert.ok(Math.abs(p.l2 - q.l2) <= 2e-3 * q.l1, `${name} l2 ${p.l2} vs ${q.l2}`);
      assert.ok(Math.abs(p.rRel - q.Rrel) < 2e-3, `${name} Rrel ${p.rRel} vs ${q.Rrel}`);
    }
    const sh = ct.verdictShares(f);
    for (const k of ['flat', 'edge', 'corner'] as const) assert.ok(Math.abs(sh[k] - g.fractions[k]) < 2e-4, `${k} share at ${win}`);
  }
});

test('the tensor shown for the probe equals the weighted sum over the scattered gradient vectors it draws', () => {
  for (const [x, y, win] of [[89, 103, 9], [0, 0, 9], [255, 255, 25], [30, 80, 5], [200, 20, 13]]) {
    const f = field(win), p = ct.probeAt(grads, f, x, y), t = ct.tensorFromVectors(p);
    assert.equal(p.vectors.length, win * win);
    assert.ok(Math.abs(t.xx - p.xx) <= 2e-3 * p.l1 + 1e-6 && Math.abs(t.yy - p.yy) <= 2e-3 * p.l1 + 1e-6 && Math.abs(t.xy - p.xy) <= 2e-3 * p.l1 + 1e-6, `(${x},${y}) W${win}`);
  }
});

test('default probe: the strongest corner of the image at window 9, with both eigenvalues large', () => {
  const f = field(ct.DEFAULT_PROBE.win), p = ct.probeAt(grads, f, ct.DEFAULT_PROBE.x, ct.DEFAULT_PROBE.y);
  assert.equal(f.R.indexOf(f.rmax), ct.DEFAULT_PROBE.y * 256 + ct.DEFAULT_PROBE.x);
  assert.equal(p.verdict, 'corner'); assert.equal(p.rRel, 1);
  assert.ok(p.l2 > 0.2 * p.l1, `ratio ${p.ratio}`);
  const text = ct.describe('en', p, f.l1max);
  assert.ok(text.includes('Verdict: corner') && text.includes(`λ1 = ${Math.round(p.l1)}`) && text.includes(`λ2 = ${Math.round(p.l2)}`), text);
  assert.ok(text.includes('R is 100% of the largest R in the image') && text.includes('cutoff 155'), text);
  assert.ok(ct.describe('zh-tw', p, f.l1max).includes('判定：角點') && ct.describe('zh-tw', p, f.l1max).includes('的 100%'));
  // The cutoff is 5% of the largest lambda1: 3101 gray levels squared at window 9.
  assert.equal(Math.round(0.05 * hc.byWindow['9'].l1max), 155);
});

test('flat, edge, corner: A, C, B have the eigenvalue patterns and Harris signs the lab teaches (window 9)', () => {
  const f = field(9), at = (k: string) => { const m = ct.MARKS.find(m => m.key === k)!; return ct.probeAt(grads, f, m.x, m.y); };
  const A = at('A'), B = at('B'), C = at('C');
  assert.deepEqual([A.verdict, B.verdict, C.verdict], ['flat', 'corner', 'edge']);
  // Flat: a point cloud near the origin (largest gradient in the window below 1 gray level per pixel; strong edges reach 74).
  assert.ok(Math.max(...A.vectors.map(v => Math.hypot(v.ix, v.iy))) < 1, 'sky gradients are tiny');
  assert.ok(A.l1 < 1 && Math.abs(A.rRel) < 1e-6);
  // Edge: all gradients along one line (lambda2 / lambda1 below 0.001) and R negative.
  assert.ok(C.ratio < 0.001 && C.rRel < -0.1, `C ratio ${C.ratio} R ${C.rRel}`);
  // Corner: two directions (ratio above 0.2) and R positive.
  assert.ok(B.ratio > 0.2 && B.rRel > 0.3, `B ratio ${B.ratio} R ${B.rRel}`);
  // The numbers quoted in the predict explanation.
  const ex = ct.lab.predict.explain;
  for (const s of [`λ1 = ${Math.round(B.l1)} and λ2 = ${Math.round(B.l2)}`, `λ1 = ${Math.round(C.l1)}, but λ2 = ${(Math.round(C.l2 * 100) / 100)}`.replace(', but', ' but'), `λ1 = ${(Math.round(A.l1 * 100) / 100)}`]) assert.ok(ex.en.includes(s), `${s} in explain`);
  for (const s of [`λ1 = ${Math.round(B.l1)}、λ2 = ${Math.round(B.l2)}`, `λ1 = ${Math.round(C.l1)}，但 λ2 = ${(Math.round(C.l2 * 100) / 100)}`, `λ1 = ${(Math.round(A.l1 * 100) / 100)}`]) assert.ok(ex['zh-tw'].includes(s), `${s} in zh explain`);
  assert.equal(ct.lab.predict.answer, 'B');
  // The marks are the only corner among the three at the default window, and the sliders reach them.
  assert.ok(ct.MARKS.every(m => m.x >= 0 && m.x <= 255 && m.y >= 0 && m.y <= 255));
});

test('challenge: both tripod legs are edges with lambda2 / lambda1 below 0.0025, and their lines point in different directions', () => {
  const f = field(9), L1 = ct.probeAt(grads, f, 122, 232), L2 = ct.probeAt(grads, f, 183, 220);
  assert.equal(L1.verdict, 'edge'); assert.equal(L2.verdict, 'edge');
  assert.ok(L1.ratio < 0.0025 && L2.ratio < 0.0025, `${L1.ratio} ${L2.ratio}`);
  const turn = Math.abs(L1.angle - L2.angle) * 180 / Math.PI;
  assert.ok(turn > 40 && turn < 50, `the principal axis turns by ${turn} degrees`);
  const g = hc.byWindow['9'].points;
  assert.ok(Math.abs(L1.angle * 180 / Math.PI - g.legL.angleDeg) < 0.2 && Math.abs(L2.angle * 180 / Math.PI - g.legR.angleDeg) < 0.2);
  assert.ok(ct.lab.challenge.en.includes('0.0025') && ct.lab.challenge['zh-tw'].includes('0.0025'));
  assert.ok(ct.lab.challenge.en.includes('x 122, y 232') && ct.lab.challenge.en.includes('x 183, y 220'));
});

test('challenge: a bigger window labels a bigger share of the image as corner (1.4% at 9, 8.9% at 25)', () => {
  const s9 = ct.verdictShares(field(9)).corner, s25 = ct.verdictShares(field(25)).corner;
  assert.equal((100 * s9).toFixed(1), '1.4'); assert.equal((100 * s25).toFixed(1), '8.9');
  for (const lang of ['en', 'zh-tw'] as const) assert.ok(ct.lab.challenge[lang].includes('1.4%') && ct.lab.challenge[lang].includes('8.9%'));
  // Maximum gradient quoted in the source comment of AXIS: strong edges reach 74 gray levels per pixel, inside the plot range.
  assert.ok(Math.abs(hc.maxAbsGradient.ix - 74.0) < 0.1 && ct.AXIS > hc.maxAbsGradient.ix);
  assert.ok(Math.max(maxAbs(grads.ix), maxAbs(grads.iy)) * 255 < ct.AXIS);
});

test('the chapter text and the lab agree: k = 0.05, sigma_d = 1, W = 9 gives the chapter integration sigma 1.5', () => {
  assert.equal(ct.HARRIS_K, 0.05); assert.equal(ct.SIGMA_D, 1); assert.equal(ct.windowSigma(9), 1.5);
  // The chapter code truncates at 4 sigma (radius 6) rather than at the 9-pixel box (radius 4); the response barely changes.
  const chapter = hc.chapterR, f9 = field(9);
  assert.deepEqual(chapter.argmax, [89, 103]);
  assert.equal(f9.R.indexOf(f9.rmax), 103 * 256 + 89);
  assert.ok(Math.abs(chapter.at.B / chapter.max - hc.byWindow['9'].points.B.Rrel) < 0.03);
});

// ------------------------------------------------------------------------------------------- receptive field

test('recurrence and exact interval trace agree for every stack, every prefix and every unit position', () => {
  for (const st of rf.STACKS) for (let depth = 1; depth <= st.layers.length; depth++) {
    const layers = st.layers.slice(0, depth), rec = rf.recurrence(layers), sizes = rf.gridSizes(layers, 256);
    for (const unit of [0, 1, 7, Math.floor(sizes[depth] / 2), sizes[depth] - 1]) {
      const t = rf.traceInterval(layers, depth, unit);
      assert.equal(t.hi - t.lo + 1, rec.rows[depth].r, `${st.key} depth ${depth} unit ${unit}`);
      assert.equal((t.lo + t.hi) / 2, unit * rec.rows[depth].j, 'the receptive field is centered on index times jump');
    }
    const a = rf.traceInterval(layers, depth, 10), b = rf.traceInterval(layers, depth, 11);
    assert.equal(b.lo - a.lo, rec.rows[depth].j, 'neighboring units are one jump apart');
  }
});

test('recurrence and trace agree on random stacks, including even kernels (seeded)', () => {
  let seed = 7;
  const rnd = (n: number) => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed % n; };
  for (let t = 0; t < 200; t++) {
    const layers = Array.from({ length: 1 + rnd(7) }, () => ({ k: [1, 2, 3, 5, 7][rnd(5)], s: 1 + rnd(3), d: 1 + rnd(4) }));
    const rec = rf.recurrence(layers), u = 5 + rnd(20), tr = rf.traceInterval(layers, layers.length, u);
    assert.equal(tr.hi - tr.lo + 1, rec.r, JSON.stringify(layers));
  }
});

test('the numbers in the chapters: DL 01 and ML 03 recurrence examples, and the ML 03 figure (7, 13, 29)', () => {
  assert.equal(rf.recurrence(rf.STACKS.find(s => s.key === 'plain3')!.layers).r, 7);
  assert.equal(rf.recurrence(rf.STACKS.find(s => s.key === 'plain6')!.layers).r, 13);
  assert.equal(rf.recurrence(rf.STACKS.find(s => s.key === 'stride-every-2nd')!.layers).r, 29);
  const dl01 = [[3, 1], [3, 1], [2, 2], [3, 1], [3, 1], [2, 2], [3, 1]].map(([k, s]) => ({ k, s, d: 1 }));
  assert.equal(rf.recurrence(dl01).r, 24);
  assert.equal(rf.traceInterval(dl01, 7, 20).hi - rf.traceInterval(dl01, 7, 20).lo + 1, 24);
});

test('a dilated layer can leave holes in the exact set, but none of the stacks in the select does', () => {
  const single = [{ k: 3, s: 1, d: 2 }];
  assert.deepEqual(rf.traceSet(single, 1, 10), [8, 10, 12]);
  assert.equal(rf.recurrence(single).r, 5);
  for (const st of rf.STACKS) {
    const set = rf.traceSet(st.layers, st.layers.length, 20), span = rf.traceInterval(st.layers, st.layers.length, 20);
    assert.equal(set.length, span.hi - span.lo + 1, `${st.key} has no holes`);
  }
});

test('the boxes agree with the independent SciPy adjoint (brute-force checked): extents and pixel counts', () => {
  const g = golden('receptive');
  assert.deepEqual(Object.keys(g).sort(), rf.STACKS.map(s => s.key).sort());
  for (const st of rf.STACKS) {
    const q = g[st.key], mid = q.centerUnit, j = rf.recurrence(st.layers).j;
    assert.deepEqual(q.layers.map((l: number[]) => ({ k: l[0], s: l[1], d: l[2] })), st.layers, `${st.key} layers`);
    for (const [name, ux, uy] of [['center', mid, mid], ['corner', 0, 0], ['last', q.grid - 1, q.grid - 1]] as [string, number, number][]) {
      const u = rf.unitAt(st.layers, 256, ux * j, uy * j), e = q[name];
      assert.equal(u.grid, q.grid, `${st.key} grid`);
      assert.deepEqual([u.clipped.x0, u.clipped.x1], e.x, `${st.key} ${name} x`);
      assert.deepEqual([u.clipped.y0, u.clipped.y1], e.y, `${st.key} ${name} y`);
      assert.equal((u.clipped.x1 - u.clipped.x0 + 1) * (u.clipped.y1 - u.clipped.y0 + 1), e.pixels, `${st.key} ${name} pixels`);
    }
    const nb = rf.unitAt(st.layers, 256, (mid + 1) * j, mid * j);
    assert.deepEqual([nb.clipped.x0, nb.clipped.x1], q.neighbor.x);
  }
});

test('predict: stride 2 in the first layer grows the field most (11, against 9 and 9; stride in the last layer leaves 7)', () => {
  const r = (k: string) => rf.recurrence(rf.stackByKey(k).layers).r;
  assert.deepEqual(['plain3', 'plain4', 'stride-first', 'dilate-first', 'stride-last'].map(r), [7, 9, 11, 9, 7]);
  assert.equal(rf.lab.predict.answer, 'stride');
  const ex = rf.lab.predict.explain;
  for (const s of ['from 7 to 11', 'against 9 for a fourth layer and 9 for dilation 2', 'adds 4 pixels instead of 2', 'adds only 2 pixels', 'the field stays 7']) assert.ok(ex.en.includes(s), s);
  for (const s of ['由 7 增加到 11', '都只增加到 9', '各增加 4 像素，而不是 2 像素', '只增加 2 像素', '感受野仍是 7']) assert.ok(ex['zh-tw'].includes(s), s);
  // The jumps after each layer of stride-first are 2, 2, 2: the two later layers each add (3 - 1) * 2 = 4.
  assert.deepEqual(rf.recurrence(rf.stackByKey('stride-first').layers).rows.map(x => x.j), [1, 2, 2, 2]);
  assert.deepEqual(rf.recurrence(rf.stackByKey('stride-first').layers).rows.map(x => x.r), [1, 3, 7, 11]);
});

test('default state and challenge numbers on the real image: 29 x 29 (1.3%), jump 8, 32 x 32 units; dilated 31 x 31 (1.5%), jump 1', () => {
  const st = rf.stackByKey(rf.DEFAULT.stack), u = rf.unitAt(st.layers, cam.w, rf.DEFAULT.x, rf.DEFAULT.y);
  assert.deepEqual([u.r, u.jump, u.grid, u.cx, u.cy], [29, 8, 32, 120, 72]);
  assert.equal(rf.pct(u.covered), '1.3%');
  assert.equal(u.covered, 29 * 29 / 65536);
  const text = rf.describe('en', st, u);
  assert.ok(text.includes('29 × 29 pixel box, 1.3% of the image') && text.includes('8 pixels apart') && text.includes('32 × 32'), text);
  assert.ok(rf.describe('zh-tw', st, u).includes('占影像的 1.3%'));
  const dst = rf.stackByKey('dilated-1248'), d = rf.unitAt(dst.layers, 256, 120, 72), corner = rf.unitAt(dst.layers, 256, 0, 0);
  assert.deepEqual([d.r, d.jump, d.grid], [31, 1, 256]); assert.equal(rf.pct(d.covered), '1.5%');
  assert.ok(corner.covered < d.covered / 3, 'the corner box is clipped by the image border');
  assert.equal(rf.pct(corner.covered), '0.39%');
  for (const lang of ['en', 'zh-tw'] as const) { const c = rf.lab.challenge[lang]; for (const s of ['29', '1.3%', '31', '1.5%']) assert.ok(c.includes(s), `${lang} ${s}`); }
  assert.ok(rf.lab.challenge.en.includes('8 pixels apart') && rf.lab.challenge['zh-tw'].includes('相距 8 像素'));
});

test('both labs: controls are within limits, defaults valid, bilingual, and ids match the old demos', () => {
  for (const lab of [ct.lab, rf.lab]) {
    assert.ok(lab.controls.length <= 3);
    for (const c of lab.controls) assert.ok(c.kind === 'range' ? c.min! <= Number(c.value) && Number(c.value) <= c.max! : c.options!.some(o => o.value === String(c.value)));
    assert.ok(lab.predict.options.length >= 2 && lab.predict.options.length <= 4 && lab.predict.options.some(o => o.value === lab.predict.answer));
    for (const f of [lab.title, lab.description, lab.challenge, lab.note, lab.predict.question, lab.predict.explain]) assert.ok(f.en.length > 10 && f['zh-tw'].length > 5);
    for (const f of [lab.title, lab.description, lab.challenge, lab.note, lab.predict.question, lab.predict.explain]) assert.ok(!/[<{]/.test(f.en + f['zh-tw']), 'no < or { in text');
  }
  assert.deepEqual([ct.lab.id, rf.lab.id], ['corner-tensor', 'receptive-field']);
});
