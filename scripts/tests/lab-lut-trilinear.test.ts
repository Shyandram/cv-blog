// Tests for the lab lut-trilinear (DLCV 05A). The reference values come from scripts/figures/playground_goldens_fusion_lut.py
// (NumPy, SciPy map_coordinates). Every number quoted in the lab's text is asserted here.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { readRgb, asset } from './helpers/png.ts';
import * as lab from '../../src/lib/playground/labs/lut-trilinear.ts';

const golden = JSON.parse(readFileSync(new URL('./golden/fusion-lut.json', import.meta.url), 'utf8')).lut;
const photo = readRgb(asset('astronaut-256.png')), gradient = lab.gradientImage(256, 256);
const near = (a: number, b: number, tol: number, msg = '') => assert.ok(Math.abs(a - b) <= tol, `${msg} ${a} vs ${b}`);

test('the colour matrix and the exact transform match the NumPy definition', () => {
  lab.colourMatrix().forEach((v, i) => near(v, golden.matrix[i], 1e-12, `matrix ${i}`));
  for (const c of golden.colours) lab.exactColour(...(c.in as [number, number, number])).forEach((v, i) => near(v, c.out[i], 1e-12, `colour ${c.in}`));
  assert.deepEqual(lab.exactColour(0, 0, 0), [0, 0, 0]);
  const white = lab.exactColour(1, 1, 1);
  white.forEach(v => near(v, 1, 1e-9));
});

test('the tone curve is monotone and fixes 0 and 1; the saturation matrix keeps grays gray', () => {
  let prev = -1;
  for (let i = 0; i <= 1000; i++) { const v = lab.toneCurve(i / 1000); assert.ok(v >= prev); prev = v; }
  assert.equal(lab.toneCurve(0), 0); near(lab.toneCurve(1), 1, 1e-12);
  const [r, g, b] = lab.exactColour(0.4, 0.4, 0.4);
  near(r, g, 1e-9); near(g, b, 1e-9);
});

test('trilinear interpolation at random colours agrees with scipy.ndimage.map_coordinates; nearest agrees with indexing', () => {
  for (const p of golden.trilinearPoints) {
    const lut = lab.bakeLut(p.M), src = Float32Array.from(p.in);
    lab.applyLut(lut, p.M, src, 'trilinear').forEach((v, i) => near(v, p.out[i], 1e-6));
    lab.applyLut(lut, p.M, src, 'nearest').forEach((v, i) => near(v, p.nearest[i], 1e-6));
  }
});

test('a node is reproduced exactly by both lookups, and trilinear is exact for the identity table', () => {
  const M = 5, lut = lab.bakeLut(M), d = M - 1, nodes = Float32Array.from([1 / d, 2 / d, 3 / d, 0, 1, 0.5]);
  for (const mode of ['nearest', 'trilinear'] as const) {
    const out = lab.applyLut(lut, M, nodes, mode), ex = lab.exactImage(nodes);
    out.forEach((v, i) => near(v, ex[i], 1e-6, mode));
  }
  const id = new Float64Array(M * M * M * 3);
  for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) for (let k = 0; k < M; k++) { const o = ((i * M + j) * M + k) * 3; id[o] = i / d; id[o + 1] = j / d; id[o + 2] = k / d; }
  const x = Float32Array.from([0.13, 0.77, 0.999, 1, 0, 0.5]);
  lab.applyLut(id, M, x, 'trilinear').forEach((v, i) => near(v, x[i], 1e-6));
});

test('error statistics in 8-bit levels match NumPy for every table size on both images', () => {
  for (const [name, img] of [['gradient', gradient], ['photo', photo]] as const) for (const M of [2, 3, 5, 9, 17, 33]) {
    const r = lab.run(img.data, M), g = golden.stats[name][String(M)];
    near(r.eNearest.mean, g.nearest.mean, 0.02, `${name} M=${M} nearest mean`); near(r.eNearest.max, g.nearest.max, 1, `${name} M=${M} nearest max`);
    near(r.eTrilinear.mean, g.trilinear.mean, 0.02, `${name} M=${M} trilinear mean`); near(r.eTrilinear.max, g.trilinear.max, 1, `${name} M=${M} trilinear max`);
  }
});

test('default state (M = 9, gradient): nearest bands, trilinear is near-exact', () => {
  const state = Object.fromEntries(lab.lab.controls.map(c => [c.key, c.value]));
  assert.equal(state.m, '9'); assert.equal(state.image, 'gradient'); assert.equal(state.view, 'nearest');
  const r = lab.run(gradient.data, 9);
  assert.equal(r.eNearest.mean.toFixed(2), '9.84'); assert.equal(r.eNearest.max, 26);
  assert.equal(r.eTrilinear.mean.toFixed(2), '0.32'); assert.equal(r.eTrilinear.max, 4);
  // banding: a nearest lookup can return only the 729 colours stored in the table; the exact result has thousands of distinct colours
  const colours = (a: Uint8Array) => { const set = new Set<number>(); for (let i = 0; i < a.length; i += 3) set.add((a[i] << 16) | (a[i + 1] << 8) | a[i + 2]); return set.size; };
  assert.ok(colours(r.nearest) <= 729, `${colours(r.nearest)}`);
  assert.ok(colours(r.exact) > 5000, `${colours(r.exact)}`);
});

test('predict answer: trilinear with M = 9 beats nearest with M = 33 on the gradient (and on the photo)', () => {
  for (const img of [gradient, photo]) {
    const tri9 = lab.run(img.data, 9).eTrilinear, nn33 = lab.run(img.data, 33).eNearest;
    assert.ok(tri9.mean < nn33.mean / 2, `${tri9.mean} ${nn33.mean}`);
  }
  const g = lab.run(gradient.data, 33).eNearest, t = lab.run(gradient.data, 9).eTrilinear;
  assert.equal(g.mean.toFixed(2), '2.46'); assert.equal(g.max, 7);
  assert.equal(t.mean.toFixed(2), '0.32'); assert.equal(t.max, 4);
  assert.deepEqual([lab.lutSize(33).nodes, lab.lutSize(9).nodes], [35937, 729]);
  assert.equal(Math.round(35937 / 729), 49);
});

test('errors shrink as M grows (trilinear) and the table size is 3 M^3', () => {
  let prev = Infinity;
  for (const M of [2, 3, 5, 9, 17, 33]) { const e = lab.run(gradient.data, M).eTrilinear.mean; assert.ok(e < prev); prev = e; }
  assert.equal(lab.lutSize(9).values, 2187);
});

test('metadata: three controls, predict answer is an option, both languages filled', () => {
  assert.ok(lab.lab.controls.length <= 3);
  assert.ok(lab.lab.predict.options.some(o => o.value === lab.lab.predict.answer));
  for (const lang of ['en', 'zh-tw'] as const) for (const k of ['title', 'description', 'challenge', 'note'] as const) assert.ok(lab.lab[k][lang].length > 20);
  assert.ok(!/[数据图网络视频默认质样]/.test(JSON.stringify(lab.lab)), 'no Simplified characters');
});
