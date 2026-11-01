// Tests for the lab image-fusion (DLCV 05B). The reference values come from scripts/figures/playground_goldens_fusion_lut.py
// (NumPy, SciPy, scikit-image on the exact PNG assets). Every number quoted in the lab's text is asserted here.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { readGray, asset } from './helpers/png.ts';
import { psnr } from '../../src/lib/imagekit/metrics.ts';
import * as lab from '../../src/lib/playground/labs/image-fusion.ts';

const golden = JSON.parse(readFileSync(new URL('./golden/fusion-lut.json', import.meta.url), 'utf8')).fusion;
const A = readGray(asset('image-fusion-a.png')), B = readGray(asset('image-fusion-b.png')), T = readGray(asset('image-fusion-truth.png')), F = readGray(asset('image-fusion-focus.png'));
const focus = Uint8Array.from(F.data, v => (v > 0.5 ? 1 : 0)), tex = lab.texturedPixels(T.data, T.w, T.h);
const near = (a: number, b: number, tol: number, msg = '') => assert.ok(Math.abs(a - b) <= tol, `${msg} ${a} vs ${b}`);
const go = (kind: lab.Measure, sigma: number, smooth: number) => lab.run(A, B, T, focus, tex, kind, sigma, smooth);

test('the assets are 256 x 256 and the focus disk is the true decision map', () => {
  for (const g of [A, B, T, F]) { assert.equal(g.w, 256); assert.equal(g.h, 256); }
  near(focus.reduce((s, v) => s + v, 0) / focus.length, Math.PI * 75 * 75 / (256 * 256), 0.002, 'disk area');
});

test('laplace is scipy.ndimage.laplace (reflect borders)', () => {
  const out = lab.laplace(Float32Array.from(golden.laplaceSample.input), 6, 5);
  for (let i = 0; i < out.length; i++) near(out[i], golden.laplaceSample.output[i], 1e-3, `i=${i}`);
});

test('the average is blurry: PSNR and the single shots match scikit-image', () => {
  near(psnr(T.data, lab.average(A.data, B.data)), golden.psnrAvg, 1e-4);
  near(psnr(T.data, A.data), golden.psnrA, 1e-4);
  near(psnr(T.data, B.data), golden.psnrB, 1e-4);
  assert.ok(golden.psnrAvg > golden.psnrA && golden.psnrAvg > golden.psnrB, 'the average beats either single shot, which is half blurred');
});

test('every focus-rule case agrees with the NumPy / SciPy pipeline', () => {
  for (const c of golden.cases) {
    const r = go(c.kind, c.sigma, c.smooth);
    near(r.psnrFused, c.psnr, 0.05, `${c.kind} sigma ${c.sigma} smooth ${c.smooth}: PSNR`);
    near(r.agree, c.agree, 0.003, `${c.kind} sigma ${c.sigma} smooth ${c.smooth}: agreement`);
  }
});

test('default state: the average is blurry (about 30 dB) while the focus rule is sharp (about 39 dB)', () => {
  const state = Object.fromEntries(lab.lab.controls.map(c => [c.key, c.value]));
  assert.equal(state.measure, 'laplacian'); assert.equal(state.window, 2); assert.equal(state.smooth, 0);
  const r = go('laplacian', 2, 0);
  assert.equal(r.psnrAvg.toFixed(2), '30.22');
  assert.equal(r.psnrFused.toFixed(2), '39.20');
  assert.ok(r.psnrFused - r.psnrAvg > 8.5);
  assert.equal((100 * r.agree).toFixed(1), '94.7');
});

test('a window that is too small makes the decision map noisy and the result worse; local variance suffers more', () => {
  const small = go('laplacian', 0.5, 0), mid = go('laplacian', 2, 0), varSmall = go('variance', 0.5, 0), varMid = go('variance', 2, 0);
  assert.ok(small.psnrFused < mid.psnrFused - 1.5 && small.agree < mid.agree - 0.05, `${small.psnrFused} ${mid.psnrFused}`);
  assert.ok(varSmall.psnrFused < small.psnrFused - 3, 'local variance at 0.5 is much worse than Laplacian energy at 0.5');
  assert.ok(varSmall.psnrFused > varSmall.psnrAvg + 3, 'but still better than averaging');
  assert.ok(varMid.psnrFused > 39);
  // Smoothing the map repairs most of the damage at a small window.
  assert.ok(go('laplacian', 0.5, 2).psnrFused > small.psnrFused + 1.5);
});

test('predict answer: the average keeps about 61% of the fine contrast inside the disk, the focus rule 101%', () => {
  const r = go('laplacian', 2, 0), truth = lab.detailContrast(T.data, 256, 256, focus);
  const ra = lab.detailContrast(r.avg, 256, 256, focus) / truth, rf = lab.detailContrast(r.fused, 256, 256, focus) / truth;
  near(ra, golden.detail.ratioAverage, 0.003); near(rf, golden.detail.ratioFused, 0.003);
  assert.equal(Math.round(100 * ra), 61); assert.equal(Math.round(100 * rf), 101);
});

test('blend with weights 1 returns shot A and with 0 returns shot B; clipping keeps [0, 1]', () => {
  const ones = new Float32Array(A.data.length).fill(1), zeros = new Float32Array(A.data.length);
  assert.deepEqual(Array.from(lab.blend(A.data, B.data, ones)), Array.from(A.data));
  assert.deepEqual(Array.from(lab.blend(A.data, B.data, zeros)), Array.from(B.data));
  const out = lab.blend(Float32Array.from([2, -1]), Float32Array.from([2, -1]), Float32Array.from([0.5, 0.5]));
  assert.deepEqual(Array.from(out), [1, 0]);
});

test('metadata: three controls, predict answer is an option, both languages filled', () => {
  assert.ok(lab.lab.controls.length <= 3);
  assert.ok(lab.lab.predict.options.some(o => o.value === lab.lab.predict.answer));
  for (const lang of ['en', 'zh-tw'] as const) for (const k of ['title', 'description', 'challenge', 'note'] as const) assert.ok(lab.lab[k][lang].length > 20);
  assert.ok(!/[数据图网络视频默认质样]/.test(JSON.stringify(lab.lab)), 'no Simplified characters');
});
