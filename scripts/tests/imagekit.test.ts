import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fft2d, fft1d, radialLowpass, filterFrequency, logSpectrum, freqIndex } from '../../src/lib/imagekit/fft.ts';
import { gaussianBlur, median3, sobel, downsample, reflectIndex } from '../../src/lib/imagekit/filter.ts';
import { otsuThreshold, localThreshold, histogram } from '../../src/lib/imagekit/threshold.ts';
import { psnr, mse, iou } from '../../src/lib/imagekit/metrics.ts';
import { dct8, idct8, blockDct, blockIdct, quantizeUniform, keepLargest, haar2d, ihaar2d } from '../../src/lib/imagekit/transform.ts';
import { uniformField, normalField, invNormal, poissonFromUniform } from '../../src/lib/imagekit/rng.ts';

const G = JSON.parse(readFileSync(new URL('./golden/imagekit.json', import.meta.url), 'utf8'));
const maxDiff = (a: ArrayLike<number>, b: ArrayLike<number>) => { assert.equal(a.length, b.length); let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const near = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number, label: string) => assert.ok(maxDiff(a, b) < tol, `${label}: max difference ${maxDiff(a, b)} >= ${tol}`);
const f32 = (a: number[]) => Float32Array.from(a);

test('FFT matches numpy and inverts exactly', () => {
  const { w, h, input } = G.fft;
  const re = Float64Array.from(input), im = new Float64Array(w * h);
  fft2d(re, im, w, h);
  near(re.subarray(0, 64), G.fft.fft_re_sample, 1e-9, 'fft real');
  near(im.subarray(0, 64), G.fft.fft_im_sample, 1e-9, 'fft imag');
  fft2d(re, im, w, h, true);
  near(re, input, 1e-9, 'round trip');
  assert.ok(Math.max(...im.map(Math.abs)) < 1e-9);
  assert.throws(() => fft1d(new Float64Array(6), new Float64Array(6), 6));
});

test('radial low-pass filters reproduce the numpy result for ideal, Butterworth and Gaussian', () => {
  const { w, h, input } = G.fft;
  for (const kind of ['ideal', 'butterworth', 'gaussian'] as const) {
    const out = filterFrequency(f32(input), w, h, radialLowpass(w, h, kind, 6, 2));
    near(out, G.fft.filters[kind].output, 1e-5, kind);
  }
  assert.equal([0, 1, 16, 17, 31].map(k => freqIndex(k, 32)).join(), '0,1,16,-15,-1');
});

test('an ideal low-pass rings while a Gaussian never overshoots a step edge', () => {
  const n = 64, step = new Float32Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) step[y * n + x] = x < n / 2 ? 0.2 : 0.8;
  const out = (kind: 'ideal' | 'gaussian') => filterFrequency(step, n, n, radialLowpass(n, n, kind, 8));
  const over = (a: Float32Array) => a.reduce((m, v) => Math.max(m, v - 0.8, 0.2 - v), 0);
  assert.ok(over(out('ideal')) > 0.03, 'ideal filter should overshoot');
  assert.ok(over(out('gaussian')) < 1e-3, 'gaussian filter should not overshoot');
});

test('log spectrum is centred and normalised', () => {
  const n = 16, flat = new Float32Array(n * n).fill(0.5);
  const s = logSpectrum(flat, n, n);
  assert.equal(s[(n / 2) * n + n / 2], 1);
  assert.equal(s.reduce((a, v) => a + (v > 0 ? 1 : 0), 0), 1);
});

test('8×8 DCT matches scipy, is orthonormal and inverts', () => {
  near(dct8(G.dct8.input), G.dct8.output, 1e-9, 'dct');
  near(idct8(dct8(G.dct8.input)), G.dct8.input, 1e-12, 'idct');
  const energy = (a: ArrayLike<number>) => Array.from(a as number[]).reduce((s, v) => s + v * v, 0);
  assert.ok(Math.abs(energy(dct8(G.dct8.input)) - energy(G.dct8.input)) < 1e-10);
});

test('block DCT quantization matches scipy and reports the nonzero fraction', () => {
  const { w, h, input, step } = G.blockdct;
  const coef = blockDct(f32(input), w, h);
  near(coef, G.blockdct.coef, 1e-6, 'block coefficients');
  const { q, nonzero } = quantizeUniform(coef, step);
  assert.ok(Math.abs(nonzero - G.blockdct.nonzero) < 1e-12);
  near(blockIdct(q, w, h), G.blockdct.recon, 1e-5, 'reconstruction');
  const { q: top, nonzero: kept } = keepLargest(coef, 0.25);
  assert.equal(kept, 0.25);
  assert.ok(top.every((v, i) => v === 0 || v === coef[i]));
});

test('Haar transform matches PyWavelets subbands, preserves energy and inverts', () => {
  const { w, h, levels, input } = G.haar;
  const c = haar2d(f32(input), w, h, levels), ref = G.haar.pywt_array as number[];
  const block = (a: ArrayLike<number>, x0: number, y0: number, n: number) => { const o: number[] = []; for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) o.push(a[(y0 + y) * w + x0 + x]); return o; };
  near(block(c, 0, 0, 4), block(ref, 0, 0, 4), 1e-5, 'LL');
  near(block(c, 4, 4, 4), block(ref, 4, 4, 4), 1e-5, 'HH level 2');
  near(block(c, 8, 8, 8), block(ref, 8, 8, 8), 1e-5, 'HH level 1');
  const tr = block(c, 8, 0, 8), bl = block(c, 0, 8, 8), rtr = block(ref, 8, 0, 8), rbl = block(ref, 0, 8, 8);
  assert.ok((maxDiff(tr, rtr) < 1e-5 && maxDiff(bl, rbl) < 1e-5) || (maxDiff(tr, rbl) < 1e-5 && maxDiff(bl, rtr) < 1e-5), 'HL/LH subbands match up to naming');
  const energy = (a: ArrayLike<number>) => Array.from(a as number[]).reduce((s, v) => s + v * v, 0);
  assert.ok(Math.abs(energy(c) - energy(input)) < 1e-4);
  near(ihaar2d(c, w, h, levels), input, 1e-5, 'inverse');
});

test('Gaussian blur matches scipy including reflected borders and a wide sigma', () => {
  const { w, h, input } = G.gaussian;
  for (const c of G.gaussian.cases) near(gaussianBlur(f32(input), w, h, c.sigma), c.output, 2e-6, `sigma ${c.sigma}`);
  assert.equal([-1, -2, 0, 5, 6, 7].map(i => reflectIndex(i, 6)).join(), '0,1,0,5,5,4');
});

test('median 3×3 and Sobel match scipy with edge replication', () => {
  near(median3(f32(G.median3.input), G.median3.w, G.median3.h), G.median3.output, 1e-6, 'median');
  const { ix, iy } = sobel(f32(G.sobel.input), G.sobel.w, G.sobel.h);
  near(ix, G.sobel.ix, 1e-5, 'sobel x'); near(iy, G.sobel.iy, 1e-5, 'sobel y');
});

test('Otsu and the Gaussian local threshold match scikit-image', () => {
  assert.ok(Math.abs(otsuThreshold(G.otsu.input) - G.otsu.threshold) < 1e-9, `otsu ${otsuThreshold(G.otsu.input)} vs ${G.otsu.threshold}`);
  for (const c of G.local.cases) near(localThreshold(f32(G.local.input), G.local.w, G.local.h, c.block, c.offset), c.threshold, 1e-5, `local block ${c.block}`);
});

test('PSNR, MSE and IoU match scikit-image', () => {
  const { a, b } = G.metrics;
  assert.ok(Math.abs(psnr(a, b) - G.metrics.psnr) < 1e-9);
  assert.ok(Math.abs(mse(a, b) - G.metrics.mse) < 1e-12);
  assert.ok(Math.abs(iou(G.metrics.mask_a, G.metrics.mask_b) - G.metrics.iou) < 1e-12);
  assert.equal(psnr(a, a), Infinity);
  assert.equal(iou([0, 0], [0, 0]), 1);
});

test('downsample averages blocks and histogram counts every value', () => {
  const d = downsample({ w: 4, h: 2, data: Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7]) }, 2);
  assert.deepEqual(Array.from(d.data), [2.5, 4.5]);
  const h = histogram([0, 0.1, 0.5, 0.99, 1], 4);
  assert.equal(h.reduce((a, v) => a + v, 0), 5);
});

test('seeded fields are reproducible; normal and Poisson samples have the right moments', () => {
  assert.deepEqual(Array.from(uniformField(5, 7)), Array.from(uniformField(5, 7)));
  assert.notDeepEqual(Array.from(uniformField(5, 7)), Array.from(uniformField(5, 8)));
  const z = normalField(40000, 1), m = z.reduce((s, v) => s + v, 0) / z.length, v = z.reduce((s, x) => s + (x - m) ** 2, 0) / z.length;
  assert.ok(Math.abs(m) < 0.03 && Math.abs(v - 1) < 0.04, `normal moments ${m} ${v}`);
  assert.ok(Math.abs(invNormal(0.975) - 1.959964) < 1e-5 && Math.abs(invNormal(0.5)) < 1e-9);
  const u = uniformField(40000, 3);
  for (const lam of [2, 20, 100]) {
    const s = Array.from(u, x => poissonFromUniform(lam, x)), mean = s.reduce((a, b) => a + b, 0) / s.length, variance = s.reduce((a, b) => a + (b - mean) ** 2, 0) / s.length;
    assert.ok(Math.abs(mean - lam) < 0.05 * lam + 0.05, `poisson mean ${lam}: ${mean}`);
    assert.ok(Math.abs(variance - lam) < 0.1 * lam + 0.1, `poisson variance ${lam}: ${variance}`);
  }
});
