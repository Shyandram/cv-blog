import test from 'node:test';
import assert from 'node:assert/strict';
import { lit, globalMask, localMask } from '../../src/lib/playground/labs/segmentation-threshold.ts';
import { capture, enhance, FULL_SCALE } from '../../src/lib/playground/labs/lowlight-photons.ts';
import { uniformField, normalField } from '../../src/lib/imagekit/rng.ts';
import { iou, psnr } from '../../src/lib/imagekit/metrics.ts';
import { radialLowpass, fft2d } from '../../src/lib/imagekit/fft.ts';
import { labs } from '../../src/lib/playground/labs/registry.ts';
import { labLoaders } from '../../src/lib/playground/labs/loaders.ts';

/** A small synthetic page: paper 0.88, ink 0.22 in stripes, so the true mask is known. */
function page(w = 96, h = 48) {
  const data = new Float32Array(w * h), truth = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const ink = y % 8 < 3 && x > 4 && x < w - 5; truth[y * w + x] = ink ? 1 : 0; data[y * w + x] = ink ? 0.22 : 0.88; }
  return { w, h, data, truth };
}

test('even light: global Otsu and the local rule both recover the text exactly', () => {
  const p = page(), im = lit(p, 0);
  assert.equal(iou(globalMask(im).mask, p.truth), 1);
  assert.ok(iou(localMask(im, p.w, p.h, 21, 0.25), p.truth) > 0.9);
});

test('uneven light breaks the global threshold, but the local rule follows the light', () => {
  const p = page(), im = lit(p, 0.85);
  const g = iou(globalMask(im).mask, p.truth), l = iou(localMask(im, p.w, p.h, 21, 0.25), p.truth);
  assert.ok(g < 0.7, `global IoU should collapse, got ${g}`);
  assert.ok(l > 0.95, `local IoU should hold, got ${l}`);
});

test('lighting only dims: it never changes the left edge and scales the right edge by 1 − ramp', () => {
  const p = page(), im = lit(p, 0.6);
  assert.equal(im[0], p.data[0]);
  assert.ok(Math.abs(im[p.w - 1] - p.data[p.w - 1] * 0.4) < 1e-6);
});

test('digital gain leaves the signal-to-noise ratio unchanged; more photons improve it', () => {
  const n = 64 * 64, scene = Float32Array.from({ length: n }, (_, i) => 0.2 + 0.5 * (i % 64) / 63);
  const u = uniformField(n, 11), z = normalField(n, 12);
  const a = capture(scene, 8, 1, u, z), b = capture(scene, 8, 4, u, z), c = capture(scene, 60, 1, u, z);
  assert.ok(Math.abs(a.snrDb - b.snrDb) < 1e-9, 'gain must not change SNR');
  assert.ok(c.snrDb > a.snrDb + 5, `60 photons should beat 8 photons by well over 5 dB: ${a.snrDb} vs ${c.snrDb}`);
  assert.ok(b.clipped >= a.clipped);
  // Same seeds: the realisation is identical, so the electrons do not change with gain.
  assert.deepEqual(Array.from(a.electrons), Array.from(b.electrons));
});

test('a denoiser raises PSNR on a noisy capture and a 1x gain at full exposure is unclipped on a mid-gray scene', () => {
  const n = 64 * 64, scene = new Float32Array(n).fill(0.4);
  const u = uniformField(n, 11), z = normalField(n, 12), cap = capture(scene, 20, 5, u, z);
  const raw = cap.display, smooth = enhance('gaussian', raw, 64, 64);
  assert.ok(psnr(scene.map(v => v * 20 * 5 / FULL_SCALE), smooth) > psnr(scene.map(v => v * 20 * 5 / FULL_SCALE), raw));
  assert.equal(capture(scene, FULL_SCALE, 1, u, z).clipped, 0);
});

test('an ideal low-pass rings but a Gaussian does not (the property the frequency lab teaches)', () => {
  const n = 64, edge = new Float32Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) edge[y * n + x] = x < 32 ? 0.1 : 0.9;
  const run = (kind: 'ideal' | 'gaussian') => { const re = Float64Array.from(edge), im = new Float64Array(n * n), H = radialLowpass(n, n, kind, 10); fft2d(re, im, n, n); for (let i = 0; i < re.length; i++) { re[i] *= H[i]; im[i] *= H[i]; } fft2d(re, im, n, n, true); return Array.from(re); };
  assert.ok(run('ideal').some(v => v > 0.93 || v < 0.07));
  assert.ok(run('gaussian').every(v => v < 0.9 + 1e-3 && v > 0.1 - 1e-3));
});

test('every lab has a loader, at most three controls, a valid predict answer and bilingual text', () => {
  for (const lab of labs) {
    assert.ok(labLoaders[lab.id], `loader for ${lab.id}`);
    assert.ok(lab.controls.length >= 1 && lab.controls.length <= 3, `${lab.id} controls`);
    assert.ok(lab.predict.options.some(o => o.value === lab.predict.answer), `${lab.id} predict answer`);
    for (const field of [lab.title, lab.description, lab.challenge, lab.note, lab.predict.question, lab.predict.explain]) assert.ok(field.en.length > 10 && field['zh-tw'].length > 5, `${lab.id} text`);
    assert.ok(lab.panes.length >= 2 && lab.related.length >= 1);
    assert.equal(new Set(lab.panes.map(p => p.key)).size, lab.panes.length);
    for (const c of lab.controls) assert.ok(c.kind === 'range' ? c.min! <= Number(c.value) && Number(c.value) <= c.max! : c.options!.some(o => o.value === String(c.value)), `${lab.id}.${c.key} default`);
  }
});
