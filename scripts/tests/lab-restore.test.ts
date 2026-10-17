// Facts for the IP 05 lab (inverse-regularization) and the IP 09 lab (morphology-shapes): every number quoted in their
// text is asserted here by running the labs' own TypeScript pipelines on the real asset PNGs, and compared with the
// references that scripts/figures/playground_goldens_restore.py computed with NumPy on the same files.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { readGray, asset } from './helpers/png.ts';
import { normalField } from '../../src/lib/imagekit/rng.ts';
import {
  makeProblem, evaluate, psnrCurve, bestOnGrid, gaussianOtf, restorationGain, peakGain, kOf, LOGK_GRID, LOGK_OFF, EPS, lab,
} from '../../src/lib/playground/labs/inverse-regularization.ts';

const G = JSON.parse(readFileSync(new URL('./golden/restore.json', import.meta.url), 'utf8'));
const cam = readGray(asset('camera-256.png'));
const p = makeProblem(cam);
const r2 = (v: number) => Math.round(v * 100) / 100;

test('the fixed noise field is the same in TypeScript and in the Python reference', () => {
  const z = normalField(256 * 256, G.seed);
  G.noise_head.forEach((v: number, i: number) => assert.ok(Math.abs(z[i] - v) < 1e-6, `noise[${i}] ${z[i]} vs ${v}`));
  let s = 0, s2 = 0;
  for (const v of z) { s += v; s2 += v * v; }
  const mean = s / z.length;
  assert.ok(Math.abs(Math.sqrt(s2 / z.length - mean * mean) - G.noise_std) < 1e-6);
});

test('the pipeline reproduces the NumPy PSNRs and gains on camera-256', () => {
  for (const c of G.cases) {
    const ev = evaluate(p, c.sigma, c.noise, kOf(c.logk));
    assert.ok(Math.abs(ev.psnrDegraded - c.psnr_degraded) < 0.005, `degraded ${JSON.stringify(c)} -> ${ev.psnrDegraded}`);
    // A noise-free exact inverse sits at machine precision (float32 rounding decides the digits), so only "above 100 dB" is compared.
    if (c.psnr_restored > 100) assert.ok(ev.psnrRestored > 100, `restored ${JSON.stringify(c)} -> ${ev.psnrRestored}`);
    else assert.ok(Math.abs(ev.psnrRestored - c.psnr_restored) < 0.01, `restored ${JSON.stringify(c)} -> ${ev.psnrRestored}`);
    assert.ok(Math.abs(ev.gain / c.gain - 1) < 1e-6, `gain ${JSON.stringify(c)} -> ${ev.gain}`);
  }
});

test('the PSNR curves and their best K match NumPy', () => {
  for (const key of ['default', 'noise0', 'noise02', 'sigma1', 'sigma3']) {
    const ref = G[`curve_${key}`], curve = psnrCurve(p, ref.sigma, ref.noise), best = bestOnGrid(curve);
    curve.forEach((v, i) => assert.ok(Math.abs(v - ref.psnr[i]) < 0.01, `${key}[${i}] ${v} vs ${ref.psnr[i]}`));
    assert.equal(best.logk, ref.best_logk, key);
  }
});

test('facts quoted in the predict explanation and the challenge hold for the default state', () => {
  const sigma = Number(lab.controls[0].value), noise = Number(lab.controls[1].value), logk = Number(lab.controls[2].value);
  assert.deepEqual([sigma, noise, logk], [2, 0.005, LOGK_OFF]);
  const def = evaluate(p, sigma, noise, kOf(logk));
  assert.equal(r2(def.psnrDegraded), 23.95);
  assert.equal(r2(def.psnrRestored), 4.79);
  assert.ok(def.psnrRestored < def.psnrDegraded - 15, 'the plain inverse must be far worse than the blurred image');
  assert.equal(r2(evaluate(p, 2, 0, 0).psnrDegraded), 23.98);
  assert.equal(r2(evaluate(p, 2, 0, 0).psnrRestored), 34.94);
  assert.equal(Math.round(evaluate(p, 2, 0.00001, 0).psnrRestored * 10) / 10, 6.1);
  // The answer to the predict question and the "win" the challenge asks for.
  assert.equal(lab.predict.answer, 'noisy');
  const curve = psnrCurve(p, sigma, noise), best = bestOnGrid(curve);
  assert.equal(best.logk, -2.75);
  assert.equal(r2(best.psnr), 26.8);
  assert.ok(best.psnr > def.psnrDegraded + 2.5, 'the best K must beat the blurred image by a clear margin');
  // The curve rises from K = 0, peaks once, and falls again.
  const i = best.index;
  for (let k = 1; k <= i; k++) assert.ok(curve[k] > curve[k - 1], `rising at ${k}`);
  for (let k = i + 1; k < curve.length; k++) assert.ok(curve[k] < curve[k - 1], `falling at ${k}`);
  // Noise 0.02 moves the best K up, and noise 0 makes K = 0 the best.
  assert.equal(bestOnGrid(psnrCurve(p, 2, 0.02)).logk, -1.75);
  assert.equal(bestOnGrid(psnrCurve(p, 2, 0)).logk, LOGK_OFF);
});

test('gain properties: at most 1/(2√K) for K > 0, at most 1/EPS for K = 0, and K = 0 divides exactly where |H| ≥ EPS', () => {
  const H = gaussianOtf(256, 256, 2);
  assert.equal(H[0], 1);
  assert.ok(Math.abs(H[1] - Math.exp(-2 * Math.PI ** 2 * 4 / 65536)) < 1e-12, 'H(1,0)');
  for (const lk of [-6, -4, -2.75, -1, 0]) {
    const K = 10 ** lk, g = peakGain(restorationGain(H, K));
    assert.ok(g <= 1 / (2 * Math.sqrt(K)) * (1 + 1e-12), `gain ${g} at logK ${lk}`);
  }
  assert.ok(Math.abs(peakGain(restorationGain(H, 10 ** -6)) / 500 - 1) < 1e-6, 'a small K caps the gain at 1/(2√K)');
  const W = restorationGain(H, 0);
  assert.ok(peakGain(W) <= 1 / EPS * (1 + 1e-12) && peakGain(W) > 0.99 / EPS);
  for (let i = 0; i < H.length; i += 997) assert.equal(W[i], H[i] < EPS ? 0 : 1 / H[i]);
});

test('without noise a mild blur is undone almost exactly; the blur itself is a circular Gaussian convolution', () => {
  assert.ok(evaluate(p, 1, 0, 0).psnrRestored > 100);
  // Direct circular convolution with a sampled Gaussian (σ = 2, radius 12) agrees with the transfer-function blur.
  const sigma = 2, r = 12, k = Float64Array.from({ length: 2 * r + 1 }, (_, i) => Math.exp(-((i - r) ** 2) / (2 * sigma * sigma)));
  const ks = k.reduce((a, b) => a + b, 0);
  const n = 256, tmp = new Float64Array(n * n), out = new Float64Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { let s = 0; for (let i = -r; i <= r; i++) s += k[i + r] * cam.data[y * n + ((x + i + n) % n)]; tmp[y * n + x] = s / ks; }
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { let s = 0; for (let j = -r; j <= r; j++) s += k[j + r] * tmp[((y + j + n) % n) * n + x]; out[y * n + x] = s / ks; }
  const ev = evaluate(p, 2, 0, 0);
  let worst = 0;
  for (let i = 0; i < out.length; i++) worst = Math.max(worst, Math.abs(out[i] - Math.min(1, Math.max(0, ev.degraded[i]))));
  assert.ok(worst < 2e-3, `max difference ${worst}`);
});

test('the noise slider only scales one fixed field: the noisy part of the degraded image is linear in the noise level', () => {
  const a = evaluate(p, 2, 0.01, 1e-9), b = evaluate(p, 2, 0, 1e-9), c = evaluate(p, 2, 0.02, 1e-9);
  let linear = 0, n = 0;
  for (let i = 0; i < a.degraded.length; i++) {
    const v0 = b.degraded[i];
    if (v0 > 0.1 && v0 < 0.9) { linear = Math.max(linear, Math.abs((c.degraded[i] - v0) - 2 * (a.degraded[i] - v0))); n++; }
  }
  assert.ok(n > 1000 && linear < 1e-5, `max deviation ${linear} over ${n} pixels`);
});

test('the lab renders English and Chinese text for every K on the slider without throwing', () => {
  const canvas = () => ({ width: 0, height: 0, getContext: () => ({ createImageData: (w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4) }), putImageData() {} }) }) as unknown as HTMLCanvasElement;
  for (const l of ['en', 'zh-tw'] as const) {
    const inst = lab.create({ lang: l, images: { 'camera-256': cam } });
    for (const lk of [LOGK_OFF, -3, 0]) {
      const out = inst.render({ sigma: 2, noise: 0.005, logk: lk }, { original: canvas(), degraded: canvas(), restored: canvas() });
      assert.equal(out.metrics.length, 3);
      assert.ok(out.plots!.curve.includes('<svg') && !/NaN|undefined/.test(out.plots!.curve + out.summary), `${l} ${lk}`);
    }
  }
  assert.equal(LOGK_GRID.length, 29);
});

// ---------------------------------------------------------------------------------------------------------------
// IP 09 lab (morphology-shapes): kit goldens against SciPy / scikit-image, and the lab's facts on the real mask asset.
import { structuringElement, elementGrid, erode, dilate, open as openMask, close as closeMask, gradient, applyOperation, type ElementShape, type Operation } from '../../src/lib/imagekit/morphology.ts';
import { labelComponents, countHoles } from '../../src/lib/imagekit/label.ts';
import { gaussianBlur } from '../../src/lib/imagekit/filter.ts';
import { otsuThreshold } from '../../src/lib/imagekit/threshold.ts';
import { lab as morph, inputStats, analyse, maskFromGray, SMALL } from '../../src/lib/playground/labs/morphology-shapes.ts';

const M = JSON.parse(readFileSync(new URL('./golden/morphology.json', import.meta.url), 'utf8'));
const fromBits = (s: string) => Uint8Array.from(s, c => (c === '1' ? 1 : 0));
const SHAPES: ElementShape[] = ['disk', 'square', 'hline', 'vline'], OPS: Operation[] = ['erosion', 'dilation', 'opening', 'closing', 'gradient'];

test('structuring elements equal skimage.morphology.disk and rectangles for sizes 1 to 12', () => {
  for (const shape of SHAPES) for (let r = 1; r <= 12; r++) {
    // elementGrid is always (2r+1) x (2r+1); skimage's line footprints are 1 x (2r+1) or (2r+1) x 1, i.e. the middle row or column.
    const n = 2 * r + 1, grid = elementGrid(structuringElement(shape, r));
    const got = shape === 'hline' ? grid.slice(r * n, (r + 1) * n) : shape === 'vline' ? Uint8Array.from({ length: n }, (_, j) => grid[j * n + r]) : grid;
    if (shape === 'hline' || shape === 'vline') assert.equal(grid.reduce((a, b) => a + b, 0), n, `${shape} ${r} has only the middle line`);
    assert.deepEqual(Array.from(got), Array.from(fromBits(M.elements[`${shape}|${r}`])), `${shape} ${r}`);
  }
});

test('erosion, dilation, opening, closing and gradient equal scipy.ndimage on random masks (all shapes, sizes 1, 2, 3, 5)', () => {
  let checked = 0;
  for (const img of M.images) {
    const mask = fromBits(img.mask);
    for (const shape of SHAPES) for (const r of [1, 2, 3, 5]) for (const op of OPS) {
      const got = applyOperation(op, mask, img.w, img.h, structuringElement(shape, r)), want = fromBits(img.ops[`${op}|${shape}|${r}`]);
      assert.deepEqual(Array.from(got), Array.from(want), `${op} ${shape} ${r}`);
      checked++;
    }
  }
  assert.equal(checked, 3 * 4 * 4 * 5);
});

test('labeling reproduces skimage.measure.label exactly for 4- and 8-connectivity, and holes match', () => {
  for (const ref of M.labels) {
    const img = M.images[ref.image], got = labelComponents(fromBits(img.mask), img.w, img.h, ref.connectivity);
    assert.equal(got.count, ref.count);
    assert.deepEqual(Array.from(got.labels), ref.labels, `image ${ref.image} connectivity ${ref.connectivity}`);
    assert.equal(got.areas.reduce((a, b) => a + b, 0), ref.labels.filter((v: number) => v > 0).length);
  }
  for (const img of M.images) assert.equal(countHoles(fromBits(img.mask), img.w, img.h, 8), img.holes8);
});

test('algebra on the real mask: erosion shrinks, dilation grows, opening is anti-extensive and idempotent, closing the reverse', () => {
  const src = readGray(asset('coins-mask.png')), mask = maskFromGray(src), { w, h } = src, se = structuringElement('disk', 4);
  const e = erode(mask, w, h, se), d = dilate(mask, w, h, se), o = openMask(mask, w, h, se), c = closeMask(mask, w, h, se);
  for (let i = 0; i < mask.length; i++) { assert.ok(e[i] <= mask[i] && mask[i] <= d[i] && o[i] <= mask[i] && mask[i] <= c[i]); }
  assert.deepEqual(Array.from(openMask(o, w, h, se)), Array.from(o));
  assert.deepEqual(Array.from(closeMask(c, w, h, se)), Array.from(c));
  const g = gradient(mask, w, h, se);
  for (let i = 0; i < mask.length; i++) assert.equal(g[i], d[i] && !e[i] ? 1 : 0);
});

test('the lab pipeline reproduces the SciPy facts on coins-mask.png for every reference case', () => {
  const src = readGray(asset('coins-mask.png')), mask = maskFromGray(src), { w, h } = src, st = inputStats(mask, w, h), F = M.lab;
  assert.deepEqual([src.w, src.h], [F.w, F.h]);
  assert.equal(st.components, F.comps8);
  assert.equal(st.holes, F.holes_in);
  assert.equal(st.small.length, F.small_areas.length);
  assert.equal(labelComponents(mask, w, h, 4).count, F.comps4);
  for (const [key, ref] of Object.entries<any>(F.cases)) {
    const [op, shape, r] = key.split('|'), a = analyse(mask, w, h, st, op as Operation, shape as ElementShape, Number(r));
    assert.equal(st.area, ref.area_in, key);
    assert.equal(a.areaOut, ref.area_out, key);
    assert.equal(a.componentsOut, ref.comps_out, key);
    assert.equal(a.smallGone, ref.small_gone, key);
    assert.equal(a.holesOut, ref.holes_out, key);
    assert.ok(Math.abs(a.areaChangePct - ref.area_change_pct) < 1e-9, key);
  }
});

test('facts quoted in the morphology lab: defaults, predict explanation, note and challenge', () => {
  const src = readGray(asset('coins-mask.png')), mask = maskFromGray(src), { w, h } = src, st = inputStats(mask, w, h), F = M.lab;
  assert.deepEqual(morph.controls.map(c => String(c.value)), ['opening', 'disk', '4']);
  // The note: 24 coins, 40 specks, 10 scratches, 18 holes; clutter is every piece under SMALL pixels; the smallest coin is above a thousand.
  assert.equal(SMALL, 200);
  const lab8 = labelComponents(mask, w, h, 8), coins = lab8.areas.filter(a => a >= SMALL), small = lab8.areas.filter(a => a < SMALL);
  assert.equal(coins.length, 24); assert.equal(small.length, 50); assert.equal(st.holes, 18);
  assert.ok(Math.min(...coins) > 1000 && Math.max(...small) < 100, `smallest coin ${Math.min(...coins)}, largest clutter ${Math.max(...small)}`);
  assert.equal(F.clean.holes_added + 0, 18); assert.equal(F.clean.specks_added, 40); assert.equal(F.clean.scratches_added, 10); assert.equal(F.clean.components, 24); assert.equal(F.clean.holes, 0);
  // The predict explanation: opening with a disk of radius 4.
  const a = analyse(mask, w, h, st, 'opening', 'disk', 4);
  assert.equal(st.components, 74); assert.equal(a.componentsOut, 24); assert.equal(a.smallGone, 50); assert.equal(st.small.length, 50); assert.equal(a.holesOut, 18);
  const coinPixels = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i++) coinPixels[i] = lab8.areas[st.labels[i] - 1] >= SMALL && st.labels[i] ? 1 : 0;
  let before = 0, after = 0;
  for (let i = 0; i < mask.length; i++) if (coinPixels[i]) { before++; after += a.output[i]; }
  assert.equal((100 * (after - before) / before).toFixed(2), '-0.10');
  assert.equal(morph.predict.answer, 'clean');
  // Challenge: a horizontal line of size 4 keeps the 5 horizontal scratches, so 45 of the 50 small pieces go and 29 components remain.
  const hl = analyse(mask, w, h, st, 'opening', 'hline', 4);
  assert.equal(hl.smallGone, 45); assert.equal(hl.componentsOut, 29);
  // Challenge: disk radius 9 still keeps all 24 coins, radius 10 loses the first one; closing fills all 18 holes at size 4 but not at size 3.
  assert.equal(analyse(mask, w, h, st, 'opening', 'disk', 9).componentsOut, 24);
  assert.equal(analyse(mask, w, h, st, 'opening', 'disk', 10).componentsOut, 23);
  assert.equal(analyse(mask, w, h, st, 'closing', 'disk', 4).holesOut, 0);
  assert.ok(analyse(mask, w, h, st, 'closing', 'disk', 3).holesOut > 0);
});

test('the clean mask in the note is what the kit gives: smooth (σ 2), subtract a σ 50 background, Otsu', () => {
  const coinsImg = readGray(asset('coins.png')), mask = readGray(asset('coins-mask.png')), { w, h } = coinsImg;
  const a = gaussianBlur(coinsImg.data, w, h, 2), b = gaussianBlur(coinsImg.data, w, h, 50), d = a.map((v, i) => v - b[i]), t = otsuThreshold(d);
  let differing = 0;
  for (let i = 0; i < d.length; i++) if ((d[i] > t ? 1 : 0) !== (mask.data[i] > 0.5 ? 1 : 0)) differing++;
  // The asset differs from the clean mask exactly at the added clutter (the count comes from the Python build).
  assert.equal(differing, M.lab.clean.pixels_differing_from_input, `${differing} pixels differ from the clean mask; Python says ${M.lab.clean.pixels_differing_from_input}`);
});

test('the morphology lab renders every operation and shape in both languages without NaN', () => {
  const ctx = new Proxy({ createImageData: (w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4) }) } as Record<string, unknown>, { get: (t, k: string) => (k in t ? t[k] : () => {}), set: () => true });
  const canvas = () => ({ width: 0, height: 0, getContext: () => ctx }) as unknown as HTMLCanvasElement;
  const src = readGray(asset('coins-mask.png'));
  for (const l of ['en', 'zh-tw'] as const) {
    const inst = morph.create({ lang: l, images: { 'coins-mask': src } });
    for (const op of OPS) for (const shape of SHAPES) for (const size of [1, 4, 20]) {
      const out = inst.render({ op, shape, size }, { input: canvas(), output: canvas(), element: canvas() });
      assert.equal(out.metrics.length, 4);
      assert.ok(!/NaN|undefined/.test(out.summary + out.metrics.map(m => m.value).join()), `${l} ${op} ${shape} ${size}`);
    }
  }
});
