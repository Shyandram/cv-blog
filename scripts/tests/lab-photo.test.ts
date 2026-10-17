// Tests for the labs array-dtype, pixel-sampling and color-clipping.
// Every number quoted in their text is asserted here against scripts/tests/golden/photo.json, which
// scripts/figures/playground_goldens_photo.py computes with NumPy, SciPy, OpenCV and scikit-image on the exact PNG assets.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { readGray, readRgb, asset } from './helpers/png.ts';
import { psnr } from '../../src/lib/imagekit/metrics.ts';
import { rgbToHsv, hueDistance } from '../../src/lib/imagekit/color.ts';
import * as dtype from '../../src/lib/playground/labs/array-dtype.ts';
import * as sampling from '../../src/lib/playground/labs/pixel-sampling.ts';
import * as clipping from '../../src/lib/playground/labs/color-clipping.ts';

const golden = JSON.parse(readFileSync(new URL('./golden/photo.json', import.meta.url), 'utf8'));
const camera = readGray(asset('camera-256.png')), zone = readGray(asset('zoneplate-256.png')), coffee = readRgb(asset('coffee-256.png'));
const near = (a: number, b: number, tol: number, msg = '') => assert.ok(Math.abs(a - b) <= tol, `${msg} ${a} vs ${b}`);
const f1 = (x: number) => (Math.round(x * 10) / 10).toString();

/** A canvas stand-in that records nothing: enough for render() to run in Node. */
function fakeCanvases(keys: string[]) {
  return Object.fromEntries(keys.map(k => [k, {
    width: 0, height: 0, style: {} as Record<string, string>,
    getContext: () => ({ createImageData: (w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4) }), putImageData() {} }),
  } as unknown as HTMLCanvasElement]));
}
function defaults(lab: { controls: { key: string; value: string | number }[] }) { return Object.fromEntries(lab.controls.map(c => [c.key, c.value])); }
/** Render a lab in every state of its controls (range ends, every select option) and both languages; no NaN or undefined may leak into text or SVG. */
function exercise(lab: typeof dtype.lab, images: Record<string, any>) {
  let variants: Record<string, string | number>[] = [defaults(lab)];
  for (const c of lab.controls) {
    const values = c.kind === 'range' ? [c.min!, c.max!, c.value] : c.options!.map(o => o.value);
    variants = variants.flatMap(v => values.map(x => ({ ...v, [c.key]: x })));
  }
  for (const lang of ['en', 'zh-tw'] as const) {
    const inst = lab.create({ lang, images }), canvases = fakeCanvases(lab.panes.filter(p => p.kind === 'canvas').map(p => p.key));
    for (const v of variants) {
      const out = inst.render(v, canvases), blob = JSON.stringify([out.summary, out.metrics, out.plots]);
      assert.ok(out.summary.length > 20 && out.metrics.length >= 3, `${lab.id} ${lang} ${JSON.stringify(v)}`);
      assert.ok(!/NaN|undefined|Infinity/.test(blob), `${lab.id} ${lang} ${JSON.stringify(v)}: ${blob.match(/.{30}(NaN|undefined|Infinity).{30}/)?.[0]}`);
    }
  }
  return variants.length;
}

// ------------------------------------------------------------------ array-dtype
test('array-dtype: wraparound is exactly NumPy uint8 arithmetic on the real photo', () => {
  const src = dtype.toBytes(camera), g = golden.dtype;
  assert.equal(src.length, g.n);
  const wrapped = dtype.brighten(src, 60, 'wrap'), sat = dtype.brighten(src, 60, 'saturate'), flt = dtype.brighten(src, 60, 'float'), ref = dtype.correct(src, 60);
  const st = dtype.stats(src, wrapped, 60, 'wrap');
  near(st.overflow * g.n, g.ge196, 0, 'pixels at 196 or more');
  near(st.darker * g.n, g.wrapped_darker, 0, 'pixels that came out darker');
  near(st.meanIn, g.mean_in, 1e-9); near(st.meanOut, g.mean_wrap, 1e-9); near(dtype.stats(src, ref, 60, 'saturate').meanOut, g.mean_clip, 1e-9);
  near(st.psnr, g.psnr_wrap_vs_clip, 1e-9, 'PSNR of the wrapped result');
  // Every wrapped pixel moves by exactly -196, every other pixel by +60.
  const deltas = new Set<number>(), others = new Set<number>();
  src.forEach((v, i) => (v >= 196 ? deltas : others).add(wrapped[i] - v));
  assert.deepEqual([...deltas], g.wrapped_delta_values); assert.deepEqual([...others], g.other_delta_values);
  // Saturating and floating-point arithmetic give the same bytes as the correct result for whole-number amounts (OpenCV cv2.add as well).
  assert.ok(g.cv2_add_equals_clip && g.float_clip_equals_clip);
  assert.deepEqual(Array.from(sat), Array.from(ref)); assert.deepEqual(Array.from(flt), Array.from(ref));
  assert.equal(dtype.stats(src, sat, 60, 'saturate').psnr, Infinity);
  assert.equal(dtype.stats(src, sat, 60, 'saturate').wrapped, 0);
  near(dtype.stats(src, sat, 60, 'saturate').clipped * g.n, g.ge196, 0);
});
test('array-dtype: negative amounts wrap the dark end up to bright', () => {
  const src = dtype.toBytes(camera), g = golden.dtype;
  near(dtype.stats(src, dtype.brighten(src, -100, 'wrap'), -100, 'wrap').overflow * g.n, g.lt100, 0);
  near(dtype.stats(src, dtype.brighten(src, -100, 'wrap'), -100, 'wrap').meanOut, g.mean_wrap_m100, 1e-9);
  near(dtype.stats(src, dtype.brighten(src, -100, 'saturate'), -100, 'saturate').meanOut, g.mean_clip_m100, 1e-9);
});
test('array-dtype: the numbers quoted in the predict explanation hold', () => {
  const g = golden.dtype, en = dtype.lab.predict.explain.en, zh = dtype.lab.predict.explain['zh-tw'];
  const frac = `${f1(100 * g.ge196 / g.n)}%`, full = `${f1(100 * g.full_ge196 / g.full_n)}%`;
  for (const t of [en, zh]) {
    for (const s of [frac, full, '17,855', '65,536', f1(g.mean_in), f1(g.mean_wrap), f1(g.mean_clip), '196', '60']) assert.ok(t.includes(s), `explain should quote ${s}`);
  }
  assert.equal(frac, '27.2%'); assert.equal(full, '27.5%'); assert.equal(g.ge196, 17855); assert.equal(g.full_ge196, 72087); assert.equal(g.full_n, 262144);
  assert.equal(256 - 60, 196); assert.equal(60 - 256, -196);
  // The 72,087 in the Images as Arrays chapter is the full-size count; the lab note says the fractions differ slightly.
  assert.ok(dtype.lab.note.en.includes('differ slightly'));
});
test('array-dtype: render runs in every state in both languages', () => {
  assert.equal(exercise(dtype.lab, { 'camera-256': camera }), 18);
});

// ------------------------------------------------------------------ pixel-sampling
test('pixel-sampling: the TypeScript pipeline reproduces the NumPy/SciPy results on both assets', () => {
  for (const c of golden.sampling.cases) {
    const a = sampling.samplePhoto(camera, c.stride, c.bits, !!c.pre), b = sampling.samplePhoto(zone, c.stride, c.bits, !!c.pre);
    const pa = psnr(camera.data, a.up), pb = psnr(zone.data, b.up), tag = JSON.stringify([c.stride, c.bits, c.pre]);
    if (c.psnr_cam === null) assert.equal(pa, Infinity, tag); else near(pa, c.psnr_cam, 0.02, `photo ${tag}`);
    if (c.psnr_zp === null) assert.equal(pb, Infinity, tag); else near(pb, c.psnr_zp, 0.02, `zone ${tag}`);
    const fp = sampling.falsePatternContrast(b, c.stride);
    if (c.false === null) assert.ok(Number.isNaN(fp), tag); else near(fp, c.false, 0.004, `false ${tag}`);
  }
});
test('pixel-sampling: kept samples, gray levels and the alias radius', () => {
  const s = sampling.samplePhoto(camera, 4, 8, false);
  assert.equal(s.sw * s.sh, golden.sampling.kept_stride4); assert.equal(s.sw, 64);
  near(sampling.aliasRadius(4), golden.sampling.alias_radius['4'], 1e-9);
  for (const k of [1, 2, 3, 8]) near(sampling.aliasRadius(k), golden.sampling.alias_radius[String(k)], 1e-9);
  // The chapter's quantizer: 4 bits leaves 16 values; bin centres.
  assert.equal(new Set(sampling.samplePhoto(camera, 1, 4, false).small).size, 16);
  assert.equal(sampling.quantizeByte(0, 1), 64); assert.equal(sampling.quantizeByte(255, 1), 192);
  assert.equal(sampling.quantizeByte(77, 8), 77);
  // Stride 1 at 8 bits and no prefilter reproduces the photo exactly.
  assert.equal(psnr(camera.data, sampling.samplePhoto(camera, 1, 8, false).up), Infinity);
  // Nearest-neighbour enlargement: every s x s block holds one value.
  const up = sampling.samplePhoto(camera, 4, 8, false).up;
  assert.equal(up[0], up[1]); assert.equal(up[0], up[3 * 256 + 3]); assert.equal(up[0], camera.data[0]);
});
test('pixel-sampling: the numbers quoted in the predict explanation hold', () => {
  const at = (stride: number, bits: number, pre: number) => golden.sampling.cases.find((c: any) => c.stride === stride && c.bits === bits && c.pre === pre);
  const off = at(4, 8, 0), on = at(4, 8, 1), en = sampling.lab.predict.explain.en, zh = sampling.lab.predict.explain['zh-tw'];
  const quoted = [f1(off.psnr_cam), f1(on.psnr_cam), f1(off.psnr_zp), f1(on.psnr_zp), (Math.round(off.false * 100) / 100).toFixed(2), (Math.round(on.false * 100) / 100).toFixed(2), String(Math.round(golden.sampling.alias_radius['4']))];
  assert.deepEqual(quoted, ['19.6', '21.6', '6.4', '9.2', '0.35', '0.02', '45']);
  for (const t of [en, zh]) for (const q of quoted) assert.ok(t.includes(q), `explain should quote ${q}`);
  // The answer: with the prefilter PSNR rises on both images, and the false rings nearly vanish (ratio below 10 %).
  assert.ok(on.psnr_cam > off.psnr_cam && on.psnr_zp > off.psnr_zp && on.false < 0.1 * off.false);
  // The honest-limits sentence: in every combination of stride 2..8 and bits 2..8 the prefilter helps; only the 1-bit zone plate gets worse.
  assert.ok(golden.sampling.blur_lowers_psnr.every((x: any[]) => x[1] === 1 && x[2] === 'zone'));
  let worse: string[] = [];
  for (let s = 2; s <= 8; s++) for (let b = 1; b <= 8; b++) for (const [name, img] of [['camera', camera], ['zone', zone]] as const)
    if (psnr(img.data, sampling.samplePhoto(img, s, b, true).up) < psnr(img.data, sampling.samplePhoto(img, s, b, false).up)) worse.push(`${s},${b},${name}`);
  assert.ok(worse.every(w => w.split(',')[1] === '1' && w.endsWith('zone')), `the prefilter should only hurt the 1-bit zone plate: ${worse}`);
  assert.ok(worse.length > 0 && sampling.lab.predict.explain.en.includes('only at 1 bit'));
});
test('pixel-sampling: the challenge claim, 8 gray levels at every pixel beat 256 levels at every 4th (and 2nd) pixel in PSNR', () => {
  const p = (stride: number, bits: number) => psnr(camera.data, sampling.samplePhoto(camera, stride, bits, false).up);
  assert.ok(p(1, 3) > p(4, 8) && p(1, 3) > p(2, 8), `${p(1, 3)} vs ${p(4, 8)} and ${p(2, 8)}`);
});
test('pixel-sampling: render runs in every state in both languages', () => {
  assert.ok(exercise(sampling.lab, { 'camera-256': camera, 'zoneplate-256': zone }) > 10);
});

// ------------------------------------------------------------------ color-clipping
test('color-clipping: HSV matches scikit-image rgb2hsv', () => {
  for (const { rgb, hsv } of golden.color.hsv) {
    const [h, s, v] = rgbToHsv(rgb[0], rgb[1], rgb[2]);
    near(s, hsv[1], 1e-9, 'saturation'); near(v, hsv[2], 1e-9, 'value');
    if (hsv[1] > 0) near(hueDistance(h, hsv[0]), 0, 1e-6, `hue of ${rgb}`);
  }
  assert.equal(hueDistance(350, 10), 20); assert.equal(hueDistance(0, 180), 180);
});
test('color-clipping: the pipeline reproduces the NumPy results on the coffee photo', () => {
  assert.deepEqual([coffee.h, coffee.w], golden.color.shape.slice(0, 2));
  for (const c of golden.color.cases) {
    const { out, clipped } = clipping.applyGain(coffee.data, c.gain, c.method), rep = clipping.hueReport(coffee.data, out, c.gain), tag = `${c.method} ${c.gain}`;
    near(rep.overflowFraction, c.overflow, 1e-6, `overflow ${tag}`); assert.equal(rep.coloured, c.coloured, `coloured ${tag}`);
    near(clipped, c.clipped_after, 1e-6, `clipped after ${tag}`);
    if (c.mean_dh === null) assert.ok(Number.isNaN(rep.meanHueChange), tag); else near(rep.meanHueChange, c.mean_dh, 0.01, `hue ${tag}`);
    if (c.lost === null) assert.ok(Number.isNaN(rep.lostFraction), tag); else near(rep.lostFraction, c.lost, 1e-4, `lost ${tag}`);
    if (c.sat_before !== null) { near(rep.satBefore, c.sat_before, 1e-4, `sat before ${tag}`); near(rep.satAfter, c.sat_after, 1e-4, `sat after ${tag}`); }
  }
});
test('color-clipping: rescaling keeps hue exactly; clipping does not; gain 1 changes nothing', () => {
  const run = (g: number, m: clipping.Overflow) => { const { out } = clipping.applyGain(coffee.data, g, m); return clipping.hueReport(coffee.data, out, g); };
  assert.ok(run(2, 'scale').meanHueChange < 1e-3 && run(2, 'scale').lostFraction === 0);
  assert.ok(run(2, 'clip').meanHueChange > 10);
  assert.ok(run(1, 'clip').overflowFraction === 0);
  for (const m of ['clip', 'scale', 'tone'] as const) assert.deepEqual(Array.from(clipping.applyGain(coffee.data, 1, m).out), Array.from(coffee.data), `${m} at gain 1 is the identity`);
  // At gain 3 the clipped hues pile up in the 60-70 degree bin (pure yellow), as the challenge says.
  const rep = run(3, 'clip');
  assert.equal(rep.hueAfter.indexOf(Math.max(...rep.hueAfter)), 6);
  assert.ok(rep.hueBefore.slice(0, 4).reduce((a, b) => a + b, 0) > 0.99 * rep.hueBefore.reduce((a, b) => a + b, 0), 'the overflowing colours start in the red to orange range');
});
test('color-clipping: the orange pixel of the predict question', () => {
  const o = golden.color.orange, px = Float32Array.of(200 / 255, 100 / 255, 40 / 255);
  const clip = clipping.applyGain(px, 2, 'clip').out, scale = clipping.applyGain(px, 2, 'scale').out;
  assert.deepEqual(Array.from(clip, v => Math.round(v * 255)), [255, 200, 80]);
  const [h0, s0] = rgbToHsv(px[0], px[1], px[2]), [h1, s1] = rgbToHsv(clip[0], clip[1], clip[2]), [h2] = rgbToHsv(scale[0], scale[1], scale[2]);
  near(h0, o.clip.hue_before, 1e-4); near(h1, o.clip.hue_after, 1e-4); near(s0, o.clip.sat_before, 1e-6); near(s1, o.clip.sat_after, 1e-6); near(h2, 22.5, 1e-3);
  near(scale[1] * 255, 127.5, 1e-3); near(scale[2] * 255, 51, 1e-3);
  assert.equal(f1(h0), '22.5'); assert.equal(f1(h1), '41.1'); assert.equal(s0.toFixed(2), '0.80'); assert.equal(s1.toFixed(2), '0.69'); assert.equal((255 / 200).toFixed(1), '1.3');
  for (const t of [clipping.lab.predict.explain.en, clipping.lab.predict.explain['zh-tw']]) for (const q of ['(400, 200, 80)', '(255, 200, 80)', '22.5°', '41.1°', '0.80', '0.69', '1.3', '127.5', '51', '2 : 1']) assert.ok(t.includes(q), `explain should quote ${q}`);
  assert.ok(h1 > h0 && h1 < 60, 'hue moves toward yellow (60 degrees)');
});
test('color-clipping: render runs in every state in both languages', () => {
  assert.ok(exercise(clipping.lab, { 'coffee-256': coffee }) > 10);
});
