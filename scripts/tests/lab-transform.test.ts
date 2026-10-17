// Facts for the `transform-coding` lab: the lab's own TypeScript pipeline on the real camera-256.png, compared with
// scripts/tests/golden/transform-coding.json (NumPy/SciPy/PyWavelets, scripts/figures/playground_goldens_transform.py).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { readGray, asset } from './helpers/png.ts';
import {
  lab, codeImage, busyMask, meanAbsWhere, meanAbsInBox, seamRatio, coefficientMap, errorView, zoomCrop, forward, REGIONS, ZOOM, SCALE, RULER, SEAM_VISIBLE, type TransformKind,
} from '../../src/lib/playground/labs/transform-coding.ts';

const G = JSON.parse(readFileSync(new URL('./golden/transform-coding.json', import.meta.url), 'utf8')) as {
  w: number; h: number; original_seam: number;
  rows: { transform: TransformKind; step: number; nonzero: number; psnr: number; psnr_clipped: number; mae_smooth: number; mae_busy: number; seam: number; coef_sq_error: number; pixel_sq_error: number; box_sky: number; box_edge: number; box_grass: number }[];
};
const image = readGray(asset('camera-256.png'));
const busy = busyMask(image);
const row = (t: TransformKind, step: number) => G.rows.find(r => r.transform === t && Math.abs(r.step - step) < 1e-9)!;
const near = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b}`);

test('the asset is the 256 x 256 camera image the golden file was computed on', () => {
  assert.deepEqual([image.w, image.h, G.w, G.h], [256, 256, 256, 256]);
});

test('TS pipeline matches NumPy/SciPy/PyWavelets for every transform and step', () => {
  for (const r of G.rows) {
    const c = codeImage(r.transform, image, r.step);
    const tag = `${r.transform} step ${r.step}`;
    near(c.nonzero, r.nonzero, 1e-4, `${tag} nonzero`);
    near(c.psnr, r.psnr, 0.01, `${tag} psnr`);
    near(meanAbsWhere(c.error, busy, 0) * 255, r.mae_smooth, 0.02, `${tag} smooth MAE`);
    near(meanAbsWhere(c.error, busy, 1) * 255, r.mae_busy, 0.02, `${tag} busy MAE`);
    near(seamRatio(c.recon, 256, 256), r.seam, 0.01, `${tag} seam ratio`);
    for (const k of ['sky', 'edge', 'grass'] as const) near(meanAbsInBox(c.error, 256, REGIONS[k].x, REGIONS[k].y) * 255, r[`box_${k}` as const], 0.06, `${tag} box ${k}`);
  }
  near(seamRatio(image.data, 256, 256), G.original_seam, 0.01, 'seam ratio of the original');
});

test('both transforms are orthonormal: squared error in coefficients equals squared error in pixels (the ML 05A point)', () => {
  for (const t of ['dct', 'haar'] as const) for (const step of [0.1, 0.6]) {
    const coef = forward(t, image.data, 256, 256), c = codeImage(t, image, step, coef);
    let ce = 0, pe = 0;
    for (let i = 0; i < coef.length; i++) { ce += (coef[i] - c.q[i]) ** 2; pe += c.error[i] ** 2; }
    assert.ok(Math.abs(ce - pe) / pe < 1e-4, `${t} ${step}: ${ce} vs ${pe}`);
  }
});

test('default state: DCT, step 0.2, sky box; numbers quoted in the predict explanation hold', () => {
  const d = Object.fromEntries(lab.controls.map(c => [c.key, c.value]));
  assert.deepEqual(d, { transform: 'dct', step: 0.2, zoom: 'sky' });
  const c = codeImage('dct', image, 0.2), smooth = meanAbsWhere(c.error, busy, 0) * 255, busier = meanAbsWhere(c.error, busy, 1) * 255;
  assert.equal(smooth.toFixed(1), '2.8');
  assert.equal(busier.toFixed(1), '7.5');
  assert.equal((busier / smooth).toFixed(1), '2.7');
  for (const s of ['2.8', '7.5', '2.7']) assert.ok(lab.predict.explain.en.includes(s) && lab.predict.explain['zh-tw'].includes(s), `explain mentions ${s}`);
  assert.equal(lab.predict.answer, 'busy');
});

test('predict answer: error is larger on the busiest half than on the smoothest half at every step, for both transforms', () => {
  for (const r of G.rows) {
    const c = codeImage(r.transform, image, r.step);
    assert.ok(meanAbsWhere(c.error, busy, 1) > 1.5 * meanAbsWhere(c.error, busy, 0), `${r.transform} ${r.step}`);
  }
});

test('challenge: at step 0.6 Haar keeps fewer nonzero coefficients than DCT at about the same PSNR; grass texture is gone', () => {
  const d = codeImage('dct', image, 0.6), h = codeImage('haar', image, 0.6);
  assert.ok(h.nonzero < d.nonzero && Math.abs(h.psnr - d.psnr) < 0.1, `${h.nonzero} < ${d.nonzero}, ${h.psnr} vs ${d.psnr}`);
  // Grass: the error is the texture itself, so the reconstruction box is nearly flat while the original is not.
  const { x, y } = REGIONS.grass, rec: number[] = [], org: number[] = [];
  for (let j = y; j < y + ZOOM; j++) for (let i = x; i < x + ZOOM; i++) { rec.push(d.recon[j * 256 + i]); org.push(image.data[j * 256 + i]); }
  const sd = (a: number[]) => { const m = a.reduce((p, q) => p + q) / a.length; return Math.sqrt(a.reduce((p, q) => p + (q - m) ** 2, 0) / a.length); };
  assert.ok(sd(rec) < 0.1 * sd(org), `reconstruction sd ${sd(rec)} vs original ${sd(org)}`);
});

test('block seams: faint up to step 0.1, clearly visible (ratio at least 2) from step 0.3 for the DCT', () => {
  assert.ok(row('dct', 0.1).seam < 1.5 && row('dct', 0.3).seam >= SEAM_VISIBLE && row('dct', 0.6).seam > 2.5);
  assert.ok(G.original_seam < 1.1);
});

test('helpers: regions lie on the block grid, the zoom strip has the pane size, the maps stay in range', () => {
  for (const r of Object.values(REGIONS)) assert.ok(r.x % 8 === 0 && r.y % 8 === 0 && r.x + ZOOM <= 256 && r.y + ZOOM <= 256);
  const z = zoomCrop(image.data, 256, REGIONS.edge.x, REGIONS.edge.y, true);
  for (const key of ['zoomOriginal', 'zoomRecon']) { const pane = lab.panes.find(p => p.key === key)!; assert.deepEqual([z.w, z.h], [pane.width, pane.height]); }
  assert.deepEqual([z.w, z.h], [ZOOM * SCALE, ZOOM * SCALE + RULER]);
  const c = codeImage('dct', image, 0.2), m = coefficientMap(c.q, 0.2);
  assert.ok(m.every(v => v >= 0 && v <= 1));
  assert.equal(m.filter(v => v > 0).length / m.length, c.nonzero);
  assert.ok(errorView(c.error).every(v => v >= -1 && v <= 1));
});

// Near-ties: a few of the 65,536 coefficients lie within float32 rounding of a half step, so TS and NumPy may round them
// differently; the tolerances above allow for that (at most a handful of coefficients).

// A minimal canvas stand-in so that render() can run in Node.
function fakeCanvases() {
  const mk = () => {
    const c: any = { width: 0, height: 0 };
    c.getContext = () => ({ createImageData: (w: number, h: number) => ({ data: new Uint8ClampedArray(4 * w * h) }), putImageData() {}, strokeRect() {} });
    return c;
  };
  return Object.fromEntries(lab.panes.map(p => [p.key, mk()]));
}
test('render works in both languages for every control combination and reports the golden numbers by default', () => {
  for (const lang of ['en', 'zh-tw'] as const) {
    const inst = lab.create({ lang, images: { 'camera-256': image } });
    for (const transform of ['dct', 'haar']) for (const step of [0.02, 0.2, 0.8]) for (const zoom of ['sky', 'edge', 'grass']) {
      const out = inst.render({ transform, step, zoom }, fakeCanvases());
      assert.ok(out.summary.length > 20 && !/NaN|undefined/.test(out.summary + out.metrics.map(m => m.label + m.value).join()), `${lang} ${transform} ${step} ${zoom}`);
      assert.equal(out.metrics.length, 4);
    }
    const out = inst.render({ transform: 'dct', step: 0.2, zoom: 'sky' }, fakeCanvases()), g = row('dct', 0.2);
    assert.equal(out.metrics[0].value, `${(100 * g.nonzero).toFixed(1)}%`);
    assert.equal(out.metrics[1].value, `${g.psnr.toFixed(1)} dB`);
    assert.equal(out.metrics[2].value, `${g.mae_smooth.toFixed(1)} / ${g.mae_busy.toFixed(1)}`);
    assert.equal(out.metrics[3].value, g.box_sky.toFixed(1));
  }
});
