import assert from 'node:assert/strict';
import test from 'node:test';
// Node's built-in TypeScript runner requires an explicit extension.
import { samplePixels, lowpassSignal, diffusionSignal, intersectionOverUnion, suppressBoxes, visionExperiments } from '../../src/lib/playground/vision.ts';

test('pixel quantization uses the chosen levels and dense 8-bit sampling has bounded error', () => {
  const binary = samplePixels(1, 1);
  assert.ok(binary.output.every((x: number) => x === 0 || x === 1));
  const dense = samplePixels(1, 8);
  assert.ok(dense.error <= (0.5 / 255) ** 2);
  const coarse = samplePixels(4, 8);
  assert.ok(coarse.error > dense.error);
  assert.equal(coarse.output[0], coarse.output[3]);
  assert.equal(coarse.output[0], coarse.output[3 * 24 + 3]);
});

test('ideal low-pass has exact frequency boundaries and Parseval mean-square loss', () => {
  const all = lowpassSignal(12), low = lowpassSignal(11), none = lowpassSignal(1);
  assert.equal(all.error, 0);
  assert.deepEqual(low.retained, [1, 0]);
  assert.ok(Math.abs(low.error - 0.45 ** 2 / 2) < 1e-12);
  assert.ok(Math.abs(none.error - (1 + 0.45 ** 2) / 2) < 1e-12);
  assert.ok(none.filtered.every((v: number) => v === 0));
});

test('all oracle targets invert the forward identity whenever identifiable', () => {
  for (const alpha of [0, 0.01, 0.5, 0.99, 1]) {
    for (const target of ['epsilon', 'x0', 'v']) {
      const r = diffusionSignal(alpha, target);
      if (alpha === 0 && target === 'epsilon') assert.equal(r.error, null);
      else assert.ok(r.error !== null && r.error < 1e-25);
    }
  }
  assert.deepEqual(diffusionSignal(0.1, 'v').noise, diffusionSignal(0.9, 'epsilon').noise);
  const clean = diffusionSignal(1, 'epsilon');
  assert.deepEqual(clean.xt, clean.x0);
});

test('IoU geometry and greedy NMS distinguish score filtering, strict overlap and suppressed boxes', () => {
  const a = { id: 'A', x: 0, y: 0, width: 10, height: 10, score: 0.9 };
  const b = { ...a, id: 'B', x: 5, score: 0.8 };
  const c = { ...a, id: 'C', x: 10, score: 0.7 };
  assert.equal(intersectionOverUnion(a, a), 1);
  assert.equal(intersectionOverUnion(a, c), 0);
  assert.equal(intersectionOverUnion(a, b), 1 / 3);
  assert.deepEqual(suppressBoxes([c, b, a], 0.3, 0).kept.map((box: { id: string }) => box.id), ['A', 'C']);
  assert.deepEqual(suppressBoxes([a, b], 1 / 3, 0).status, { A: 'kept', B: 'kept' });
  assert.equal(suppressBoxes([a, b], 0.3, 0.85).status.B, 'filtered');
  assert.equal(suppressBoxes([a, b], 0.3, 1).kept.length, 0);
});

test('all bilingual experiment defaults and control endpoints produce finite SVGs', () => {
  for (const experiment of visionExperiments) {
    const state = Object.fromEntries(experiment.controls.map((c: { key: string; value: string | number }) => [c.key, c.value]));
    for (const lang of ['en', 'zh-tw'] as const) {
      for (const control of experiment.controls) {
        const values = control.kind === 'range' ? [control.min!, control.max!] : control.options!.map((o: { value: string }) => o.value);
        for (const value of values) {
          const result = experiment.compute({ ...state, [control.key]: value }, lang);
          assert.match(result.svg, /^<svg/);
          assert.doesNotMatch(result.svg, /NaN|Infinity|undefined/);
          assert.ok(result.summary.length > 20);
        }
      }
    }
  }
});
