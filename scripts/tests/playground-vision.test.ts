import assert from 'node:assert/strict';
import test from 'node:test';
// Node's built-in TypeScript runner requires an explicit extension.
import { intersectionOverUnion, suppressBoxes, visionExperiments } from '../../src/lib/playground/vision.ts';

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
