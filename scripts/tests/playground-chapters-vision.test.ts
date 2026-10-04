import assert from "node:assert/strict";
import { test } from "node:test";
import {
  regularizedInverse,
  colorGain,
  haarForward,
  haarInverse,
  dctQuantize,
  binaryMorph,
  thresholdScene,
  radialSnake,
  structureTensor,
  prototypeDistances,
  receptiveField,
  classificationThreshold,
  ambiguousRestoration,
  photonNoise,
  trackingAssignment,
  worldRollout,
  chapterVisionExperiments,
} from "../../src/lib/playground/chapters-vision.ts";
const near = (a: number, b: number, tol = 1e-9) =>
  assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`);
const mse = (a: number[], b: number[]) =>
  a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length;
test("Inverse restoration is exact without perturbations; regularization controls amplified noise", () => {
  for (const blur of [0, 3, 6]) {
    const r = regularizedInverse(blur, 0, 0);
    near(mse(r.clean, r.restored), 0);
  }
  const raw = regularizedInverse(6, 0.03, 0),
    reg = regularizedInverse(6, 0.03, 0.02);
  assert.ok(reg.gain < raw.gain);
  assert.ok(mse(reg.clean, reg.restored) < mse(raw.clean, raw.restored));
});
test("Common linear-light gain preserves color ratios only before clipping", () => {
  near(colorGain([0.7, 0.3, 0.1], 1.1).shift, 0);
  const r = colorGain([0.7, 0.3, 0.1], 2);
  assert.equal(r.count, 1);
  assert.ok(r.shift > 0);
});
test("Haar transform is invertible, energy preserving and keeps the mean under detail removal", () => {
  const a = [0.2, 0.3, 0.7, 0.8, 0.1, 0.9, 0.4, 0.6],
    c = haarForward(a),
    b = haarInverse(c);
  a.forEach((v, i) => near(v, b[i]));
  near(
    a.reduce((s, v) => s + v * v, 0),
    c.reduce((s, v) => s + v * v, 0),
  );
  const smooth = haarInverse(c.map((v, i) => (i ? 0 : v)));
  smooth.forEach((v) => near(v, a.reduce((s, x) => s + x, 0) / 8));
});
test("DCT quantization obeys Parseval and the zero-step roundtrip", () => {
  near(dctQuantize(0).error, 0);
  for (const step of [0.025, 0.15, 1]) {
    const r = dctQuantize(step);
    near(r.error, mse(r.coeff, r.quant));
    assert.ok(
      r.quant.every((v) => Math.abs(v / step - Math.round(v / step)) < 1e-9),
    );
  }
});
test("Morphology: dilation contains input, erosion is contained, opening is idempotent", () => {
  const a = Array.from({ length: 100 }, (_, i) =>
    Number(
      i % 10 >= 3 &&
        i % 10 <= 6 &&
        Math.floor(i / 10) >= 3 &&
        Math.floor(i / 10) <= 6,
    ),
  );
  for (const shape of ["square", "diamond"]) {
    const eroded = binaryMorph(a, 10, 1, shape, "erode"),
      dilated = binaryMorph(a, 10, 1, shape, "dilate"),
      opened = binaryMorph(a, 10, 1, shape, "open");
    a.forEach((v, i) => {
      assert.ok(eroded[i] <= v);
      assert.ok(dilated[i] >= v);
    });
    assert.deepEqual(binaryMorph(opened, 10, 1, shape, "open"), opened);
  }
});
test("Otsu perfectly separates the uniform-lit scene but not severe illumination", () => {
  near(thresholdScene(0, 0.4, true).iou, 1);
  assert.ok(thresholdScene(1, 0.4, true).iou < 1);
  assert.equal(
    thresholdScene(0, 1, false).mask.reduce((a, b) => a + b, 0),
    0,
  );
});
test("Circular contour descent never raises energy and can miss narrow edge attraction", () => {
  for (const start of [0.1, 0.5, 1, 1.8])
    for (const tension of [0, 0.15, 1]) {
      const r = radialSnake(start, tension, 80);
      r.history
        .slice(1)
        .forEach((v, i) =>
          assert.ok(r.energy(v) <= r.energy(r.history[i]) + 1e-10),
        );
      assert.ok(r.radius >= 0 && r.radius <= 1.8);
    }
  const outside = radialSnake(0.5, 0.15, 80),
    nearEdge = radialSnake(1.2, 0.15, 80);
  assert.ok(outside.radius < 0.5);
  assert.ok(nearEdge.radius > 0.9);
});
test("Structure tensor eigenvalues are rotation invariant and contrast scales response quartically", () => {
  for (const kind of ["flat", "edge", "corner"]) {
    const a = structureTensor(kind, 0, 1),
      b = structureTensor(kind, 75, 1),
      c = structureTensor(kind, 75, 2);
    a.eigen.forEach((v, i) => near(v, b.eigen[i]));
    near(c.response, 16 * a.response);
  }
  assert.ok(structureTensor("edge", 0, 1).response < 0);
  assert.ok(structureTensor("corner", 0, 1).response > 0);
});
test("Scaling a feature can change nearest-prototype predictions", () => {
  const a = prototypeDistances(0.6, 0.65, 0.25),
    b = prototypeDistances(0.6, 0.65, 4);
  assert.ok(a[1] < a[0]);
  assert.ok(b[0] < b[1]);
});
test("Receptive-field recurrence accumulates preceding stride correctly", () => {
  assert.equal(receptiveField(3, 1, 1).field, 7);
  assert.equal(receptiveField(3, 2, 1).field, 15);
  assert.equal(receptiveField(3, 2, 1).jump, 8);
  assert.equal(receptiveField(3, 1, 3).field, 19);
});
test("Threshold confusion counts, undefined precision and weighted error cost", () => {
  const all = classificationThreshold(0, 1),
    none = classificationThreshold(1, 1),
    mid = classificationThreshold(0.5, 1);
  assert.deepEqual([all.tp, all.fp, all.fn, all.tn], [5, 5, 0, 0]);
  assert.deepEqual([none.tp, none.fp, none.fn, none.tn], [0, 0, 5, 5]);
  assert.equal(none.precision, null);
  near(none.recall, 0);
  near(none.f1, 0);
  assert.deepEqual([mid.tp, mid.fp, mid.fn, mid.tn], [4, 2, 1, 3]);
  near(mid.precision!, 2 / 3);
  near(mid.recall, 0.8);
  near(mid.f1, 8 / 11);
  near(mid.cost, 0.3);
  const costly = classificationThreshold(0.5, 5);
  assert.deepEqual(costly.prediction, mid.prediction);
  near(costly.cost, 1.1);
  for (const t of [0, 0.47, 0.5, 0.82, 1]) {
    const r = classificationThreshold(t, 1);
    assert.equal(r.tp + r.fp + r.fn + r.tn, 10);
  }
});
test("Conditional mean minimizes expected MSE over the ambiguity slider", () => {
  for (const p of [0.1, 0.5, 0.9]) {
    const base = ambiguousRestoration(p, 0);
    for (const mix of [0.25, 0.5, 1]) {
      assert.ok(ambiguousRestoration(p, mix).risk >= base.risk - 1e-12);
    }
  }
  near(ambiguousRestoration(0.5, 0).risk, 0.25);
  near(ambiguousRestoration(0.5, 1).risk, 0.5);
});
test("Digital gain preserves SNR while more exposure improves it", () => {
  const a = photonNoise(40, 1, 2),
    b = photonNoise(40, 8, 2),
    c = photonNoise(80, 1, 2);
  near(a.snr, b.snr);
  near(b.mean, 8 * a.mean);
  near(b.std, 8 * a.std);
  assert.ok(c.snr > a.snr);
});
test("Two-object matching enumerates globally minimal cost; appearance can prevent swaps", () => {
  assert.deepEqual(trackingAssignment(1, 0.1, 1).assignment, [1, 0]);
  assert.deepEqual(trackingAssignment(0, 0.1, 1).assignment, [0, 1]);
  for (const w of [0, 0.5, 1])
    for (const error of [0, 0.5, 1]) {
      const r = trackingAssignment(w, error, 1),
        cost = r.cost[0][r.assignment[0]] + r.cost[1][r.assignment[1]];
      near(cost, Math.min(r.direct, r.swapped));
    }
});
test("World model exact dynamics remove drift, refresh every step equals teacher forcing", () => {
  const exact = worldRollout(0.96, 60, 0.1, 5);
  near(exact.openError, 0);
  near(exact.oneError, 0);
  const error = worldRollout(1.02, 60, 0.1, 1);
  near(error.correctedError, error.oneError);
  assert.ok(error.openError > error.oneError);
  const off = worldRollout(1.02, 60, 0.1, 0);
  near(off.openError, off.correctedError);
});
test("All fifteen bilingual experiments render finite SVGs at defaults and individual endpoint settings", () => {
  assert.equal(chapterVisionExperiments.length, 15);
  const ids = new Set<string>();
  for (const e of chapterVisionExperiments) {
    assert.ok(!ids.has(e.id));
    ids.add(e.id);
    const defaults = Object.fromEntries(
      e.controls.map((c) => [c.key, c.value]),
    );
    const states = [defaults];
    for (const c of e.controls) {
      if (c.kind === "range")
        for (const value of [c.min!, c.max!])
          states.push({ ...defaults, [c.key]: value });
      if (c.options)
        for (const o of c.options)
          states.push({ ...defaults, [c.key]: o.value });
    }
    for (const state of states)
      for (const lang of ["en", "zh-tw"] as const) {
        const r = e.compute(state, lang);
        assert.match(r.svg, /^<svg/);
        assert.doesNotMatch(r.svg, /NaN|Infinity|undefined/);
        assert.ok(r.summary.length > 20);
        assert.ok(r.metrics.length >= 2);
      }
  }
});
