import assert from "node:assert/strict";
import { test } from "node:test";
import {
  radialSnake,
  classificationThreshold,
  ambiguousRestoration,
  trackingAssignment,
  worldRollout,
  chapterVisionExperiments,
} from "../../src/lib/playground/chapters-vision.ts";
const near = (a: number, b: number, tol = 1e-9) =>
  assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`);
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
test("All five bilingual experiments render finite SVGs at defaults and individual endpoint settings", () => {
  assert.equal(chapterVisionExperiments.length, 5);
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
