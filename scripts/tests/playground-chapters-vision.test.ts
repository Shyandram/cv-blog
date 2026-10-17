import assert from "node:assert/strict";
import { test } from "node:test";
import {
  classificationThreshold,
  rocCurve,
  ambiguousRestoration,
  totalVariation,
  worldRollout,
  chapterVisionExperiments,
} from "../../src/lib/playground/chapters-vision.ts";
const near = (a: number, b: number, tol = 1e-9) =>
  assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`);
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
test("ROC and precision-recall curves come from the same scores; the operating point lies on them", () => {
  const { roc, auc, positives, negatives } = rocCurve();
  assert.equal(positives, 5);
  assert.equal(negatives, 5);
  assert.deepEqual([roc[0].fpr, roc[0].tpr], [0, 0]);
  assert.deepEqual([roc.at(-1)!.fpr, roc.at(-1)!.tpr], [1, 1]);
  roc.slice(1).forEach((p, i) => {
    assert.ok(p.fpr >= roc[i].fpr && p.tpr >= roc[i].tpr);
  });
  // AUC equals the fraction of (positive, negative) pairs that the scores order correctly.
  const { scores, truth } = classificationThreshold(0.5, 1);
  let ordered = 0;
  scores.forEach((a, i) =>
    scores.forEach((b, j) => {
      if (truth[i] && !truth[j] && a > b) ordered++;
    }),
  );
  near(auc, ordered / 25);
  near(auc, 0.84);
  // Any slider threshold lands exactly on the vertex of the smallest score that is >= threshold.
  for (const t of [0, 0.1, 0.3, 0.5, 0.67, 0.9, 0.95]) {
    const r = classificationThreshold(t, 1);
    const vertex = roc.find(
      (p) => Math.abs(p.fpr - r.fp / 5) < 1e-12 && Math.abs(p.tpr - r.tp / 5) < 1e-12,
    );
    assert.ok(vertex, `threshold ${t} is not on the curve`);
    if (r.tp + r.fp) near(vertex!.precision, r.precision!);
  }
  // The predict question: lowering 0.5 -> 0.3 moves from (0.4, 0.8) to (0.6, 1.0).
  const before = classificationThreshold(0.5, 1),
    after = classificationThreshold(0.3, 1);
  assert.deepEqual([before.fp / 5, before.tp / 5], [0.4, 0.8]);
  assert.deepEqual([after.fp / 5, after.tp / 5], [0.6, 1]);
  assert.deepEqual([before.tp, before.fp, after.tp, after.fp], [4, 2, 5, 3]);
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
test("Restoration default shows the point: the blurry mean has lower risk, the sharp sample is sharper but costs more", () => {
  const e = chapterVisionExperiments.find((x) => x.id === "restoration-ambiguity")!;
  const d = Object.fromEntries(e.controls.map((c) => [c.key, Number(c.value)]));
  const r = ambiguousRestoration(d.probability, d.mix);
  // At the defaults the reconstruction is a sharp copy of A.
  assert.deepEqual(r.output, r.a);
  assert.ok(r.riskMean < r.risk);
  near(r.riskMean, 0.25);
  near(r.risk, 0.5);
  assert.ok(totalVariation(r.output) > totalVariation(r.mean));
  assert.equal(totalVariation(r.output), 3);
  assert.equal(totalVariation(r.mean), 0);
  // Predict question: at P(A) = 0.9 the mean still wins (0.09 vs 0.10).
  const hi = ambiguousRestoration(0.9, 1);
  near(hi.riskMean, 0.09);
  near(hi.risk, 0.1);
  assert.ok(hi.riskMean < hi.risk);
  assert.equal(e.predict!.answer, "mean");
  for (const p of [0.1, 0.3, 0.5, 0.7, 0.9]) {
    const q = ambiguousRestoration(p, 1);
    near(q.riskMean, p * (1 - p));
    assert.ok(q.riskMean < q.risk);
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
test("All three bilingual experiments render finite SVGs at defaults and individual endpoint settings", () => {
  assert.equal(chapterVisionExperiments.length, 3);
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
