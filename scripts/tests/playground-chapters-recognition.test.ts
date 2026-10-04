import test from 'node:test';
import assert from 'node:assert/strict';
import { kernelDensity, densityScore, ridgeFit, REGRESSION_X, REGRESSION_Y, nearestVote, projectionStats, alignSequences, mixtureEM, variationalComparison, ensembleVariance, rejectionUnderShift, chapterRecognitionExperiments } from '../../src/lib/playground/chapters-recognition.ts';
const close = (actual: number, expected: number, tolerance = 1e-9) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);

test('kernel density integrates to one and leave-one-out excludes self prediction', () => {
  for (const h of [0.1, 0.5, 2]) {
    let integral = 0;
    for (let i = 0; i <= 5000; i++) integral += kernelDensity(-20 + i * 0.008, h) * 0.008;
    close(integral, 1, 1e-8);
    assert.ok(Number.isFinite(densityScore(h)));
  }
  assert.ok(densityScore(0.5) > densityScore(0.1));
});
test('ridge solutions satisfy normal equations and regularization shrinks coefficients', () => {
  for (const degree of [1, 3, 7]) for (const lambda of [0, 0.1, 5]) {
    const fit = ridgeFit(degree, lambda);
    for (let j = 0; j <= degree; j++) {
      const derivative = REGRESSION_X.reduce((s, x, i) => s + (fit.predict(x) - REGRESSION_Y[i]) * x ** j, 0) + (j ? lambda * fit.weights[j] : 0);
      close(derivative, 0, 1e-8);
    }
  }
  assert.ok(ridgeFit(7, 5).penalty < ridgeFit(7, 0).penalty);
  assert.ok(ridgeFit(7, 0).trainingMse < ridgeFit(1, 0).trainingMse);
});
test('nearest-neighbor metric affects rankings and odd votes never tie', () => {
  const a = nearestVote(1.2, 0.5, 5, 0.2), b = nearestVote(1.2, 0.5, 5, 3);
  assert.notDeepEqual(a.neighbors.map(p => p.i), b.neighbors.map(p => p.i));
  close(nearestVote(0.85, 0, 1, 1).neighbors[0].distance, 0);
  for (const k of [1, 5, 23]) {
    const r = nearestVote(0.2, 1.1, k, 1);
    assert.equal(r.neighbors.length, k);
    assert.notEqual(r.fraction, 0.5);
    close(r.fraction * k, r.neighbors.reduce((s, p) => s + p.label, 0));
  }
});
test('orthogonal projections conserve variance and demonstrate PCA label loss', () => {
  for (const angle of [0, 30, 90, 180]) {
    const r = projectionStats(angle);
    close(r.variance + r.reconstructionMse, r.totalVariance);
    close(r.u[0] ** 2 + r.u[1] ** 2, 1);
  }
  close(projectionStats(0).gap, 0);
  close(projectionStats(90).gap, 2);
  assert.ok(projectionStats(0).variance > projectionStats(90).variance);
});
test('DTW obeys path constraints and cannot worsen when the band widens', () => {
  close(alignSequences(1, 3).dtwCost, 0);
  for (const warp of [0.5, 1.7, 2.5]) {
    const diagonal = alignSequences(warp, 0);
    close(diagonal.dtwCost, diagonal.diagonalCost);
    let previous = Infinity;
    for (const band of [0, 1, 3, 11]) {
      const r = alignSequences(warp, band);
      assert.ok(r.dtwCost <= previous + 1e-10); previous = r.dtwCost;
      assert.deepEqual(r.route[0], [0, 0]); assert.deepEqual(r.route.at(-1), [11, 11]);
      close(r.route.reduce((s, [i, j]) => s + r.costs[i][j], 0), r.dtwCost);
      r.route.forEach(([i, j], index) => {
        assert.ok(Math.abs(i - j) <= band);
        if (index) { const [a, b] = r.route[index - 1]; assert.ok(i - a >= 0 && i - a <= 1 && j - b >= 0 && j - b <= 1 && i + j > a + b); }
      });
    }
  }
});
test('constrained EM increases likelihood and retains identical-mean symmetry', () => {
  for (const separation of [0, 0.2, 4]) for (const sigma of [0.4, 0.8, 2]) {
    const r = mixtureEM(separation, sigma, 20);
    r.history.slice(1).forEach((v, i) => assert.ok(v >= r.history[i] - 1e-9));
    assert.ok(r.responsibilities.every(v => v >= 0 && v <= 1));
  }
  const symmetric = mixtureEM(0, 0.8, 20);
  close(symmetric.means[0], symmetric.means[1]);
  symmetric.responsibilities.forEach(v => close(v, 0.5));
});
test('KL comparison vanishes for identical normals and distinguishes mode coverage', () => {
  const exact = variationalComparison(0, 0.65, 0);
  close(exact.forward, 0); close(exact.reverse, 0);
  const mode = variationalComparison(-2, 0.65, 2), cover = variationalComparison(0, 2.1, 2);
  assert.ok(cover.forward < mode.forward);
  assert.ok(mode.reverse < cover.reverse);
  for (const r of [exact, mode, cover, variationalComparison(3, 3, 3)]) {
    assert.ok(r.forward >= 0 && r.reverse >= 0);
    assert.ok(r.pMass > 0.999 && r.qMass > 0.998);
  }
});
test('ensemble variance matches independent and fully correlated limits', () => {
  for (const m of [1, 5, 50]) {
    close(ensembleVariance(m, 0, 0).variance, 1 / m);
    close(ensembleVariance(m, 1, 0).variance, 1);
    close(ensembleVariance(m, 0.3, 0.4).mse - ensembleVariance(m, 0.3, 0).mse, 0.16);
  }
});
test('rejection distinguishes no decisions from no error and exposes distribution shift', () => {
  const all = rejectionUnderShift(0.5, 0), selective = rejectionUnderShift(0.9, 0), shifted = rejectionUnderShift(0.9, 2), none = rejectionUnderShift(1, 0);
  close(all.coverage, 1); close(all.risk!, 0.158655, 0.001);
  assert.ok(selective.coverage < all.coverage);
  assert.ok(selective.risk! < all.risk!);
  assert.ok(shifted.risk! > selective.risk!);
  close(none.coverage, 0); assert.equal(none.risk, null);
  close(rejectionUnderShift(0.8, -1).risk!, rejectionUnderShift(0.8, 1).risk!, 1e-8);
});
test('all nine chapter experiments render finite bilingual results at control boundaries', () => {
  assert.equal(chapterRecognitionExperiments.length, 9);
  for (const experiment of chapterRecognitionExperiments) {
    const initial = Object.fromEntries(experiment.controls.map(c => [c.key, c.value]));
    for (const lang of ['en', 'zh-tw'] as const) {
      for (const state of [initial, ...experiment.controls.flatMap(c => c.kind === 'range' ? [{ ...initial, [c.key]: c.min! }, { ...initial, [c.key]: c.max! }] : [])]) {
        const result = experiment.compute(state, lang);
        assert.ok(!/NaN|Infinity|undefined/.test(result.svg), experiment.id);
        assert.ok(result.summary.length > 20); assert.ok(result.metrics.length >= 2);
        assert.match(result.svg, /viewBox="0 0 640/);
      }
    }
    assert.equal(experiment.related.length, 1);
  }
});
