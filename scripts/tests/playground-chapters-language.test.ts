import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  chapterLanguageExperiments, scopeTruth, trainBpe, poolingExample, sequenceDecode,
  lexicalRetrieval, decodingDistribution, maskingHeads, finiteDifference,
  mlpProbability, selectiveMemory, contrastiveExample, vaeObjective,
  flowTransform, transportSolve, binaryAutoregression, temperatureCalibration,
} from '../../src/lib/playground/chapters-language.ts';
import type { State } from '../../src/lib/playground/types.ts';

const close = (actual: number, expected: number, eps = 1e-10) => assert.ok(Math.abs(actual - expected) < eps, `${actual} != ${expected}`);
const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);

test('quantifier order: shared witness implies individual witnesses, but not conversely', () => {
  assert.deepEqual(scopeTruth([1, 2, 4]), { eachSome: true, shared: false });
  assert.deepEqual(scopeTruth([3, 1, 7]), { eachSome: true, shared: true });
  assert.deepEqual(scopeTruth([0, 1, 7]), { eachSome: false, shared: false });
  for (let a = 0; a < 8; a++) for (let b = 0; b < 8; b++) for (let c = 0; c < 8; c++) {
    const truth = scopeTruth([a, b, c]); assert.ok(!truth.shared || truth.eachSome);
  }
});

test('BPE fits training frequencies only, preserves spelling and reduces training length', () => {
  let last = Infinity;
  for (let n = 0; n <= 12; n++) {
    const result = trainBpe(n, 'slowest');
    assert.ok(result.trainingTokens <= last); last = result.trainingTokens;
    assert.equal(result.pieces.join(''), 'slowest▁');
    assert.deepEqual(result.merges, trainBpe(n, 'unknown').merges);
  }
  assert.equal(trainBpe(1, 'low').merges[0].count, 13);
  assert.equal(trainBpe(12, 'low').pieces.join(''), 'low▁');
});

test('pooling exposes order invariance and padding denominator contamination', () => {
  close(poolingExample(false, 'mean', 0).pooled, poolingExample(true, 'mean', 9).pooled);
  assert.notEqual(poolingExample(false, 'position', 0).pooled, poolingExample(true, 'position', 0).pooled);
  close(poolingExample(false, 'unmasked', 3).pooled, poolingExample(false, 'mean', 0).pooled / 2);
});

test('global sequence optimum respects BIO and improves over any legal local path', () => {
  for (const evidence of [-2, 0.8, 4]) for (const bonus of [-2, 1, 3]) {
    const r = sequenceDecode(evidence, bonus, true);
    assert.ok(r.bestValid);
    if (evidence === 4) assert.equal(r.localValid, false);
    if (r.localValid) assert.ok(r.bestScore >= r.localScore - 1e-10);
    const free = sequenceDecode(evidence, bonus, false);
    assert.ok(free.bestScore >= r.bestScore - 1e-10);
  }
});

test('retrieval has no matches for OOV and k1=0 reduces each matched term to IDF', () => {
  assert.ok(lexicalRetrieval('unseenword', 1.2, 0.75).every(d => d.score === 0));
  const docs = lexicalRetrieval('red', 0, 1).filter(d => d.score > 0);
  assert.equal(docs.length, 3);
  docs.forEach(d => close(d.score, Math.log(2)));
  assert.deepEqual(lexicalRetrieval('red red', 1, 0), lexicalRetrieval('red', 1, 0));
});

test('decoding normalizes truncated probabilities, and top-k precedes top-p', () => {
  for (const t of [0.2, 1, 3]) for (const k of [1, 3, 5]) for (const p of [0.1, 0.9, 1]) {
    const r = decodingDistribution(t, k, p); close(sum(r.distribution), 1);
    assert.ok(r.keep <= k); assert.equal(r.distribution.filter(v => v > 0).length, r.keep);
    if (k === 1) { close(r.distribution[0], 1); close(r.entropy, 0); }
    if (p === 1) assert.equal(r.keep, k);
  }
});

test('right context cannot leak through the causal visibility mask', () => {
  assert.deepEqual(maskingHeads(true, -1, 'masked').probabilities, maskingHeads(true, 1, 'masked').probabilities);
  assert.notDeepEqual(maskingHeads(false, -1, 'masked').probabilities, maskingHeads(false, 1, 'masked').probabilities);
  close(sum(maskingHeads(false, 1, 'sequence').probabilities), 1);
});

test('finite differences are accurate at useful steps and tiny steps can cancel', () => {
  assert.ok(finiteDifference(1, -4, true).relative < 1e-8);
  assert.ok(finiteDifference(1, -4, true).relative < finiteDifference(1, -4, false).relative);
  assert.ok(finiteDifference(1, -16, true).relative > finiteDifference(1, -4, true).relative);
});

test('hand constructed nonlinear network solves XOR, linear network cannot', () => {
  const corners = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
  for (const [x, y, label] of corners) assert.equal(Number(mlpProbability(x, y, 'relu', 1, 0) >= 0.5), label);
  for (const bias of [-3, 0, 3]) assert.ok(corners.some(([x, y, label]) => Number(mlpProbability(x, y, 'linear', 1, bias) >= 0.5) !== label));
});

test('selective recurrence rejects distractors and gate endpoints have clear semantics', () => {
  assert.deepEqual(selectiveMemory(0.25, 0, true).states, selectiveMemory(0.25, 2, true).states);
  assert.notDeepEqual(selectiveMemory(0.25, 0, false).states, selectiveMemory(0.25, 2, false).states);
  assert.ok(selectiveMemory(1, 1, true).states.every(h => h === 0));
  close(selectiveMemory(0, 1, true).states.at(-1)!, -1);
  close(selectiveMemory(0.5, 1, false).derivative, 0.5 ** 11);
});

test('contrastive collapse has log candidate-count loss and hard negatives penalize low temperature', () => {
  for (const t of [0.05, 0.2, 1]) close(contrastiveExample(20, 60, t, true).loss, Math.log(4));
  assert.ok(contrastiveExample(60, 0, 0.05, false).loss > contrastiveExample(60, 0, 1, false).loss);
  close(sum(contrastiveExample(20, 60, 0.2, false).probabilities), 1);
});

test('Gaussian VAE KL is nonnegative and zero at prior; beta acts only on KL', () => {
  close(vaeObjective(0, 1, 1).kl, 0);
  for (const mu of [-2, 0, 2]) for (const sigma of [0.1, 1, 2]) {
    const base = vaeObjective(mu, sigma, 0), weighted = vaeObjective(mu, sigma, 2);
    assert.ok(base.kl >= 0); close(weighted.total - base.total, 2 * base.kl);
  }
});

test('flow density includes positive Jacobian and numerically preserves probability mass', () => {
  for (const scale of [0.3, 1, 2]) for (const bend of [0, 1, 2]) {
    let area = 0; let prev = flowTransform(-5, scale, bend);
    for (let i = 1; i <= 2000; i++) { const curr = flowTransform(-5 + i / 200, scale, bend); assert.ok(curr.x > prev.x); area += (curr.x - prev.x) * (curr.density + prev.density) / 2; prev = curr; }
    close(area, 1, 1e-5);
    close(flowTransform(0, scale, bend).density * (scale + bend), 1 / Math.sqrt(2 * Math.PI));
  }
});

test('transport integrates constant field exactly and refinement reduces curved-field error', () => {
  close(transportSolve(0.5, 0, 1, false).error, 0);
  assert.ok(transportSolve(0.5, 0.8, 20, false).error < transportSolve(0.5, 0.8, 2, false).error);
  assert.ok(transportSolve(0.5, 0.8, 4, true).error < transportSolve(0.5, 0.8, 4, false).error);
});

test('autoregressive chain normalizes all images in either order and uniform conditionals give one bit per pixel', () => {
  for (const copy of [0.05, 0.5, 0.95]) for (const column of [true, false]) {
    const r = binaryAutoregression(copy, [0, 0, 1, 1], column); close(sum(r.distribution), 1);
    close(r.distribution[3], r.probability);
    if (copy === 0.5) close(r.bitsPerPixel, 1);
  }
  assert.notEqual(binaryAutoregression(0.8, [0, 0, 1, 1], false).probability, binaryAutoregression(0.8, [0, 0, 1, 1], true).probability);
});

test('temperature changes probability scoring without changing any decision', () => {
  for (const t of [0.2, 1, 4]) close(temperatureCalibration(t).accuracy, 0.75);
  assert.notEqual(temperatureCalibration(1).nll, temperatureCalibration(4).nll);
  assert.ok(temperatureCalibration(0.2).nll > temperatureCalibration(1).nll);
});

test('all chapter demos render finite bilingual results at range endpoints and select options', () => {
  assert.equal(chapterLanguageExperiments.length, 16);
  for (const experiment of chapterLanguageExperiments) {
    const base = Object.fromEntries(experiment.controls.map(c => [c.key, c.value]));
    const states: State[] = [base];
    for (const control of experiment.controls) {
      if (control.kind === 'range') for (const value of [control.min!, control.max!]) states.push({ ...base, [control.key]: value });
      if (control.kind === 'select') for (const option of control.options!) states.push({ ...base, [control.key]: option.value });
      if (control.kind === 'text') states.push({ ...base, [control.key]: '<script>alert("x")</script>' });
    }
    for (const state of states) for (const lang of ['en', 'zh-tw'] as const) {
      const result = experiment.compute(state, lang);
      assert.ok(result.summary.length > 30, experiment.id);
      assert.ok(result.svg.startsWith('<svg'), experiment.id);
      assert.ok(!/NaN|Infinity/.test(result.svg) && !/<script/i.test(result.svg), experiment.id);
      assert.ok(result.metrics.every(m => !/NaN|Infinity/.test(m.value)), experiment.id);
    }
  }
});
