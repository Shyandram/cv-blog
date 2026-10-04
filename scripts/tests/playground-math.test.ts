import test from 'node:test';
import assert from 'node:assert/strict';
import { bayesDecision, quadraticDescent, binaryEntropy, binaryChannel, huffmanCode, mathExperiments } from '../../src/lib/playground/math.ts';
const close = (actual: number, expected: number, tolerance = 1e-10) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);

test('Bayes threshold is the equal-risk point and responds correctly to cost/prior', () => {
  for (const prior of [0.05, 0.5, 0.95]) for (const cost of [0.25, 1, 5]) {
    const threshold = bayesDecision(prior, cost, 0).threshold;
    const boundary = bayesDecision(prior, cost, threshold);
    close(boundary.risk0, boundary.risk1);
    close(boundary.posterior, cost / (1 + cost));
    assert.equal(bayesDecision(prior, cost, threshold - 0.1).decision, 0);
    assert.equal(bayesDecision(prior, cost, threshold + 0.1).decision, 1);
  }
  assert.ok(bayesDecision(0.5, 5, 0).threshold > 0);
  assert.ok(bayesDecision(0.8, 1, 0).threshold < 0);
});
test('quadratic updates converge, oscillate at the boundary and diverge beyond it', () => {
  close(quadraticDescent(1, 1, 1).objective, 0);
  const stable = quadraticDescent(10, 0.1, 40);
  assert.ok(stable.objective < 0.001);
  close(quadraticDescent(10, 0.2, 40).points.at(-1)![1], 2);
  assert.ok(quadraticDescent(10, 0.3, 40).objective > 1e20);
  assert.ok(Number.isFinite(quadraticDescent(20, 1, 40).objective));
});
test('binary channel matches noiseless, useless and deterministic-input limits', () => {
  close(binaryEntropy(0), 0); close(binaryEntropy(1), 0); close(binaryEntropy(0.5), 1);
  for (const q of [0, 0.1, 0.5, 0.9, 1]) {
    close(binaryChannel(q, 0).mutualInformation, binaryEntropy(q));
    close(binaryChannel(q, 0.5).mutualInformation, 0);
    for (const p of [0, 0.1, 0.5]) {
      const channel = binaryChannel(q, p);
      assert.ok(channel.mutualInformation <= channel.capacity + 1e-10);
      assert.ok(channel.mutualInformation >= -1e-10);
    }
  }
  for (const p of [0, 0.1, 0.5]) close(binaryChannel(0.5, p).mutualInformation, binaryChannel(0.5, p).capacity);
  close(binaryChannel(0, 0.2).mutualInformation, 0);
});
test('Huffman codes are prefix free, satisfy bounds and reconstruct messages', () => {
  for (const dominant of [0.1, 0.25, 0.4, 0.7, 0.9]) {
    const result = huffmanCode([dominant, ...Array(3).fill((1 - dominant) / 3)]);
    const words = Object.values(result.codes);
    for (const first of words) for (const second of words) if (first !== second) assert.ok(!second.startsWith(first));
    close(words.reduce((sum, word) => sum + 2 ** -word.length, 0), 1);
    assert.ok(result.averageLength >= result.entropy - 1e-10);
    assert.ok(result.averageLength < result.entropy + 1);
    const encoded = [...'ABACABAD'].map(symbol => result.codes[symbol]).join('');
    let decoded = '', pending = '';
    for (const bit of encoded) {
      pending += bit;
      const match = Object.entries(result.codes).find(([, word]) => word === pending);
      if (match) { decoded += match[0]; pending = ''; }
    }
    assert.equal(decoded, 'ABACABAD'); assert.equal(pending, '');
  }
  close(huffmanCode([0.25, 0.25, 0.25, 0.25]).averageLength, 2);
  close(huffmanCode([0.7, 0.1, 0.1, 0.1]).averageLength, 1.5);
});
test('all math labs render both languages at control extremes without invalid coordinates', () => {
  for (const experiment of mathExperiments) for (const lang of ['en', 'zh-tw'] as const) {
    for (const edge of ['value', 'min', 'max'] as const) {
      const state = Object.fromEntries(experiment.controls.map(control => [control.key, control[edge] ?? control.value]));
      const result = experiment.compute(state, lang);
      assert.ok(result.svg.startsWith('<svg'));
      assert.ok(!/NaN|Infinity|undefined/.test(result.svg + result.summary));
      assert.ok(result.metrics.length >= 3);
    }
  }
});
