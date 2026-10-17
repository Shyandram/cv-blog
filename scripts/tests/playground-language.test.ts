import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ngramDistribution, stableSoftmax, attentionExample, languageExperiments, ATTENTION_V } from '../../src/lib/playground/language.ts';

const close = (actual: number, expected: number, tolerance = 1e-12) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
const sum = (items: number[]) => items.reduce((a, b) => a + b, 0);

test('n-gram counts respect line boundaries and include EOS, never BOS targets', () => {
  const result = ngramDistribution('a b\na c', 'a', 2, 0);
  assert.equal(result.contextCount, 2);
  close(result.probabilities[result.vocabulary.indexOf('b')], 0.5);
  close(result.probabilities[result.vocabulary.indexOf('c')], 0.5);
  assert.ok(!result.vocabulary.includes('[BOS]'));
  const end = ngramDistribution('a b\na c', 'b', 2, 0);
  close(end.probabilities[end.vocabulary.indexOf('[EOS]')], 1);
  const start = ngramDistribution('a b\na c', '', 3, 0);
  assert.deepEqual(start.context, ['[BOS]', '[BOS]']);
  close(start.probabilities[start.vocabulary.indexOf('a')], 1);
});

test('unseen query never expands vocabulary and zero-count MLE is undefined', () => {
  const result = ngramDistribution('a b', 'newword', 2, 0);
  assert.equal(result.defined, false);
  assert.deepEqual(result.context, ['[UNK]']);
  assert.ok(!result.vocabulary.includes('newword'));
  const smooth = ngramDistribution('a b', 'newword', 2, 0.5);
  assert.equal(smooth.defined, true);
  smooth.probabilities.forEach(p => close(p, 0.25));
  close(sum(smooth.probabilities), 1);
});

test('unigram ignores supplied context; empty corpus remains undefined', () => {
  const result = ngramDistribution('a a\nb', 'unseen query', 1, 0);
  assert.equal(result.contextCount, 5);
  close(result.probabilities[result.vocabulary.indexOf('a')], 2 / 5);
  close(result.probabilities[result.vocabulary.indexOf('[EOS]')], 2 / 5);
  assert.deepEqual(result.context, []);
  assert.deepEqual(ngramDistribution(' \n ', 'a', 2, 1).vocabulary, []);
});

test('softmax remains stable for large negative scores and completely masked rows', () => {
  const result = stableSoftmax([-10000, -10001]);
  close(sum(result), 1);
  close(result[0], 1 / (1 + Math.exp(-1)));
  assert.deepEqual(stableSoftmax([1, 2], [false, false]), [0, 0]);
  assert.deepEqual(stableSoftmax([1, 20000], [true, false]), [1, 0]);
});

test('causal attention excludes future values and returns an actual weighted sum', () => {
  const first = attentionExample(0, true, 1);
  assert.deepEqual(first.weights, [1, 0, 0, 0, 0]);
  assert.deepEqual(first.output, ATTENTION_V[0]);
  const result = attentionExample(2, true, 0.5);
  assert.equal(result.weights[3], 0);
  assert.equal(result.weights[4], 0);
  close(sum(result.weights), 1);
  result.output.forEach((value, d) => close(value, result.weights.reduce((a, w, i) => a + w * ATTENTION_V[i][d], 0)));
  assert.ok(attentionExample(2, false, 1).weights[4] > 0);
  assert.ok(Math.max(...attentionExample(2, false, 0.2).weights) > Math.max(...attentionExample(2, false, 2).weights));
});

test('all bilingual experiments compute finite SVG and escape entered text', () => {
  for (const experiment of languageExperiments) {
    const state = Object.fromEntries(experiment.controls.map(c => [c.key, c.value]));
    for (const lang of ['en', 'zh-tw'] as const) {
      const result = experiment.compute(state, lang);
      assert.ok(result.svg.startsWith('<svg'));
      assert.ok(!/NaN|undefined/.test(result.svg));
      assert.ok(result.summary.length > 0);
    }
  }
  const result = languageExperiments[0].compute({ corpus: '<script> a\n<img> b', context: '', order: 1, alpha: 0.1 }, 'en');
  assert.ok(!result.svg.includes('<script>'));
  assert.ok(!result.svg.includes('<img>'));
  assert.ok(result.svg.includes('&lt;'));
});
