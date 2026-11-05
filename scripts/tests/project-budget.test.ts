import test from 'node:test';
import assert from 'node:assert/strict';
import { projectBudget } from '../../src/lib/project-budget.ts';

test('published planning scenario includes development once and annual contingency', () => {
  const result = projectBudget();
  assert.deepEqual(result.develop.usd, [425, 1275]);
  assert.ok(Math.abs(result.annual.usd[0] - 127) < 1e-9);
  assert.equal(result.annual.usd[1], 635);
  assert.deepEqual(result.total.usd, [552, 1910]);
  assert.deepEqual(result.total.twd, [17664, 61120]);
});

test('usage scales API costs, retaining fixed annual allowance and separate currency conversion', () => {
  assert.deepEqual(projectBudget(0).total.usd, [25, 125]);
  assert.deepEqual(projectBudget(2).develop.usd, [850, 2550]);
  assert.deepEqual(projectBudget(1, 40).total.twd, [22080, 76400]);
  assert.deepEqual(projectBudget(1, 40).total.usd, projectBudget().total.usd);
});

test('invalid planning input cannot produce a displayed NaN or negative budget', () => {
  for (const [scale, exchange] of [[-1, 32], [21, 32], [NaN, 32], [1, 0], [1, Infinity], [1, 1001]]) {
    assert.throws(() => projectBudget(scale, exchange), RangeError);
  }
});
