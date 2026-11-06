import test from 'node:test';
import assert from 'node:assert/strict';
import { experiments, experimentGroups, experimentGroupFor, type ExperimentGroup } from '../../src/lib/playground/catalog.ts';
import { experimentLoaders } from '../../src/lib/playground/experiment-loaders.ts';

test('each server-rendered experiment is available from its browser loader', async () => {
  assert.deepEqual(Object.keys(experimentLoaders).sort(), Object.keys(experimentGroups).sort());
  const loaded = [];
  for (const [group, load] of Object.entries(experimentLoaders)) {
    const entries = await load();
    assert.deepEqual(entries.map(e => e.id), experimentGroups[group as ExperimentGroup].map(e => e.id));
    for (const entry of entries) {
      assert.equal(experimentGroupFor.get(entry.id), group);
      loaded.push(entry.id);
    }
  }
  assert.equal(new Set(loaded).size, loaded.length, 'no experiment can be assigned to multiple modules');
  assert.deepEqual(loaded.sort(), experiments.map(e => e.id).sort());
});

test('browser modules reproduce the server defaults in both languages', async () => {
  for (const original of experiments) {
    const entries = await experimentLoaders[experimentGroupFor.get(original.id)!]();
    const entry = entries.find(e => e.id === original.id)!;
    assert.deepEqual(entry.controls, original.controls);
    const state = Object.fromEntries(original.controls.map(c => [c.key, c.value]));
    for (const lang of ['en', 'zh-tw'] as const) {
      assert.deepEqual(entry.compute(state, lang), original.compute(state, lang), `${original.id} / ${lang}`);
    }
  }
});
