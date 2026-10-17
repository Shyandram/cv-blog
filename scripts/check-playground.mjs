/**
 * Chapter coverage and rendering smoke checks. Run after adding, renaming or removing chapters or experiments.
 *
 * Rules enforced here (see src/lib/playground/coverage.ts):
 *  - every chapter either lists 1-2 experiments in `chapterPlan` or is listed as reading-only;
 *  - no chapter lists more than MAX_EXPERIMENTS_PER_CHAPTER experiments;
 *  - every plan entry and alias points at a real experiment, and every experiment is used by a chapter.
 * Experiments with a `compute()` are rendered in every control state (EN and zh-tw); image labs
 * (`kind: 'image'`) are exercised by scripts/check-labs.py in a real browser.
 */
import { build } from 'esbuild';
import { readdirSync, readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
const bundled = await build({
  stdin: {
    contents: "export * from './src/lib/playground/coverage'; export * from './src/lib/playground/catalog';",
    resolveDir: process.cwd(), loader: 'ts',
  },
  bundle: true, write: false, platform: 'node', format: 'esm',
});
const m = await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`);
const { experiments, labs = [], playgroundEntries, chapterExperiments, chaptersForExperiment, chapterPlan, readingOnly, idAliases, MAX_EXPERIMENTS_PER_CHAPTER } = m;

const ids = new Set(playgroundEntries.map(e => e.id));
assert.equal(ids.size, playgroundEntries.length, 'Duplicate experiment ID');

const chapters = [];
for (const track of readdirSync('src/content/tutorials/en')) for (const file of readdirSync(`src/content/tutorials/en/${track}`)) {
  if (!file.endsWith('.mdx') || file.startsWith('_')) continue;
  const body = readFileSync(`src/content/tutorials/en/${track}/${file}`, 'utf8');
  if (/^draft:\s*true\s*$/m.test(body)) continue;
  chapters.push(`${track}/${file.slice(0, -4)}`);
}
for (const slug of chapters) {
  const planned = chapterPlan[slug] ?? [];
  assert.ok(planned.length || readingOnly[slug], `Chapter has neither an experiment nor a reading-only entry: ${slug}`);
  assert.ok(!(planned.length && readingOnly[slug]), `Chapter is both planned and reading-only: ${slug}`);
  assert.ok(planned.length <= MAX_EXPERIMENTS_PER_CHAPTER, `Chapter lists more than ${MAX_EXPERIMENTS_PER_CHAPTER} experiments: ${slug} (${planned.join(', ')})`);
  assert.equal(new Set(planned).size, planned.length, `Duplicate experiment in plan for ${slug}`);
  assert.equal(chapterExperiments(slug).length, planned.length, `Plan for ${slug} names an unknown experiment: ${planned.join(', ')}`);
}
for (const slug of [...Object.keys(chapterPlan), ...Object.keys(readingOnly)]) assert.ok(chapters.includes(slug), `Plan or reading-only entry for a missing chapter: ${slug}`);
for (const [from, to] of Object.entries(idAliases)) {
  assert.ok(!ids.has(from), `Alias ${from} shadows a live experiment`);
  assert.ok(ids.has(to), `Alias ${from} points at unknown experiment ${to}`);
}
for (const e of playgroundEntries) assert.ok(chaptersForExperiment(e.id).some(slug => chapters.includes(slug)), `No current chapter uses ${e.id}`);

let states = 0;
for (const e of experiments) {
  assert.equal(new Set(e.controls.map(c => c.key)).size, e.controls.length, `Duplicate control key in ${e.id}`);
  const defaults = Object.fromEntries(e.controls.map(c => [c.key, c.value]));
  let variants = [defaults];
  for (const c of e.controls) {
    if (c.kind === 'range') { assert.ok(c.min <= Number(c.value) && Number(c.value) <= c.max, `Default outside range in ${e.id}.${c.key}`); variants = variants.flatMap(s => [s, { ...s, [c.key]: c.min }, { ...s, [c.key]: c.max }]); }
    else if (c.kind === 'select') variants = variants.flatMap(s => c.options.map(o => ({ ...s, [c.key]: o.value })));
  }
  if (e.predict) {
    assert.ok(e.predict.options.some(o => o.value === e.predict.answer), `Predict answer is not an option in ${e.id}`);
    assert.ok(e.predict.options.length >= 2, `Predict needs at least two options in ${e.id}`);
  }
  for (const state of variants) for (const lang of ['en', 'zh-tw']) {
    const result = e.compute(state, lang); states++;
    assert.ok(!/NaN|Infinity|undefined/.test(result.svg), `Invalid SVG in ${e.id}: ${JSON.stringify(state)}`);
    assert.ok(result.summary.trim() && result.metrics.length, `Missing interpretation in ${e.id}`);
  }
}
for (const lab of labs) {
  assert.equal(new Set(lab.controls.map(c => c.key)).size, lab.controls.length, `Duplicate control key in lab ${lab.id}`);
  assert.ok(lab.predict, `Image lab ${lab.id} needs a predict step`);
  assert.ok(lab.predict.options.some(o => o.value === lab.predict.answer), `Predict answer is not an option in lab ${lab.id}`);
  assert.ok(lab.controls.length <= 3, `Image lab ${lab.id} has more than three controls`);
}
console.log(JSON.stringify({
  chapters: chapters.length,
  withExperiments: chapters.filter(s => (chapterPlan[s] ?? []).length).length,
  readingOnly: chapters.filter(s => readingOnly[s]).length,
  experiments: playgroundEntries.length, imageLabs: labs.length, renderStates: states,
}, null, 2));
