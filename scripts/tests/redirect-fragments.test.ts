import test, { type TestContext } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { runInNewContext } from 'node:vm';
import { preserveRedirectFragments } from '../preserve-redirect-fragments.mjs';
import { indexSearch } from '../index-search.mjs';

async function fixture(t: TestContext) {
  const out = await mkdtemp(join(tmpdir(), 'cvblog-redirects-'));
  t.after(() => rm(out, { recursive: true, force: true }));
  return out;
}
function follow(html: string, hash: string) {
  let redirected = '';
  const script = html.match(/<script data-preserve-fragment>([\s\S]*?)<\/script>/)![1];
  runInNewContext(script, { URL, location: { href: `https://example.com/en/old/${hash}`, hash, replace: (url: string) => { redirected = url; } } });
  return redirected;
}

test('native host redirects without HTML get a usable static fallback before indexing', async (t) => {
  const out = await fixture(t);
  await mkdir(join(out, 'en/new'), { recursive: true });
  await writeFile(join(out, 'en/new/index.html'), '<html lang="en"><body><main data-pagefind-body><h1>Learning images</h1><p>Pixels and images.</p></main></body></html>');
  await preserveRedirectFragments(out, { '/en/old/': '/en/new/' });
  const html = await readFile(join(out, 'en/old/index.html'), 'utf8');
  assert.equal(follow(html, '#exercise'), 'https://example.com/en/new/#exercise');
  assert.match(html, /<noscript><meta http-equiv="refresh" content="0;url=\/en\/new\/">/);
  assert.equal(await indexSearch(out), 1, 'redirect fallback stays out of search results');
  await readFile(join(out, 'pagefind/pagefind.js'));
});

test('existing static redirects retain their markup and are safe to process twice', async (t) => {
  const out = await fixture(t);
  await mkdir(join(out, 'en/old'), { recursive: true });
  await writeFile(join(out, 'en/old/index.html'), '<html><head><title>Original</title><meta http-equiv="refresh" content="0;url=/en/new/"></head><body>Original text</body></html>');
  const routes = { '/en/old/': '/en/new/' };
  await preserveRedirectFragments(out, routes);
  const once = await readFile(join(out, 'en/old/index.html'), 'utf8');
  await preserveRedirectFragments(out, routes);
  assert.equal(await readFile(join(out, 'en/old/index.html'), 'utf8'), once);
  assert.match(once, /<title>Original<\/title>/);
  assert.match(once, /<body>Original text<\/body>/);
});

test('explicit destination anchors win and fallback HTML escapes target URLs', async (t) => {
  const out = await fixture(t);
  await preserveRedirectFragments(out, { '/zh-tw/old/': '/zh-tw/new/?a=1&b="<tag>"#target' });
  const html = await readFile(join(out, 'zh-tw/old/index.html'), 'utf8');
  assert.match(html, /lang="zh-Hant"/);
  assert.match(html, /&amp;b=&quot;&lt;tag&gt;&quot;/);
  assert.equal(new URL(follow(html, '#incoming')).hash, '#target');
  assert.equal((html.match(/<script /g) ?? []).length, 1);
});

test('filesystem failures other than a missing redirect are still reported', async (t) => {
  const out = await fixture(t);
  await mkdir(join(out, 'en/old/index.html'), { recursive: true });
  await assert.rejects(preserveRedirectFragments(out, { '/en/old/': '/en/new/' }), { code: 'EISDIR' });
});
