import test from 'node:test';
import assert from 'node:assert/strict';
import { compile } from '@mdx-js/mdx';
import rehypeCodeFolding from '../../src/lib/rehypeCodeFolding.mjs';

test('code stays literal, with language/line metadata and no default open attribute', async () => {
  let tree: any;
  await compile({ value: '```python\nif x < 2:\n    print("<b> & {x}")\n```', path: '/src/content/tutorials/zh-tw/test.mdx' }, {
    rehypePlugins: [rehypeCodeFolding, () => (root) => { tree = root; }],
  });
  const wrap = tree.children[0], fold = wrap.children[0];
  assert.equal(fold.tagName, 'details');
  assert.equal(fold.properties.open, undefined);
  assert.equal(fold.children[0].children[0].children[0].value, '程式碼 · Python · 2 行');
  assert.equal(fold.children[1].children[0].children[0].value, 'if x < 2:\n    print("<b> & {x}")\n');
});

test('preserves existing bookmarks, avoids collisions and does not double-wrap', () => {
  const code = { type: 'element', tagName: 'code', properties: {}, children: [{ type: 'text', value: 'x\n' }] };
  const original = { type: 'element', tagName: 'pre', properties: { id: 'existing-code' }, children: [structuredClone(code)] };
  const tree: any = { type: 'root', children: [
    { type: 'element', tagName: 'h2', properties: { id: 'code-example-1' }, children: [] },
    original,
    { type: 'element', tagName: 'pre', properties: {}, children: [code] },
  ] };
  const transform = rehypeCodeFolding();
  transform(tree, { path: '/en/test.md' });
  assert.equal(tree.children[1].children[0].children[1], original);
  assert.equal(original.properties.id, 'existing-code');
  assert.equal(tree.children[2].children[0].children[1].properties.id, 'code-example-2');
  const once = structuredClone(tree);
  transform(tree, { path: '/en/test.md' });
  assert.deepEqual(tree, once);
});
