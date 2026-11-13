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

test('in tutorials, a text block right after code becomes its attached, unfolded output', () => {
  const pre = (lang: string, value: string) => ({ type: 'element', tagName: 'pre', properties: { dataLanguage: lang }, children: [{ type: 'element', tagName: 'code', properties: {}, children: [{ type: 'text', value }] }] });
  const make = (): any => ({ type: 'root', children: [pre('python', 'print(1)\n'), { type: 'text', value: '\n' }, pre('text', '1\n'), { type: 'text', value: '\n' }, pre('text', 'grid\n')] });
  const tree = make();
  const transform = rehypeCodeFolding();
  transform(tree, { path: '/src/content/tutorials/en/x/y.mdx' });
  assert.equal(tree.children.length, 3);
  const wrap = tree.children[0];
  assert.equal(wrap.children[1].properties.className[0], 'rt-output');
  assert.equal(wrap.children[1].children[0].children[0].value, 'Output');
  assert.equal(wrap.children[1].children[1].children[0].children[0].value, '1\n');
  // A second text block is not output: it stays a folded block of its own.
  assert.equal(tree.children[2].children[0].tagName, 'details');
  const once = structuredClone(tree);
  transform(tree, { path: '/src/content/tutorials/en/x/y.mdx' });
  assert.deepEqual(tree, once);
  // Outside tutorials nothing is paired.
  const post = make();
  transform(post, { path: '/src/content/posts/en/p.md' });
  assert.equal(post.children.filter((n: any) => n.tagName === 'div').length, 3);
});
