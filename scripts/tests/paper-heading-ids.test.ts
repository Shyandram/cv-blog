import test from 'node:test';
import assert from 'node:assert/strict';
import {compile} from '@mdx-js/mdx';
import {rehypeHeadingIds} from '@astrojs/markdown-remark';
import rehypePaperHeadingIds from '../../src/lib/rehypePaperHeadingIds.mjs';

type Heading = {depth: number; slug: string; text: string};
async function headings(body: string, file: string): Promise<Heading[]> {
  const result = await compile({value: body, path: file}, {rehypePlugins: [rehypePaperHeadingIds, rehypeHeadingIds]});
  return (result.data as {astro?: {headings?: Heading[]}}).astro?.headings ?? [];
}
function sections(all: Heading[], guide: string) {
  const start = all.findIndex(h => h.depth === 3 && h.text === guide);
  const after = all.findIndex((h, i) => i > start && h.depth <= 3);
  return all.slice(start+1, after < 0 ? undefined : after).map(h => h.slug);
}

test('inserting an earlier English paper keeps Retinex-Net bookmarks from the rendered course', async () => {
  const guide = 'Retinex-Net: decompose, adjust and denoise';
  const body = `## Paper guides\n\n### A newly inserted paper\n\n#### Challenge\n\n#### Method\n\n### ${guide}\n\n#### Challenge\n\n#### Method\n`;
  const result = await headings(body, '/workspace/src/content/tutorials/en/deep-learning/dl-05-low-light-enhancement.mdx');
  assert.deepEqual(sections(result, guide), ['challenge', 'method']);
  assert.ok(sections(result, 'A newly inserted paper').every(id => id.startsWith('paper-a-newly-inserted-paper--')));
  assert.equal(new Set(result.map(h => h.slug)).size, result.length);
});

test('the corresponding Traditional Chinese bookmarks remain unchanged', async () => {
  const guide = 'Retinex-Net：分解、調光、去雜訊';
  const body = `## 論文導讀\n\n### 新增導讀\n\n#### 挑戰\n\n#### 方法\n\n### ${guide}\n\n#### 挑戰\n\n#### 方法\n`;
  const result = await headings(body, '/workspace/src/content/tutorials/zh-tw/deep-learning/dl-05-low-light-enhancement.mdx');
  assert.deepEqual(sections(result, guide), ['挑戰', '方法']);
  assert.equal(new Set(result.map(h => h.slug)).size, result.length);
});

test('new guide subsection IDs stay attached to their paper when papers are reordered', async () => {
  const item = (name: string) => `### ${name}\n\n#### Method\n\n#### Results\n\n`;
  const file = 'src/content/tutorials/en/deep-learning/new-guide-fixture.mdx';
  const forward = await headings('## Paper guides\n\n'+item('First paper')+item('Second paper'), file);
  const backward = await headings('## Paper guides\n\n'+item('Second paper')+item('First paper'), file);
  for (const paper of ['First paper', 'Second paper']) assert.deepEqual(sections(forward, paper), sections(backward, paper));
  assert.notDeepEqual(sections(forward, 'First paper'), sections(forward, 'Second paper'));
});

test('explicit anchors and ordinary tutorial sections retain their existing meaning', () => {
  const h = (depth: number, text: string, id?: string) => ({
    type: 'element', tagName: `h${depth}`, properties: id ? {id} : {},
    children: [{type: 'text', value: text}],
  });
  const ordinary = h(4, 'Method');
  const explicit = h(4, 'Chosen title', 'chosen-anchor');
  const tree = {type: 'root', children: [h(2, 'Main'), ordinary, h(2, 'Paper guides'), h(3, 'New paper'), explicit]};
  rehypePaperHeadingIds()(tree, {path: 'src/content/tutorials/en/deep-learning/new-guide-fixture.mdx'});
  assert.equal(explicit.properties.id, 'chosen-anchor');
  assert.equal(ordinary.properties.id, undefined);
});
