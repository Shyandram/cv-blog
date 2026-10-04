// Run: node --experimental-strip-types scripts/check-content-stats.mjs
import assert from 'node:assert/strict';
import { proseCount } from '../src/lib/content-stats.ts';

const body = [
  'import Demo from "./demo.astro";',
  '# Hello **world**',
  '[Read this](https://example.com/words-not-counted) and `ignored code` $x + y$.',
  '```python\nnot prose\n```',
  '<Demo title="ignored attributes">影像處理 test</Demo>',
  '{/* ignored comment */}',
].join('\n\n');
assert.deepEqual(proseCount(body, 'mdx'), { words: 6, characters: 4 });
assert.deepEqual(proseCount('Traditional 中文 and [a link](/path/)\n\n$$x = y$$', 'md'), { words: 4, characters: 2 });
assert.deepEqual(proseCount('', 'md'), { words: 0, characters: 0 });
console.log('Prose count checks passed: MDX, Markdown, mixed languages, and excluded syntax.');
