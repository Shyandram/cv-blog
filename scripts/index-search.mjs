import { readdir, readFile } from 'node:fs/promises';
import * as pagefind from 'pagefind';
import { join, relative } from 'node:path';

/** Index content pages, leaving redirects out of search results. */
export async function indexSearch(out) {
  const { index, errors } = await pagefind.createIndex({ excludeSelectors: ['.katex-html', '.katex-mathml annotation', '.rt-copy', '.fig-badge'] });
  if (!index) throw new Error(`Pagefind failed: ${errors.join('; ')}`);
  let count = 0;
  async function walk(dir) {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) { if (entry.name !== 'pagefind') await walk(path); continue; }
      if (!entry.name.endsWith('.html')) continue;
      const content = await readFile(path, 'utf8');
      if (/http-equiv=["']refresh["']/i.test(content) || /data-pagefind-ignore=["']all["']/i.test(content)) continue;
      const url = '/' + relative(out, path).replace(/index\.html$/, '');
      const result = await index.addHTMLFile({ content, url });
      if (result.errors.length) throw new Error(result.errors.join('; '));
      count++;
    }
  }
  try { await walk(out); await index.writeFiles({ outputPath: join(out, 'pagefind') }); }
  finally { await pagefind.close(); }
  return count;
}
