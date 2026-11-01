import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import redirects from '../src/content/legacy-chapter-redirects.json' with { type: 'json' };
/** Static meta refresh drops fragments in Chromium; carry old bookmark hashes to the new chapter. */
export async function preserveRedirectFragments(out) {
  for (const [from, to] of Object.entries(redirects)) {
    const file = join(out, from, 'index.html');
    const html = await readFile(file, 'utf8');
    if (html.includes('data-preserve-fragment')) continue;
    const script = `<script data-preserve-fragment>location.replace(${JSON.stringify(to)} + location.hash)</script>`;
    await writeFile(file, html.replace(/<meta http-equiv="refresh"[^>]*>/, (meta) => script + '<noscript>' + meta + '</noscript>'));
  }
}
