import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import redirects from '../src/content/legacy-chapter-redirects.json' with { type: 'json' };
/**
 * Static meta refresh drops fragments in Chromium; carry old bookmark hashes to the new chapter.
 * @param {string} out
 * @param {Record<string, string>} [routes]
 */
export async function preserveRedirectFragments(out, routes = redirects) {
  const escape = (value) => value.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
  for (const [from, to] of Object.entries(routes)) {
    const file = join(out, from, 'index.html');
    let html;
    try {
      html = await readFile(file, 'utf8');
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      // Some hosts emit native redirects without HTML. Keep a static fallback too:
      // this hook must not fail before Pagefind can index the actual course pages.
      const target = escape(to);
      const lang = from.startsWith('/zh-tw/') ? 'zh-Hant' : 'en';
      html = `<!doctype html><html lang="${lang}"><head><meta charset="utf-8"><meta name="robots" content="noindex"><title>Redirect</title><meta http-equiv="refresh" content="0;url=${target}"></head><body><a href="${target}">${target}</a></body></html>`;
      await mkdir(dirname(file), { recursive: true });
    }
    if (html.includes('data-preserve-fragment')) continue;
    const script = `<script data-preserve-fragment>const target=new URL(${JSON.stringify(to).replaceAll('<', '\\u003c')},location.href);if(!target.hash)target.hash=location.hash;location.replace(target.href)</script>`;
    await writeFile(file, html.replace(/<meta http-equiv="refresh"[^>]*>/, (meta) => script + '<noscript>' + meta + '</noscript>'));
  }
}
