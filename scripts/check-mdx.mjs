// Fast syntax check for content files without a full site build.
// Usage: node scripts/check-mdx.mjs <file.mdx|file.md> [...]
// Compiles each file with the same remark plugins the site uses and checks that
// frontmatter has the required tutorial/post fields. Exit code 1 on any error.
import { compile } from '@mdx-js/mdx';
import remarkMath from 'remark-math';
import remarkCjkFriendly from 'remark-cjk-friendly';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import { readFileSync } from 'node:fs';

let failed = false;
for (const file of process.argv.slice(2)) {
  const src = readFileSync(file, 'utf8');
  const fm = src.match(/^---\n([\s\S]*?)\n---\n/);
  const body = fm ? src.slice(fm[0].length) : src;
  const problems = [];
  if (!fm) problems.push('missing frontmatter');
  else {
    const need = file.includes('/tutorials/')
      ? ['title', 'description', 'track', 'order', 'level', 'published']
      : ['title', 'description', 'published'];
    for (const k of need) if (!new RegExp(`^${k}:`, 'm').test(fm[1])) problems.push(`frontmatter missing "${k}"`);
  }
  // Components must be imported at the top level: an import inside a code fence compiles but fails at build time.
  if (file.endsWith('.mdx')) {
    const prose = body.replace(/^(```|~~~)[\s\S]*?^\1/gm, '').replace(/`[^`\n]*`/g, '').replace(/\$\$[\s\S]*?\$\$/g, '').replace(/(?<!\\)\$[^$\n]*\$/g, '');
    const imported = new Set([...prose.matchAll(/^import\s+(?:\{([^}]*)\}|(\w+))\s+from\s+/gm)].flatMap(m => (m[1] ?? m[2]).split(',').map(s => s.trim().split(/\s+as\s+/).pop())));
    for (const tag of new Set([...prose.matchAll(/<([A-Z][A-Za-z0-9]*)\b/g)].map(m => m[1]))) if (!imported.has(tag)) problems.push(`component <${tag}> is used but not imported at the top level`);
  }
  try {
    const out = await compile(body, {
      format: file.endsWith('.md') ? 'md' : 'mdx',
      remarkPlugins: [remarkGfm, remarkMath, remarkCjkFriendly],
      rehypePlugins: [[rehypeKatex, { throwOnError: true, strict: false }]],
    });
    // rehype-katex reports bad formulas as file messages instead of throwing.
    const offset = fm ? fm[0].split('\n').length - 1 : 0;
    for (const m of out.messages) problems.push(`${m.line ? `line ${m.line + offset}: ` : ''}${m.reason}`);
  } catch (e) {
    problems.push(`${e.line ? `line ${e.line + (fm ? fm[0].split('\n').length - 1 : 0)}: ` : ''}${e.reason ?? e.message}`);
  }
  if (problems.length) { failed = true; console.log(`✗ ${file}\n  - ${problems.join('\n  - ')}`); }
  else console.log(`✓ ${file}`);
}
process.exit(failed ? 1 : 0);
