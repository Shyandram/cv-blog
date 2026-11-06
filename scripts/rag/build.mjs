import { chapterPath } from '../../src/lib/chapter-paths.mjs';
// Builds a retrieval corpus (for RAG and for agents) from the course source: every tutorial (en and zh-tw) and blog post is split
// into section-sized chunks with their place in the course, the site URL with anchor (a note or paper guide: its own page,
// …/notes/<id>/ or …/papers/<id>/), and the source file and line range.
//
// Usage: node scripts/rag/build.mjs [--out .rag] [--max-words 700]
// Output: <out>/chunks.jsonl (one JSON object per line, schema in docs/rag.md) and <out>/manifest.json.
// Anchors come from docs/content-index.json (run `npm run index` first if headings changed); no network, no dependencies.
import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { join, relative } from 'node:path';

const root = new URL('../..', import.meta.url).pathname;
const arg = (name, dflt) => { const i = process.argv.indexOf(name); return i > 0 ? process.argv[i + 1] : dflt; };
const OUT = join(root, arg('--out', '.rag'));
const MAX_WORDS = Number(arg('--max-words', 700));
const ZONE = { 'supplementary notes': 'notes', '補充資料': 'notes', 'paper guides': 'papers', '論文導讀': 'papers', references: 'references', '參考文獻': 'references' };

const index = JSON.parse(readFileSync(join(root, 'docs/content-index.json'), 'utf8'));
const bySlug = Object.fromEntries(index.tutorials.map((t) => [t.slug, t]));

// Approximate size: words for Latin text, characters for CJK.
const size = (s) => (s.match(/[A-Za-z0-9_]+/g)?.length ?? 0) + (s.match(/[㐀-鿿]/g)?.length ?? 0) / 1.6;
const plain = (s) => s.replace(/\*\*|__|`/g, '').replace(/\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/\s+/g, ' ').trim();

function frontmatter(src) {
  const m = src.match(/^---\n([\s\S]*?)\n---\n/);
  const fm = {};
  if (m) for (const line of m[1].split('\n')) { const kv = line.match(/^(\w+):\s*(.*)$/); if (kv) fm[kv[1]] = kv[2].trim().replace(/^["']|["']$/g, ''); }
  return { fm, offset: m ? m[0].split('\n').length - 1 : 0, body: m ? src.slice(m[0].length) : src };
}

// Clean an MDX block for retrieval: drop imports and JSX tags but keep their text (captions, alt text, callout bodies).
function clean(text) {
  return text
    .replace(/^import\s.+$/gm, '')
    .replace(/\{\/\*[\s\S]*?\*\/\}/g, '')
    .replace(/<Figure[^>]*?alt="([^"]*)"[^>]*?caption="([^"]*)"[^>]*\/>/g, '[Figure: $2 — $1]')
    .replace(/<Figure[^>]*?caption="([^"]*)"[^>]*?alt="([^"]*)"[^>]*\/>/g, '[Figure: $1 — $2]')
    .replace(/<\/?[A-Z][A-Za-z]*[^>]*>/g, '')
    .replace(/<\/?(details|summary)>/g, '')
    .replace(/^(\s*(?:[-*+]|\d+\.) )\[(core|key|detail)\]\s*/gm, '$1') // outline priority markers (src/lib/rehypeOutline.mjs)
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function anchorFor(t, lang, heading) {
  if (!t) return null;
  const want = plain(heading);
  const s = t.sections.find((x) => plain(lang === 'en' ? x.text : x.textZh ?? '') === want);
  return s ? (lang === 'en' ? s.url : s.urlZh) : null;
}

function chunkFile(file, meta) {
  const src = readFileSync(file, 'utf8');
  const { fm, offset, body } = frontmatter(src);
  const lines = body.split('\n');
  const chunks = [];
  let h2 = '', h3 = '', zone = 'main', start = 0, buf = [], inFence = false;
  const pageUrl = meta.url;
  // Tutorials: the displayed title from the course tree (number + title text); posts: their own title.
  const title = meta.tutorial ? (meta.lang === 'en' ? meta.tutorial.title : meta.tutorial.titleZh || meta.tutorial.title) : fm.title ?? '';
  const flush = (end) => {
    const text = clean(buf.join('\n'));
    if (text && size(text) > 3) {
      const heading = h3 || h2;
      const pieces = [];
      // Split long sections at paragraph boundaries.
      let cur = [], curSize = 0;
      for (const para of text.split(/\n\s*\n/)) {
        const s = size(para);
        if (curSize + s > MAX_WORDS && cur.length) { pieces.push(cur.join('\n\n')); cur = []; curSize = 0; }
        cur.push(para); curSize += s;
      }
      if (cur.length) pieces.push(cur.join('\n\n'));
      const url = (heading && anchorFor(meta.tutorial, meta.lang, heading)) || (zone === 'notes' ? pageUrl + 'notes/' : zone === 'papers' ? pageUrl + 'papers/' : pageUrl);
      pieces.forEach((p, k) => chunks.push({
        id: `${meta.id}#${chunks.length}`, kind: meta.kind, lang: meta.lang, track: meta.track ?? null, slug: meta.slug ?? null,
        title, parent: meta.tutorial?.parent ?? null, section: [h2, h3].filter(Boolean).map(plain), zone, url,
        file: relative(root, file), lines: [start + offset + 1, end + offset], part: pieces.length > 1 ? `${k + 1}/${pieces.length}` : null,
        text: p,
      }));
    }
    buf = [];
  };
  lines.forEach((line, i) => {
    if (/^(```|~~~)/.test(line)) inFence = !inFence;
    const m = !inFence && line.match(/^(#{2,3})\s+(.*)$/);
    if (m) {
      flush(i);
      const text = m[2].trim();
      if (m[1] === '##') { h2 = text; h3 = ''; zone = ZONE[plain(text).toLowerCase()] ?? (zone === 'references' ? 'references' : 'main'); }
      else h3 = text;
      start = i;
    }
    buf.push(line);
  });
  flush(lines.length);
  // A first chunk with the tutorial's own summary helps routing questions to the right page.
  if (meta.kind === 'tutorial' && meta.tutorial) {
    const t = meta.tutorial;
    chunks.unshift({
      id: `${meta.id}#summary`, kind: 'summary', lang: meta.lang, track: t.track, slug: t.slug, title, parent: t.parent ?? null, section: [], zone: 'main',
      url: pageUrl, file: relative(root, file), lines: [1, offset], part: null,
      text: `${title}. ${fm.description ?? ''} Prerequisites: ${t.prereqs.join(', ') || 'none'}. Sections: ${t.sections.filter((s) => s.depth === 2 && s.zone === 'main').map((s) => (meta.lang === 'en' ? s.text : s.textZh ?? s.text)).join('; ')}.`,
    });
  }
  return chunks;
}

const all = [];
for (const t of index.tutorials) {
  for (const lang of ['en', 'zh-tw']) {
    const file = lang === 'en' ? join(root, t.files.en) : t.files.zh && join(root, t.files.zh);
    if (!file || !existsSync(file)) continue;
    all.push(...chunkFile(file, { id: `${lang}/${t.slug}`, kind: 'tutorial', lang, track: t.track, slug: t.slug, tutorial: t, url: chapterPath(`/${lang}/learn/${t.slug}/`) }));
  }
}
for (const lang of ['en', 'zh-tw']) {
  const dir = join(root, 'src/content/posts', lang);
  if (!existsSync(dir)) continue;
  for (const f of readdirSync(dir).filter((x) => /\.mdx?$/.test(x) && !x.startsWith('_')).sort()) {
    const slug = f.replace(/\.mdx?$/, '');
    all.push(...chunkFile(join(dir, f), { id: `${lang}/blog/${slug}`, kind: 'post', lang, url: `/${lang}/blog/${slug}/` }));
  }
}

mkdirSync(OUT, { recursive: true });
writeFileSync(join(OUT, 'chunks.jsonl'), all.map((c) => JSON.stringify(c)).join('\n') + '\n');
const manifest = {
  generatedBy: 'scripts/rag/build.mjs', schema: 'docs/rag.md', chunks: all.length, maxWords: MAX_WORDS,
  byLang: Object.fromEntries(['en', 'zh-tw'].map((l) => [l, all.filter((c) => c.lang === l).length])),
  byZone: Object.fromEntries(['main', 'notes', 'papers', 'references'].map((z) => [z, all.filter((c) => c.zone === z).length])),
  tutorials: index.tutorials.length,
};
writeFileSync(join(OUT, 'manifest.json'), JSON.stringify(manifest, null, 1) + '\n');
console.log(`Wrote ${all.length} chunks to ${relative(root, OUT)}/chunks.jsonl`);
