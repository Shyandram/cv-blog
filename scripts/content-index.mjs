import { chapterPath } from '../src/lib/chapter-paths.mjs';
// Builds a repository-side map of the course for people and research agents who work on this repo: which tutorial covers
// what, where its file, figures, script, notes, paper guides and references are, and the exact anchors the site generates.
//
// Usage:
//   node scripts/content-index.mjs           write docs/content-map/ (README, one page per track, topics) and docs/content-index.json
//   node scripts/content-index.mjs --check   exit 1 if those files are out of date (used by CI)
//
// Anchors are computed with the same unified pipeline the site uses (remark-math → rehype-katex → Astro's rehypeHeadingIds),
// so headings that contain math get the same slugs as on the built site. Each ### after "## Supplementary notes" / "## 補充資料"
// is a page of its own at …/<slug>/notes/<h3-id>/ (index …/<slug>/notes/), each ### after "## Paper guides" / "## 論文導讀" at
// …/<slug>/papers/<h3-id>/; an id inside such an item lives on that item's page (…/notes/<h3-id>/#<id>).
import { compile } from '@mdx-js/mdx';
import remarkMath from 'remark-math';
import remarkCjkFriendly from 'remark-cjk-friendly';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import { rehypeHeadingIds } from '@astrojs/markdown-remark';
import rehypePaperHeadingIds from '../src/lib/rehypePaperHeadingIds.mjs';
import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { join, relative } from 'node:path';
import { curriculum } from '../src/content/curriculum.ts';
import { placesOf, displayTitle, validateCurriculum } from '../src/lib/curriculum.ts';

const root = new URL('..', import.meta.url).pathname;
const CHECK = process.argv.includes('--check');
const TRACKS = ['foundations', 'image-processing', 'nlp', 'pattern-recognition', 'ml', 'deep-learning', 'optimization', 'information-theory'];
const TRACK_NAME = {
  foundations: 'Foundations', 'image-processing': 'Image Processing (IP)', nlp: 'Natural Language Processing (NLP)',
  'pattern-recognition': 'Pattern Recognition (PR)', ml: 'Machine Learning (ML)', 'deep-learning': 'Deep Learning for Computer Vision (DLCV)',
  optimization: 'Optimization (OPT)', 'information-theory': 'Information Theory (IT)',
};
const ZONE = { 'supplementary notes': 'notes', '補充資料': 'notes', 'paper guides': 'papers', '論文導讀': 'papers', references: 'refs', '參考文獻': 'refs' };

const rel = (p) => relative(root, p);
const textOf = (n) => (n.type === 'text' ? n.value : (n.children ?? []).map(textOf).join(''));

function frontmatter(src) {
  const m = src.match(/^---\n([\s\S]*?)\n---\n/);
  const fm = {};
  if (!m) return { fm, body: src };
  for (const line of m[1].split('\n')) {
    const kv = line.match(/^(\w+):\s*(.*)$/);
    if (!kv) continue;
    let v = kv[2].trim();
    if (v.startsWith('[')) v = v.slice(1, -1).split(',').map((s) => s.trim().replace(/^["']|["']$/g, '')).filter(Boolean);
    else v = v.replace(/^["']|["']$/g, '');
    fm[kv[1]] = v;
  }
  return { fm, body: src.slice(m[0].length) };
}

// Headings with the ids the site generates, plus the zone each heading belongs to and, inside the notes and paper guides,
// the item (### id) whose page holds it.
async function headings(path, body) {
  const out = [];
  const collect = () => (tree) => {
    let zone = 'main';
    let item = '';
    const walk = (n) => {
      if (n.type === 'element' && /^h[1-6]$/.test(n.tagName)) {
        const depth = Number(n.tagName[1]);
        const text = textOf(n).replace(/\s+/g, ' ').trim();
        const id = n.properties?.id ?? '';
        if (depth === 2) { zone = ZONE[text.toLowerCase()] ?? (zone === 'refs' ? 'refs' : 'main'); item = ''; }
        if (depth === 3 && (zone === 'notes' || zone === 'papers')) item = id;
        out.push({ depth, text: plainHeading(n), id, zone, item: depth >= 3 ? item : '' });
      }
      for (const c of n.children ?? []) walk(c);
    };
    walk(tree);
  };
  await compile({ value: body, path }, {
    format: path.endsWith('.md') ? 'md' : 'mdx',
    remarkPlugins: [remarkGfm, remarkMath, remarkCjkFriendly],
    rehypePlugins: [[rehypeKatex, { strict: false }], rehypePaperHeadingIds, rehypeHeadingIds, collect],
  });
  return out;
}
// Heading text as written (math shown as its TeX source instead of KaTeX's tripled text).
function plainHeading(n) {
  const walk = (x) => {
    if (x.type === 'text') return x.value;
    const cls = x.properties?.className ?? [];
    if (Array.isArray(cls) && cls.includes('katex')) {
      const ann = findAnnotation(x);
      return ann ? `$${ann}$` : '';
    }
    return (x.children ?? []).map(walk).join('');
  };
  return walk(n).replace(/\s+/g, ' ').trim();
}
function findAnnotation(x) {
  if (x.type === 'element' && x.tagName === 'annotation') return textOf(x);
  for (const c of x.children ?? []) { const r = findAnnotation(c); if (r) return r; }
  return '';
}

function references(body) {
  const i = body.search(/^## (References|參考文獻)\s*$/m);
  if (i < 0) return [];
  return [...body.slice(i).matchAll(/^(\d+)\.\s+(.*)$/gm)].map((m) => {
    const url = m[2].match(/\]\((https?:[^)\s]+)\)/)?.[1] ?? '';
    return { n: Number(m[1]), text: m[2].replace(/\s*\[[^\]]*\]\([^)]*\)/g, '').trim(), url };
  });
}

function coverage() {
  const src = readFileSync(join(root, 'src/lib/playground/coverage.ts'), 'utf8');
  const plan = {};
  for (const m of src.matchAll(/'([a-z-]+\/[a-z0-9-]+)':\s*\[([^\]]*)\]/g)) plan[m[1]] = [...m[2].matchAll(/'([^']+)'/g)].map((x) => x[1]);
  return plan;
}

// The course tree (src/content/curriculum.ts) gives the order, the nesting and the displayed numbers.
const places = placesOf(curriculum);
{
  const files = [];
  for (const lang of ['en', 'zh-tw']) for (const track of TRACKS) {
    const dir = join(root, 'src/content/tutorials', lang, track);
    if (existsSync(dir)) for (const f of readdirSync(dir).filter((x) => /\.mdx?$/.test(x) && !x.startsWith('_'))) files.push({ lang, slug: `${track}/${f.replace(/\.mdx?$/, '')}`, track: frontmatter(readFileSync(join(dir, f), 'utf8')).fm.track ?? track });
  }
  const errors = validateCurriculum(curriculum, files);
  if (errors.length) { console.error(`The course tree is not valid:\n- ${errors.join('\n- ')}`); process.exit(1); }
}

const plan = coverage();
const tutorials = [];
for (const track of TRACKS) {
  const dir = join(root, 'src/content/tutorials/en', track);
  if (!existsSync(dir)) continue;
  for (const name of readdirSync(dir).filter((f) => f.endsWith('.mdx') && !f.startsWith('_')).sort()) {
    const slug = `${track}/${name.replace(/\.mdx$/, '')}`;
    const en = join(dir, name);
    const zh = join(root, 'src/content/tutorials/zh-tw', track, name);
    const srcEn = readFileSync(en, 'utf8');
    const { fm, body } = frontmatter(srcEn);
    const hs = await headings(en, body);
    const place = places.get(slug);
    let zhTitle = '';
    let hsZh = [];
    if (existsSync(zh)) {
      const z = frontmatter(readFileSync(zh, 'utf8'));
      zhTitle = displayTitle(z.fm.title ?? '', place.number, 'zh-tw');
      hsZh = await headings(zh, z.body);
    }
    const figures = [...srcEn.matchAll(/^import\s+\w+\s+from\s+['"]([^'"]+\.(?:png|jpe?g|webp|svg))['"]/gm)].map((m) => m[1].replace(/^(\.\.\/)+/, 'src/'));
    const script = `scripts/figures/${name.replace(/\.mdx$/, '').replace(/-/g, '_')}.py`;
    // Main text: …/<slug>/#id. Notes and paper guides: the section index …/notes/, an item's page …/notes/<item>/, or an id
    // inside an item …/notes/<item>/#id.
    const url = (lang, h) => {
      const base = chapterPath(`/${lang}/learn/${slug}/`);
      if (h.zone !== 'notes' && h.zone !== 'papers') return `${base}#${h.id}`;
      if (!h.item) return `${base}${h.zone}/`;
      return `${base}${h.zone}/${h.item}/${h.id === h.item ? '' : `#${h.id}`}`;
    };
    tutorials.push({
      slug, track, number: place.number, depth: place.depth, parent: place.parent, children: place.children, position: place.index,
      level: fm.level ?? '', title: displayTitle(fm.title ?? '', place.number, 'en'), titleZh: zhTitle,
      description: fm.description ?? '', prereqs: fm.prereqs ?? [], tags: fm.tags ?? [], published: fm.published ?? '', updated: fm.updated ?? '',
      files: { en: rel(en), zh: existsSync(zh) ? rel(zh) : null, figureScript: existsSync(join(root, script)) ? script : null, figures },
      pages: { main: chapterPath(`/en/learn/${slug}/`), notes: hs.some((h) => h.zone === 'notes') ? chapterPath(`/en/learn/${slug}/notes/`) : null, papers: hs.some((h) => h.zone === 'papers') ? chapterPath(`/en/learn/${slug}/papers/`) : null },
      experiments: plan[slug] ?? [],
      sections: hs.filter((h) => h.depth >= 2 && h.depth <= 3 && h.zone !== 'refs').map((h, i) => ({ depth: h.depth, text: h.text, zone: h.zone, url: url('en', h), urlZh: hsZh[hs.indexOf(h)] ? url('zh-tw', hsZh[hs.indexOf(h)]) : null, textZh: hsZh[hs.indexOf(h)]?.text ?? null })),
      references: references(body),
    });
  }
}
tutorials.sort((a, b) => a.position - b.position);
const bySlug = Object.fromEntries(tutorials.map((t) => [t.slug, t]));
const short = (t) => t.title.split(/ · |・/)[0];
// What the map prints next to a section link: its anchor, or for a note or paper guide its page (notes/<item-id>/).
const anchor = (s) => (s.url.includes('#') ? s.url.split('#')[1] : s.url.replace(/^.*\/((?:notes|papers)\/(?:[^/]+\/)?)$/, '$1'));

const posts = [];
for (const lang of ['en', 'zh-tw']) {
  const dir = join(root, 'src/content/posts', lang);
  if (!existsSync(dir)) continue;
  for (const f of readdirSync(dir).filter((x) => /\.mdx?$/.test(x) && !x.startsWith('_')).sort()) {
    const { fm } = frontmatter(readFileSync(join(dir, f), 'utf8'));
    posts.push({ lang, file: rel(join(dir, f)), title: fm.title ?? f, published: fm.published ?? '' });
  }
}

// ---------- Markdown output ----------
const files = {};
const HEADER = '<!-- Generated by scripts/content-index.mjs. Do not edit by hand: run `node scripts/content-index.mjs`. -->\n\n';
const row = (cells) => `| ${cells.map((c) => String(c).replace(/\|/g, '\\|')).join(' | ')} |`;

let readme = HEADER + `# Content map (for contributors and research agents)

A generated, repository-side guide to the course. Use it to find **which file and which section** covers a topic before you read or
edit anything. Everything here is derived from the source files; regenerate with \`node scripts/content-index.mjs\` after adding or
renaming a tutorial or heading (CI runs it with \`--check\`). Rules for editing live in [AGENTS.md](../../AGENTS.md).

## How to look things up fast

- **By topic:** search [topics.md](topics.md) (every \`##\`/\`###\` heading of every tutorial, alphabetical, with the file and the exact
  site anchor), or \`grep -i "<term>" docs/content-map/*.md\`.
- **By tutorial:** open the track page below; each tutorial lists its files, prerequisites, sections, Supplementary notes, Paper
  guides, figures, figure script and playground experiments.
- **Machine-readable:** [../content-index.json](../content-index.json) has the same data plus every reference entry (number, text,
  URL) — e.g. \`node -e "const d=require('./docs/content-index.json');console.log(d.tutorials.filter(t=>t.references.some(r=>/ArcFace/.test(r.text))).map(t=>t.slug))"\`.
- **By question (RAG):** \`npm run rag -- "how is PQ computed"\` ranks section chunks (BM25, English and Chinese) and prints the
  file, line range and anchor of each; \`--json\` for tools. Format and options: [../rag.md](../rag.md).
- **Course tree:** [src/content/curriculum.ts](../../src/content/curriculum.ts) is the single source of truth for the order, the
  nesting (main chapter → sub-chapter → …) and the numbers of the tutorials. Numbers come from the position in the tree
  (\`DLCV 04\`, its children \`DLCV 04A\`, \`DLCV 04B\`, the next level \`DLCV 04A.1\`); the displayed title is that number plus the
  frontmatter title without its old prefix. To add a chapter, add \`{ slug: '<track>/<file>' }\` where it belongs (or run
  \`node scripts/curriculum.mjs\`, which adds new files by the letter rule); to move one, move its entry. The build and
  \`node scripts/curriculum.mjs --check\` fail when a tutorial is missing from the tree. In this map, sub-chapters are nested under
  their main chapter; \`content-index.json\` gives each tutorial's \`number\`, \`depth\`, \`parent\` and \`children\`.
- **Anchors and pages:** links between pages must use the URLs shown here. Each item (\`###\`) of a Supplementary-notes section
  is a page of its own at \`…/<slug>/notes/<item-id>/\` (the section's index is \`…/<slug>/notes/\`), each Paper guide at
  \`…/<slug>/papers/<item-id>/\`; an id inside an item is \`…/notes/<item-id>/#<id>\`. Headings containing math have KaTeX-derived
  slugs (e.g. \`rank-kkk\`); copy them from this map rather than guessing.

## Tracks

${row(['Track', 'Tutorials', 'Map page'])}
${row(['---', '---', '---'])}
${TRACKS.map((tr) => row([TRACK_NAME[tr], tutorials.filter((t) => t.track === tr).length, `[${tr}.md](${tr}.md)`])).join('\n')}

Total: ${tutorials.length} tutorials (${tutorials.filter((t) => t.files.zh).length} with a zh-tw twin), ${posts.filter((p) => p.lang === 'en').length} English and ${posts.filter((p) => p.lang === 'zh-tw').length} zh-tw blog posts.

## Where things are in the repo

${row(['Path', 'What'])}
${row(['---', '---'])}
${[
  ['src/content/tutorials/<lang>/<track>/<slug>.mdx', 'tutorials (en and zh-tw twins share the slug); files starting with `_` are drafts'],
  ['src/content/posts/<lang>/', 'blog posts (the owner\'s own writing; do not edit their text)'],
  ['src/assets/<track>/<slug>/', 'generated figures (PNG), one folder per tutorial'],
  ['scripts/figures/<slug_with_underscores>.py', 'figure generators'],
  ['src/lib/playground/coverage.ts', 'which interactive experiment belongs to which chapter (`chapterPlan`, `readingOnlyChapters`)'],
  ['src/lib/playground/labs/, catalog.ts, chapters-*.ts', 'interactive experiments (image labs and SVG demos)'],
  ['src/content/curriculum.ts, src/lib/curriculum.ts, scripts/curriculum.mjs', 'the course tree (order, nesting, numbers), its logic, and the script that adds new files to it'],
  ['src/lib/rehypeSupplementZones.mjs, src/components/zones/', 'splits each tutorial into its main page and one page per note and paper guide (plus an index page for each section)'],
  ['src/lib/rehypeCitations.mjs', 'turns [n] citations into links to the reference list'],
  ['src/components/', 'MDX components (Callout, Figure, ImageCompare, KernelPlayground, Lab, …) and page widgets'],
  ['src/pages/[lang]/', 'routes: home, learn (+ [...slug]: main page, notes/papers index pages and item pages), blog, playground, search, download, about'],
  ['src/i18n/ui.ts, src/lib/site.ts, src/lib/taxonomy.ts', 'UI strings, site helpers, track list'],
  ['scripts/check-mdx.mjs, check-playground.mjs, tests/', 'validators (see AGENTS.md section 6)'],
  ['scripts/book/', 'PDF book export (four volumes per language)'],
  ['scripts/rag/, docs/rag.md', 'local retrieval interface (chunked corpus + BM25 search) for agents and RAG pipelines'],
  ['docs/', 'design notes, this content map and content-index.json'],
].map(row).join('\n')}

## Blog posts

${posts.map((p) => `- \`${p.file}\` — ${p.title}${p.published ? ` (${p.published})` : ''}`).join('\n')}
`;
files['docs/content-map/README.md'] = readme;

for (const tr of TRACKS) {
  const list = tutorials.filter((t) => t.track === tr);
  let md = HEADER + `# ${TRACK_NAME[tr]}\n\n[← Content map](README.md)\n\n`;
  // Sub-chapters are nested under their main chapter (course tree).
  const indent = (t) => (t.depth > 1 ? `${'&emsp;'.repeat(Math.min(t.depth - 1, 3))}↳ ` : '');
  md += `${row(['#', 'Tutorial', 'Level', 'Needs', 'File'])}\n${row(['---', '---', '---', '---', '---'])}\n`;
  md += list.map((t) => row([t.number || '—', `${indent(t)}[${t.title}](#${t.slug.split('/')[1]})`, t.level, t.prereqs.map((p) => (bySlug[p] ? short(bySlug[p]) : p)).join(', ') || '—', `\`${t.files.en}\``])).join('\n') + '\n\n';
  for (const t of list) {
    md += `## ${t.slug.split('/')[1]}\n\n**${t.title}**${t.titleZh ? ` · ${t.titleZh}` : ''}  \n${t.description}\n\n`;
    md += `- Files: \`${t.files.en}\`${t.files.zh ? `, \`${t.files.zh}\`` : ' (no zh-tw twin)'}\n`;
    md += `- Pages: \`${t.pages.main}\`${t.pages.notes ? `, \`${t.pages.notes}\`` : ''}${t.pages.papers ? `, \`${t.pages.papers}\`` : ''}\n`;
    if (t.parent || t.children.length) md += `- Course tree: ${t.parent ? `under \`${t.parent}\`` : 'main chapter'}${t.children.length ? ` · sub-chapters: ${t.children.map((c) => `\`${c}\``).join(', ')}` : ''}\n`;
    md += `- Level ${t.level} · depth ${t.depth} · needs: ${t.prereqs.length ? t.prereqs.map((p) => `\`${p}\``).join(', ') : 'none'}${t.tags.length ? ` · tags: ${t.tags.join(', ')}` : ''}\n`;
    md += `- Figures: ${t.files.figures.length}${t.files.figureScript ? ` (script \`${t.files.figureScript}\`)` : ''} · references: ${t.references.length}${t.experiments.length ? ` · experiments: ${t.experiments.join(', ')}` : ''}\n\n`;
    for (const zone of ['main', 'notes', 'papers']) {
      const ss = t.sections.filter((s) => s.zone === zone && !(zone !== 'main' && s.depth === 2));
      if (!ss.length) continue;
      if (zone !== 'main') md += `*${zone === 'notes' ? 'Supplementary notes' : 'Paper guides'} (one page each, index \`${zone}/\`):*\n\n`;
      md += ss.map((s) => `${s.depth === 3 && zone === 'main' ? '  ' : ''}- [${s.text}](${s.url}) \`${anchor(s)}\``).join('\n') + '\n\n';
    }
  }
  files[`docs/content-map/${tr}.md`] = md;
}

const topics = tutorials.flatMap((t) => t.sections.filter((s) => !(s.zone !== 'main' && s.depth === 2)).map((s) => ({ s, t })))
  .filter(({ s }) => !/^(what you'll learn|the big picture|modern view|key takeaways|exercises|references|supplementary notes|paper guides)$/i.test(s.text))
  .sort((a, b) => a.s.text.localeCompare(b.s.text, 'en', { sensitivity: 'base' }));
let tmd = HEADER + `# Topic index\n\n[← Content map](README.md) · every section heading (main text, Supplementary notes and Paper guides), alphabetical.\n\n`;
let letter = '';
for (const { s, t } of topics) {
  const L = (s.text.match(/[A-Za-z0-9]/)?.[0] ?? '#').toUpperCase();
  if (L !== letter) { letter = L; tmd += `\n## ${L}\n\n`; }
  tmd += `- ${s.text} — ${short(t)}${s.zone !== 'main' ? ` (${s.zone})` : ''} · \`${t.files.en}\` · \`${s.url}\`\n`;
}
files['docs/content-map/topics.md'] = tmd;
files['docs/content-index.json'] = JSON.stringify({ generatedBy: 'scripts/content-index.mjs', tutorials, posts }, null, 1) + '\n';

let stale = [];
for (const [p, content] of Object.entries(files)) {
  const full = join(root, p);
  if (CHECK) { if (!existsSync(full) || readFileSync(full, 'utf8') !== content) stale.push(p); continue; }
  mkdirSync(join(full, '..'), { recursive: true });
  writeFileSync(full, content);
}
if (CHECK) {
  if (stale.length) { console.error(`Content map is out of date: ${stale.join(', ')}\nRun: node scripts/content-index.mjs`); process.exit(1); }
  console.log('Content map is up to date.');
} else console.log(`Wrote ${Object.keys(files).length} files for ${tutorials.length} tutorials.`);
