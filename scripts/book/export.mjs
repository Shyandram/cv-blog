import { chapterPath } from '../../src/lib/chapter-paths.mjs';
// Convert published MDX to a portable LaTeX book without executing its expressions.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { createProcessor } from '@mdx-js/mdx';
import { toMarkdown } from 'mdast-util-to-markdown';
import { mathToMarkdown } from 'mdast-util-math';
import { gfmToMarkdown } from 'mdast-util-gfm';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import remarkCjkFriendly from 'remark-cjk-friendly';
import { parse as yaml } from 'yaml';
import GithubSlugger from 'github-slugger';
import { volumes, stem as volumeStem } from './volumes.mjs';
import { curriculum } from '../../src/content/curriculum.ts';
import { placesOf, displayTitle } from '../../src/lib/curriculum.ts';

const root = path.resolve(import.meta.dirname, '../..');
const out = path.resolve(process.env.BOOK_OUTPUT || path.join(root, 'book-output'));
const pandoc = process.env.PANDOC || 'pandoc';
const fontDir = process.env.BOOK_FONTS;
// Pandoc 3.8 renamed --no-highlight to --syntax-highlighting=none; accept either.
const pandocVersion = execFileSync(pandoc, ['--version'], { encoding: 'utf8' }).match(/pandoc(?:\.exe)? (\d+)\.(\d+)/);
const noHighlight = pandocVersion && (Number(pandocVersion[1]) > 3 || (Number(pandocVersion[1]) === 3 && Number(pandocVersion[2]) >= 8)) ? ['--syntax-highlighting=none'] : ['--no-highlight'];
const langs = process.argv.slice(2).length ? process.argv.slice(2) : ['en', 'zh-tw'];
const tracks = ['foundations', 'image-processing', 'nlp', 'pattern-recognition', 'ml', 'deep-learning', 'optimization', 'information-theory'];
const names = {
 en: ['Foundations', 'Image Processing', 'Natural Language Processing', 'Pattern Recognition', 'Machine Learning', 'Deep Learning for Computer Vision', 'Optimization', 'Information Theory'],
 'zh-tw': ['基礎', '影像處理', '自然語言處理', '圖形識別', '機器學習', '電腦視覺深度學習', '最佳化', '資訊理論'],
};
const tex = s => String(s).replace(/[\\{}$&#%_~^]/g, c => ({'\\':'\\textbackslash{}','{':'\\{','}':'\\}','$':'\\$','&':'\\&','#':'\\#','%':'\\%','_':'\\_','~':'\\textasciitilde{}','^':'\\textasciicircum{}'}[c]));
const text = value => ({ type: 'text', value });
const para = children => ({ type: 'paragraph', children });
const strong = value => ({ type: 'strong', children: [text(value)] });
const SITE = 'https://blog.shyandram.dev';
const absolute = href => href.startsWith('/') ? `${SITE}${href}` : href;
// The course tree (src/content/curriculum.ts) orders the tutorials, nests sub-chapters under their main chapter and gives
// the displayed numbers. The optional sections (Supplementary notes, Paper guides) are online only: they are left out of
// the book, and main-text links to them point at their pages on the site.
const places = placesOf(curriculum);
const OPTIONAL = /^(supplementary notes|補充資料|paper guides|論文導讀)$/i;
const plain = node => node.type === 'text' || node.type === 'inlineCode' ? node.value : (node.children ?? []).map(plain).join('');
const label = (kind, slug) => `book-${kind}-${slug.replaceAll('/', '-')}`;
fs.mkdirSync(path.join(out, 'assets/fonts'), { recursive: true });
if (!fontDir) throw new Error('Set BOOK_FONTS to a directory containing the Noto CJK OTF files and licenses.');
for (const required of ['NotoSerifCJKtc-Regular.otf', 'NotoSerifCJKtc-Bold.otf', 'NotoSansCJKtc-Regular.otf', 'NotoSansCJKtc-Bold.otf', 'DejaVuSansMono.ttf']) {
 if (!fs.existsSync(path.join(fontDir, required))) throw new Error(`Missing required font: ${required}`);
}
for (const file of fs.readdirSync(fontDir)) fs.copyFileSync(path.join(fontDir, file), path.join(out, 'assets/fonts', file));
const revision = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim();
const date = process.env.BOOK_DATE || new Date().toISOString().slice(0, 10);
const manifest = { date, sourceCommit: revision, books: {} };

for (const lang of langs) {
 if (!names[lang]) throw new Error(`Unsupported language: ${lang}`);
 const zh = lang === 'zh-tw';
 const allEntries = [];
 for (const kind of ['tutorials', 'posts']) {
  const dir = path.join(root, 'src/content', kind, lang);
  for (const rel of fs.readdirSync(dir, { recursive: true }).filter(f => /\.(md|mdx)$/.test(f) && !path.basename(f).startsWith('_'))) {
   const file = path.join(dir, rel), source = fs.readFileSync(file, 'utf8');
   const fm = source.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/);
   if (!fm) throw new Error(`Missing frontmatter: ${file}`);
   const data = yaml(fm[1]);
   if (data.draft) continue;
   const slug = rel.replace(/\.(md|mdx)$/, '');
   const place = kind === 'tutorials' ? places.get(slug) : undefined;
   if (kind === 'tutorials' && !place) throw new Error(`${slug} is not in src/content/curriculum.ts (run node scripts/curriculum.mjs)`);
   if (place) data.title = displayTitle(data.title, place.number, lang);
   allEntries.push({ file, rel, kind, slug, data, place, body: source.slice(fm[0].length) });
  }
 }
 allEntries.sort((a, b) => (a.kind === b.kind ? 0 : a.kind === 'tutorials' ? -1 : 1) ||
  (a.kind === 'tutorials' ? a.place.index - b.place.index : new Date(a.data.published) - new Date(b.data.published)) || a.slug.localeCompare(b.slug));
 for (const volume of volumes) {
  const entries = allEntries.filter(e => e.kind === 'posts' ? volume.posts : volume.tracks.includes(e.data.track));
  const stats = { tutorials: entries.filter(e => e.kind === 'tutorials').length, posts: entries.filter(e => e.kind === 'posts').length, figures: 0, comparisons: 0, interactiveLinks: 0, codeBlocks: 0, displayEquations: 0, chapters: [] };
  const routeLabels = new Map(entries.map(e => [`/${lang}/${e.kind === 'tutorials' ? 'learn' : 'blog'}/${e.slug}/`, label(e.kind, e.slug)]));
  let lastPart;
  const content = [];
  for (const entry of entries) {
   const { file, data, kind, slug } = entry;
   const current = kind === 'posts' ? 'posts' : data.track;
   if (current !== lastPart) {
    const title = current === 'posts' ? (zh ? '部落格文章' : 'Blog Essays') : names[lang][tracks.indexOf(current)];
    content.push(`\\part{${tex(title)}}`); lastPart = current;
   }
   const online = chapterPath(`https://blog.shyandram.dev/${lang}/${kind === 'tutorials' ? 'learn' : 'blog'}/${slug}/`);
   // A main chapter is a book chapter; a sub-chapter (tree depth 2, 3, …) is a section (subsection, …) under it in the
   // contents and the PDF bookmarks, and its own headings move down by as many levels.
   const depth = entry.place?.depth ?? 1;
   const shift = Math.min(depth - 1, 2);
   const opening = depth === 1 ? `\\bookchapter{${tex(data.title)}}{${label(kind, slug)}}`
     : `\\booksubchapter{${tex(data.title)}}{${label(kind, slug)}}{${['section', 'subsection', 'subsubsection'][Math.min(depth, 4) - 2]}}{${Math.min(depth, 4) - 1}}`;
   content.push(`${opening}\n\n${tex(data.description)}\n\n\\booksource{${zh ? '網站版本' : 'Online edition'}}{${online}}\n\n`);
   if (data.notebook) content.push(`\\booksource{${zh ? '筆記本' : 'Notebook'}}{https://github.com/Shyandram/cv-blog/blob/${revision}/${data.notebook}}\n\n`);
   const parser = createProcessor({ format: file.endsWith('.mdx') ? 'mdx' : 'md', remarkPlugins: [remarkMath, remarkGfm, remarkCjkFriendly] });
   const ast = parser.parse(entry.body), imports = new Map();
   // Leave out the optional sections; remember which page on the site holds each of their ids (an item is a ### and
   // everything under it; the ids follow the site's heading slugs).
   const onlineOnly = new Map();
   {
    const slugger = new GithubSlugger();
    let zone = null, item = null;
    const kept = [];
    for (const node of ast.children) {
     if (node.type === 'heading') {
      const text = plain(node).trim(), id = slugger.slug(text);
      if (node.depth === 2) { const m = text.match(OPTIONAL); zone = m ? (/^(paper guides|論文導讀)$/i.test(text) ? 'papers' : 'notes') : null; item = null; if (zone) { onlineOnly.set(id, { title: text, url: `${online}${zone}/` }); continue; } }
      else if (zone && node.depth === 3) item = { title: text, url: `${online}${zone}/${id}/` };
      if (zone && item) onlineOnly.set(id, node.depth === 3 ? item : { title: text, url: `${item.url}#${id}` });
     }
     if (!zone) kept.push(node);
    }
    ast.children = kept;
   }
   // A main-text pointer line to an optional item ("*Optional: [X](#x) (in the Supplementary notes). …*") becomes
   // "Online only: X — <URL of its page>"; any other link to such an item keeps its text and shows the URL.
   const isPointer = p => p.type === 'paragraph' && p.children.length === 1 && p.children[0].type === 'emphasis' && /^\s*(Optional:|選讀：)/.test(plain(p.children[0]));
   const targets = n => n.type === 'link' && n.url.startsWith('#') && onlineOnly.has(decodeURIComponent(n.url.slice(1))) ? [onlineOnly.get(decodeURIComponent(n.url.slice(1)))] : (n.children ?? []).flatMap(targets);
   const fixLinks = n => {
    if (!n.children) return;
    n.children = n.children.flatMap(c => {
     if (isPointer(c)) {
      const found = targets(c);
      if (found.length) return [para([{ type: 'emphasis', children: [text(zh ? '僅線上版：' : 'Online only: '), ...found.flatMap((x, i) => [...(i ? [text(zh ? '；' : '; ')] : []), text(`${x.title} — `), { type: 'link', url: x.url, children: [text(x.url)] }])] }])];
     }
     if (c.type === 'link' && c.url.startsWith('#') && onlineOnly.has(decodeURIComponent(c.url.slice(1)))) {
      const x = onlineOnly.get(decodeURIComponent(c.url.slice(1)));
      return [...c.children, text(' ('), { type: 'link', url: x.url, children: [text(x.url)] }, text(')')];
     }
     fixLinks(c);
     return [c];
    });
   };
   fixLinks(ast);
   for (const node of ast.children.filter(n => n.type === 'mdxjsEsm')) {
    for (const match of node.value.matchAll(/import\s+(\w+)\s+from\s+['"]([^'"]+)['"]/g)) imports.set(match[1], path.resolve(path.dirname(file), match[2]));
   }
   function asset(attr) {
    const source = typeof attr === 'string' ? path.resolve(path.dirname(file), attr) : imports.get(attr?.value);
    if (!source || !source.startsWith(path.join(root, 'src/assets/') )) throw new Error(`Unresolved image in ${file}: ${JSON.stringify(attr)}`);
    const dest = path.join('assets/figures', path.relative(path.join(root, 'src/assets'), source));
    fs.mkdirSync(path.dirname(path.join(out, dest)), { recursive: true });
    fs.copyFileSync(source, path.join(out, dest));
    return dest;
   }
   function figure(src, caption) {
    return [para([{ type: 'image', url: asset(src), alt: '', title: null }]), para([{ type: 'emphasis', children: [text(caption || '')] }])];
   }
   function convert(node) {
    if (node.type === 'mdxjsEsm') return [];
    if (node.type === 'code') stats.codeBlocks++;
    if (node.type === 'math') stats.displayEquations++;
    if (node.type === 'mdxJsxFlowElement' || node.type === 'mdxJsxTextElement') {
     const attrs = Object.fromEntries(node.attributes.map(a => [a.name, a.value]));
     const children = node.children.flatMap(convert);
     if (node.name === 'strong') return [{ type: 'strong', children }];
     if (node.name === 'Callout') return [{ type: 'blockquote', children: [para([strong(attrs.title || (zh ? '提示' : 'Note'))]), ...children] }];
     if (node.name === 'details') return [{ type: 'blockquote', children }];
     if (node.name === 'summary') return [para([{ type: 'strong', children }])];
     if (node.name === 'Figure') { stats.figures++; return figure(attrs.src, attrs.caption || attrs.alt); }
     if (node.name === 'ImageCompare' && attrs.before && attrs.after) {
      stats.comparisons++;
      return [...figure(attrs.before, attrs.beforeLabel), ...figure(attrs.after, attrs.afterLabel), para([text(attrs.alt || '')])];
     }
     if (node.name === 'KernelPlayground' || node.name === 'ImageCompare' || node.name === 'Lab') {
      stats.interactiveLinks++;
      return [{ type: 'blockquote', children: [para([strong(zh ? '互動示範（網站版）' : 'Interactive demonstration (online)'), text(zh ? '：請開啟本篇教學的網站版本，操作此示範。' : ': open the online edition of this tutorial to use this demonstration.'), { type: 'link', url: online, children: [text(zh ? '開啟示範' : 'Open demonstration')] }])] }];
     }
     throw new Error(`Unsupported component ${node.name} in ${file}`);
    }
    if (node.type === 'heading') return [{ ...node, depth: Math.min(6, Math.max(1, node.depth - 1) + shift), children: node.children.flatMap(convert) }];
    if (node.type === 'link') {
     const base = node.url.split('#')[0];
     // Chapter links stay inside the PDF; section links keep the precise web destination.
     const href = routeLabels.has(base) && !node.url.includes('#') ? `#${routeLabels.get(base)}` : node.url.startsWith('#') ? online + node.url : absolute(node.url);
     return [{ ...node, url: href, children: node.children.flatMap(convert) }];
    }
    if (node.type.startsWith('mdx')) throw new Error(`Unsupported MDX node ${node.type} in ${file}`);
    return [{ ...node, ...(node.children ? { children: node.children.flatMap(convert) } : {}) }];
   }
   const converted = { type: 'root', children: ast.children.flatMap(convert) };
   content.push(toMarkdown(converted, { extensions: [mathToMarkdown(), gfmToMarkdown()], bullet: '-', fences: true, emphasis: '*', rule: '-' }));
   stats.chapters.push({ title: data.title, source: path.relative(root, file), url: online, label: label(kind, slug) });
  }
  const stem = volumeStem(lang, volume), number = volumes.indexOf(volume) + 1;
  fs.writeFileSync(path.join(out, `${stem}.md`), content.join('\n\n'));
  const metadata = { booktitle: 'Pixels → Perception', edition: `${zh ? '繁體中文版' : 'English Edition'} · ${zh ? `第 ${number} 冊` : `Volume ${number}`}`, volumeNumber: number, volumeCount: volumes.length, volumeTitle: volume.title[lang], subtitle: zh ? '從像素到理解：視覺、語言與學習' : 'From Pixels to Understanding', tagline: zh ? '基礎・影像・語言・學習・理論' : 'Foundations · Vision · Language · Learning · Theory', date, revision, zh, tutorials: stats.tutorials, posts: stats.posts };
  fs.writeFileSync(path.join(out, `${stem}.json`), JSON.stringify(metadata, null, 2));
  execFileSync(pandoc, [path.join(out, `${stem}.md`), '--from=markdown+raw_tex-implicit_figures', '--to=latex', '--standalone', '--top-level-division=section', ...noHighlight, `--template=${path.join(root, 'scripts/book/template.tex')}`, `--lua-filter=${path.join(root, 'scripts/book/layout.lua')}`, `--metadata-file=${path.join(out, `${stem}.json`)}`, '-o', path.join(out, `${stem}.tex`)], { cwd: out, stdio: 'inherit' });
  manifest.books[stem] = { lang, volume: volume.id, number, title: volume.title, tracks: volume.tracks, ...stats };
  console.log(`${stem}: ${stats.tutorials} tutorials, ${stats.posts} posts, ${stats.figures} figures, ${stats.codeBlocks} code blocks`);
 }
}
fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 2));
console.log(`LaTeX sources written to ${out}`);
