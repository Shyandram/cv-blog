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
const absolute = href => href.startsWith('/') ? `https://blog.shyandram.dev${href}` : href;
const label = (kind, slug) => `book-${kind}-${slug.replaceAll('/', '-')}`;
fs.mkdirSync(path.join(out, 'assets/fonts'), { recursive: true });
if (!fontDir) throw new Error('Set BOOK_FONTS to a directory containing the Noto CJK OTF files and licenses.');
for (const required of ['NotoSerifCJKtc-Regular.otf', 'NotoSerifCJKtc-Bold.otf', 'NotoSansCJKtc-Regular.otf', 'NotoSansCJKtc-Bold.otf', 'DejaVuSansMono.ttf']) {
 if (!fs.existsSync(path.join(fontDir, required))) throw new Error(`Missing required font: ${required}`);
}
for (const file of fs.readdirSync(fontDir)) fs.copyFileSync(path.join(fontDir, file), path.join(out, 'assets/fonts', file));
const revision = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim();
const date = process.env.BOOK_DATE || new Date().toISOString().slice(0, 10);
const manifest = { date, sourceCommit: revision, editions: {} };

for (const lang of langs) {
 if (!names[lang]) throw new Error(`Unsupported language: ${lang}`);
 const zh = lang === 'zh-tw';
 const entries = [];
 for (const kind of ['tutorials', 'posts']) {
  const dir = path.join(root, 'src/content', kind, lang);
  for (const rel of fs.readdirSync(dir, { recursive: true }).filter(f => /\.(md|mdx)$/.test(f) && !path.basename(f).startsWith('_'))) {
   const file = path.join(dir, rel), source = fs.readFileSync(file, 'utf8');
   const fm = source.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/);
   if (!fm) throw new Error(`Missing frontmatter: ${file}`);
   const data = yaml(fm[1]);
   if (data.draft) continue;
   const slug = rel.replace(/\.(md|mdx)$/, '');
   entries.push({ file, rel, kind, slug, data, body: source.slice(fm[0].length) });
  }
 }
 entries.sort((a, b) => (a.kind === b.kind ? 0 : a.kind === 'tutorials' ? -1 : 1) ||
  (a.kind === 'tutorials' ? tracks.indexOf(a.data.track) - tracks.indexOf(b.data.track) || a.data.order - b.data.order : new Date(a.data.published) - new Date(b.data.published)) || a.slug.localeCompare(b.slug));
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
  const online = `https://blog.shyandram.dev/${lang}/${kind === 'tutorials' ? 'learn' : 'blog'}/${slug}/`;
  content.push(`\\bookchapter{${tex(data.title)}}{${label(kind, slug)}}\n\n${tex(data.description)}\n\n\\booksource{${zh ? '網站版本' : 'Online edition'}}{${online}}\n\n`);
  if (data.notebook) content.push(`\\booksource{${zh ? '筆記本' : 'Notebook'}}{https://github.com/Shyandram/cv-blog/blob/${revision}/${data.notebook}}\n\n`);
  const parser = createProcessor({ format: file.endsWith('.mdx') ? 'mdx' : 'md', remarkPlugins: [remarkMath, remarkGfm, remarkCjkFriendly] });
  const ast = parser.parse(entry.body), imports = new Map();
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
   if (node.type === 'heading') return [{ ...node, depth: Math.max(1, node.depth - 1), children: node.children.flatMap(convert) }];
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
 const stem = `pixels-perception-${lang}`;
 fs.writeFileSync(path.join(out, `${stem}.md`), content.join('\n\n'));
 const metadata = { booktitle: 'Pixels → Perception', edition: zh ? '繁體中文版' : 'English Edition', subtitle: zh ? '從像素到理解：視覺、語言與學習' : 'From Pixels to Understanding', tagline: zh ? '基礎・影像・語言・學習・理論' : 'Foundations · Vision · Language · Learning · Theory', date, revision, zh, tutorials: stats.tutorials, posts: stats.posts };
 fs.writeFileSync(path.join(out, `${stem}.json`), JSON.stringify(metadata, null, 2));
 execFileSync(pandoc, [path.join(out, `${stem}.md`), '--from=markdown+raw_tex-implicit_figures', '--to=latex', '--standalone', '--top-level-division=section', ...noHighlight, `--template=${path.join(root, 'scripts/book/template.tex')}`, `--lua-filter=${path.join(root, 'scripts/book/layout.lua')}`, `--metadata-file=${path.join(out, `${stem}.json`)}`, '-o', path.join(out, `${stem}.tex`)], { cwd: out, stdio: 'inherit' });
 manifest.editions[lang] = stats;
 console.log(`${lang}: ${stats.tutorials} tutorials, ${stats.posts} posts, ${stats.figures} figures, ${stats.codeBlocks} code blocks`);
}
fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest, null, 2));
console.log(`LaTeX sources written to ${out}`);
