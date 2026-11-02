// Mechanical coverage checks complement source review; they do not establish scientific correctness.
import fs from 'node:fs';
import path from 'node:path';
import {createProcessor} from '@mdx-js/mdx';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import remarkCjkFriendly from 'remark-cjk-friendly';
const root = process.cwd();
const read = file => JSON.parse(fs.readFileSync(path.join(root, file), 'utf8'));
const index = read('docs/content-index.json');
const registry = new Map();
const sources = read('docs/paper-figure-sources.json');
const maps = read('docs/paper-guide-reading-maps.json');
const teaching = read('docs/paper-guide-teaching-diagrams.json');
const issues = [], rows = [], used = new Set();
const fail = message => issues.push(message);
for (const item of [...sources, ...maps, ...teaching]) {
  if (registry.has(item.output)) fail(`Duplicate diagram output: ${item.output}`);
  registry.set(item.output, item);
}
const sourceKeys = new Set();
for (const item of sources) {
  if (sourceKeys.has(item.key)) fail(`Duplicate source key: ${item.key}`);
  sourceKeys.add(item.key);
  if (!/^https:\/\//.test(item.sourceUrl)) fail(`Invalid source URL: ${item.key}`);
  const hash = item.kind === 'original-author-figure' ? item.sourceSha256 : item.pdfSha256;
  if (!/^[a-f0-9]{64}$/.test(hash ?? '')) fail(`Missing pinned source bytes: ${item.key}`);
  if (item.kind === 'original-paper-excerpt' && (!(item.page > 0) || item.crop?.length !== 4 || item.crop[2] <= item.crop[0] || item.crop[3] <= item.crop[1])) fail(`Invalid PDF excerpt: ${item.key}`);
}
const processor = createProcessor({remarkPlugins: [remarkMath, remarkGfm, remarkCjkFriendly]});
const plain = n => n.value ?? (n.children ?? []).map(plain).join('');
function walk(n, fn) { fn(n); for (const c of n.children ?? []) walk(c, fn); }
const imageStats = new Map();
function checkImage(file) {
  if (imageStats.has(file)) return imageStats.get(file);
  const full = path.join(root, file);
  if (!fs.existsSync(full)) { fail(`Missing image: ${file}`); return null; }
  const data = fs.readFileSync(full);
  if (data.length >= 1_000_000) fail(`Image exceeds 1 MB: ${file}`);
  const png = data.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10]));
  if (!png) { fail(`Expected a PNG figure: ${file}`); return null; }
  const width = data.readUInt32BE(16), height = data.readUInt32BE(20);
  if (width > 1600 || width < 1 || height < 1) fail(`Invalid figure dimensions ${width}x${height}: ${file}`);
  const result = {width, height, bytes: data.length}; imageStats.set(file, result); return result;
}
function guides(file) {
  const source = fs.readFileSync(path.join(root, file), 'utf8');
  const body = source.replace(/^---\n[\s\S]*?\n---\n/, '');
  const ast = processor.parse(body);
  const imports = new Map([...body.matchAll(/^import\s+(\w+)\s+from\s+['"]([^'"]+)['"]/gm)]
    .map(m => [m[1], path.relative(root, path.resolve(path.dirname(path.join(root, file)), m[2]))]));
  let inPapers = false, guide;
  const result = [];
  for (const node of ast.children) {
    if (node.type === 'heading' && node.depth === 2) { inPapers = /^(Paper guides|論文導讀)$/.test(plain(node)); guide = null; continue; }
    if (!inPapers) continue;
    if (node.type === 'heading' && node.depth === 3) {
      guide = {title: plain(node), nodes: [], figures: [], subheadings: [], citations: new Set(), text: ''}; result.push(guide); continue;
    }
    if (!guide) continue;
    guide.nodes.push(node);
    if (node.type === 'heading' && node.depth === 4) guide.subheadings.push(plain(node));
    walk(node, n => {
      if (n.type === 'text') {
        guide.text += n.value+' ';
        for (const m of n.value.matchAll(/\[(\d+)\]/g)) guide.citations.add(Number(m[1]));
      }
      if (n.type === 'mdxJsxFlowElement' && n.name === 'Figure') {
        const attrs = new Map(n.attributes.filter(a => a.type === 'mdxJsxAttribute').map(a => [a.name, a.value]));
        const variable = attrs.get('src')?.value?.trim();
        const image = imports.get(variable);
        if (!image) fail(`${file}: ${guide.title}: unresolved figure ${variable}`);
        else {
          checkImage(image); used.add(image);
          const description = attrs.get('caption');
          if (typeof description !== 'string' || description.trim().length < 30) fail(`${file}: ${guide.title}: missing explanatory caption`);
          if (typeof attrs.get('alt') !== 'string' || attrs.get('alt').trim().length < 10) fail(`${file}: ${guide.title}: missing useful alt text`);
          guide.figures.push(image);
        }
      }
    });
  }
  return result;
}
for (const chapter of index.tutorials) {
  const en = guides(chapter.files.en), zh = guides(chapter.files.zh);
  if (en.length !== zh.length) { fail(`${chapter.slug}: bilingual guide counts differ`); continue; }
  const refs = new Set(chapter.references.map(r => r.n));
  const sections = chapter.sections.filter(s => s.zone === 'papers' && s.depth === 3);
  en.forEach((a, i) => {
    const b = zh[i], label = `${chapter.slug}: ${a.title}`;
    if (!a.figures.length || !b.figures.length) fail(`${label}: missing guide-local figure`);
    if (JSON.stringify(a.figures) !== JSON.stringify(b.figures)) fail(`${label}: bilingual figure order differs`);
    if (a.subheadings.length !== b.subheadings.length) fail(`${label}: bilingual section counts differ`);
    for (const [lang, guide] of [['en', a], ['zh-tw', b]]) for (const n of guide.citations) if (!refs.has(n)) fail(`${label} (${lang}): reference [${n}] is missing`);
    const ca = [...a.citations].sort((x,y)=>x-y), cb = [...b.citations].sort((x,y)=>x-y);
    if (JSON.stringify(ca) !== JSON.stringify(cb)) fail(`${label}: bilingual citations differ: ${ca} / ${cb}`);
    const words = a.text.match(/[\p{L}\p{N}][\p{L}\p{N}'’–-]*/gu)?.length ?? 0;
    if (words < 250) fail(`${label}: only ${words} prose words; review depth`);
    for (const [role, pattern] of [
      ['thesis', /challeng|problem|question|argu|aim|survey|review|cover/i],
      ['method or reasoning', /method|idea|approach|defin|operator|architect|framework|algorithm|taxonom|organiz|represent|pipeline/i],
      ['evidence', /result|show|evaluat|compar|experiment|benchmark|proof|prove|theor|evidence|conclud/i],
      ['limits', /limit|problem|assum|fail|caution|unseen|scope|bias|protocol|does not|depend|cannot|do not|conditional|uncertainty/i],
    ]) if (!pattern.test(a.text)) fail(`${label}: no ${role} cue; review content`);
    rows.push({chapter: chapter.slug, title: a.title, titleZh: b.title, url: sections[i]?.url ?? null, urlZh: sections[i]?.urlZh ?? null,
      proseWordsEn: words, sections: a.subheadings.length, citations: ca,
      figures: a.figures.map(file => ({file, kind: registry.get(file)?.kind ?? 'existing-teaching-figure', ...imageStats.get(file)}))});
  });
}
for (const [file] of registry) if (!used.has(file)) fail(`Registered diagram is not used in a paper guide: ${file}`);
const report = {schemaVersion: 1, purpose: 'Mechanical coverage only; source accuracy and readability require review.',
  summary: {guides: rows.length, chapters: new Set(rows.map(r=>r.chapter)).size, withoutFigures: rows.filter(r=>!r.figures.length).length,
    originalSourceFigures: sources.length, comparativeMaps: maps.length, supplementalTeachingDiagrams: teaching.length}, guides: rows};
if (issues.length) { console.error(issues.join('\n')); process.exit(1); }
const output = path.join(root, 'docs/paper-guide-coverage.json');
if (process.argv.includes('--check')) {
  if (!fs.existsSync(output) || fs.readFileSync(output, 'utf8') !== JSON.stringify(report, null, 2)+'\n') throw Error('Paper-guide coverage report is stale; run node scripts/check-paper-guides.mjs');
} else fs.writeFileSync(output, JSON.stringify(report, null, 2)+'\n');
console.log(JSON.stringify(report.summary));
