// Local BM25 search over the retrieval corpus built by scripts/rag/build.mjs. No network, no dependencies.
//
// Usage: node scripts/rag/search.mjs "query" [--k 8] [--lang en|zh-tw] [--track deep-learning] [--zone main|notes|papers|references]
//                                            [--kind tutorial|summary|post] [--json] [--full]
// English text is tokenized into lower-case words; CJK text into overlapping character bigrams, so Chinese queries work too.
// Builds the corpus first if .rag/chunks.jsonl is missing (or pass --rebuild).
import { readFileSync, existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';

const root = new URL('../..', import.meta.url).pathname;
const argv = process.argv.slice(2);
const opt = (name) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined; };
const flag = (name) => argv.includes(name);
const query = argv.filter((a, i) => !a.startsWith('--') && !(i > 0 && argv[i - 1].startsWith('--') && !['--json', '--full', '--rebuild'].includes(argv[i - 1]))).join(' ');
if (!query) { console.error('usage: node scripts/rag/search.mjs "query" [--k 8] [--lang en|zh-tw] [--track …] [--zone …] [--kind …] [--json] [--full]'); process.exit(2); }

const corpus = join(root, opt('--dir') ?? '.rag', 'chunks.jsonl');
if (!existsSync(corpus) || flag('--rebuild')) execFileSync(process.execPath, [join(root, 'scripts/rag/build.mjs')], { stdio: 'inherit' });
let docs = readFileSync(corpus, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
for (const [key, name] of [['lang', '--lang'], ['track', '--track'], ['zone', '--zone'], ['kind', '--kind']]) {
  const v = opt(name); if (v) docs = docs.filter((d) => d[key] === v);
}

const STOP = new Set('a an and are as at be by for from how in is it of on or that the this to was what when where which why with does do can'.split(' '));
function tokens(s) {
  const out = [];
  for (const w of s.toLowerCase().match(/[a-z0-9]+(?:[-'][a-z0-9]+)*/g) ?? []) if (!STOP.has(w)) out.push(w.length > 4 ? w.replace(/(ies|es|s)$/, '') : w);
  for (const run of s.match(/[㐀-鿿]+/g) ?? []) {
    if (run.length === 1) out.push(run);
    for (let i = 0; i + 1 < run.length; i++) out.push(run.slice(i, i + 2));
  }
  return out;
}

const field = (d) => `${d.title} ${d.section.join(' ')} ${d.section.join(' ')} ${d.text}`; // headings count twice
const N = docs.length, df = new Map(), lens = [], tfs = [];
for (const d of docs) {
  const tf = new Map();
  for (const t of tokens(field(d))) tf.set(t, (tf.get(t) ?? 0) + 1);
  tfs.push(tf); lens.push([...tf.values()].reduce((a, b) => a + b, 0));
  for (const t of tf.keys()) df.set(t, (df.get(t) ?? 0) + 1);
}
const avg = lens.reduce((a, b) => a + b, 0) / Math.max(N, 1), k1 = 1.4, b = 0.75;
const q = [...new Set(tokens(query))];
const scored = docs.map((d, i) => {
  let s = 0;
  for (const t of q) {
    const f = tfs[i].get(t); if (!f) continue;
    const idf = Math.log(1 + (N - df.get(t) + 0.5) / (df.get(t) + 0.5));
    s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * lens[i] / avg));
  }
  return { d, s };
}).filter((x) => x.s > 0).sort((a, c) => c.s - a.s).slice(0, Number(opt('--k') ?? 8));

if (flag('--json')) { console.log(JSON.stringify(scored.map(({ d, s }) => ({ score: +s.toFixed(3), ...d, text: flag('--full') ? d.text : d.text.slice(0, 600) })), null, 1)); process.exit(0); }
if (!scored.length) { console.log('No matches.'); process.exit(0); }
for (const { d, s } of scored) {
  const where = [d.title.split(/ · |・/)[0], ...d.section].filter(Boolean).join(' › ');
  console.log(`\n[${s.toFixed(2)}] ${where}${d.zone !== 'main' ? `  (${d.zone})` : ''}${d.part ? `  part ${d.part}` : ''}`);
  console.log(`    ${d.file}:${d.lines[0]}-${d.lines[1]}   ${d.url}`);
  const text = d.text.replace(/\s+/g, ' ');
  console.log(`    ${flag('--full') ? text : text.slice(0, 260) + (text.length > 260 ? '…' : '')}`);
}
