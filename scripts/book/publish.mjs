// Build the PDF volumes (four per language) and publish them to the site: public/books/*.pdf plus src/data/books.json (read by the download page).
//
//   BOOK_FONTS=/path/to/fonts TECTONIC=/path/to/tectonic npm run book              # export, compile, publish all volumes of both languages
//   BOOK_FONTS=... TECTONIC=... node scripts/book/publish.mjs en                    # one language
//   BOOK_OUTPUT=/path/to/existing/book-output node scripts/book/publish.mjs --skip-build   # only publish PDFs already compiled
//
// See scripts/book/README.md for the requirements (Pandoc, Tectonic, Noto CJK fonts). Cloudflare Pages rejects files over
// 25 MiB, so this script stops if a PDF is larger than that.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { volumes, stem as volumeStem } from './volumes.mjs';

const root = path.resolve(import.meta.dirname, '../..');
const out = path.resolve(process.env.BOOK_OUTPUT || path.join(root, 'book-output'));
const tectonic = process.env.TECTONIC || 'tectonic';
const args = process.argv.slice(2), skipBuild = args.includes('--skip-build');
const langs = args.filter(a => !a.startsWith('--')).length ? args.filter(a => !a.startsWith('--')) : ['en', 'zh-tw'];
const LIMIT = 25 * 1024 * 1024;

const python = process.env.PYTHON || (fs.existsSync(path.join(root, '.venv/bin/python')) ? path.join(root, '.venv/bin/python') : 'python3');
if (!skipBuild) {
  execFileSync('node', [path.join(root, 'scripts/book/export.mjs'), ...langs], { stdio: 'inherit', env: { ...process.env, BOOK_OUTPUT: out } });
  // Shrink the figures in the output folder (never the originals), then point the LaTeX sources at any renamed files.
  const map = path.join(out, 'figures-map.json');
  execFileSync(python, [path.join(root, 'scripts/book/optimize_figures.py'), path.join(out, 'assets/figures'), map], { stdio: 'inherit' });
  const renamed = JSON.parse(fs.readFileSync(map, 'utf8'));
  for (const lang of langs) {
    for (const volume of volumes) {
      const file = path.join(out, `${volumeStem(lang, volume)}.tex`);
      let source = fs.readFileSync(file, 'utf8');
      for (const [from, to] of Object.entries(renamed)) source = source.split(from).join(to);
      fs.writeFileSync(file, source);
    }
  }
  for (const lang of langs) for (const volume of volumes) {
    console.log(`Compiling ${volumeStem(lang, volume)} with Tectonic (the first run downloads LaTeX packages)…`);
    execFileSync(tectonic, ['--untrusted', '--keep-logs', '--keep-intermediates', `${volumeStem(lang, volume)}.tex`], { cwd: out, stdio: 'inherit' });
  }
}

const manifest = JSON.parse(fs.readFileSync(path.join(out, 'manifest.json'), 'utf8'));
const dest = path.join(root, 'public/books');
fs.mkdirSync(dest, { recursive: true });
const dataFile = path.join(root, 'src/data/books.json');
const previous = fs.existsSync(dataFile) ? JSON.parse(fs.readFileSync(dataFile, 'utf8')) : { editions: {} };
const books = { date: manifest.date, sourceCommit: manifest.sourceCommit, editions: { ...previous.editions } };
for (const lang of langs) {
  const published = [];
  for (const volume of volumes) {
    const key = volumeStem(lang, volume), name = `${key}.pdf`, src = path.join(out, name);
    if (!fs.existsSync(src)) throw new Error(`Missing ${src}: compile the book first (or drop --skip-build).`);
    const bytes = fs.statSync(src).size;
    if (bytes > LIMIT) throw new Error(`${name} is ${(bytes / 2 ** 20).toFixed(1)} MiB; Cloudflare Pages accepts at most 25 MiB per file. Reduce figure size or split the volume further.`);
    fs.copyFileSync(src, path.join(dest, name));
    const stats = manifest.books[key] ?? {};
    published.push({ id: volume.id, number: volumes.indexOf(volume) + 1, title: volume.title[lang], tracks: volume.tracks, file: name, bytes, tutorials: stats.tutorials, posts: stats.posts, figures: stats.figures, codeBlocks: stats.codeBlocks, displayEquations: stats.displayEquations });
    console.log(`Published ${name} (${(bytes / 2 ** 20).toFixed(1)} MiB)`);
  }
  books.editions[lang] = { date: manifest.date, sourceCommit: manifest.sourceCommit, volumes: published };
  // Remove PDFs of earlier layouts (the single-volume books) so the site never links to a stale file.
  const old = path.join(dest, `pixels-perception-${lang}.pdf`);
  if (fs.existsSync(old)) fs.rmSync(old);
}
fs.mkdirSync(path.dirname(dataFile), { recursive: true });
fs.writeFileSync(dataFile, JSON.stringify(books, null, 2) + '\n');
console.log(`Wrote ${path.relative(root, dataFile)}`);
