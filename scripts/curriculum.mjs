// Maintains src/content/curriculum.ts, the course tree (hierarchy, order and numbering of every tutorial).
//
// Usage:
//   node scripts/curriculum.mjs           add tutorial files that are not in the tree yet (a lettered chapter such as
//                                         dl-04d-… goes under dl-04-…, placed among its siblings by frontmatter `order`),
//                                         drop entries whose files are gone, keep `enOnly` in step; existing entries keep
//                                         their place. Prints what changed.
//   node scripts/curriculum.mjs --check   validate only; exit 1 with a message naming each offending slug (CI)
//   node scripts/curriculum.mjs --init    rebuild the whole tree from the numbering rule and frontmatter `order`
//                                         (discards any hand-made nesting; use only to start over)
//   node scripts/curriculum.mjs --titles  also list every tutorial whose computed title differs from its frontmatter title
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { addMissing, treeFromNumbers, serializeTree, validateCurriculum, placesOf, displayTitle } from '../src/lib/curriculum.ts';

const root = new URL('..', import.meta.url).pathname;
const FILE = join(root, 'src/content/curriculum.ts');
const args = process.argv.slice(2);

/** Every tutorial file (both languages) with the frontmatter fields the tree needs. */
export function tutorialFiles() {
  const out = [];
  for (const lang of ['en', 'zh-tw']) {
    const dir = join(root, 'src/content/tutorials', lang);
    if (!existsSync(dir)) continue;
    for (const track of readdirSync(dir)) {
      if (!existsSync(join(dir, track)) || track.includes('.')) continue;
      for (const f of readdirSync(join(dir, track)).filter((x) => /\.mdx?$/.test(x) && !x.startsWith('_')).sort()) {
        const src = readFileSync(join(dir, track, f), 'utf8');
        const fm = src.match(/^---\n([\s\S]*?)\n---/)?.[1] ?? '';
        const get = (k) => fm.match(new RegExp(`^${k}:\\s*(.*)$`, 'm'))?.[1].trim().replace(/^["']|["']$/g, '');
        out.push({ lang, slug: `${track}/${f.replace(/\.mdx?$/, '')}`, track: get('track') ?? track, order: Number(get('order') ?? 0), title: get('title') ?? '' });
      }
    }
  }
  return out;
}

const files = tutorialFiles();
const metas = [...new Map(files.filter((f) => f.lang === 'en').map((f) => [f.slug, f])).values()]
  .map((f) => ({ slug: f.slug, track: f.track, order: f.order, enOnly: !files.some((z) => z.lang === 'zh-tw' && z.slug === f.slug) }));
// A zh-tw file without an English twin still needs a tree entry (validation reports the missing English file).
for (const f of files) if (f.lang === 'zh-tw' && !metas.some((m) => m.slug === f.slug)) metas.push({ slug: f.slug, track: f.track, order: f.order });

const current = existsSync(FILE) ? (await import(FILE)).curriculum : {};
if (args.includes('--check')) {
  const errors = validateCurriculum(current, files);
  if (errors.length) { console.error(`The course tree is not valid:\n- ${errors.join('\n- ')}`); process.exit(1); }
  console.log(`Course tree OK: ${placesOf(current).size} tutorials.`);
} else {
  const { tree, added, removed } = args.includes('--init') ? { tree: treeFromNumbers(metas), added: ['(whole tree)'], removed: [] } : addMissing(current, metas);
  const text = serializeTree(tree);
  if (!existsSync(FILE) || readFileSync(FILE, 'utf8') !== text) writeFileSync(FILE, text);
  for (const s of added) console.log(`added   ${s}`);
  for (const s of removed) console.log(`removed ${s}`);
  const errors = validateCurriculum(tree, files);
  if (errors.length) console.error(`Still to fix:\n- ${errors.join('\n- ')}`);
  console.log(`${placesOf(tree).size} tutorials in src/content/curriculum.ts.`);
}
if (args.includes('--titles')) {
  const tree = (await import(`${FILE}?t=${Date.now()}`)).curriculum;
  const places = placesOf(tree);
  let n = 0;
  for (const f of files) {
    const p = places.get(f.slug);
    const shown = p ? displayTitle(f.title, p.number, f.lang) : f.title;
    if (shown !== f.title) { n++; console.log(`${f.lang} ${f.slug}: "${f.title}" → "${shown}"`); }
  }
  console.log(`${n} computed titles differ from the frontmatter.`);
}
