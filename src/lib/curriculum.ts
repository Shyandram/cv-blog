// The curriculum tree (src/content/curriculum.ts) is the single source of truth for the course's hierarchy, order and
// numbering. This module has no Astro dependency, so the site (src/lib/site.ts), the scripts (content index, RAG corpus, PDF
// book, scripts/curriculum.mjs) and the tests all use the same logic.
//
// Numbering, by position in the tree: a top-level chapter is "<PREFIX> NN" (DLCV 04), its children add a letter (DLCV 04A),
// the level below adds ".n" (DLCV 04A.1), and every deeper level adds ".n" again (DLCV 04A.1.2). A track without a prefix
// (foundations) has no numbers. The displayed title is that number plus the frontmatter title with any "<PREFIX> NN[X] · "
// or "<PREFIX> NN[X]・" prefix removed, so a file may keep or drop its old prefix.
import { TRACKS } from './taxonomy.ts';

export type CurriculumNode = {
  /** "<track>/<slug>", e.g. "deep-learning/dl-04a-super-resolution". */
  slug: string;
  /** The tutorial exists in English only (no zh-tw twin yet); the zh-tw site shows the English text. */
  enOnly?: boolean;
  children?: CurriculumNode[];
};
export type CurriculumTree = Record<string, CurriculumNode[]>;

export const PREFIX: Record<string, string> = {
  foundations: '', 'image-processing': 'IP', nlp: 'NLP', 'pattern-recognition': 'PR', ml: 'ML', 'deep-learning': 'DLCV',
  optimization: 'OPT', 'information-theory': 'IT',
};

export type Place = {
  slug: string;
  track: string;
  /** "DLCV 04A", or "" in a track without a prefix. */
  number: string;
  /** 1 for a top-level chapter. */
  depth: number;
  parent: string | null;
  children: string[];
  /** From the top-level chapter down to the parent. */
  ancestors: string[];
  /** Position in the whole course (pre-order, tracks in site order). */
  index: number;
  enOnly: boolean;
};

const letters = (i: number): string => (i < 26 ? String.fromCharCode(65 + i) : letters(Math.floor(i / 26) - 1) + letters(i % 26));

/** The displayed number of the node at `path` (0-based positions from the top level down). */
export function numberOf(prefix: string, path: number[]): string {
  if (!prefix || path.length === 0) return '';
  const [top, second, ...rest] = path;
  return `${prefix} ${String(top + 1).padStart(2, '0')}${second === undefined ? '' : letters(second)}${rest.map((n) => `.${n + 1}`).join('')}`;
}

/** Every node of the tree with its place, in pre-order (tracks in site order, then any other key of the tree). */
export function placesOf(tree: CurriculumTree): Map<string, Place> {
  const out = new Map<string, Place>();
  const keys = [...TRACKS.filter((t) => t in tree), ...Object.keys(tree).filter((t) => !(TRACKS as readonly string[]).includes(t))];
  let index = 0;
  for (const track of keys) {
    const walk = (nodes: CurriculumNode[], path: number[], ancestors: string[]) => nodes.forEach((node, i) => {
      const p = [...path, i];
      out.set(node.slug, {
        slug: node.slug, track, number: numberOf(PREFIX[track] ?? '', p), depth: p.length, parent: ancestors.at(-1) ?? null,
        children: (node.children ?? []).map((c) => c.slug), ancestors, index: index++, enOnly: !!node.enOnly,
      });
      walk(node.children ?? [], p, [...ancestors, node.slug]);
    });
    walk(tree[track] ?? [], [], []);
  }
  return out;
}

// "IP 05 · ", "DLCV 04A・", "ML 03A · ", "DLCV 09A.1 · " at the start of a title (the separator is kept as written).
const NUMBERED = /^\s*[A-Z][A-Za-z]* \d{2}[A-Z]*(?:\.\d+)*(\s*[·・]\s*)/;

/** The title without its number, and the separator it used (if any). */
export function splitTitle(title: string): { text: string; sep: string | null } {
  const m = NUMBERED.exec(title);
  return m ? { text: title.slice(m[0].length), sep: m[1] } : { text: title, sep: null };
}

/** The title shown everywhere: the tree's number, the file's separator (or the language's usual one) and the title text. */
export function displayTitle(title: string, number: string, lang: string): string {
  const { text, sep } = splitTitle(title);
  if (!number) return text;
  return `${number}${sep ?? (lang === 'zh-tw' ? '・' : ' · ')}${text}`;
}

export type TutorialFile = { lang: string; slug: string; track: string };

/** Problems with the tree, as messages that name the slug and what to do; empty when the tree is valid. */
export function validateCurriculum(tree: CurriculumTree, files: TutorialFile[], file = 'src/content/curriculum.ts'): string[] {
  const errors: string[] = [];
  const langsOf = new Map<string, Set<string>>();
  const trackOf = new Map<string, string>();
  for (const f of files) {
    if (!langsOf.has(f.slug)) langsOf.set(f.slug, new Set());
    langsOf.get(f.slug)!.add(f.lang);
    trackOf.set(f.slug, f.track);
  }
  const seen = new Set<string>();
  for (const [track, nodes] of Object.entries(tree)) {
    if (!(TRACKS as readonly string[]).includes(track)) errors.push(`${file}: unknown track "${track}" (tracks: ${TRACKS.join(', ')})`);
    const walk = (list: CurriculumNode[]) => {
      for (const node of list) {
        const slug = node.slug;
        if (seen.has(slug)) errors.push(`${slug} appears more than once in ${file}; keep one entry`);
        seen.add(slug);
        if (!slug.startsWith(`${track}/`)) errors.push(`${slug} is listed under track "${track}" in ${file}; move it to "${slug.split('/')[0]}"`);
        const langs = langsOf.get(slug);
        if (!langs) errors.push(`${slug} is listed in ${file} but there is no tutorial file src/content/tutorials/en/${slug}.mdx; remove the entry or fix the slug`);
        else {
          if (trackOf.get(slug) !== slug.split('/')[0]) errors.push(`${slug}: frontmatter track "${trackOf.get(slug)}" does not match its folder`);
          if (!langs.has('en')) errors.push(`${slug} has no English file src/content/tutorials/en/${slug}.mdx`);
          if (!langs.has('zh-tw') && !node.enOnly) errors.push(`${slug} has no zh-tw file; add src/content/tutorials/zh-tw/${slug}.mdx or mark the entry { enOnly: true } in ${file}`);
          if (langs.has('zh-tw') && node.enOnly) errors.push(`${slug} is marked enOnly in ${file} but has a zh-tw file; remove enOnly`);
        }
        walk(node.children ?? []);
      }
    };
    walk(nodes);
  }
  for (const slug of [...langsOf.keys()].sort()) if (!seen.has(slug)) errors.push(`${slug} is not in the curriculum tree: add ${slug} to ${file} (or run node scripts/curriculum.mjs)`);
  return errors;
}

// ---- Building the tree from the files (scripts/curriculum.mjs) ----

export type TutorialMeta = { slug: string; track: string; order: number; enOnly?: boolean };

/** "dl-04a-super-resolution" → { num: "04", letter: "a" }; files without a number give null. */
export function slugNumber(slug: string): { num: string; letter: string } | null {
  const m = /^[a-z]+-(\d{2})([a-z]?)-/.exec(slug.split('/').pop() ?? '');
  return m ? { num: m[1], letter: m[2] } : null;
}

/** The parent the numbering rule gives: a lettered chapter (dl-04a-…) goes under the plain-number chapter (dl-04-…). */
function ruleParent(slug: string, candidates: string[]): string | undefined {
  const n = slugNumber(slug);
  if (!n || !n.letter) return undefined;
  return candidates.find((c) => { const m = slugNumber(c); return c !== slug && m?.num === n.num && m.letter === ''; });
}

/** The tree the numbering rule gives: tracks in site order, chapters by frontmatter order, lettered chapters under their parent. */
export function treeFromNumbers(tutorials: TutorialMeta[]): CurriculumTree {
  return addMissing({}, tutorials).tree;
}

/**
 * Adds tutorials that are not in the tree yet (by the numbering rule, placed among their siblings by frontmatter order),
 * removes entries whose files are gone, and keeps `enOnly` in step with the files. Existing entries keep their place.
 */
export function addMissing(tree: CurriculumTree, tutorials: TutorialMeta[]): { tree: CurriculumTree; added: string[]; removed: string[] } {
  const bySlug = new Map(tutorials.map((t) => [t.slug, t]));
  const orderOf = (slug: string) => bySlug.get(slug)?.order ?? Number.POSITIVE_INFINITY;
  const removed: string[] = [];
  const prune = (nodes: CurriculumNode[]): CurriculumNode[] => nodes.flatMap((n) => {
    const t = bySlug.get(n.slug);
    if (!t) { removed.push(n.slug); return prune(n.children ?? []); }
    const out: CurriculumNode = { slug: n.slug };
    if (t.enOnly) out.enOnly = true;
    const children = prune(n.children ?? []);
    if (children.length) out.children = children;
    return [out];
  });
  const next: CurriculumTree = {};
  for (const track of TRACKS) if (tree[track]?.length || tutorials.some((t) => t.track === track)) next[track] = prune(tree[track] ?? []);
  const present = new Set(placesOf(next).keys());
  const added: string[] = [];
  const missing = tutorials.filter((t) => !present.has(t.slug))
    .sort((a, b) => TRACKS.indexOf(a.track as never) - TRACKS.indexOf(b.track as never) || a.order - b.order || a.slug.localeCompare(b.slug));
  // Plain-number chapters first, so lettered ones find their parent.
  missing.sort((a, b) => Number(!!slugNumber(a.slug)?.letter) - Number(!!slugNumber(b.slug)?.letter));
  const find = (nodes: CurriculumNode[], slug: string): CurriculumNode | undefined => {
    for (const n of nodes) { if (n.slug === slug) return n; const f = find(n.children ?? [], slug); if (f) return f; }
    return undefined;
  };
  for (const t of missing) {
    next[t.track] ??= [];
    const inTrack = [...placesOf({ [t.track]: next[t.track] }).keys()];
    const parentSlug = ruleParent(t.slug, inTrack);
    const parent = parentSlug ? find(next[t.track], parentSlug) : undefined;
    const siblings = parent ? (parent.children ??= []) : next[t.track];
    const node: CurriculumNode = t.enOnly ? { slug: t.slug, enOnly: true } : { slug: t.slug };
    const at = siblings.findIndex((s) => orderOf(s.slug) > t.order);
    siblings.splice(at < 0 ? siblings.length : at, 0, node);
    added.push(t.slug);
  }
  return { tree: next, added, removed };
}

/** The tree as the TypeScript source of src/content/curriculum.ts. */
export function serializeTree(tree: CurriculumTree): string {
  const node = (n: CurriculumNode, pad: string): string => {
    const extra = n.enOnly ? ', enOnly: true' : '';
    if (!n.children?.length) return `${pad}{ slug: '${n.slug}'${extra} },`;
    return `${pad}{ slug: '${n.slug}'${extra}, children: [\n${n.children.map((c) => node(c, `${pad}  `)).join('\n')}\n${pad}] },`;
  };
  const body = Object.entries(tree).map(([track, nodes]) => `  '${track}': [\n${nodes.map((n) => node(n, '    ')).join('\n')}\n  ],`).join('\n');
  return `// The course tree: the single source of truth for which chapters exist, their order, their nesting and their numbers.
// Chapter numbers are computed from the position in this tree (src/lib/curriculum.ts): a top-level entry is "<PREFIX> NN",
// its children "<PREFIX> NNA", "NNB", …, the next level "NNA.1", "NNA.2", and so on.
// To add a chapter: add { slug: '<track>/<file name without .mdx>' } where it belongs (or run node scripts/curriculum.mjs,
// which adds new files by the letter rule). To move one: move its entry (with its children). Mark a chapter that has no
// zh-tw file yet with enOnly: true. The build and node scripts/curriculum.mjs --check fail if a tutorial is missing here.
import type { CurriculumTree } from '../lib/curriculum.ts';

export const curriculum: CurriculumTree = {
${body}
};
`;
}
