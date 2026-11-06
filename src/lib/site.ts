import { chapterPath } from './chapter-paths.mjs';
import { getCollection, render, type CollectionEntry } from 'astro:content';
import { LANGS, DEFAULT_LANG, type Lang } from '../i18n/ui';
import { curriculum } from '../content/curriculum';
import { placesOf, displayTitle, validateCurriculum, type Place } from './curriculum';
import { zonesOf, type Zones } from './zones';

// GitHub repo for "Edit on GitHub" and "Open in Colab" links.
export const REPO = 'Shyandram/cv-blog';
export const BRANCH = 'main';

/** Prefix an internal path with the deploy base (e.g. "/cv-blog/"). */
export function url(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${base}${chapterPath(p)}`;
}

export function splitId(id: string): { lang: Lang; slug: string } {
  const [lang, ...rest] = id.split('/');
  return { lang: lang as Lang, slug: rest.join('/') };
}

export function isLang(x: string | undefined): x is Lang {
  return !!x && (LANGS as readonly string[]).includes(x);
}

type Tut = CollectionEntry<'tutorials'>;
type Post = CollectionEntry<'posts'>;

const visible = (e: { data: { draft: boolean } }) => import.meta.env.DEV || !e.data.draft;

const PLACES = placesOf(curriculum);
let treeChecked = false;
/** Fails the build with a message naming each slug when src/content/curriculum.ts and the tutorial files disagree. */
async function checkTree() {
  if (treeChecked) return;
  const files = (await getCollection('tutorials')).map((e) => ({ ...splitId(e.id), track: e.data.track }));
  const errors = validateCurriculum(curriculum, files);
  if (errors.length) throw new Error(`The course tree (src/content/curriculum.ts) does not match the tutorials:\n- ${errors.join('\n- ')}`);
  treeChecked = true;
}

/**
 * Tutorials for a language, in course order (the tree in src/content/curriculum.ts), each with its place in the tree and
 * its displayed title (the tree's number + the frontmatter title text; `entry.data.title` is replaced by it, so every page
 * shows the computed title). A tutorial missing in `lang` falls back to the English version, flagged with
 * `fallback: true` so the page can show a notice.
 */
export async function tutorialsFor(lang: Lang) {
  await checkTree();
  const all = (await getCollection('tutorials')).filter(visible);
  const bySlug = new Map<string, { entry: Tut; fallback: boolean }>();
  for (const e of all) {
    const { lang: l, slug } = splitId(e.id);
    if (l === lang) bySlug.set(slug, { entry: e, fallback: false });
  }
  if (lang !== DEFAULT_LANG) {
    for (const e of all) {
      const { lang: l, slug } = splitId(e.id);
      if (l === DEFAULT_LANG && !bySlug.has(slug)) bySlug.set(slug, { entry: e, fallback: true });
    }
  }
  return [...bySlug.entries()]
    .map(([slug, v]) => {
      const place = PLACES.get(slug)!;
      const title = displayTitle(v.entry.data.title, place.number, v.fallback ? DEFAULT_LANG : lang);
      return { slug, ...v, entry: { ...v.entry, data: { ...v.entry.data, title } } as Tut, place };
    })
    .sort((a, b) => a.place.index - b.place.index);
}
export type TutorialItem = Awaited<ReturnType<typeof tutorialsFor>>[number];

/** A tutorial with the tutorials nested under it (children of a hidden draft move up to the nearest shown ancestor). */
export type TutorialNode<T> = { item: T; children: TutorialNode<T>[]; depth: number };
export function nestTutorials<T extends { slug: string; place: Place }>(list: T[]): TutorialNode<T>[] {
  const shown = new Map(list.map((x) => [x.slug, { item: x, children: [] as TutorialNode<T>[], depth: 1 }]));
  const roots: TutorialNode<T>[] = [];
  for (const x of list) {
    const node = shown.get(x.slug)!;
    const up = [...x.place.ancestors].reverse().find((a) => shown.has(a));
    if (up) { const parent = shown.get(up)!; node.depth = parent.depth + 1; parent.children.push(node); } else roots.push(node);
  }
  return roots;
}

async function prepareTutorial(entry: Tut) {
  const result = await render(entry);
  return { ...result, zones: zonesOf(result.headings) };
}
// A chapter and all its optional pages reuse the same compiled component and metadata.
// Development stays uncached so edits cannot leave stale headings or content behind.
const tutorialCache = new Map<string, ReturnType<typeof prepareTutorial>>();
export function renderTutorial(entry: Tut) {
  if (import.meta.env.DEV) return prepareTutorial(entry);
  let result = tutorialCache.get(entry.id);
  if (!result) { result = prepareTutorial(entry); tutorialCache.set(entry.id, result); }
  return result;
}

/** The Supplementary notes and Paper guides of a tutorial (each item is a page of its own). */
export async function zonesFor(entry: Tut): Promise<Zones> {
  return (await renderTutorial(entry)).zones;
}

/** The chain of tutorials above `slug` (top-level first) among those shown. */
export function ancestorsOf<T extends { slug: string; place: Place }>(list: T[], slug: string): T[] {
  const bySlug = new Map(list.map((x) => [x.slug, x]));
  return (bySlug.get(slug)?.place.ancestors ?? []).map((a) => bySlug.get(a)).filter((x): x is T => !!x);
}

/**
 * Posts written in `lang`. With `includeUntranslated`, also posts that exist only in
 * another language, so readers of either language see everything; `lang` on each item
 * is the language the post is written in, and its URL lives under that language.
 */
export async function postsFor(lang: Lang, { includeUntranslated = false } = {}) {
  const all = (await getCollection('posts')).filter(visible).map((p: Post) => ({ ...splitId(p.id), entry: p }));
  const own = new Set(all.filter((p) => p.lang === lang).map((p) => p.slug));
  return all
    .filter((p) => p.lang === lang || (includeUntranslated && !own.has(p.slug)))
    .sort((a, b) => b.entry.data.published.valueOf() - a.entry.data.published.valueOf());
}

/** Unrounded reading time of a Markdown/MDX fragment in minutes (code blocks and display math are skipped). */
function readingUnits(body: string, lang: Lang): number {
  const text = body.replace(/```[\s\S]*?```/g, '').replace(/\$\$[\s\S]*?\$\$/g, '');
  // English ~220 words/min; Chinese ~400 characters/min.
  return lang === 'zh-tw' ? (text.match(/[一-鿿]/g) ?? []).length / 400 + text.split(/\s+/).filter((w) => /[a-z]/i.test(w)).length / 220 : text.split(/\s+/).length / 220;
}

export function readingMinutes(body: string | undefined, lang: Lang): number {
  if (!body) return 1;
  return Math.max(1, Math.round(readingUnits(body, lang)));
}

/** Sections that are not part of a tutorial's main reading line: optional pages of their own, and the reference list. */
export const OPTIONAL_SECTION = /^(supplementary notes|補充資料|paper guides|論文導讀|references|參考文獻)$/i;

/**
 * Reading time of each "## " section of a tutorial body, in order (headings inside fenced code are ignored). Optional
 * sections (notes, paper guides, references) get 0 minutes because the main page only shows a short card or a list.
 */
type SectionTime = { title: string; minutes: number; units: number };
const sectionCache = new Map<Lang, Map<string, SectionTime[]>>();
export function sectionMinutes(body: string | undefined, lang: Lang): SectionTime[] {
  if (!body) return [];
  const cache = import.meta.env.DEV ? undefined : sectionCache.get(lang) ?? new Map<string, SectionTime[]>();
  if (cache && !sectionCache.has(lang)) sectionCache.set(lang, cache);
  const cached = cache?.get(body);
  if (cached) return cached;
  const out: { title: string; lines: string[] }[] = [];
  let fence = false;
  for (const line of body.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) fence = !fence;
    if (!fence && /^## /.test(line)) { out.push({ title: line.slice(3).trim(), lines: [] }); continue; }
    out.at(-1)?.lines.push(line);
  }
  const sections = out.map(({ title, lines }) => {
    const units = OPTIONAL_SECTION.test(title) ? 0 : readingUnits(lines.join('\n'), lang);
    return { title, units, minutes: units > 0 ? Math.max(1, Math.round(units)) : 0 };
  });
  cache?.set(body, sections);
  return sections;
}

/** Reading time of a tutorial's main line: every section except the optional ones. */
export function tutorialMinutes(body: string | undefined, lang: Lang): number {
  const sections = sectionMinutes(body, lang);
  if (sections.length === 0) return readingMinutes(body, lang);
  return Math.max(1, Math.round(sections.reduce((a, x) => a + x.units, 0)));
}

/** Heading of a tutorial's exercise section. */
export const EXERCISES = /^(exercises|練習)$/i;

/** Number of exercises in a tutorial: each exercise has exactly one hint (a <details> block) in the "## Exercises" section. */
export function exerciseCount(body: string | undefined): number {
  const section = sectionText(body, EXERCISES);
  return (section.match(/^\s*<details\b/gm) ?? []).length;
}

/**
 * A one-line summary of each "## " section, in order, for the section map at the top of a tutorial: the first sentence
 * of the section's "Plain version" paragraph when it has one (the house style opens every topic with it), otherwise of
 * its first paragraph before any "### ". A section that opens with a list ("What you'll learn", "Key takeaways") is
 * summarized by its item count, the exercises by their count. Markup, citations and links are reduced to plain text;
 * inline math keeps a few readable symbols. Nothing new is written: the map only quotes the page.
 */
const PLAIN = /^[*_]{0,2}(plain version|in plain words|白話版|白話說|直觀說法)[.:。：]?[*_]{0,2}[.:。：]?\s*/i;
const TEX: [RegExp, string][] = [[/\\times/g, '×'], [/\\lt/g, '<'], [/\\gt/g, '>'], [/\\le(q)?/g, '≤'], [/\\ge(q)?/g, '≥'], [/\\cdot/g, '·'], [/\\to/g, '→'], [/\^2/g, '²'], [/\\[a-zA-Z]+/g, ''], [/[{}]/g, '']];
function plainSentence(paragraph: string, lang: Lang): string {
  let s = paragraph.replace(PLAIN, '')
    .replace(/(?<!\\)\$([^$\n]+)\$/g, (_, tex: string) => TEX.reduce((t, [re, to]) => t.replace(re, to), tex).replace(/\s+/g, ' ').trim())
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\s*\[\d+\](?:\s*[–,-]\s*\[\d+\])*/g, '')
    .replace(/\*\*|__|(?<![\w*])\*(?!\s)|(?<!\s)\*(?![\w*])|`/g, '')
    .replace(/\s+/g, ' ').trim();
  const zh = lang === 'zh-tw';
  const end = zh ? s.search(/[。！？]/) : s.search(/[.!?](?=\s+[A-Z0-9"“(]|$)/);
  if (end >= 0) s = s.slice(0, end + 1);
  const max = zh ? 72 : 160;
  if (s.length > max) {
    // Prefer ending at a clause ("X fixed one problem of its predecessors: …" keeps the claim), else at a word.
    const clause = Math.max(...[...s.slice(0, max)].map((c, i) => (/[:;：；]/.test(c) ? i : -1)));
    if (clause > max / 3) s = `${s.slice(0, clause)}${zh ? '。' : '.'}`;
    else {
      const cut = zh ? max : s.lastIndexOf(' ', max);
      s = `${s.slice(0, cut > 40 ? cut : max).replace(/[,，、;；:：\s]+$/, '')}…`;
    }
  }
  return s;
}
const summaryCache = new Map<string, string[]>();
export function sectionSummaries(body: string | undefined, lang: Lang): string[] {
  if (!body) return [];
  const key = `${lang}\n${body}`;
  const cached = import.meta.env.DEV ? undefined : summaryCache.get(key);
  if (cached) return cached;
  const zh = lang === 'zh-tw';
  const sections: { title: string; lead: string[]; all: string[] }[] = [];
  let fence = false, sub = false;
  for (const line of body.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) fence = !fence;
    if (!fence && /^## /.test(line)) { sections.push({ title: line.slice(3).trim(), lead: [], all: [] }); sub = false; continue; }
    const cur = sections.at(-1);
    if (!cur) continue;
    if (!fence && /^#{3,} /.test(line)) sub = true;
    cur.all.push(fence ? '' : line);
    if (!sub) cur.lead.push(fence ? '' : line);
  }
  const blocks = (lines: string[]) => lines.join('\n').split(/\n\s*\n/).map((b) => b.trim()).filter(Boolean);
  const isParagraph = (b: string) => !/^(<|\||[-*+] |\d+\. |\$\$|#|\*(Optional|選讀)|import |\{\/\*)/.test(b);
  const out = sections.map(({ title, lead, all }) => {
    if (OPTIONAL_SECTION.test(title)) return '';
    if (EXERCISES.test(title)) {
      const n = (all.join('\n').match(/^\s*<details\b/gm) ?? []).length;
      return n ? (zh ? `${n} 題練習，每題附提示。` : `${n} exercises, each with a hint.`) : '';
    }
    const leadBlocks = blocks(lead);
    const plain = blocks(all).find((b) => PLAIN.test(b));
    const para = plain ?? leadBlocks.find(isParagraph);
    if (para) return plainSentence(para.split('\n').join(' '), lang);
    const items = lead.filter((l) => /^([-*+] |\d+\. )/.test(l)).length;
    return items ? (zh ? `${items} 個重點。` : `${items} points.`) : '';
  });
  summaryCache.set(key, out);
  return out;
}

function sectionText(body: string | undefined, title: RegExp): string {
  if (!body) return '';
  const out: string[] = [];
  let on = false, fence = false;
  for (const line of body.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) fence = !fence;
    if (!fence && /^## /.test(line)) { on = title.test(line.slice(3).trim()); continue; }
    if (on) out.push(line);
  }
  return out.join('\n');
}

/** Short name of a tutorial for compact lists: "IP 03 · Intensity…" → "IP 03"; titles without a code keep their first part. */
export function shortTitle(title: string): string {
  return title.split(/\s*[·・]\s*/)[0];
}

/** "≈ 45 min" / "≈ 8 h" (zh-tw: 「約 45 分鐘」/「約 8 小時」) for a total reading time. */
export function formatDuration(minutes: number, lang: Lang): string {
  const zh = lang === 'zh-tw';
  if (minutes < 60) return zh ? `約 ${minutes} 分鐘` : `≈ ${minutes} min`;
  const h = Math.round(minutes / 30) / 2;
  return zh ? `約 ${h} 小時` : `≈ ${h} h`;
}

export function formatDate(d: Date, lang: Lang) {
  return d.toLocaleDateString(lang === 'zh-tw' ? 'zh-TW' : 'en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}
