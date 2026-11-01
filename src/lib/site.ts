import { getCollection, type CollectionEntry } from 'astro:content';
import { LANGS, DEFAULT_LANG, type Lang } from '../i18n/ui';
import { TRACKS } from './taxonomy';

// GitHub repo for "Edit on GitHub" and "Open in Colab" links.
export const REPO = 'Shyandram/cv-blog';
export const BRANCH = 'main';

/** Prefix an internal path with the deploy base (e.g. "/cv-blog/"). */
export function url(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  const p = path.startsWith('/') ? path : `/${path}`;
  return `${base}${p}`;
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

/**
 * Tutorials for a language. A tutorial missing in `lang` falls back to the English
 * version, flagged with `fallback: true` so the page can show a notice.
 */
export async function tutorialsFor(lang: Lang) {
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
    .map(([slug, v]) => ({ slug, ...v }))
    .sort((a, b) => TRACKS.indexOf(a.entry.data.track) - TRACKS.indexOf(b.entry.data.track) || a.entry.data.order - b.entry.data.order);
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
export function sectionMinutes(body: string | undefined, lang: Lang): { title: string; minutes: number; units: number }[] {
  if (!body) return [];
  const out: { title: string; lines: string[] }[] = [];
  let fence = false;
  for (const line of body.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) fence = !fence;
    if (!fence && /^## /.test(line)) { out.push({ title: line.slice(3).trim(), lines: [] }); continue; }
    out.at(-1)?.lines.push(line);
  }
  return out.map(({ title, lines }) => {
    const units = OPTIONAL_SECTION.test(title) ? 0 : readingUnits(lines.join('\n'), lang);
    return { title, units, minutes: units > 0 ? Math.max(1, Math.round(units)) : 0 };
  });
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
