import { getCollection, type CollectionEntry } from 'astro:content';
import { LANGS, DEFAULT_LANG, type Lang } from '../i18n/ui';
import { TRACKS } from './taxonomy';

// "<user>/<repo>" for edit and Colab links. GitHub Actions sets GITHUB_REPOSITORY automatically;
// the fallback only matters for local builds.
export const REPO = process.env.GITHUB_REPOSITORY ?? 'YOUR_GITHUB_USER/cv-blog';
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

export async function postsFor(lang: Lang) {
  const all = (await getCollection('posts')).filter(visible);
  return all
    .filter((p: Post) => splitId(p.id).lang === lang)
    .map((p: Post) => ({ slug: splitId(p.id).slug, entry: p }))
    .sort((a, b) => b.entry.data.published.valueOf() - a.entry.data.published.valueOf());
}

export function readingMinutes(body: string | undefined, lang: Lang): number {
  if (!body) return 1;
  const text = body.replace(/```[\s\S]*?```/g, '').replace(/\$\$[\s\S]*?\$\$/g, '');
  // English ~220 words/min; Chinese ~400 characters/min.
  const units = lang === 'zh-tw' ? (text.match(/[一-鿿]/g) ?? []).length / 400 + text.split(/\s+/).filter((w) => /[a-z]/i.test(w)).length / 220 : text.split(/\s+/).length / 220;
  return Math.max(1, Math.round(units));
}

export function formatDate(d: Date, lang: Lang) {
  return d.toLocaleDateString(lang === 'zh-tw' ? 'zh-TW' : 'en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}
