// The Practice section (/<lang>/practice/): every playground experiment and image lab as one "problem", with the
// chapter, level and section it belongs to. Everything shown is read from the playground registries
// (src/lib/playground) and the curriculum; nothing here adds teaching text.
//
// Study plans: one per course (track), in TRACKS order. The plan's line is computed (number of problems and of the
// chapters that use them), so the grouping is the only decision made here.
import { experiments } from './playground/catalog';
import { labs } from './playground/labs/registry';
import { chaptersForExperiment } from './playground/coverage';
import type { Localized, Predict } from './playground/types';
import { TRACKS, type Track } from './taxonomy';
import { tutorialsFor, renderTutorial, shortTitle, OPTIONAL_SECTION, type TutorialItem } from './site';
import type { Lang } from '../i18n/ui';

export type ProblemType = 'image' | 'plot';
export type Level = 'beginner' | 'intermediate' | 'advanced';

export interface ChapterRef { slug: string; code: string; title: string; href: string; level: Level; track: Track }
export interface SectionRef { chapter: ChapterRef; heading?: string; href: string }
export interface Problem {
  id: string;
  index: number;
  track: Track;
  type: ProblemType;
  title: Localized;
  description: Localized;
  challenge: Localized;
  predict?: Predict;
  /** The chapters that use this problem, in course order; the first decides the chapter column and the level. */
  chapters: ChapterRef[];
  level: Level;
  tags: string[];
}

const LEVEL_SET = new Set<string>(['beginner', 'intermediate', 'advanced']);

/** Problems in course order (the first chapter that uses each one), as on the Playground page. */
export async function problemsFor(lang: Lang): Promise<Problem[]> {
  const tutorials = await tutorialsFor(lang);
  const order = new Map(tutorials.map((x, i) => [x.slug, i]));
  const bySlug = new Map(tutorials.map((x) => [x.slug, x]));
  const ref = (x: TutorialItem): ChapterRef => {
    const level = String(x.entry.data.level);
    return {
      slug: x.slug,
      code: shortTitle(x.entry.data.title),
      title: x.entry.data.title,
      href: `/${lang}/learn/${x.slug}/`,
      level: (LEVEL_SET.has(level) ? level : 'intermediate') as Level,
      track: x.entry.data.track as Track,
    };
  };
  const entries = [...experiments.map((e) => ({ e, type: 'plot' as const })), ...labs.map((e) => ({ e, type: 'image' as const }))];
  const rows = entries.map(({ e, type }) => {
    const slugs = [...new Set([...chaptersForExperiment(e.id), ...e.related])]
      .filter((s) => bySlug.has(s))
      .sort((a, b) => order.get(a)! - order.get(b)!);
    const chapters = slugs.map((s) => ref(bySlug.get(s)!));
    const tags = [...new Set(slugs.flatMap((s) => (bySlug.get(s)!.entry.data.tags ?? []) as string[]))];
    return {
      id: e.id, index: 0, track: e.track, type, title: e.title, description: e.description, challenge: e.challenge,
      predict: e.predict, chapters, level: chapters[0]?.level ?? 'intermediate', tags,
      first: slugs.length ? order.get(slugs[0])! : 1e6,
    };
  });
  rows.sort((a, b) => TRACKS.indexOf(a.track) - TRACKS.indexOf(b.track) || a.first - b.first);
  return rows.map(({ first: _first, ...p }, index) => ({ ...p, index }));
}

export interface Plan { track: Track; count: number; chapters: number }
export function plansOf(problems: Problem[]): Plan[] {
  return TRACKS.map((track) => {
    const mine = problems.filter((p) => p.track === track);
    return { track, count: mine.length, chapters: new Set(mine.flatMap((p) => p.chapters.map((c) => c.slug))).size };
  }).filter((p) => p.count > 0);
}

/** Tags of the chapters behind at least `min` problems, most used first (topic chips). */
export function topicsOf(problems: Problem[], limit = 16): { tag: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const p of problems) for (const t of p.tags) counts.set(t, (counts.get(t) ?? 0) + 1);
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const min = ranked.filter(([, n]) => n >= 3).length >= 8 ? 3 : 2;
  return ranked.filter(([, n]) => n >= min).slice(0, limit).map(([tag, count]) => ({ tag, count }));
}

const normalize = (s: string) => s
  .replace(/\$[^$]*\$/g, '')
  .replace(/\\\{#[^}]*\\\}|\{#[^}]*\}/g, '')
  .replace(/[`*_~[\]()]/g, '')
  .toLocaleLowerCase()
  .replace(/[^\p{L}\p{N}]+/gu, '');

/**
 * For each chapter of a problem, the section of the main text that embeds `<Lab id="…">` (a link to that heading),
 * or the chapter itself when the lab is not embedded there or sits in an optional section.
 */
export async function sectionsFor(problem: Problem, lang: Lang): Promise<SectionRef[]> {
  const tutorials = await tutorialsFor(lang);
  const bySlug = new Map(tutorials.map((x) => [x.slug, x]));
  const out: SectionRef[] = [];
  for (const chapter of problem.chapters) {
    const item = bySlug.get(chapter.slug);
    const body = (item?.entry.body ?? '').replace(/^(```|~~~)[\s\S]*?^\1/gm, '');
    const at = body.search(new RegExp(`<Lab\\b[^>]*\\bid="${problem.id}"`));
    let section: SectionRef = { chapter, href: chapter.href };
    if (item && at >= 0) {
      const before = [...body.slice(0, at).matchAll(/^(#{2,4})\s+(.+?)\s*$/gm)];
      const optional = before.some((m) => m[1] === '##' && OPTIONAL_SECTION.test(m[2].trim()));
      const last = before[before.length - 1];
      if (last && !optional) {
        const { headings } = await renderTutorial(item.entry);
        const target = normalize(last[2]);
        const h = headings.find((x) => x.depth === last[1].length && normalize(x.text) === target);
        if (h) section = { chapter, heading: h.text, href: `${chapter.href}#${h.slug}` };
      }
    }
    out.push(section);
  }
  return out;
}
