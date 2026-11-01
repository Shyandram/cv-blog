// Reading progress kept in this browser only (localStorage, every access in try/catch; pages work without it).
//   pp:p:<path>  furthest scroll ratio reached on a page (a tutorial, or its …/notes/ and …/papers/ pages)
//   pp:x:<path>  exercises of a tutorial: { n: count, s: { "<index>": { at: time solved, r: reviews done } } }
//   pp:s:<path>  sections already read (ReaderTools)
//   pp:last      the last place read: { path, title, ratio, id, heading }
// Shared by the tutorial page (ReaderTools, CheckPanel), the learn index and the home page.

export const load = (key: string): string | null => { try { return localStorage.getItem(key); } catch { return null; } };
export const save = (key: string, value: string) => { try { localStorage.setItem(key, value); } catch { /* storage unavailable */ } };
const json = <T,>(key: string, fallback: T): T => { try { return (JSON.parse(load(key) ?? 'null') as T) ?? fallback; } catch { return fallback; } };

export type Solved = { at: number; r: number };
export type Exercises = { n: number; s: Record<string, Solved> };

export const exercisesOf = (path: string): Exercises => {
  const x = json<Partial<Exercises>>(`pp:x:${path}`, {});
  return { n: Number(x.n) || 0, s: x.s && typeof x.s === 'object' ? x.s : {} };
};
export const saveExercises = (path: string, x: Exercises) => save(`pp:x:${path}`, JSON.stringify(x));
export const ratioOf = (path: string): number => Math.min(1, Math.max(0, Number(load(`pp:p:${path}`)) || 0));

/** A notes or paper-guides page belongs to the tutorial at the parent path. */
export const tutorialPathOf = (path: string) => path.replace(/(?:notes|papers)\/$/, '');

export type Status = {
  ratio: number; notes: number; papers: number;
  solved: number; total: number;
  /** Finished: every exercise solved (or, for a page without exercises, read to the end). */
  done: boolean;
  started: boolean;
};

export function statusOf(path: string): Status {
  const ratio = ratioOf(path);
  const notes = ratioOf(`${path}notes/`);
  const papers = ratioOf(`${path}papers/`);
  const x = exercisesOf(path);
  const solved = Object.keys(x.s).length;
  const total = x.n;
  const done = total > 0 ? solved >= total : ratio >= 0.95;
  return { ratio, notes, papers, solved, total, done, started: done || solved > 0 || ratio > 0.02 || notes > 0.02 || papers > 0.02 };
}

export type Last = { path: string; title: string; ratio: number; id?: string; heading?: string };
export function lastPlace(): Last | null {
  const last = json<Last | null>('pp:last', null);
  return last && typeof last.path === 'string' ? last : null;
}

/** Days after solving when an exercise is due for another try (spaced review): 3, then 10, then 30 days. */
export const REVIEW_DAYS = [3, 10, 30];
const DAY = 86_400_000;
export function dueExercises(path: string, now = Date.now()): string[] {
  const { s } = exercisesOf(path);
  return Object.entries(s).filter(([, v]) => v.r < REVIEW_DAYS.length && now - v.at >= REVIEW_DAYS[v.r] * DAY).map(([k]) => k);
}

/** Fill a template such as "{k} of {n}". */
export const fill = (template: string, values: Record<string, string | number>) =>
  template.replace(/\{(\w+)\}/g, (m, k: string) => (k in values ? String(values[k]) : m));
