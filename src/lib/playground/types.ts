import type { Lang } from '../../i18n/ui';
import type { Track } from '../taxonomy';

export type Localized = { en: string; 'zh-tw': string };
export type State = Record<string, string | number>;
export interface Control {
  key: string;
  label: Localized;
  kind: 'range' | 'select' | 'text' | 'textarea';
  value: string | number;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: Localized }[];
}
export interface Result {
  svg: string;
  summary: string;
  metrics: { label: string; value: string }[];
}
/** A commit-before-reveal question shown above the controls. `answer` is the value of the correct option. */
export interface Predict {
  question: Localized;
  options: { value: string; label: Localized }[];
  answer: string;
  explain: Localized;
}
export interface Experiment {
  id: string;
  track: Track;
  title: Localized;
  description: Localized;
  challenge: Localized;
  note: Localized;
  /** Existing tutorial slugs; the page resolves current translated titles at build time. */
  related: string[];
  controls: Control[];
  predict?: Predict;
  compute: (state: State, lang: Lang) => Result;
}
export const L = (en: string, zh: string): Localized => ({ en, 'zh-tw': zh });
export const pick = (lang: Lang, en: string, zh: string) => lang === 'zh-tw' ? zh : en;
export const num = (state: State, key: string) => Number(state[key]);
export const escape = (value: string | number) => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
export const fmt = (n: number, digits = 3) => Number.isFinite(n) ? String(Number(n.toFixed(digits))) : '—';
/** SVGs use a shared palette supplied by the component, and a fixed viewBox. */
export const svg = (body: string, height = 300) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 ${height}" role="img" aria-hidden="true">${body}</svg>`;
export const text = (x: number, y: number, label: string | number, attrs = '') => `<text x="${x}" y="${y}" ${attrs}>${escape(label)}</text>`;
export const path = (points: [number, number][], attrs = '') => `<path d="${points.map(([x,y], i) => `${i ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`).join(' ')}" fill="none" ${attrs}/>`;
