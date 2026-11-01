// Keep the reading position when switching language. The English and Chinese pages mirror each other section by section,
// so the place is stored as "heading i of n, this far towards heading i+1"; when the other page has the same number of
// headings the reader lands in the same section, otherwise at the same fraction of the page. Session storage only.
const KEY = 'pp:lang-place';
const headings = () => [...document.querySelectorAll<HTMLElement>('.prose :is(h2, h3)')];
const readingLine = () => Math.min(window.innerHeight / 3, 120);
const maxScroll = () => document.documentElement.scrollHeight - window.innerHeight;

type Place = { to: string; i: number; n: number; f: number; ratio: number };
type Spot = Omit<Place, 'to'>;

const here = (): Spot | null => {
  if (window.scrollY < 40) return null;
  const hs = headings();
  const line = window.scrollY + readingLine();
  const tops = hs.map((h) => h.getBoundingClientRect().top + window.scrollY);
  let i = -1;
  tops.forEach((t, k) => { if (t <= line) i = k; });
  const from = i < 0 ? 0 : tops[i];
  const to = i + 1 < tops.length ? tops[i + 1] : document.documentElement.scrollHeight;
  return {
    i,
    n: hs.length,
    f: to > from ? Math.min(1, Math.max(0, (line - from) / (to - from))) : 0,
    ratio: maxScroll() > 0 ? window.scrollY / maxScroll() : 0,
  };
};

// Where the reader last stopped scrolling. On phones the header is not sticky, so reaching the language link means
// scrolling back to the top first; a quick flick up does not count as a stop, so the place read before it is kept.
let rested: Spot | null = null;
let restTimer: number | undefined;
window.addEventListener('scroll', () => { clearTimeout(restTimer); restTimer = window.setTimeout(() => { rested = here(); }, 1200); }, { passive: true });

document.querySelectorAll<HTMLAnchorElement>('a.lang[hreflang]').forEach((a) =>
  a.addEventListener('click', () => {
    const spot = window.scrollY >= 40 ? here() : rested;
    if (!spot) return;
    try { sessionStorage.setItem(KEY, JSON.stringify({ to: new URL(a.href).pathname, ...spot })); } catch { /* storage unavailable */ }
  }),
);

let place: Place | null = null;
try {
  place = JSON.parse(sessionStorage.getItem(KEY) ?? 'null');
  sessionStorage.removeItem(KEY);
} catch { place = null; }

if (place && place.to === location.pathname && !location.hash) {
  const p = place;
  const target = () => {
    const hs = headings();
    if (hs.length === p.n && p.n > 0) {
      const tops = hs.map((h) => h.getBoundingClientRect().top + window.scrollY);
      const from = p.i < 0 ? 0 : tops[p.i];
      const to = p.i + 1 < tops.length ? tops[p.i + 1] : document.documentElement.scrollHeight;
      return from + p.f * (to - from) - readingLine();
    }
    return p.ratio * maxScroll();
  };
  const go = () => window.scrollTo({ top: Math.max(0, Math.round(target())), behavior: 'instant' });
  go();
  // Web fonts and late layout can move the text after the first jump: follow it for a few seconds, until the reader
  // scrolls on their own.
  const ro = new ResizeObserver(() => go());
  ro.observe(document.body);
  const stop = () => { ro.disconnect(); ['wheel', 'touchstart', 'keydown', 'pointerdown'].forEach((e) => window.removeEventListener(e, stop)); };
  ['wheel', 'touchstart', 'keydown', 'pointerdown'].forEach((e) => window.addEventListener(e, stop, { passive: true }));
  window.setTimeout(stop, 4000);
}
