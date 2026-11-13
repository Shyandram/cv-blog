# Home page redesign (restyle step 2)

Rebuilt `/en/` and `/zh-tw/` from the approved mockup (`Home.dc.html`, design canvas) in the site's own stack and tokens.
Scope: the home page and the components it uses. Header, footer, Learn page and all other pages are unchanged.

## Sections

1. **Hero**: badge (free · English and 繁體中文 · runnable code in every chapter), headline with the highlighted word
   (italic Source Serif "see" in English; upright "看見" in Chinese, since CJK has no true italic), one-sentence lede, two
   buttons (first chapter of the curriculum; Learn page with the computed course count), the continue-reading link
   (StudyMap, kept from the old page) and a fan of five real chapter figures. Each card links to its chapter and shows the
   chapter number from the curriculum tree.
2. **Numbers band**: courses, chapters, paper guides, interactive experiments, 0 sign-ups or paywalls.
3. **How every chapter works**: four step cards.
4. **Try it now**: histogram equalization before/after in `ImageCompare` (new opt-in `sweep` mode), linking to the
   histogram-equalization section of IP 03 (the section id is read from that language's rendered headings).
5. **Learning paths**: beginner / intermediate / advanced, four chapters each, resolved by slug from the curriculum
   (a wrong slug fails the build); every chapter is a link, plus "Start this path" to the first.
6. **From the blog**: the three newest posts. All current posts share one publication date, so ties are broken by the
   blog's "start here" order, then series part, then slug (stable and deterministic).

## Numbers (computed at build time, in `src/pages/[lang]/index.astro`)

| Number | Source | Value (2026-11-13) |
|---|---|---|
| Courses | tracks in `TRACKS` with at least one published chapter | 8 |
| Chapters | `tutorialsFor(lang).length`; label says "each in two languages" only when every chapter has a zh-tw file (otherwise it shows how many do) | 97 (97 bilingual) |
| Paper guides | sum of `zonesFor(entry).papers.items` (the `###` items under `## Paper guides`, i.e. the guide pages) | 188 (matches `scripts/check-paper-guides.mjs`) |
| Interactive experiments | `experiments.length + labs.length + 1`, the same count as the Playground page (catalog + image labs + kernel playground) | 73 |
| Sign-ups or paywalls | static site with no accounts or payments | 0 |

The badge claims were checked: all 97 chapters have a `python` code block and an Exercises section with `<details>` hints,
in both languages. Step 4 says "exercises with hints close every chapter" rather than "experiments close every chapter",
because some chapters are reading-only on the Playground.

## Motion

All in CSS keyframes or a small inline-bundled script; only `opacity`, `translate` and `clip-path` are animated (checked in
the browser with `document.getAnimations()`).

- Hero text rises in with a stagger; the underline under "see" is drawn with `clip-path`; the badge dot pulses.
- Fan cards slide in one after another, then float out of sync. Tilt is a static `transform: rotate()`; the motion uses
  the separate `translate` property, so the tilt is kept.
- A glowing scan line moves down the edge card: the line is the top edge of a full-size overlay that moves with
  `translate: 0 100%`, so it adapts to any card size.
- Numbers start at 0 and count up once (cubic ease-out, 1.6 s) when the band is 40% visible (IntersectionObserver).
  The HTML holds the final values, so without JS they are simply shown.
- Lower sections rise in when they scroll into view. Only elements below the viewport at load are hidden first, so nothing
  that is already visible flashes.
- The before/after divider sweeps (28%–72%, 7 s cycle) while on screen and stops for good on pointer down, touch or slider
  focus. The `ImageCompare` handle now moves with `translate` instead of `left` (no visual change for tutorial pages; the
  sweep and its round knob appear only with `sweep`).
- `prefers-reduced-motion: reduce`: no animations or transitions, the scan line is hidden, numbers show final values, the
  slider does not sweep (checked: 0 running animations).

## Layout

The fan is positioned in percentages of a 1344 × 430 stage (`aspect-ratio`), so space is reserved and it scales down
without layout shift. Below 900 px it becomes a horizontally scrolling row of upright cards inside the viewport
(scroll-snap); the page itself never scrolls sideways. The SLIC card shows only the third panel of `kmeans-slic.png` by
positioning the full image inside a square, overflow-hidden box (x 783–1142, y 46–405 of 1550 × 421); no new image file.
Fan images are `astro:assets` WebP with `srcset`, eager (they are in the first screen); the slider images are lazy.

## Removed

- `HeroSobel.astro` (abstract Sobel hero) and `TrackGrid.astro`: used only by the old home page.
- i18n keys `home.title`, `home.curriculum`, `home.latest`; `home.lede` and `home.browse` reworded; new `home.*` keys in both languages.

## Deviations from the mockup

- Header and footer stay the site's global ones (out of scope); the mockup's header and dark footer are not implemented.
- Fonts: the site's tokens (Bricolage Grotesque display, system UI text, Source Serif 4 italic for "see") instead of
  Inter and Google Fonts (the CSP forbids external fonts).
- Colors use the site tokens and track colors via `color-mix`, so dark mode works; the step and level chips use track
  colors close to the mockup's.
- The demo copy says "a flat, gray photo gain contrast" instead of "a dark photo come back": the IP 03 input is a
  low-contrast mid-gray moon image, not a dark one.
- The lede adds "with the language, optimization and information theory around them", because the course has those tracks.
- The slider keeps a visible range input under the image so it can be operated with the keyboard.
- Learning-path cards are not one big link: each chapter is its own link, plus "Start this path".
- The SLIC card image box is square (the panel is square) instead of 306 × 330.

## Validation

- `npx astro check`: 0 errors, 0 warnings (3 pre-existing hints).
- `npm run build`: 1268 pages; same warning count as `main` before the change.
- `node scripts/content-index.mjs --check`: up to date. `check-content-stats`, `check-playground`, Node tests: 336/336 pass.
- `scripts/check-site-links.py`: 2233 pages, 150,335 links, 0 broken (including the language-specific IP 03 fragment).
- Playwright (Chromium): `/en/` and `/zh-tw/` at 1440 and 375 px, light and dark, plus reduced motion: no horizontal
  scroll, no console errors; keyboard Tab reaches all 29 links and the slider in `<main>`; the sweep stops on focus;
  counters are 0 before the band is visible and end at the final values; with JS disabled all numbers and sections show.
- `scripts/check-security.py` stops at a pre-existing failure unrelated to this change: legacy redirect pages
  (e.g. `dist/en/learn/ml/ml-01-introduction/index.html`) carry the fragment-preserving inline script without a CSP meta.
