# Learn page redesign (course home, step 1 of the site restyle)

Scope: `/en/learn/` and `/zh-tw/learn/` only (`src/pages/[lang]/learn/index.astro`, new `src/components/CourseUnits.astro`,
new strings in `src/i18n/ui.ts`). Other pages are unchanged. Design source: the approved "Course home, Khan-style
structure" artboard (Main.dc.html), implemented with the site's own tokens and fonts (no new web fonts).

## Decisions

- **Hero.** One headline and one sentence (`learn.hero`, `learn.heroLede`; the course count is computed from the tracks
  that have chapters), and a row of links: Start here, the whole course tree, the PDF download.
- **Continue card.** Reading progress already exists (`src/scripts/progress.ts`: `pp:last`, `pp:p:<path>`, written by
  ReaderTools on chapter pages). The card reads that record client-side (every storage access already wrapped in
  try/catch there) and shows the chapter, the last section read, a bar of how far the chapter was read and a Resume
  button. No new tracking. Without a record (or without storage/JS) it is a "New here? Start with Foundations" card
  listing the Foundations chapters from the content data.
- **Course cards.** One per track in `TRACKS`, grid of 4 / 2 / 1 columns (>1100 px / >560 px / phones). Each card has a
  code badge (the curriculum `PREFIX`; Foundations has none, so its track index "00"), the track name, the chapter
  count (plus the unit count when sub-chapters exist) and the total reading time (`tutorialMinutes`, `formatDuration`),
  all computed. Track colours appear only in the badge: text in `--t-<track>` on an 8 % tint of it, measured at
  4.5:1 or better for every track in light and dark mode. StudyMap's per-track progress (`.tprog`) still attaches to
  each card through `data-paths`.
- **Units.** Below the grid, one panel per course with the top-level entries of `src/content/curriculum.ts` (computed
  numbers, title text, chapter count of the unit including its sub-chapters), in two columns read top to bottom.
  Sub-chapters appear only when a unit's toggle is pressed (44 × 44 px button, `aria-expanded`, `aria-controls`);
  deeper levels open the same way. Each row keeps `data-path` so StudyMap's per-chapter progress still shows.
- **Selection.** Cards are real links to `#<track>` with `aria-controls`/`aria-expanded`; a click opens that course
  (another click closes it), updates the URL hash with `replaceState`, moves focus to the course heading and scrolls
  it into view when it is off screen (phones). `/learn/#<track>` opens the course on load and on `hashchange` (links
  from the home page, search page, breadcrumbs and CheckPanel keep working); `#lt-<slug>` ids from the old tree are kept
  and open the units above them. With no hash, the course of the last chapter read is opened; otherwise none is, and a
  one-line hint says to choose a course.
- **No JavaScript.** A `<noscript>` style shows every course panel and every sub-chapter list and hides the toggles,
  so all units and chapters are reachable (the CSP already allows inline styles).
- **Kept.** Course-tree link (plus a per-course link to `/learn/tree/#tree-<track>`), PDF link, Start here, spaced-review
  list (`#review`), header search, language switch and dark mode. The old `#resume` line is replaced by the card.
- **Look.** Light ground (`--ink` at 3.5 % over `--bg`) with white cards in light mode; `--surface` cards on `--bg` in
  dark mode. No monospace eyebrows on this page. The ground runs to the footer.

## Validation (2026-11-13)

- `npx astro check`: 0 errors, 0 warnings (3 pre-existing hints).
- `npm run build`: 1268 pages; the 211 Vite warnings are the same count as on `main` before the change.
- `node scripts/content-index.mjs --check`: up to date. Unit tests: 336/336 pass; content-stats and playground checks pass.
- `scripts/check-site-links.py`: 2233 pages, 150347 links, 0 broken.
- `scripts/check-course-ui.py`: 28 page layouts passed. Changed: the Learn step now opens the DLCV card, checks that
  exactly that course panel is visible and expanded, opens the first unit toggle and checks its sub-chapter list, and
  checks the `#ml` deep link; it also creates `artifacts/` and uses the preinstalled Chromium in `/opt/pw-browsers`
  when Playwright's own browser is not downloaded.
- Playwright screenshots, both languages, 1440 and 375 px, light and dark, before and after, plus DLCV opened with its
  first unit expanded: no horizontal scroll, no page errors.
- Seeded `pp:last` progress: the Continue card, the DLCV panel opening and the row's "40% read" all appear in both
  languages.
- Keyboard: Tab reaches each card in order; Enter on a card opens it and moves focus to its heading; the next Tabs reach
  unit links and toggles; Enter on a toggle expands it and the next Tab enters the sub-chapter list.
- JavaScript disabled (375 px): all 8 panels and 9/9 sub-chapter lists visible, toggles hidden, no horizontal scroll.
