# Practice section (LeetCode-style, step 6 of the site restyle)

Scope: new routes `/<lang>/practice/` (problem list) and `/<lang>/practice/<id>/` (one page per playground experiment or
image lab), built from the existing registries. Design source: the approved Practice.dc.html and Problem.dc.html
artboards, implemented with the site's own tokens and fonts and the Learn page's visual language (light ground, white
cards in light mode, `--surface` cards in dark mode, track-coloured code badges). The Playground is unchanged and stays
in the header; Practice is added next to Learn.

## Files

- `src/lib/practice.ts` (new): `problemsFor(lang)` turns `experiments` + `labs` into problems in course order; the
  chapters of a problem are `chaptersForExperiment(id)` ∪ `related`, sorted by the course tree; the first decides the
  Chapter column (curriculum code from `shortTitle`) and the level (that chapter's frontmatter `level`). `plansOf` is the
  study-plan mapping (one plan per course, TRACKS order; the line under each plan is computed: problems · chapters).
  `topicsOf` picks topic chips from the tags of the chapters behind the problems. `sectionsFor` finds, for each
  chapter, the heading above its `<Lab id="…">` embed in the main text (matched against the rendered heading ids), else
  links the chapter.
- `src/scripts/practice.ts` (new): status in `localStorage` key `pp:practice` (`{id: "attempted" | "solved"}`) through
  the try/catch helpers of `progress.ts`; "solved" is never lowered.
- `src/pages/[lang]/practice/index.astro`, `src/pages/[lang]/practice/[id].astro` (new).
- Shared files (additive only): `src/components/Header.astro` (one nav link), `src/i18n/ui.ts` (`nav.practice` and
  `practice.*` strings, EN and zh-tw), `scripts/content-index.mjs` (repo-map line naming the practice routes) and the
  regenerated `docs/content-map/README.md`.

## Decisions

- **Nothing invented.** Every problem text (title, description, challenge, prediction, explanation, note) comes from
  the registries; chapter names, codes, levels and tags from the curriculum and frontmatter. 72 problems (46 SVG
  experiments, 26 image labs). The hand-written `kernel-filter` card of the Playground is not in a registry, so it is not
  a problem (it stays on the Playground).
- **Status.** "Attempted" when a problem page is opened. "Solved" when its prediction is answered correctly. The
  `Predict` component has no custom event; the problem page observes the `data-ok` attribute that `Predict` already sets
  on its verdict (no change to `Predict.astro`). 43 problems have a prediction question, so the progress ring counts
  solved out of 43 and says so; the other 29 can only be attempted. The page states "Stored only in this browser."
- **Hints.** No lab note is already written as a list of short items, so per the brief no note is split: the Hints tab
  links the chapter sections that use the lab (33 links go to the exact heading above the embed; the others to the
  chapter).
- **Explanation.** The prediction's `explain` in a closed `<details>`; it opens after the prediction is checked.
- **Lab pane.** `ImageLab` / `ExperimentPlayground` mounted unchanged (`links=[]`, the usual sources) plus "Open in the
  Playground". The lab's own prediction and "Try this" line are hidden with page-scoped CSS because the left pane shows
  them; the left `Predict` uses id `practice-<id>` so its radio group never collides with the lab's.
- **Layout.** List: plans (4 / 2 / 2 columns), sticky right column (progress ring, topic chips) at > 1100 px; filters
  are native `<select>`s (Course, Level, Type, Status) plus search, synced to the URL (`?course=nlp` etc.; plan cards
  link there). Phones: rows become two-line cards (descriptions hidden). Problem: split view at ≥ 1024 px with the left
  pane sticky and the lab scrolling with the page; stacked below. ARIA tabs (arrow keys, Home/End); prev/next links with
  `rel` and screen-reader names.
- **Without JavaScript.** The list shows all 72 rows; filters, chips and the count are hidden (`<noscript>` style). The
  problem page hides the tab bar and shows Description, Hints and Explanation in order with their headings.
- **Gotcha found.** Pagefind's component CSS resets any element whose class starts with `pf-`, and `global.css` styles
  `.card`, `.chip`, `.pill`: the pages use `needs-js`, `pcard`, `tchip`, `ppill` instead.

## Validation (2026-11-13)

- `npx astro check`: 0 errors, 0 warnings (3 pre-existing hints). `npm run build`: 1414 pages built (146 new), 211 Vite
  warnings (same as main). `node scripts/content-index.mjs --check`: up to date. Unit tests 336/336;
  `check-playground.mjs` and `check-content-stats.mjs` pass.
- `scripts/check-site-links.py`: 2379 pages, 154488 links, 0 broken. `scripts/check-course-ui.py`: 28 layouts passed.
  `scripts/check-labs.py`: all lab checks passed. No check needed adapting.
- Playwright (scratchpad `step6-shots/`): list and three problem pages, EN and zh-tw, 1440 and 375 px, light and dark:
  no horizontal scroll, no page errors. Behaviour: opening marks attempted; a correct prediction marks solved, shows the
  status pill and opens the explanation; a wrong one stays attempted; the list shows both statuses and "1 of 43 solved";
  course, level, status, search and topic filters and `?course=` restore work; keyboard tabs work; JavaScript disabled
  shows all rows and all panels at 375 px without horizontal scroll.
