# Structure and layers inside tutorials — 2026-11-06

The owner asked to see the whole structure and the relationships inside a chapter (leveled bullet points with
indentation and priority, a rethought layout) and for progressive disclosure: deeper content only when the reader asks
for it, also at the chapter level. This pass adds four building blocks, wires the chapter map into every tutorial page
and pilots the full model on DLCV 02 (`deep-learning/dl-02-image-classification`, EN and zh-tw). Baseline: `fc52cb4`.
The unmerged prototype on `origin/wip/ux-content-layers` was mined for ideas (outline levels, priority words, the
disclosure summary) and not merged.

## What was built

| Piece | Where | API |
| --- | --- | --- |
| Leveled lists | `src/styles/global.css` | Every nested Markdown list in prose: level 2 indented under a connector line and a step smaller; level 3 smaller and muted; size stops shrinking after level 3; indent capped on phones. |
| Priority markers / outlines | `src/lib/rehypeOutline.mjs` | An item starting with `[core]`, `[key]` or `[detail]` gets a tag ("Core / 核心", "Important / 重要", "Detail / 細節"). A list with a marker becomes an outline (square level-1 markers in the track colour, wider spacing); its third level folds into a native `<details>` ("2 more details") when it has 3+ items or 240+ characters (a CJK character counts 2.5). |
| Layer 2 | `src/components/Deeper.astro` | `<Deeper title="…" kind?="precise\|example\|derivation\|code\|more" id?>` Markdown children (no headings). Native `<details>`, closed in the HTML; summary shows a kind label, the title and what is inside ("3 equations"). |
| Relations | `src/components/HowItFits.astro` | `<HowItFits center centerNote from to parts fromLabel? toLabel? title?>`; nodes `{ chapter \| href, label?, note?, tag?, children? }`. One nested list; `chapter` is a curriculum slug resolved at build time (unknown slug = build error). |
| Chapter map | `src/components/SectionMap.astro`, `sectionSummaries()` in `src/lib/site.ts`, wired in `src/pages/[lang]/learn/[...slug].astro` | Generated for every tutorial main page from the h2s, their reading time, their h3s (on demand) and a one-line summary quoted from the section (the first sentence of its "Plain version" / "In plain words" / 「白話版」 paragraph, else its first paragraph). House-style frame sections are grouped into "Start" and "Wrap up". A layer switch ("Precise versions: Show all (n)") opens or closes every `<Deeper>` on the page. |

Supporting changes: `chapter-sections.ts` also opens a disclosure that is itself the fragment target (a `<Deeper>` id);
`prose.ts` lets wide equations inside a `<Deeper>` use the column; the PDF exporter prints `<Deeper>` children inline under
a bold title, turns `<HowItFits>` into the same nested list and markers into a bold "Core:" label; the RAG builder drops
the markers; AGENTS.md §3 has a new "Showing structure and layers" subsection with the rules and an MDX example.

## Navigation: merged, not added

The page already had three outlines of the same h2s: the side "On this page" list (≥ 1200 px), the sticky section bar
(< 1200 px) and the console's section picker. The chapter map is the overview at the top; it does not compete:

- **Side list (≥ 1200 px):** hidden (`visibility`, its grid column kept, so nothing moves) while the map is on screen;
  it fades in as the "you are here" tracker once the map scrolls away. Only one outline is visible at a time.
- **Section bar (< 1200 px):** sits directly under the map as its "you are here" line and becomes sticky as before.
- **Phones (≤ 620 px):** the map is one folded 44 px row ("Chapter map · 10 topics · 47 min"); the outline opens on
  demand and each topic's summary is revealed with its h3s (one toggle). Wider screens always show the map open
  (`::details-content` keeps the body visible; browsers without it show the one-row summary, still operable).
- The console's section picker and the chapter tabs are unchanged.

## Design decisions and the guideline behind each

UI/UX Pro Max (`ui-ux-pro-max/references/quick-reference.md`) rules cited by id.

- Levels by indent, connector and size, not colour; priority by word + shape (solid / outlined / dashed) —
  `visual-hierarchy`, `color-not-only`, `color-not-decorative-only`, `weight-hierarchy`.
- Max three list levels, third level muted and folded when long; outline items spaced more than detail items —
  `whitespace-balance`, `progressive-disclosure`, `font-scale`.
- Layer 2 as native closed `<details>` in the HTML: no layout shift, keyboard and screen-reader state for free, find-in-page
  opens it — `layout-shift-avoid`, `content-jumping`, `keyboard-nav`, `voiceover-sr`.
- Phones show the map folded and topic summaries on demand — `content-priority`, `progressive-disclosure`.
- One outline visible at a time instead of a fourth navigation — `avoid-mixed-patterns`, `navigation-consistency`,
  `nav-hierarchy`.
- Deep links and search hits open the target layer — `deep-linking`.
- 44 px targets for touch controls (map head, toggles, layer switch, fold summaries); 36 px rows only for fine pointers —
  `touch-target-size`, `touch-spacing`.
- New colours reuse semantic tokens (`--ink`, `--bg`, `--muted`, `--line-strong`); the only new token, `--connector`,
  is decorative (indentation carries the level) and has a dark variant — `color-semantic`, `color-dark-mode`,
  `color-accessible-pairs`.
- Tabular figures for minutes and counts; "How it fits" keeps the reading measure and stacks vertically with ↓ arrows
  whose CSS alt text is empty — `number-tabular`, `line-length`, `icon-context`.
- Motion: only a 160 ms opacity fade on the side list, off under reduced motion — `reduced-motion`, `state-transition`.

## The pilot (DLCV 02, both languages)

Restructure only: every sentence is kept (moved into a list item or a `<Deeper>` block), the reference list is
unchanged (148 citations before and after in each language, same set of numbers) and zh-tw mirrors EN item for item.
A token diff of old vs new source loses only link-URL tokens (now generated by `<HowItFits>`) and connective words.

- "What you'll learn" and "Key takeaways": leveled, with priority markers.
- "The big picture": the prerequisite and related-chapter paragraph became a `<HowItFits>` (builds on IP 13, DLCV 01,
  ML 02 → this chapter, its four parts from the analogy — data, architecture, training recipe, evaluation — each linking
  to its sections → related DLCV 02A, DLCV 01B). The Li et al. [1] sentence stays in prose.
- Dense section 1, "The binary case: TP, FP, FN, TN": the meaning of each ratio as a leveled outline with priority
  (Chicco–Jurman and Sokolova–Lapalme as folded third-level details); the three display formulas in
  `<Deeper title="The ratios as formulas">`; the rules of thumb as an outline.
- Dense section 2, "ResNet: learning residuals": problem / idea / why it trains / bottleneck as a leveled outline; the
  residual equation, its symbols and the bottleneck weight count in a `<Deeper>`; the key figure stays visible.
- "Calibration": the formal definition and the ECE equation in a `<Deeper>`.
- Code blocks were already Layer 2 (folded) and were left as they are.

## Measurements (production build, Chromium, before = `fc52cb4`)

| Page / viewport | Page height | Interactive elements above the fold | First key point of "What you'll learn" | of "The classification problem" | of "Exercises" |
| --- | ---: | ---: | ---: | ---: | ---: |
| EN 1440×900 | 51,002 → 51,841 px | 46 → 47 | 521 → 1,119 | 1,818 → 3,104 | 40,448 → 41,288 |
| EN 375×812 | 73,699 → 74,373 px | 10 → 11 | 683 → 777 | 2,745 → 3,822 | 57,066 → 57,741 |
| zh-tw 1440×900 | 45,769 → 46,592 px | 53 → 52 | 499 → 1,088 | 1,619 → 2,857 | 35,015 → 35,838 |
| zh-tw 375×812 | 62,587 → 63,072 px | 10 → 11 | 601 → 695 | 2,194 → 3,195 | 45,841 → 46,326 |

How to read it. On desktop the map (≈ 560 px) is new content above the first section, so every section starts about
600 px lower; it replaces scanning the side list, which is hidden meanwhile, and above-the-fold controls stay the same
(the map's links replace the side list's). On phones the folded map costs one 44 px row: the first key point moves
94 px. The restructured sections themselves got shorter (closed): "The classification problem" −311 px (EN 1440),
−347 px (EN 375); "CNN architectures" −166 / −168 px. The relation view is longer than the paragraph it replaces
(+564 px EN 1440, +872 px EN 375 for "The big picture"), the price of showing ten links and four parts explicitly.
Pilot-only Layer-2 savings are modest because DLCV 02 already folds all its code; a chapter-wide rollout of `<Deeper>`
for display math is where the length drops. Raw data: scratchpad `measure-before.json` / `measure-after.json`.

## Validation

- `node scripts/check-mdx.mjs` on both pilot files: pass. `npm run build`: 1,268 pages, Pagefind 1,267.
  `npx astro check`: 0 errors, 0 warnings, 3 existing hints. Node tests: 336/336.
- `node scripts/content-index.mjs`: no change to the committed index. `node scripts/check-paper-guides.mjs`: 188 guides,
  66 chapters, 0 without figures.
- `scripts/check-site-links.py`: 2,233 pages, 156,025 links, 0 broken (covers the `<HowItFits>` links and anchors).
- New `scripts/check-content-layers.py` ([content-layers-validation.json](content-layers-validation.json)): 8 layouts
  (EN/zh-tw × light/dark × 1440/375) — no horizontal scroll, no JS errors; map open on desktop / folded on phones and
  operable; h3 toggles; side list hidden while the map is visible and shown after; 5 outlines with the right tag words;
  folded third level; 3 `<Deeper>` closed, keyboard Enter toggles, layer switch on / off / mixed; print opens every
  layer and restores; `<HowItFits>` links return 200 and anchors exist; deep link to a `<Deeper>` id opens it; without
  JavaScript the blocks work natively and the layer switch stays hidden. Contrast of the new text styles, worst case
  over both themes: 5.78:1 (muted labels, summaries, detail tags), 15.3:1 (tags and titles); `--connector` is decorative.
- Existing regressions on the pilot page: section folding 8 layouts, code folding 1,758 blocks on 358 pages + 12
  layouts, floating console 20 layouts, reading 72 layouts — all pass. `check-course-ui.py` fails at its first step on
  the learn index (`.lt-toggle` not visible) on the baseline build as well; unrelated to this change.
- A 16-page random sample of other chapters with the generated map (both languages, 375 and 1440 px): no overflow, no
  errors. All 1,879 generated summaries are free of TeX, Markdown or HTML remnants.
- PDF exporter: `scripts/book/export.mjs` runs for both languages (all 8 volumes to LaTeX with Pandoc; Tectonic is not
  installed here, so no PDF was compiled). In vol. 3 the `<Deeper>` children print inline, `<HowItFits>` is a nested
  list with internal/online links, and 27 priority labels appear in each language.
- Screenshots (light/dark, EN/zh-tw, 1440/375) were reviewed; the review led to the two-column map, the folded phone map,
  the vertical relation view and clause-aware summary cuts.

This is a web change plus a pilot; the committed PDFs were not rebuilt. Nothing was pushed or deployed.
