# Chapter page redesign (restyle step 3, d2l.ai-style book page)

Scope: the tutorial route `src/pages/[lang]/learn/[...slug].astro` (main pages, `notes/` and `papers/` index and item pages)
and the components and scripts it uses. Design source: the approved "Book chapter page, d2l.ai-style" artboard
(`D2L.dc.html`), implemented with the site's own tokens and fonts (no new web fonts; the CSP blocks external ones).

## Layout

- **Left: one book sidebar** (≥ 900 px, sticky under the header, scrolls on its own). A heading names the course (code badge
  in the track colour on an 8–9 % tint, as on the Learn page, plus the course name; it links to `/learn/#<track>`). Below
  it the curriculum tree of that course (`ChapterTree`): every chapter numbered in a column of its own ("02", "02A"; the
  course code is in the heading), the current chapter open to its sub-chapters, Supplementary notes and Paper guides, and
  the current page highlighted with a track-tinted row. Rows are compact for mouse users (34 px, targets ≥ 24 px) and stay
  44 px on touch screens. A "Hide contents" button collapses the sidebar and a "Contents" button in the chapter header
  brings it back; the choice is kept in this browser only (`pp:rail`, try/catch via `src/scripts/progress.ts`). Without
  JavaScript the sidebar is shown and both buttons stay hidden.
- **Phones (< 900 px): the "Contents" sheet.** The existing sticky section bar (`ReaderTools`, `.rt-mnav`) already holds
  "On this page", the chapter's notes and guides, and the course tree; its summary now starts with a visible "Contents" /
  「目錄」 tag. It is a native `<details>`, so it also opens without JavaScript.
- **Right: "On this page"** for the current page only, at ≥ 1200 px, 13 px type (h3s still open on demand with the 44 px
  disclosure buttons; scroll-spy and ✓ marks unchanged). Between 900 and 1199 px its role is taken by the section bar.
- **Centre: one reading column.** Article up to 880 px (code, figures, tables, labs use it); prose at a 680 px measure
  (`--measure: 42.5rem` on the page), Source Serif 20 px / 1.7 (zh-tw: system CJK, 1.85) from 621 px up, 18 px on phones
  (global rule). Print rules are unchanged (they still reset the column and line height).

## Chapter header

Breadcrumbs; the chapter number above the title in the track colour (the `<h1>` text stays "DLCV 02 · Title" for search,
progress and screen readers: the separator is visually hidden); one quiet meta line (course · level · minutes · updated
date); "Open in Colab" / "Notebook" buttons only when the frontmatter has `notebook` (today ML 03); the prerequisite line;
then one rule that holds the chapter tabs (Chapter · Supplementary notes · Paper guides) on the left and the reading tools
on the right.

## The floating console

On chapter pages the floating, draggable console is gone. Its functions are now:

| Console function | Where it is now |
|---|---|
| Contents / jump to a section, previous / next section | right "On this page" (≥ 1200 px), the section bar (< 1200 px), the left sidebar |
| Collapse / expand all sections | "Collapse all sections" button in the header rule (`.reader-console-fold-sections`) |
| Expand / collapse all code | "Expand all code" button in the header rule (`.reader-console-fold-code`; hidden on pages without code) |
| Back to top | the page's own round button (`.rt-top`, which the console used to hide) |
| Reading progress | the 3 px progress bar at the top and the ✓ marks in the tables of contents |

`src/scripts/reading-controls.ts` now uses the docked buttons when the page has a `.reader-dock`
(`src/scripts/docked-reading-tools.ts`), and the floating console otherwise, so blog posts (step 5's scope) keep it
unchanged. The button labels always name what a click does and update when a single fold is toggled.

## Code blocks

Code still starts folded (unchanged behaviour, covered by `check-code-folding.py`). The frame is cleaner: a quiet header
(label, Expand, Copy), the code on the surface colour, and, new, its **output attached below**: in tutorials, a
plain-text block that directly follows a code block (only whitespace between them) is that code's printed output, so
`rehypeCodeFolding.mjs` moves it into the same frame as `div.rt-output` with an "Output" / 「輸出」 label. The output stays
visible while the code is folded (before, it was a second folded block labelled "Code · n lines"). All 149 such pairs in
the English sources were checked: each is a Python block that prints, followed by its output (149 × 2 languages = 298 on the
site). Text blocks introduced by a sentence ("Output:", "The executed output is:") are left as they are. No tabs were added
(no chapter has equivalent alternative code). Posts are not affected (the pairing is limited to `/tutorials/` files).

## Exercises

The Exercises section gets the mockup's closing row: "Show hints" / 「顯示提示」 (opens or closes every hint of the section;
the label follows the state) and a primary "Next: <next chapter> →" link. Hints keep their native `<details>`, now with a
quiet left rule and a chevron. Fixed on the way: `CheckPanel`'s script looked for the exercises next to the heading, but
the heading had already been moved into the section's `<summary>` by `chapter-sections.ts`, so the "try first" prompt was
rendered inside the heading row (one letter per line) and no "Solved" boxes appeared. It now works on the section body,
whichever script runs first.

## Kept

Chapter tabs, notes/papers pages and their pagers, progress tracking (`pp:p`, `pp:s`, `pp:last`, `pp:x`), section folding,
code folding and copy, search (Pagefind meta and filters unchanged), language switch, "Explore this chapter" lab links and
lab embeds, KaTeX, figures, citations and reference previews, the edit link, previous/next chapter, "Check your
understanding". Every URL is unchanged.

## Files

- `src/pages/[lang]/learn/[...slug].astro` (layout, header, styles); `src/components/ChapterTree.astro` (optional number
  column); `src/components/ReaderTools.astro` ("Contents" tag); `src/components/CheckPanel.astro` (fix, closing row).
- New: `src/scripts/book-rail.ts`, `src/scripts/docked-reading-tools.ts`. Changed: `src/scripts/reading-controls.ts`,
  `src/lib/rehypeCodeFolding.mjs` (+ unit test in `scripts/tests/code-folding.test.ts`).
- Shared file: `src/i18n/ui.ts`, nine new keys in each language (`tut.contents`, `tut.hideContents`, `tut.showContents`,
  `tut.updated`, `tut.notebookShort`, `tut.readingTools`, `tut.nextLabel`, `ex.showHints`, `ex.hideHints`). Additive only.
- Tests adapted: `scripts/check-floating-tools.py` now runs the full console suite on a blog post (the console's remaining
  home; its sections action is disabled there because posts have no folding sections) and adds a docked-tools suite on
  chapter pages (both languages, 375/1024/1440 px: no console, labels, single-fold truthfulness, back to top, sidebar
  hide/show with persistence, code-free paper pages). `scripts/check-code-folding.py` counts attached outputs separately
  (they are deliberately outside the fold).

## Validation (2026-11-13)

- `npx astro check`: 0 errors, 0 warnings (3 pre-existing hints). `npm run build`: 1268 pages; 211 warnings, same as `main`.
- `node scripts/content-index.mjs --check`: up to date. Node tests: 337/337 pass (one new). `check-content-stats`: pass.
- `scripts/check-site-links.py`: 2233 pages, 151,599 links, 0 broken.
- Browser checks (Playwright, Chromium): `check-floating-tools` (20 console layouts on a post + 6 docked layouts + papers,
  blocked storage, touch drag), `check-section-folding` (8 layouts), `check-code-folding` (1476 folded blocks on 364 pages,
  298 attached outputs, 12 layouts, no-JS, print, Pagefind), `check-readability` (72 layouts: 20 px prose ≥ 621 px,
  contrast, touch targets, section bar), `check-course-ui` (28), `check-paper-guides-ui`, `check-labs`,
  `check-search-overview`: all pass.
- Pre-existing failures, unchanged by this step (they fail on `main`'s build too): `check-supplements-ui.py` expects the
  SeAFusion output text in the visible prose, but that block is introduced by a sentence and stays folded;
  `check-security.py` stops at the legacy redirect pages without a CSP meta.
- Screenshots: DLCV 02 and IP 03 before/after, EN and zh-tw, 1440 and 375 px, light and dark; six more chapters
  (Foundations, NLP, OPT, IT, PR, ML 03 with Colab) and a note page; no horizontal scroll, no page errors. Without
  JavaScript: sidebar shown (≥ 900 px), reading tools hidden, all 18 sections readable. Keyboard: Tab reaches the course
  link, "Hide contents", then the header tools in order.
