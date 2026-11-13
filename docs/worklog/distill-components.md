# Distill-style components (restyle step 4)

New, self-contained components (own styles, no change to the tutorial route) and one pilot section.

## `MarginNote` (`src/components/MarginNote.astro`)
- Written inline: `$t_i$<MarginNote>true positives plus misses</MarginNote> plus ...`; replaces a parenthetical, adds no facts.
- Margin mode needs the window to be at least 1200 px wide AND the `.prose` column at least 820 px wide. The note takes the room
  beside the text measure (`100cqw - --measure - 20px`, clamped to 120–200 px); with the d2l-style chapter layout (680 px measure in
  a column up to 880 px) that is about 128 px at 1440 px and 180 px when the column is at its full width. The column test is a CSS container query (`.prose { container-type: inline-size }`, set by this
  component's global style, so only on pages that use it), so the note never runs under a sidebar whatever the layout does.
  In margin mode the note floats right with a negative margin: the prose column does not move.
- Otherwise a small script moves each note after its paragraph (a left-ruled aside, numbered; list items and table cells get it
  inside the cell), and puts it back when the margin becomes available (matchMedia plus ResizeObserver). Without JavaScript the
  note stays in the sentence in brackets with its number: nothing is lost.
- Numbering is a CSS counter on `body`; the marker is `aria-hidden`, the note is `role="note"` and is read in place.
- Tuning: `--mn-width` (200 px at most), `--mn-gap`.

## `LabFigure` (`src/components/LabFigure.astro`)
- `<LabFigure caption="..."><Lab id="..." /></LabFigure>`: wraps the existing `Lab` (not reimplemented); a `figure` with a caption
  "Interactive figure n — ..." / "互動圖 n — ..." (CSS counter, separate from the hand-numbered static figures).
- From 1200 px the figure is centred on the column and widened to `--lab-breakout` (default 1000 px), but never beyond
  `100vw - 2 * --lab-side` (the room kept for sidebars; default 380 px) and never narrower than the column. Below 1200 px it is
  the column's full width (full width on phones).
- Interaction with the current tutorial layout: the left rail and the right ToC leave no room beside the 892 px column, so with the
  default `--lab-side` the breakout is zero at 1440 px and about 14 px per side at 1680 px. A layout without those sidebars sets
  `--lab-side: 16px` on the page (or on `.labfig`) to get the full 1000 px. Checked by overriding the grid in Playwright: no overlap,
  no horizontal scroll.

## Pilot
`dl-03b-image-segmentation` (EN and zh-tw), section "Measuring semantic segmentation": the lab is in `LabFigure` with a caption
that restates the lab's own controls and readouts (no numbers); two parentheticals of the IoU sentence ("true positives plus
misses", "false alarms" / 真陽性加上漏掉的、誤報) became margin notes.

## Validation (2026-11-13)
`check-mdx` both files; `npx astro check` 0/0 (3 existing hints); `npm run build` complete; `content-index --check` up to
date; Node tests 336/336; `check-site-links` 2233 pages, 0 broken; `check-labs` all passed. Screenshots EN and zh-tw at 1680, 1440,
1280 and 375 px (dark at 1440 and 375): no horizontal scroll, no page errors; notes in the margin at 1440 and 1680, below the
paragraph at 1280 and 375.
