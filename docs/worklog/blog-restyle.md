# Blog restyle, Medium-style (restyle step 5)

Scope: `src/pages/[lang]/blog/index.astro`, `src/pages/[lang]/blog/[...slug].astro`, new blog-only files
(`src/components/blog/PostCard.astro`, `src/lib/blog-thumb.ts`, `src/scripts/blog-filter.ts`) and new `blog.*` strings in
`src/i18n/ui.ts` (additive; shared file). Design source: `Article.dc.html`. No post text was changed.

## Post page

- 680 px column; body in the site's serif (20 px, 18 px on phones; 19 px with a looser line height for Chinese), headings in the
  site's display sans. Figures and image paragraphs break out to 1000 px (full width below that); code blocks and tables to 820 px.
- Header: breadcrumb (Blog / topic), title, subtitle (the post's `question`, else its description) and one quiet meta line:
  reading time, date, "Updated" (only when it differs from the date), "Sources checked", and a link to the other language when
  the translation exists. Tags moved to quiet chips under the text; the "originally published on" line is kept.
- Series block ("Part n of 3"): compact numbered list of the series' posts in that language, the current one marked
  `aria-current="page"`. It was previously only on the index.
- Blockquotes: pull-quote (ink rule, larger italic; upright for Chinese). Quotes that open with a bold lead-in ("In OpenClaw.")
  are upright in a quiet panel instead, because they are long labelled asides, not quotations.
- Tables: no grid, strong rule under the header, hairlines between rows.
- "Further reading" and "References" (matched by heading id in both languages): smaller, muted, ruled off.
- Footer: "Next in the series" (featured card) then "More posts" (2 cards; 3 when there is no next post). Cards come from the
  same series, then the same topic, then the rest, in that language only; each shows the first imported figure when there is one.

## Index

- Editorial list: "Start here" cards first (one per reader goal), topic filter chips with counts, then one section per topic with
  title, one-line question or description, meta line (reading time, date, updated, checked) and a thumbnail on the right.
- Thumbnail = the first image a post imports (`firstPostImage`, parses the MDX `import` of an `assets/blog/...` file and resolves it
  with `import.meta.glob`). Posts without a figure show none; no placeholders.
- Chips are links to `#<topic>`. With JavaScript they filter the list (hash updated with `replaceState`, "All" clears it, the
  hash on load and on `hashchange` is honoured, so `/blog/#agents` from a post's breadcrumb still works); without JavaScript they
  jump to the section and every post stays visible. Start-here cards hide while a topic filter is active.
- Replaces the old `<details>` topic accordions (the shared `disclosure-index` script is no longer used on this page; the Learn tree still uses it).

## Motion and accessibility

Only `translate` and colour transitions on cards and chips, removed under `prefers-reduced-motion`. Chips and title links are at
least 44 px tall; the title link covers the whole row (focus ring drawn around the row); current chip has `aria-current`.
Dark mode through the site tokens.

## Shared-file changes

`src/i18n/ui.ts`: added `blog.readMin`, `blog.filter`, `blog.all`, `blog.hint`, `blog.startHere`,
`blog.nextInSeries`, `blog.moreFrom`, `blog.tags` in both languages.
