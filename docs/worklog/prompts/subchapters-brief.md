# Brief: every Supplementary note and Paper guide becomes its own clickable sub-chapter page

Owner's request (verbatim, Chinese): 「補充應該在畫面上也可以點擊，且每個補充都應該是類似一個新章節（subchapter）」 — the optional
items must be clickable in the page's navigation, and each note / paper guide should be a page of its own, like a sub-chapter.

You work in your own git worktree (`pwd`). First: `ln -s /home/user/cv-blog/node_modules node_modules; ln -s /home/user/cv-blog/.venv .venv;
export PATH=$HOME/.nvm/versions/node/v22.23.3/bin:$PATH`. Never touch /home/user/cv-blog. Read AGENTS.md completely first (git safety:
no git stash/reset/checkout/restore/clean, explicit paths only, do not push; validation section 6). Commit your finished, validated work in
your worktree branch, message ending with
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
`Claude-Session: https://claude.ai/code/session_01LDLcNdAG6dTWpQHbdjJauG`.

## How it works today (read these files fully)
- src/lib/rehypeSupplementZones.mjs splits each tutorial into MainZone / SupplementZone (`## Supplementary notes`/`## 補充資料`) /
  PapersZone (`## Paper guides`/`## 論文導讀`) / rest (References), and rewrites in-page links across zones (main → `notes/#id`, …).
- src/pages/[lang]/learn/[...slug].astro builds the main page plus one page per zone (`…/<slug>/notes/`, `…/<slug>/papers/`) and picks
  which zone renders via components (src/components/zones/Show|Hide|NotesTeaser.astro). The main page shows a NotesTeaser card.
- The left rail lists the track's chapters; the right TOC lists the h2s (and h3s of the current section). Notes are not in either.
- scripts/content-index.mjs (docs/content-map, docs/content-index.json; CI runs `--check`) and scripts/rag/build.mjs emit note URLs as
  `…/notes/#id`. About 70 links in src/content point at `…/<slug>/notes/#id` or `…/papers/#id` of OTHER chapters. AGENTS.md documents
  the old URL rule. scripts/book/ builds PDFs from the MDX (notes at the chapter end) — must keep working.

## What to build
1. **One page per item.** Each `###` under the notes section → `/<lang>/learn/<track>/<slug>/notes/<h3-id>/`; each `###` under Paper
   guides → `/<lang>/learn/<track>/<slug>/papers/<h3-id>/`. The h3 id is the slug rehypeHeadingIds already gives (take it from
   `render(entry).headings`, which is available in getStaticPaths — do not re-slugify by hand). The page shows that item only, with the
   item title as the page h1-level title, a line saying which chapter it belongs to (link) and that it is optional, then its content,
   then the reference list (as the current notes page does), then a pager: ← previous item / next item → within the same zone, plus
   "Back to the chapter". Page <title>: "<item> · <chapter title>". Each item page is its own Pagefind entry.
2. **Index pages.** `…/notes/` and `…/papers/` become sub-chapter indexes: the section's intro sentence, then a card list of the items
   (title + the first sentence of the item, which by house rule says what it is about) linking to each item page. No item bodies on the
   index (avoid duplicate search hits). Keep the old-link forwarding: `…/notes/#<id>` (and `…/#<id>` on the main page via NotesTeaser)
   must redirect in the browser to the item page that contains that id (any id inside an item — h3, h4, figure — not only the h3).
3. **Clickable in the navigation.** In the left rail, the current chapter expands into sub-entries: its notes and paper guides (two small
   labelled groups, e.g. "Supplementary notes · 補充資料" and "Paper guides · 論文導讀"), each item a link; the current item is highlighted
   on item pages (aria-current). On the main page's right TOC, add the same two groups at the end (collapsed to the group headings if the
   list is long; your call — keep it legible). The NotesTeaser card on the main page lists items linking to the item pages. Mobile
   (< 900 px hides the rail): make sure the items are still reachable (the teaser card, the index page, and whatever mobile navigation
   ReaderTools.astro offers — check it and add the sub-chapters there if it lists sections).
4. **Links.** In-page links in the MDX (`[…](#note-id)` from the main text, note → main, note → other note, note → paper guide) are
   rewritten at build time by the rehype plugin to the right relative URL for each page (main → `notes/<item>/#id`, item → `../../#id`,
   item → `../<other>/#id`, notes item → `../../papers/<item>/#id`, …). Ids inside an item map to that item's page. Migrate the ~70
   cross-chapter links in src/content (both languages) to the new form `…/<slug>/notes/<item-id>/#<id>` (or without the hash when it
   targets the item's own h3) with a careful script; verify every one resolves in the built site. Update AGENTS.md (the Supplementary
   notes / Paper guides paragraphs: URL rule for links from another page) — only those sentences.
5. **Generated maps.** Update scripts/content-index.mjs (and the README text it writes) and scripts/rag/build.mjs so URLs for items are
   the item pages; regenerate docs/content-map and docs/content-index.json. Keep the PDF build working (run
   `node scripts/book/publish.mjs --help` or read it to see whether URLs appear; if it builds links to notes, update them).
6. Design: follow the site's existing look (CSS variables, fonts, the rail/TOC styles); no new dependencies. Legible at 375 px, no
   horizontal scroll, keyboard accessible, both languages (zh-tw UI strings in Taiwanese Traditional Chinese).

## Validate
`node scripts/check-mdx.mjs` on every changed MDX; `node scripts/check-playground.mjs`;
`node --experimental-strip-types --test scripts/tests/*.test.ts`; `npx astro check` (0 errors, 0 warnings); `npm run build`;
`node scripts/content-index.mjs --check` after regenerating; a link checker over dist/ (every internal href with or without #fragment
resolves to an existing page and id; adapt /tmp/claude-0/-home-user-cv-blog/46b65cc4-04fc-53dc-9c17-ea7809aadfc9/scratchpad/xanchors.py
for the new layout or write your own) — 0 broken; check that old URLs `…/notes/#id` forward correctly in Chromium (Playwright, one EN
and one zh-tw case, plus a main-page `#note-id` case). Screenshots with Playwright at 1440 and 375 px of: a main tutorial page (rail +
TOC + teaser), a notes index page, a note page, a paper-guide page, EN and zh-tw; look at them yourself and fix what looks wrong.
Good test chapters: deep-learning/dl-05a-lightweight-enhancement (3 notes, 4 guides), deep-learning/dl-01c-model-compression,
image-processing/dip-03-intensity-spatial-filtering.

## Final report
Worktree path, branch, commit hash; URL scheme; files changed; how link rewriting and forwarding work; how many content links were
migrated; validation results with numbers (pages built, links checked, broken = 0); screenshots taken (paths) and what you fixed;
anything uncertain.
