# Minimalist interface — 2026-11-04

The user requested a clearer, simpler overall style. Shared surfaces now use white or neutral charcoal, with quieter text colors and a restrained link accent. Typography and whitespace establish hierarchy. Track colors remain useful identifiers, while decorative colored frames, pills and repeated state backgrounds have been reduced.

Chapter headings are unboxed disclosures. Chapter tabs use a text row with an underline for the current page; prerequisites, experiment links and previous/next navigation no longer sit in separate decorative cards. Paper/supplement lists use light separators, and callouts retain an inset text treatment. The shared header, course grid and current chapter marker follow the same direction. Article text, figures and course structure were not changed.

The floating console is now a single row of four stable text labels: Contents, Sections, Code and Top. It is 54px high and at most 304px wide, with 44px minimum action targets. Decorative icons, the colored top stripe, repeated title and heavy active fills are removed. Detailed navigation stays inside Contents. Dragging, position memory, keyboard controls, section/code folding and default closed code remain available; accessible names and tooltips still explain each fold action.

Validation completed:

- Production build: 1,114 pages; 1,113 indexed for search. Astro check: 0 errors, 0 warnings, 3 existing hints.
- Reading: 72 layouts, including contrast, text size, touch controls, print sizing, reduced motion and mobile navigation.
- Console: 20 layouts plus code-free paper pages, blocked storage and real mobile touch dragging.
- Course navigation: 28 layouts. Code: 12 layouts; 1,730 blocks on 348 pages remain initially folded, with copying, search, bookmarks and no-JS/print behavior preserved.
- Additional screenshot checks: 24 home/blog/chapter layouts across both languages and themes at 375px and 1440px. Reviewed phone/desktop reading, dark mode, the homepage and the paper index visually. Local screenshots: `artifacts/minimal-*.png`.
- Internal links: 2,079 HTML pages, 139,181 links, 0 broken. Content index regenerated without changes; all 149 paper guides retain figures.

Browser checks used Chromium. This is a web interface change; the standalone PDF was not rebuilt. Changes were committed locally without pushing or deploying.
