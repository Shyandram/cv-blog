# Sitewide minimalist interface — 2026-11-04

The user confirmed that the minimalist direction applies to the entire website. This extends the reading-interface update to home, course overview/tree, blog list/articles, playground, search, downloads, about and the shared components used by those pages.

The blog index now uses a readable single-column article list. Playground entries, the chapter directory, download editions, course-tree roots, statistics and learning-progress panels use whitespace and light separators instead of nested colored cards. Search suggestions and track filters use text controls with clear hover/selected states. Home's demonstration is unboxed. Shared controls use a small corner radius and the restyled buttons retain 44px minimum targets; input boundaries, keyboard focus and data-bearing plot colors remain meaningful.

Experiment controls, predictions, inline labs, practice feedback and the resume-reading link share this quieter treatment. Functional section/code disclosures and the floating console retain the previously validated behavior. Original article text, experiment algorithms, references and PDF contents were not edited.

Validation against the production build:

- Build: 1,114 pages; Pagefind: 1,113 indexed pages. Astro check: 0 errors, 0 warnings, 3 existing hints.
- One-off browser review: 96 layouts across eight page families, both languages, both themes, and 320/375/1440px widths. No page overflow or JavaScript errors. At 375px, also exercised playground search/filtering, image-lab slider/reset, plot enlargement, search suggestions/results/clearing and a blog article. Verified the 404 page at 375/1440px. See [site-minimalist-validation.json](site-minimalist-validation.json).
- Course navigation: 28 layouts passed, including the course tree and mobile menus.
- Downloads: four layouts passed; all eight built PDF copies match their source hashes and edition metadata.
- Internal links: 2,079 HTML pages, 139,181 links, 0 broken. Content index regenerated without changes. Paper-guide coverage remains 149 guides across 40 chapters, with no missing figures.
- Visually reviewed mobile playground/about/search, desktop blog/downloads and light/dark screenshots. Local screenshots are named `artifacts/site-minimal-*.png`.

Browser review used Chromium. Existing lazy experiment activation was allowed to complete after scrolling the controls into view. The PDF files were checked but not rebuilt for this web-only change. Committed locally without pushing or deploying.
