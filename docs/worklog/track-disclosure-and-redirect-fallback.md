# Track disclosure and redirect fallback — 2026-11-06

The homepage's eight Track cards now start collapsed. Native details/summary
controls open each preview independently with pointer, Enter or Space, including
when JavaScript is disabled. The closed view retains subject, tutorial count,
reading time and existing reading-progress indicators. An explicit chapter link
opens only the selected subject on the Learn index. The Learn index and full
course tree already supported collapsed subjects.

The supplied Cloudflare log reports `ENOENT` in the `pagefind-index`
`astro:build:done` hook after redirect routes were logged without HTML output.
That hook first calls `preserveRedirectFragments`, which previously read every
legacy redirect's HTML unconditionally. Missing files reproduce the failure
before indexing begins. The helper now creates a small static fallback when a
redirect file is absent, preserving bookmark fragments, no-JavaScript navigation
and search exclusion. Destinations come from the resolved Astro configuration,
including overrides for reorganized NLP chapters. Existing HTML is retained; non-ENOENT filesystem errors
still fail. No Cloudflare redirect rules or credentials are changed.

Regression tests cover absent redirect HTML followed by successful Pagefind
indexing, existing HTML/idempotence, destination-anchor precedence, escaped
URLs and unrelated filesystem failures. All 334 tests pass. Astro check reports
zero errors, zero warnings and three existing hints. Content-index and 188
paper-guide checks pass. The standard production build passes.

Initial browser checks pass eight cases: EN/zh-tw, 375/1440 px, with/without
JavaScript. They verify default closure, independent disclosure, keyboard
operation, hidden links, no viewport overflow and no JavaScript errors. Chapter
links select the correct Learn track; the full course tree stays collapsed.
The final full build with `build.redirects: false` passes, simulating the
missing-HTML host condition: 1,268 content pages and 1,267 indexed pages. The
static checker scans 2,233 pages and 149,951 links with zero broken links.
All eight browser cases pass again against this output; desktop/mobile
screenshots were reviewed. Browser checks confirm legacy training-chapter and
reorganized NLP bookmark forwarding in both languages, working search (40
results for Bayes), and visible reading progress with a closed Track.

The Cloudflare service itself has not been rerun here; this PR provides the
validated source fix for its next build.
