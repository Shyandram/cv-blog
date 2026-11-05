# Code disclosure and reading flow — 2026-11-04

Code now starts folded in the generated HTML. Readers can follow the explanation, identify a snippet by language and line count, then expand it when ready to inspect or try the example. A separate copy button works while the snippet is closed; the page explains that examples run in the corresponding environment. No execution button is implied.

The page offers separate code and chapter controls. Expanding all code also opens its enclosing chapter or hint so the requested examples are visible. Chapter-only expansion leaves code folded. Fresh page visits return to the default state; bookmarks into code reveal the target and its ancestors.

Implementation:

- `rehypeCodeFolding.mjs` adds native `details` wrappers while retaining each original pre/code tree, syntax highlighting and existing IDs. Generated IDs avoid collisions. Summary labels stay out of search; snippet contents remain indexed.
- `code-folds.ts` adds shared copy and bulk controls to prose pages. It copies source text while hidden, preserving indentation and line breaks. Clipboard failure opens and focuses the code for manual selection, with an announced error.
- Native keyboard disclosure works without scripts. Printing opens all code and restores the previous state afterward; print CSS also exposes closed code in browsers supporting `::details-content` with JavaScript disabled.
- Controls keep 44px minimum targets and separate hit areas on phones. Original post text and tutorial snippets are not edited. The independent PDF export is unchanged.

Validation:

- Production build: 1,114 pages, 1,113 indexed for search. Astro check: 0 errors, 0 warnings, 3 existing unused-symbol hints.
- All 1,730 code blocks across 348 built pages start in native closed disclosures with bookmarks.
- 12 Chromium layouts: both languages and themes at 320, 375 and 1440px. Default state, exact clipboard contents, source preservation, syntax highlighting, separate touch targets, keyboard, bulk controls, mixed-state print restoration, direct bookmarks and clipboard failure fallback passed.
- Native disclosure and full-code printing with JavaScript disabled passed in both languages. Pagefind finds the course through `torch.manual_seed` and retains the original code in its indexed content.
- Reading regression: 72 layouts passed. Chapter folding: 8 layouts, keyboard, deep links, print restoration and no-JS reading passed. Mobile and desktop screenshots were visually inspected.
- Two transformation tests passed for literal snippet preservation, language/line labels, bookmark preservation, collision handling and idempotence.
- Content index regenerated without changes; all 149 paper guides retain figures. Internal links: 2,079 HTML pages, 139,181 links, 0 broken.

See [code-folding-validation.json](code-folding-validation.json) and [the design decisions](../design/readability.md). Browser results are from Chromium; no-JS print expansion uses the modern `::details-content` CSS feature. The existing print events provide expansion and restoration when scripts are enabled.
