# Whole-course textbook refresh — 2026-11-05

Goal: rescan all teaching material, correct substantive errors, fill conceptual
and worked-example gaps, and update the downloadable books. Preserve the
beginner learning route, bilingual parity, existing bookmarks and progressive
disclosure. This is a source/content refresh, not a reproduction of every cited
paper's training or benchmark experiment.

Baseline: `116d724` (merged PR #6), synchronized from `origin/main`. Work branch:
`feat/textbook-refresh-2026-11`. All eight tracks and 97 EN/zh-tw chapter pairs
are included, along with their supplementary notes and paper guides. Original
personal blog text is unchanged.

## Review coverage

Six independent track reviews own disjoint content files; integration checks
cover the entire corpus. Each report distinguishes reviewed chapters from those
changed, records sources, and identifies the executed examples.

| Tracks | Pairs | Detailed audit |
| --- | ---: | --- |
| Foundations and Pattern Recognition | 12 | [Audit](textbook-refresh-foundations-pr.md) |
| Image Processing | 13 | [Audit](textbook-refresh-ip.md) |
| Machine Learning | 15 | [Audit](textbook-refresh-ml.md) |
| Deep Learning for Computer Vision | 29 | [Audit](textbook-refresh-dlcv.md) |
| Natural Language Processing | 12 | [Audit](textbook-refresh-nlp.md) |
| Optimization and Information Theory | 16 | [Audit](textbook-refresh-opt-it.md) |

The baseline structural scan finds all 97 translations, matching heading depths,
fenced code and inline labs, existing referenced figures/notebooks, valid
prerequisite targets and no prerequisite cycles. Numeric-reference candidates
were checked manually: bracketed configuration arrays and example intervals are
not missing references. Restormer configuration arrays need code markup to
avoid being interpreted as citations; locator citations such as `[8, Sec. 2.5.1]`
are valid citations even when a simple numeric-only scanner misses them.

The convolution notebook's three code cells execute sequentially in the project
Python environment; the reproduced maximum difference from OpenCV is `0.0`.

## Corrections and validation

Reviewed all 97 pairs; 45 chapter pairs have targeted changes (87 source files,
including three one-language parity corrections). The other 52 pairs are
recorded as reviewed without a necessary edit. Key repairs include Fourier
phase, GVF stability, interpolation boundaries, statistical assumptions,
rank/normalization conditions, evaluation definitions, masking and sampling,
and variable reuse that contaminated later executable examples.

All 627 English Python blocks pass in the project environment, with byte-identical
bilingual code. Track reports state whether blocks were executed independently
or in a shared chapter namespace. No original research training run is claimed.
All 194 tutorial MDX files pass; full-course structure/prerequisite checks pass.
Production build: 1,268 generated pages and 1,267 Pagefind-indexed pages.
All 31 test files pass. Astro check reports zero errors and warnings (three
pre-existing hints). Content index, 188 paper guides, curriculum and playground
coverage checks pass. No legacy heading was removed.

Playwright passes all 396 course layouts (97 main chapters plus two new note
pages, in both languages at 375/1440 px): default code folding, public fold
controls, keyboard disclosure, translation routes and viewport overflow. About
budget checks pass both languages and viewports, including reset, invalid
inputs, zero/double usage, exchange rates and JavaScript-disabled rendering.
Representative screenshots were reviewed. The static link checker reports
2,233 HTML pages, 149,951 links and zero broken links (including redirects).
RAG rebuild produces 8,379 chunks and retrieves the new optional note correctly.
Downloadable-book validation and source checkpoint are recorded separately.

## About-page budget addition

The user chose AI/API development cost plus operation, rather than a human
replacement-cost valuation. The new bilingual About component presents explicit
planning usage allowances and a transparent weighted model mix. No historical
usage ledger or actual invoice is claimed. Current standard short-context prices
were opened on 2026-11-05: GPT-6.1 Sol USD 2/10, Claude Opus 5.5 USD 4/20, GPT-6
Astra USD 10/50 per million input/output tokens. The assumed 60/30/10 mix gives
USD 3.40/17.00. Development input/output allowances are 50–150M / 10–30M;
monthly maintenance allowances are 1–5M / 0.2–1M, plus an annual USD 20–100
domain/tool allocation and 25% contingency. Default first-year total is
USD 552–1,910 (TWD 17,664–61,120 at the explicit non-live planning rate of 32).
Usage and exchange-rate inputs are adjustable. Subscription allowances are not
added twice to API-equivalent costs. Labor, equipment and tax are excluded.

Primary sources: https://developers.openai.com/api/docs/pricing,
https://platform.claude.com/docs/en/about-claude/pricing,
https://developers.cloudflare.com/pages/functions/pricing/,
https://developers.cloudflare.com/pages/platform/limits/.

## Remaining evidence limit

SeAFusion original numerical/ablation tables could not be recovered from its
restricted primary full text. The existing guide continues to identify that
specific gap; later papers' comparison rows are not substituted for original
results. See the DLCV audit. Existing future-calendar update labels in some
chapters were preserved rather than backdated; the audit uses its explicit UTC
check date.

## Downloadable books

All eight PDFs were exported from committed source `2b3c30d` (2026-11-05),
including the agent and generation-tool essays previously merged in PR #6.
Supplementary notes and paper guides remain online, with links from the books.
The books total 4,461 pages and 12,938 PDF links. Chapter/code/equation/figure
coverage, embedded fonts, missing glyphs, internal/named destinations and both
horizontal and vertical page boundaries pass. All PDFs remain below 25 MiB
(largest: 14.3 MiB). English/Chinese chapter, contents and blog previews were
reviewed. Detailed counts, hashes and toolchain provenance are in
[textbook-refresh-pdfs.json](textbook-refresh-pdfs.json).

The Tectonic default package bundle returned HTTP 403. The compatible installed
XeLaTeX compiler was used instead, with downloaded TeX Gyre fonts and pinned
xeCJK source. Exported TeX received only font-path, PDF-creator metadata and
standard figure-optimizer path adjustments; teaching content was unchanged.

Final build after publishing the artifacts passes: 1,268 pages, 1,267 indexed
pages. All eight PDFs served from the built site match their source hashes and
recorded byte sizes. Both download languages at 375/1440 px show the correct
source checkpoint, eight downloads and no viewport overflow or JavaScript
errors. Final Astro check: zero errors, zero warnings, three existing hints.
