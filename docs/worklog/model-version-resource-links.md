# Model versions and benchmark resources — 2026-11-06

Three existing AI-authored selection guides were updated in both languages:
image generation, video generation, and model/benchmark selection. All original
headings remain; welcome/introductory personal blog prose is untouched.

## Scope

- Muse is now explicit in Meta's family overview, hosted versions and downloadable
  releases: Wonder1.0/Spark1.3, Spark Open and Glimmer are linked to their actual
  access/download documentation. Muse, Muse Code, API IDs and subscription-only
  Wonder Full are distinguished.
- Image selection includes Meta `muse-image-1.0`, release/model-card links for
  hosted and downloadable alternatives, and Google Muse2023/MaskGIT as a separate
  research path linked to the existing bilingual ML09 section.
- Video selection links exact hosted IDs and checkpoint variants. Google's
  retired GeminiAPI Veo3.1Preview endpoints are distinguished from documented
  Cloud GA Veo3.1 and Omni1.2Flash. MetaMuseVideo's July announcement is labeled
  as a preview without an invented public API ID; MicrosoftMuse/WHAM remains a
  separate gameplay research model.
- Benchmarks now have direct version/methodology/results links: SWE-bench,
  Terminal-Bench/Harbor, BFCL, Arena, ArtificialAnalysis, LiveCodeBench, HELM,
  GenEval/2, T2I-CompBench, DPG-Bench and VBench/2. Comparison guidance records
  exact model, suite/version, task split, evaluator, effort and sampling budget.
  No performance ranking or reproduced benchmark score is introduced.

Primary URLs were opened by the content agents before writing. Public sources
include provider documentation, official announcements/model cards, author
repositories and benchmark methodology pages. The English/Chinese external URL
sets and reference numbering agree. References: image28, video23, models35 per
language; every entry is cited. The accompanying JSON records the URLs.

## Integration verification

All six changed MDX files pass. Astro check: zero errors/warnings, three existing
hints. Final Wrangler static deployment dry-run passes in **130.14 seconds**;
no production deployment was performed. This local run follows the earlier
systems checkpoint; timing is not a hosted performance guarantee.

All 12 EN/zh-tw desktop/mobile resource layouts pass viewport, table and JS-error
checks. The benchmark table contains all seven intended rows. MuseImage and
MuseWonder version searches return results in both languages. Representative
screenshots were reviewed. The full static scan reports **2,233 pages / 150,095
links / zero broken links**.

Rendered differences are limited to the six edited posts, two blog indexes
(reading times), and two About pages (automatically derived prose counts).
No pages were added or removed. All course content remains equal to the prior
systems checkpoint. Existing bookmarks and internal Muse/ML09 links resolve.

The systems checkpoint separately records 336 passing tests,396 course layouts,
92 scalar-card interactions, failed-chunk fallback, console stress comparisons
and content-page/browser security verification.
