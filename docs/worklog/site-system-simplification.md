# Site system simplification — 2026-11-06

Baseline: `96f7ce1` (merged static deployment PR #9). This pass retains the
curriculum, chapter/guide URLs, reading-console controls and deterministic
experiments while reducing repeated work and unnecessary initial browser code.

## Changes

- ExperimentPlayground loads only a small dispatch table initially, then imports
  the relevant computation group near the viewport. Server-rendered default
  plots remain readable without JavaScript or when a chunk fails. One shared
  server-side group map assigns each experiment; coverage/default tests prevent
  client/server drift. The unused empty foundations experiment module is removed.
- The reading console uses one delegated toggle listener, cached headings and
  the existing animation-frame scheduler. Expand-all no longer recalculates all
  fold states once for every individual fold; repeated registration does not
  accumulate click handlers.
- A single production tutorial-render cache now serves both optional-page
  discovery and chapter rendering. The 194 source entries can reuse compiled
  components/metadata across 1,230 tutorial pages instead of 1,424 render calls.
  Reading-time section analysis is cached by body/language. Development bypasses
  these caches, preserving source-edit behavior. No heading parser or extra
  generated metadata pipeline is introduced.
- Type checking excludes generated book output and local validation artifacts,
  matching their ignored/generated role. Security browser checks explicitly
  scroll lazy controls into view before testing them.

## Measurements and checks

Same local Wrangler4.158.0 dry-run entry point as PR#9: **147.12 seconds**, versus
165.74 seconds in the prior recorded run. This is a local single-run comparison,
not a guarantee of hosted timing. Maximum child-process RSS: 3,930,096 KiB.

Fresh browser contexts measured raw initial script-response bodies:

| Route | Before | After |
| --- | ---: | ---: |
| Home | 204,947 B | 204,842 B |
| NLP attention chapter | 243,751 B | 243,646 B |
| Image-array chapter | 252,459 B | 252,422 B |
| Playground | 382,579 B | 216,225 B |

Playground initial JavaScript drops **43.5%**; local gzip estimates are
124,596 → 56,087 bytes. These are response-body/gzip measurements, not Cloudflare
wire-transfer measurements. Computation groups load as needed afterward.

The synthetic console stress fixture has 120 sections and 240 code folds:
per-fold toggle listeners **360 → 0** (one delegated listener remains),
expand-all group-state calculations **361 → 3**. Eight EN/zh-tw 375/1440px
old/new comparisons pass folding/mixed states, section navigation, Escape,
keyboard movement/reset and repeated registration.

Integration verification:

- All **336 tests** pass, including two new experiment-loader parity tests.
- Astro check: zero errors/warnings, three existing hints.
- All **1,267 content pages** retain normalized text, exact code text, ordered
  bookmark IDs and links compared with the baseline. No content page added/removed
  during this systems-only checkpoint.
- All **396 bilingual desktop/mobile course/note layouts** pass.
- All 46 scalar experiment cards pass activation/input/reset in both languages
  (**92 interactions**); a blocked computation chunk retains the default SVG and
  displays the error message. Image-lab computations remain covered by their
  existing tests; the 73 total playground items include 26 image labs and the
  kernel widget.
- Content-page CSP/hash audit and existing browser security checks pass, including
  lazy controls, search, themes/language and injection blocking.
- Content index and 188 paper-guide checks pass. Static link scan: 2,233 HTML
  pages / 149,951 links / zero broken links.

The user subsequently requested expanded model-version/benchmark links. Those
editorial additions are a separate checkpoint; the content equality above refers
specifically to the systems changes.
