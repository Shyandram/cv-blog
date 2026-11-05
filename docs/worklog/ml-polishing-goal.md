# ML polishing goal

Status: completed — editorial pass and validation (2026-11-06). Baseline: `c42c407`. Scope: all 15 ML chapters and their Traditional Chinese counterparts (30 pages).

Make the ML course easier to enter and review: concise objectives, direct explanations, clear assumptions and useful takeaways. Continue the [DLCV editorial standard](cvdl-polishing-goal.md), while retaining beginner-friendly examples and analogies.

Completion criteria:

- Edit every chapter's objectives and takeaways, plus selected explanatory paragraphs throughout the track; record actual scope and before/after prose counts.
- Mirror changes in English and Traditional Chinese; qualify misleading shortcuts without adding unsupported methods, benchmarks or claims.
- Preserve headings, URLs, prerequisites, technical derivations, code, figures, references and detailed optional material. Repeated inline formulas in summaries may be shortened when the derivation remains in the body.
- Check MDX/KaTeX, generated index, paper-guide coverage, Astro diagnostics, build, internal links and bilingual desktop/mobile reading controls.
- Verify built copy matches the final source, record results and commit the validated work locally.

This is an editorial pass, not a new paper-source audit or benchmark reproduction. Existing source limitations remain tracked separately.

## Editorial result

All 15 chapter pairs have shorter main-path prose. Each now has five learning objectives and six takeaways. The pass rewrites both summary sections and 46 selected explanatory paragraphs per language (92 paragraphs total); it preserves other paragraphs and the introductory analogies. It is not a line-by-line rewrite of every paragraph.

Corrections qualify common shortcuts: stationary points versus minima; SVD reflections and signed determinants; graphical dependence versus causation; assumptions behind PCA equivalence, equivariance and optimization; and the limits of calibration, generation and scaling claims. The detailed mathematics and examples remain available below the concise explanations.

| Scope | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| English main-path word tokens | 88,331 | 85,049 | 3,282 (3.7%) |
| Chinese main-path Han characters | 140,291 | 135,139 | 5,152 (3.7%) |
| English objectives and takeaways, word tokens | 4,426 | 1,899 | 2,527 (57.1%) |
| Chinese objectives and takeaways, Han characters | 7,566 | 3,512 | 4,054 (53.6%) |

Counts exclude optional notes, paper guides, references, code fences, math and component attributes. Chinese counts are Han characters, not space-separated words. The [per-chapter audit](ml-polishing-audit.json) records the exact counting convention and coverage.

All 30 source baselines match `c42c407`. Every heading, import, code fence, display equation and figure component is unchanged. The entire optional-notes/paper-guides/references tail is byte-for-byte unchanged. Reversing only the planned summary, paragraph and date replacements recovers each baseline exactly. No previously present citation number disappears entirely. New Chinese prose passes the repository terminology scan.

## Validation

- All 30 changed tutorials pass MDX/KaTeX validation.
- The content index was regenerated and its freshness check passes; only 15 chapter update dates changed in the generated index.
- Paper-guide coverage remains 167 guides across 51 chapters: zero missing figures, 139 original source figures, 22 comparative maps and seven supplemental teaching diagrams. Four heading-ID regression checks pass.
- Astro check: 176 files, zero errors, zero warnings, three existing unused-code hints.
- Production build passes: 1,172 generated pages, 1,171 indexed by Pagefind. Redirect HTML accounts for the larger link-checker page count.
- Internal links: 2,137 HTML pages, 143,547 links, zero broken pages or fragments.
- All 422 replacement bullets/paragraphs appear in the final built HTML, allowing only typographic-apostrophe and whitespace normalization.
- Browser checks pass for all 15 chapters in both languages at 1440 px and 375 px (60 layouts): five objectives, six takeaways, no horizontal page overflow, code initially collapsed, working floating section/code controls, keyboard section toggling and no JavaScript page errors. See [the browser report](ml-polishing-browser-validation.json).
- English desktop and Traditional Chinese mobile objective sections were visually inspected. No UI implementation changes were needed.
- `git diff --check` passes. Executable examples are unchanged; this editorial pass does not rerun training or paper benchmarks.

## Delivery

Committed locally on `local/site-subchapters-tree`; no deployment is part of this checkpoint. Per-chapter counts and preservation checks are recorded in the audit linked above.
