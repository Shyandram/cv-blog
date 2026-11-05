# CVDL / DLCV polishing goal

Status: completed — editorial pass and validation (2026-11-06). Starting checkpoint: `31fbd79`. Scope: all 29 tutorials in `deep-learning`, with their 29 Traditional Chinese counterparts. “CVDL” in this request refers to the site's Deep Learning for Computer Vision (DLCV) track.

## Goal

讓整個電腦視覺深度學習課程更精煉、清楚、方便循序學習：每章讓讀者快速知道要解決的任務、需要的先備知識、方法如何選擇、結果如何評估，以及下一步。

Polish learning objectives, selected main-path explanations and takeaways in every chapter. Remove repeated model lists and redundant explanations; use direct sentences and consistent terminology. Preserve the details needed to understand equations, reproduce examples, interpret evidence and recognize limitations.

## Completion criteria

- Review and edit every chapter pair, including specialized and nested chapters; record per-chapter changes and before/after prose counts.
- Shorten actual teaching prose rather than merely changing metadata or adding summaries. Keep each chapter's task, assumptions and evaluation distinctions explicit.
- Mirror the English changes in natural Traditional Chinese. Keep the same headings, section order, main derivations, display equations, executable examples, figures and reference numbering. Repeated inline formulas may be removed from objectives and summaries when the full explanation remains in the body.
- Keep substantive paper-guide evidence, source figures and detailed optional explanations available. This editorial pass is not a new scientific audit or benchmark reproduction.
- Preserve public URLs, existing heading IDs, prerequisite navigation, code folding and the floating reading console.
- Pass MDX/KaTeX validation, content-index freshness, paper-guide coverage, Astro check, build, internal links and representative bilingual desktop/mobile reading checks.
- Record the final scope, counts, validation and any remaining source limitations; commit the validated work locally.

The earlier [course navigation goal](course-navigation-goal.md) and [NLP paper-guide batch](nlp-paper-guides.md) remain separate completed checkpoints. Existing source limitations, including the SeAFusion numerical-table gap, remain tracked in [remaining-supplements-audit.md](remaining-supplements-audit.md).

## Editorial result

All 29 chapter pairs have shorter main-path prose. Every learning-objective and takeaway section was rewritten; 86 selected explanatory paragraphs were rewritten in each language (172 paragraphs total). Other main-path paragraphs were preserved. This is a track-wide editorial pass, not a line-by-line rewrite of every paragraph.

| Scope | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| English main-path word tokens | 222,646 | 213,899 | 8,747 (3.9%) |
| Chinese main-path Han characters | 327,725 | 315,427 | 12,298 (3.8%) |
| English objectives and takeaways, word tokens | 10,264 | 3,703 | 6,561 (63.9%) |
| Chinese objectives and takeaways, Han characters | 16,419 | 7,111 | 9,308 (56.7%) |

Counts exclude optional notes, paper guides, references, code fences, math and component attributes. Chinese counts are Han characters, not space-separated words. Per-chapter counts and the exact counting convention are in [the editorial audit](cvdl-polishing-audit.json).

The edits make task definitions and evaluation conditions explicit, shorten model-name inventories in summaries, and qualify overbroad statements about speed, generative fidelity, fusion artifacts, depth calibration and attention interpretation. Every chapter now has five learning objectives and six or seven takeaways.

Preservation checks compared each file against `31fbd79`: every heading, code fence, display equation, import and figure component is unchanged; the entire optional-notes/paper-guides/references tail is byte-for-byte unchanged. Reversing the intended summary, paragraph and update-date replacements recovers each original file exactly. No previously present citation number disappears entirely.

## Validation

- MDX/KaTeX: all 58 changed tutorials pass; the 29 Chinese files were checked again after the final terminology adjustment.
- Content index: regenerated; freshness check passes. Only the 29 chapter update dates change in the generated index.
- Paper guides: 167 guides across 51 chapters; zero missing figures, 139 original source figures, 22 comparative maps, seven supplemental teaching diagrams. Four heading-ID regression checks pass.
- Astro check: 176 files, zero errors, zero warnings; three existing unused-code hints.
- Final production build: passes; 1,172 generated pages, with 1,171 pages indexed by Pagefind. These build counts exclude some redirect HTML files counted by the link checker.
- Built internal links: 2,137 HTML pages, 143,547 links, zero broken links or fragments.
- Rendered copy: all 828 replacement bullets/paragraphs appear in the final built HTML, allowing only typographic apostrophe and whitespace normalization. The final build includes the last Chinese terminology adjustment.
- Browser: all 29 chapters in both languages at 1440 px and 375 px (116 layouts) pass. Each has five objectives, six or seven takeaways, no horizontal page overflow, code initially collapsed, working section/code console controls, keyboard section toggling and no JavaScript page errors. See [the browser report](cvdl-polishing-browser-validation.json).
- Visually inspected the English desktop and Traditional Chinese mobile objective sections; layout and controls remain readable. No UI implementation changes were needed.
- `git diff --check`: passes. Tutorial code is byte-for-byte unchanged; no training or benchmark reruns were needed for this prose-only change.

## Delivery and limits

Committed locally on `local/site-subchapters-tree`; no deployment is part of this editorial checkpoint. Existing paper-source limitations remain tracked in the earlier audit. This pass preserves detailed guides and evidence; it does not claim to resolve the SeAFusion numerical-table gap or reproduce every paper's experiments.
