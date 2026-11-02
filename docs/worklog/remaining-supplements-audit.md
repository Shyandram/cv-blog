# Remaining supplements audit — 2026-11-02

Scope: all 97 tutorial pairs, the paused chapter tasks, curriculum navigation, optional notes/paper guides and downloadable books. This audit follows the completed source checkpoint `2edb518` and PDF checkpoint `23315dd`; the earlier chronological worklog remains historical evidence.

## Corrections

- Added the missing `receptive-field` inline lab to the Traditional Chinese CNN chapter. The English chapter already embedded it; the chapter plan alone did not detect the omission. Playground validation now compares the actual inline lab IDs and order in both translations for every chapter.
- Matched 39 Chinese code blocks to their English counterparts across seven tutorials. Python AST comparison first confirmed that every difference was a comment or docstring; no executable behavior changed.
- Replaced the FusionGAN original-table limitation with verified original evidence: Table 1's three GPU timings and the qualitative adversarial-loss comparison in Section 4.3 / Figure 10. The original title, five authors, volume, pages and year match reference 39. Explicitly retains the mixed CPU/GPU comparison caveat and the absence of numerical ablation deltas.
- Added a self-contained, executed CPU example to both SeAFusion guides. It evaluates the released intensity/gradient objective on four synthetic candidate images and explains why pixelwise maximum does not minimize the gradient objective. It is clearly identified as a local loss calculation, not a trained-network result or original-paper ablation.
- Fixed PDF notebook URLs that pointed at an unpublished local source checkpoint. They now use the public `main` notebook, matching the website; the full public repository tree confirms that every declared notebook path exists. Book edition metadata still records the exact content checkpoint.
- Updated the navigation goal's stale 95-tutorial status and obsolete descriptions of unfinished/merged chapters.

## Whole-course audit

All 97 English/Traditional Chinese pairs have matching heading depths, complete identical fenced code blocks, display-equation counts, figure counts and inline-lab IDs/order. Every declared notebook exists. A scan for untranslated prose found only an original paper title and an intentional English linguistic example. The detailed equation differences were inspected: they are translated text, punctuation and whitespace, not changed mathematical definitions. The setup/reading-path guide is intentionally exempt from the tutorial-style Modern view section.

No TODO/FIXME content placeholders or empty leaf sections were found. Parent headings followed immediately by child headings and sections containing only an executable example are valid structure, not missing prose. Historical pause/progress records are preserved rather than rewritten as if work had always been finished. Original personal posts remain unchanged.

## Source verification still open

**SeAFusion original numerical comparison and ablation tables.** The journal record and official code are verified, but the original numerical tables have not been obtained. The author's publication page links to the publisher's restricted article. A ResearchGate page labeled with the SeAFusion title exposes a different paper (GAN-FM) in its public text/figures; that material is rejected as evidence. Its author-upload PDF could not be fetched. Later papers' SeAFusion comparison rows are not substituted for original results. No original mIoU, timing, parameter count or numerical ablation improvement is invented.

Opened sources:

- [FusionGAN original article public transcription](https://www.scribd.com/document/1060859850/End-1-FusionGAN), title/author/publication identity checked against [the original DOI](https://doi.org/10.1016/j.inffus.2018.09.004).
- [SeAFusion author's publication list](https://linfeng-tang.github.io/publications/), [official implementation](https://github.com/Linfeng-Tang/SeAFusion) and [original DOI](https://doi.org/10.1016/j.inffus.2021.12.004).
- [ResearchGate record with mismatched public full text](https://www.researchgate.net/publication/357507432_Image_fusion_in_the_loop_of_high-level_vision_tasks_A_semantic-aware_real-time_infrared_and_visible_image_fusion_network).

## Validation

Source validation passed: all ten changed MDX files compile; all 97 tutorial pairs pass the structural/code/lab audit; curriculum and playground checks pass (73 experiments, 26 image labs, 2,016 render states). The missing-Chinese-lab regression is rejected by the strengthened checker. All 28 existing test files pass. Astro check reports zero errors, zero warnings and three existing hints. Build: 1,076 content pages and 1,075 Pagefind pages. Link check: 2,041 HTML pages, 130,517 internal links, zero broken. Twelve focused browser layouts pass at 1440/375 px across both languages, including the live CNN lab, original FusionGAN table, SeAFusion calculation and folding controls. RAG rebuilt to 7,925 chunks. All eight PDF editions are refreshed from `fde8eaa`: 4,392 pages and 12,145 links pass chapter/code/equation/figure coverage, font/glyph, page-bound and named-destination validation. Three notebook links use the verified public `main` path. Largest file: 14.40 MiB. File hashes and per-volume statistics are recorded in `pdf-validation-2026-11-02.json`; representative Chinese pages were visually inspected. The final website build also passes (1,076 content pages / 1,075 Pagefind pages). All four bilingual desktop/mobile download layouts pass; all eight built PDF copies match the validated hashes and identify source `fde8eaa`. Final internal-link check again passes 2,041 HTML pages and 130,517 links with zero failures. No remote push, merge or deployment.
