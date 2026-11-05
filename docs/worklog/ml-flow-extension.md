# Beyond Flow Matching — 2026-11-06 (Asia/Taipei)

Status: complete. Baseline: `cc06fc9`, the completed [17-guide ML core batch](ml-paper-guides.md).

The user requested more coverage after Flow Matching, specifically MeanFlow, JiT and related work. This extension adds **MeanFlow → Improved MeanFlow**, then **JiT → Pixel MeanFlow**, distinguishing interval modeling from the network's direct output. ML now has 21 bilingual paper guides across its 15 chapters; ML 08 has six guides.

## Course changes

A new foldable section immediately after Flow Matching explains the reversed time convention, instantaneous versus average velocity, the iMF prediction-input change, flexible guidance, JiT's separate output/loss spaces, and the pMF conversion from an image-like field through average velocity to instantaneous-velocity loss. It relates these choices to existing reflow and consistency-model coverage without presenting JiT as a one-step method.

A short, tested Python example follows the exact curve `(t, t²)`. With `t=1`, `r=0`, one instantaneous Euler jump gives `(0.0, -1.0)` while the interval-average jump gives the correct `(0.0, 0.0)`. It is explicitly an algebra illustration, not a reproduction of learned MeanFlow. The snippet is identical in both languages and uses the existing default-folded code UI.

The four new paper guides each have six foldable sections, source figures, defined symbols, controlled experimental comparisons, design analysis, a reading question and limitations. Flow Matching's next-reading link enters this continuation. ML 01 adds a short 4A reading stage with prerequisites and a checkpoint; the route now covers all 21 guides in six stacked stages.

The existing JiT discussion and all earlier headings remain. Its original subsection now also points to the new guide. The old MeanFlow overview bullet is replaced by a concise pointer to the expanded treatment. Existing reference numbers stay fixed; iMF and pMF are appended as ML 08 references 49 and 50.

## Verified source material

| Guide | Primary publication | Original figure | Evidence emphasized |
| --- | --- | --- | --- |
| MeanFlow | Geng et al., *Mean Flows for One-step Generative Modeling*, NeurIPS 2025 proceedings | Fig. 3, PDF p. 4 | Interval-ratio and JVP ablations; B/4, 80-epoch ablations distinguished from XL/2, 240-epoch headline results; pretrained latent tokenizer disclosed |
| Improved MeanFlow (iMF) | Geng et al., *Improved Mean Flows: On the Challenges of Fastforward Generative Models*, CVPR 2026 proceedings | Fig. 1, PDF p. 1 | Predicted marginal JVP direction versus sampled conditional direction; objective/guidance/conditioning ablations separated; final models are not compute-matched |
| JiT | Li & He, *Back to Basics: Let Denoising Generative Models Denoise*, CVPR 2026 proceedings | Fig. 3, PDF p. 5 | Controlled output-space/loss-space comparisons at different patch dimensions; pixel architecture does not imply one-step sampling |
| Pixel MeanFlow (pMF) | Lu et al., *One-step Latent-free Image Generation with Pixel Mean Flows*, ICML 2026, PMLR 306 | Fig. 1, PDF p. 1 | Image versus average-velocity output at matched sequence length; perceptual losses separated from field conversion; intermediate manifold claim is approximate |

The four publication pages and PDFs were opened and the relevant methods, tables, figure captions and conclusions inspected. PMLR's PDF link points to its official `mlresearch/v306` GitHub asset; that resolved source is pinned rather than the nonexistent conventional PDF path. The original PDFs, hashes, crops and page/figure numbers are recorded in [the figure manifest](../paper-figure-sources.json). No replacement teaching diagram was needed. Rectified Flow and Consistency Models remain supporting comparisons with their existing references 11 and 12.

The notation makes the time reversal explicit: MeanFlow/iMF/pMF use data at zero and noise at one; JiT uses the opposite direction. The pMF guide distinguishes its two-time induced image-like field from the particular clean sample used to create a training pair. It also discloses pretrained perceptual features in the stronger pMF recipe, so “latent-free” is not misreported as “no pretraining anywhere.”

Reproduce the four source excerpts with:

```bash
python scripts/figures/extract_paper_figures.py --only \
  ml-meanflow ml-imf ml-jit ml-pixel-meanflow
```

## Validation

- The four changed MDX files pass syntax, frontmatter and KaTeX rendering checks. New display equations and the new Python snippet match across languages; the snippet runs with the documented outputs. Existing mathematical text translations in the older chapter are preserved.
- All four source PDFs match their pinned hashes; extracted images are at most 1,600 pixels wide and below 1 MB. Original pages and final crops were visually inspected.
- Content index and coverage generation/checks pass. Global coverage is 188 guides across 66 chapters, 160 original-source figures, 22 comparative maps, seven teaching diagrams and zero missing figures. The four new guides contain 455–478 English prose words each and six matching bilingual subsections/citation sets.
- All four heading-ID regression tests pass. `npx astro check` reports zero errors and zero warnings, with three pre-existing unused-import hints.
- Production build passes: 1,244 pages, with 1,243 indexed by Pagefind. The existing Shiki/CSP advisory is unchanged.
- The built-site audit checks 2,209 HTML pages and 148,973 internal links: zero broken links.
- All 21 ML guides pass 84 bilingual desktop/mobile layouts at 1,440/375 pixels, including source image loading/zoom links, language/parent links, folds, keyboard, fragment reveal, viewport bounds and no JavaScript errors. All 828 legacy subsection IDs/texts and 12 browser forwarding routes are retained. See [the guide report](ml-flow-extension-browser-validation.json).
- The six-stage intro route passes four layouts and 84 paper-link clicks, with fresh-session code folding and floating-console controls. See [the route report](ml-flow-route-validation.json).

- The new walkthrough passes four bilingual desktop/mobile layouts and 16 guide-pointer clicks. Its new code starts folded; floating-console code/section controls, keyboard toggles, fragment reveal, links to the earlier JiT discussion and few-step overview, viewport bounds and JavaScript checks all pass. See [the walkthrough report](ml-flow-walkthrough-validation.json).

Desktop JiT, mobile pMF and the Traditional Chinese mobile walkthrough were visually reviewed. The original figure views retain full-size zoom links. Browser test selectors account for relative paper URLs and percent-encoded Chinese fragments; these were test-harness adjustments, not content failures.

No model training or benchmark reproduction is claimed. Changes remain local until separately published.
