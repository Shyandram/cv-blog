# ML paper guides — 2026-11-06 (Asia/Taipei)

Status: complete for the 17-guide core batch. Baseline: `9d5de87`; branch `local/site-subchapters-tree`.

The rescan found zero standalone paper guides in all 15 ML chapters. This batch adds **17 core guides spanning every chapter**, with two each in generative models and diffusion/flow matching. It does not claim a separate guide for every existing bibliography entry. Existing main lessons, exercises, snippets, numbering and heading-ID baseline are preserved.

Each guide has six real, foldable H4 sections: question/thesis, method and original figure, evidence, design analysis, reading checkpoint, and limitations/next step. English guides contain 399–502 prose words by the coverage check, excluding math and figure markup. Traditional Chinese mirrors their equations, figure order, section depth and citation numbers. The linked next step connects each guide to another guide or its parent chapter.

ML 01 adds one foldable reading route with five stacked stages: problem/mathematics, training/architecture, representations, generation, and evaluation. All 17 guides are linked. Chapter prerequisites are clickable; checkpoints ask readers to distinguish assumptions, controlled comparisons and evaluation targets. These are optional thematic readings after the main chapters, not additional course prerequisites. The route avoids a wide comparison table on phones.

## Sources and reading emphasis

All figures are original PDF excerpts. Architecture/flow figures are used where appropriate; theoretical, survey and evaluation papers use their original conceptual or evidence figures instead of invented network diagrams. Page numbers below are PDF pages, not journal pagination.

| Chapter | Paper | Inspected source version and figure | Reading emphasis |
| --- | --- | --- | --- |
| ML 01 | Domingos, *A Few Useful Things to Know About Machine Learning* | CACM 55(10), 2012; author-hosted published article; fig. 1, p. 4 | Representation/evaluation/optimization; dartboards are an analogy, while fig. 2 has a stated experimental protocol |
| ML 01A | Jolliffe & Cadima, *Principal component analysis: a review and recent developments* | Philosophical Transactions A 374, 2016; Europe PMC published copy; fig. 1, p. 5 | Scores versus loadings, variance versus usefulness, scaling and the fossil-teeth example |
| ML 01B | Shannon, *A Mathematical Theory of Communication* | 1948 corrected reprint of both BSTJ parts, Harvard copy; fig. 1, p. 2 | Discrete entropy, coding theorems and units; not semantic meaning or a neural benchmark |
| ML 01C | *Mixed Precision Training* | ICLR 2018 manuscript, arXiv:1710.03740v3; fig. 1, p. 3 | Master-weight updates versus gradient underflow; task-specific scaling and resource caveats |
| ML 02 | *Adam: A Method for Stochastic Optimization* | ICLR 2015 manuscript, arXiv:1412.6980v9; fig. 4, p. 8 | Raw second moment, initialization correction, training curves and the later convergence counterexample |
| ML 03 | *Deep Residual Learning for Image Recognition* | CVPR 2016 proceedings; fig. 2, p. 2 | Post-addition ReLU, shape matching, depth-matched ImageNet comparison and optimization degradation |
| ML 04 | *Long Short-Term Memory* | Neural Computation 9(8), 1997; published Stanford copy; fig. 1, p. 10 | Original cell has no forget gate; protected state path; sequence length differs from minimum relevant delay |
| ML 05 | *An Image is Worth 16x16 Words* | ICLR 2021 manuscript, arXiv:2010.11929v2; fig. 1, p. 3 | Token count, position, inductive bias and pretraining-scale comparisons |
| ML 06 | *A Simple Framework for Contrastive Learning of Visual Representations* | ICML 2020, PMLR 119; fig. 2, p. 2 | Positive-pair assumptions, projection-head ablation and linear probing versus fine-tuning |
| ML 06A | *Probabilistic Principal Component Analysis* | JRSS B 61(3), 1999; Oxford author manuscript dated September 27, 1999; fig. 1, p. 6 | Linear Gaussian assumptions, rotational ambiguity and missing-data evidence |
| ML 07 | *Auto-Encoding Variational Bayes* | ICLR 2014 manuscript, arXiv:1312.6114v11; fig. 1, p. 2 | ELBO gap, reparameterization, encoder versus prior sampling and distinct evaluation quantities |
| ML 07 | *Generative Adversarial Nets* | NeurIPS 2014 proceedings; fig. 1, p. 4 | Distribution-level optimum versus finite training; Parzen evaluation is a surrogate |
| ML 08 | *Denoising Diffusion Probabilistic Models* | NeurIPS 2020 proceedings; fig. 2, p. 2 | Noise prediction, reverse chain and the reported quality–likelihood-bound tradeoff |
| ML 08 | *Flow Matching for Generative Modeling* | ICLR 2023 paper, arXiv:2210.02747v2 manuscript; fig. 3, p. 6 | Conditional/marginal objectives, endpoint smoothing, conditional OT versus global coupling, NFE versus runtime |
| ML 09 | *Neural Discrete Representation Learning* | NeurIPS 2017 proceedings; fig. 1, p. 4 | Codebook/commitment gradient recipients, straight-through approximation and separately learned prior |
| ML 10 | *Learning Transferable Visual Models From Natural Language Supervision* | ICML 2021, PMLR 139; fig. 1, p. 2 | Image/text positives, prompt-based classifier, specific 27-dataset baseline and zero-shot scope |
| ML 11 | *On Calibration of Modern Neural Networks* | ICML 2017, PMLR 70; fig. 4, p. 8 | Held-out temperature fitting, unchanged argmax, bin-dependent ECE and the stochastic-depth figure variant |

Titles/authors, publication records and the relevant method/result sections were inspected. Publisher download failures for PCA/PPCA and an unavailable LSTM author server were resolved with Europe PMC, Oxford and Stanford copies. Source bytes, versions, pages, figure numbers and crop rectangles are pinned in [the figure manifest](../paper-figure-sources.json). All 17 extracted images were inspected, including corrected boundaries for PCA, Shannon, ResNet, VAE, Flow Matching and CLIP. No teaching diagram or reading-map substitution was needed.

Existing reference numbers are unchanged. Domingos is appended as ML 01 reference 26. Reddi, Kale & Kumar, *On the Convergence of Adam and Beyond* (ICLR 2018; arXiv:1904.09237), is appended as ML 02 reference 30 and used as a supporting correction, not counted as an eighteenth standalone guide. The original Adam claim is not presented as a universal convergence guarantee.

Reproduce the source-figure extraction with:

```bash
python scripts/figures/extract_paper_figures.py --only \
  ml-domingos ml-pca ml-shannon ml-mixed-precision ml-adam ml-resnet \
  ml-lstm ml-vit ml-simclr ml-ppca ml-vae ml-gan ml-ddpm \
  ml-flow-matching ml-vqvae ml-clip ml-calibration
```

## Validation

- All 30 changed tutorial MDX files pass syntax, frontmatter and KaTeX rendering checks.
- The 17 guide pairs have identical display equations; coverage checks confirm matching figures, six-part structure and citation sets. No new simplified-Chinese terminology was found; `演算法` is the correct Traditional Chinese term and not a forbidden `算法` occurrence.
- All 18 primary metadata links (17 guides plus the Adam correction) return HTTP 200. All source PDF hashes match the manifest, and all extracted images are at most 1,600 pixels wide and below 1 MB.
- Global paper coverage: 184 guides across 66 chapters; 156 original-source figures, 22 comparative maps, seven supplemental teaching diagrams; zero missing figures.
- The four heading-ID regression tests pass. The original subsection-ID baseline is unchanged.
- `npx astro check`: zero errors, zero warnings; three pre-existing unused-import hints.

- Production build passes: 1,236 generated pages; Pagefind indexes 1,235. Existing Shiki/CSP compatibility advisory remains unchanged.
- Content index and coverage `--check` both pass. All original lines remain in order in the 30 changed lessons; additions are imports, optional pointers, the route, guides and appended references.
- Built-site link audit: 2,201 HTML pages, 148,173 internal links, zero broken links.
- All 17 guides pass 68 bilingual desktop/mobile layouts at 1,440/375 pixels: six folds, full-size source images, language/parent links, horizontal bounds, console fold/open controls, keyboard operation, fragment reveal and no JavaScript errors. All 828 legacy subsection IDs/texts and 12 browser forwarding routes are retained. See [the guide browser report](ml-paper-browser-validation.json).
- Four intro layouts pass 68 paper-link clicks, route keyboard folding, five stacked stages, fresh-session code folding and floating-console code controls. See [the route report](ml-reading-route-validation.json). Screenshots of desktop Adam, mobile Flow Matching and the Traditional Chinese mobile reading route were visually reviewed.

No model training or benchmark reproduction is claimed by this reading-guide task. Changes are local; publishing remains a separate action. The subsequent request for MeanFlow/JiT and related methods is an extension to this validated core batch.
