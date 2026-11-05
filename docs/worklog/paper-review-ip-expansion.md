# New-paper review and IP algorithm expansion — 2026-11-04

## Scope and acceptance

This pass substantively re-read the 22 user-named paper guides against their pinned primary sources and added six independent image-processing algorithm guides, mirrored in English and Traditional Chinese. The existing whole-site inventory was 149 guide units; the resulting inventory is 155. Global structural validation is separate from scientific review: this pass does **not** claim a new scientific audit or experimental reproduction of all 155 papers.

For an introductory paper guide, “sufficient” means a clear problem and thesis, a useful attributed source figure, a method with defined symbols, evidence tied to its actual protocol, an appropriate design/ablation discussion, and limitations that bound the conclusion. The 22 existing guides already meet that structure and contain substantive original-table discussion. The improvement here is a specific figure-reading and baseline-selection checkpoint in every guide. Proposed diagnostic checks are identified as suggestions, not original-paper experiments.

## The 22 reviewed guides

All 22 cached PDFs match the source hashes in `docs/paper-figure-sources.json`. Results/ablation decimal tokens were checked for presence in those PDFs as an additional transcription check; that mechanical check alone cannot verify the table row, metric or scientific interpretation. The review retains the protocol distinctions below and adds a method-specific reading checkpoint to each bilingual guide.

| Paper | Evidence and interpretation retained | Reading checkpoint added |
| --- | --- | --- |
| NAFNet | Original restoration tables and block ablations; training/crop/inference conversion matter | Trace SimpleGate and channel reweighting; multiplication remains nonlinear; hold the training recipe fixed |
| Restormer | Task-specific restoration and progressive-training comparisons | Distinguish channel attention from pixel attention, matrix memory from feature-map cost, architecture from checkpoint |
| AirNet | Contrastive representation and restoration ablations; noise settings contribute separate mean columns | Separate training crop pairs from inference inputs; examine scene-content confounding |
| TransWeather | Weather-specific test sets and query/component experiments | Learned weather queries are vectors, not text instructions or measured atmosphere |
| PromptIR | Dynamic versus fixed prompts and restoration results | Separate prompt components from their input-dependent mixing weights; no user language input |
| AdaIR | Frequency-module ablations, five-task versus single-task settings; documented table/prose arithmetic discrepancy | Neither high nor low frequency is inherently clean; inspect broad changes and fine texture separately |
| DA-CLIP | Predicted versus oracle conditioning and perceptual/fidelity metrics | Trace content and degradation signals separately; training labels are not deployment oracles |
| InstructIR | Instruction gates, frozen sentence encoder and task protocols | Follow language modulation; distinguish pre-upsampled SR input from native low-resolution input |
| AutoDIR | Assessment, diffusion and consistency-module comparisons; added snow/JPEG tasks use fine-tuning | Separate diagnosis errors, generated plausibility and structure consistency; avoid zero-shot overclaim |
| AgenticIR | Planning/rollback studies, evaluator recall and selected rollback-subset bias | Follow stored state, execution, evaluation and rollback; log tool calls and latency |
| 4KAgent | Profile-dependent fidelity/perceptual trade-offs and tool selection | Profile choice is not a fixed output rectangle; quality feedback does not establish true texture |
| PromptFix | Instruction/diagnosis and high-frequency adaptation studies | Input high frequencies contain artifacts too; hold decoder adaptation fixed in comparisons |
| DPS | Noisy/nonlinear inverse-problem evidence and guidance settings | Compute a measurement-space residual and differentiate through the clean estimate; noise need not be fitted exactly |
| DDNM | Known linear operator, pseudoinverse, noiseless projection versus DDNM+ | Separate measured and null-space components; pixel replacement is only a special example |
| BlindDPS | Image/kernel metrics, learned operator-prior and sparsity studies | Follow both unknowns into one likelihood; small residual does not identify the true pair |
| KinD | Paired exposure training, restoration/illumination modules and original results | Brightening can expose reflectance defects; decomposition is not uniquely measured physics |
| IAT | Local/global branch studies and downstream results that can worsen | Separate local affine maps from a shared color/gamma transform; hold downstream models fixed |
| Multinex | Conference paper/supplement versus released checkpoint settings | Follow multiplicative residual interaction without asserting unique physical factors |
| CPGA-Net | Pointwise channel priors, gamma/fusion ablations and demonstration-weight caveat | Channel minimum is not a spatial dark-channel minimum; learned fusion is not literal set intersection |
| CPGA-Net+ | Attention, BDSF inheritance, plus/further-fine-tuned settings | Sum-to-one coefficients need not form a convex blend; checkpoint lineage matters |
| Latent Diffusion | Autoencoder compression studies, generation tables and guidance/sampling settings | Check reconstruction before blaming the latent sampler; restoration needs measurement constraints |
| SD3 | ICML research system, objective/time sampling, caption data, latent channels and preference optimization | Trace separate modality transformations and joint attention; distinguish reconstruction from alignment scores |

The primary-source versions and original numerical excerpts remain documented in [requested-paper-guides.md](requested-paper-guides.md), the guides themselves and the figure manifest. No reported model scores were replaced by newly invented results.

## Six added IP guides

Each new guide has six foldable sections, its own original paper figure, defined equations, evidence interpretation, limitations and bilingual reading pointers from the main chapter. Parameter studies and analytical evidence are called what they are; older papers are not given fabricated modern ablation tables.

| Chapter | Added guide | Primary source and figure | Main teaching distinction |
| --- | --- | --- | --- |
| IP 03 | Bilateral Filter | Tomasi & Manduchi, ICCV 1998, figure 1, PDF p. 3 | Spatial versus appearance weights; smoothing and texture removal are not automatically restoration |
| IP 03 | Guided Filter | He, Sun & Tang, TPAMI 2013 author PDF, figure 1, p. 3 | Filtering input versus guidance; figure 2 PSNR compares two filters, not a clean target |
| IP 05 | Non-Local Means | Buades, Coll & Morel, IPOL 2011, figure 1, PDF p. 4 | Pixelwise versus patchwise aggregation; the reproducible IPOL variant is distinguished from CVPR 2005 |
| IP 05 | BM3D | Dabov et al., TIP 2007 accepted author manuscript, figure 3, p. 5 | The basic estimate guides the second pass, while original noisy coefficients remain its measurement input |
| IP 11 | Chan–Vese | Chan & Vese, TIP 2001, figure 1, PDF p. 3 / printed p. 268 | Fitting terms versus complete energy; level-set sign conventions and initialization dependence |
| IP 12 | SIFT | Lowe, IJCV 2004 author manuscript, figure 7, PDF p. 15 | The illustration is 2×2 cells; experiments use 4×4×8 entries; ratio-test rates have a specific protocol |

Primary PDF title/authors/publication information was inspected. Crossref additionally confirmed the Guided Filter, IPOL NLM, BM3D, Chan–Vese and SIFT records. The bilateral Crossref request was rate-limited; its accessible original ICCV PDF supplies the verified bibliographic evidence. The IPOL article reference is appended as IP 05 reference 23 in both languages; prior citation numbers are unchanged.

Six source excerpts are reproducible with:

```bash
python scripts/figures/extract_paper_figures.py \
  --cache-dir /tmp/cvblog-paper-cache \
  --only bilateral guided-pami nlm bm3d chanvese sift
```

The manifest pins source URLs, SHA-256 hashes, PDF page/crop, figure number and version. All six excerpts were visually inspected. The NLM paper has a result panel rather than an architecture diagram; the guide explains its weights and aggregation directly. Its source photograph credit and license are retained. Chan–Vese's original scanned illustration is lower resolution than the vector diagrams; its four cases are explained in prose rather than relabeled as a newly drawn figure.

Main-text corrections accompany the guides: Chan–Vese no longer implies success from almost any initialization, and SIFT no longer implies unchanged descriptors under arbitrary viewpoint/illumination changes. The paper's positive-inside convention is explicitly distinguished from the chapter's negative-inside convention. Existing guide titles, public routes and the legacy subsection-ID baseline are preserved.

## Validation

All checks below passed on the final source state:

- All 18 changed MDX files compile, including KaTeX; nine chapter pairs have matching heading depths, figure counts and display-equation counts. Added Traditional Chinese prose passes the terminology scan.
- `node scripts/content-index.mjs` regenerated the map; `--check` confirms it is current. `node scripts/check-paper-guides.mjs` reports 155 guides across 40 chapters, zero missing figures, matching bilingual sections/citations, 126 original-source figures, 22 comparative maps and seven teaching diagrams. Six new excerpts are at most 1,591 pixels wide and below 1 MB each.
- `node scripts/tests/paper-heading-ids.test.ts` passes all four regression cases. The legacy ID baseline is unchanged.
- `npx astro check`: zero errors, zero warnings, three pre-existing hints. Final `npm run build`: 1,126 content pages and 1,125 Pagefind pages.
- `python scripts/check-site-links.py`: 2,091 HTML pages, 140,563 internal links, zero broken links.
- The existing paper-guide browser checker, restricted to these 28 guide units, passes all 112 English/Traditional Chinese desktop/mobile layouts at 1,440/375 pixels, including image loading/full-size links, language/parent links, folds, keyboard operation, fragment reveal and horizontal bounds. Its full static legacy audit retains all 828 subsection IDs; 12 legacy browser routes also pass. There are no JavaScript errors. See [the scoped browser report](paper-review-ip-browser-validation.json) for the exact titles and matrix.
- Source crops and representative rendered pages were visually inspected, including mobile Guided Filter, BM3D and Chan–Vese and desktop SIFT. The final build includes the corrected dark-object description for the original Chan–Vese illustration.

No network training or original-model benchmark reproduction is claimed. No remote push, merge or deployment was performed.

## Remaining source limitation

The previously documented **SeAFusion original numerical comparison/ablation tables** remain unverified; see [remaining-supplements-audit.md](remaining-supplements-audit.md). They are outside these 22 named-paper rechecks and six added IP guides. Its available method/figure discussion and teaching calculation are not substitutes for authentic original tables. This pass does not close that gap or claim all full-text numerical evidence is now available.
