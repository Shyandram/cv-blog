# DLCV textbook refresh — 2026-11-05

Scope: all 29 English / Traditional Chinese tutorial pairs in `src/content/tutorials/{en,zh-tw}/deep-learning`, starting from `116d724`. Reviewed the main chapter, supplementary-note and paper-guide structure together, with targeted checks of assumptions, symbols, evaluation claims, source limitations and executable examples. This is a structured chapter-wide scan with targeted primary-source verification, not a claim to have independently reproduced every cited benchmark or re-opened every existing citation.

Nine pairs received substantive corrections; the other 20 pairs were reviewed without adding filler. Existing headings, anchors, figures and paper-guide organization are preserved. Shared navigation, generated indexes, notebooks, PDFs and the final site build are handled by the coordinating agent.

## Chapter-by-chapter findings and actions

All Python counts below refer to complete fenced Python blocks, executed in document order in one shared namespace per chapter, with a fresh process for the next chapter. The Chinese blocks are byte-identical to the English blocks and were not redundantly rerun.

| Chapter slug | Review focus and finding | Action | Python blocks / final result |
|---|---|---|---|
| `dl-01-deep-learning-for-image-processing` | CNN shapes, convolution/normalization assumptions, core-to-note/guide coverage | Reviewed; no concrete correction identified | 7 / pass |
| `dl-01a-training-practice` | Valid-item weighting, accumulation, distributed reductions | Corrected sum-versus-average DDP normalization and undefined zero-item mean | 12 / pass |
| `dl-01b-building-blocks` | Architecture configuration, RegNet quantization, source rendering | RegNet width multiplier must exceed 1; marked Restormer vectors as code so citation processing cannot reinterpret them | 7 / pass |
| `dl-01c-model-compression` | Quantization/distillation/pruning assumptions, speed versus parameter counts | Reviewed; no concrete correction identified | 8 / pass |
| `dl-02-image-classification` | Classification objectives, calibration/threshold metrics, modern guides | Reviewed; no concrete correction identified | 14 / pass |
| `dl-02a-face-metric-learning` | Verification operating points, ROC/DET interpretation | Logarithmic FAR is an optional useful scale, not the definition of ROC; zero FAR cannot be plotted on it | 9 / pass |
| `dl-03-object-detection` | Small-object geometry, COCO evaluation, runnable box-regression training | Corrected area-versus-side-length argument and category-wise maxDets; used one CPU thread and refreshed observed training output | 8 / pass after timeout fix |
| `dl-03a-real-time-detection` | Assignment/NMS distinctions, latency comparisons, detector guides | Reviewed; no concrete correction identified | 13 / pass |
| `dl-03b-image-segmentation` | Dice derivatives, empty-mask behavior, sigmoid/softmax distinction | Separated probability and logit derivatives; included correct sigmoid chain rule and multiclass Jacobian caveat | 10 / pass |
| `dl-04-image-restoration` | Restoration assumptions, PSNR/SSIM/perceptual evaluation, modern restorers | Reviewed; no concrete correction identified | 7 / pass |
| `dl-04a-super-resolution` | Degradation models, evaluation/cropping assumptions, paper comparisons | Reviewed; no concrete correction identified | 7 / pass |
| `dl-04b-video-super-resolution` | Alignment and temporal consistency, temporal versus framewise evaluation | Reviewed; no concrete correction identified | 6 / pass |
| `dl-04c-generative-restoration` | Posterior assumptions, perceptual/distortion trade-off, calibration, sequential code | Corrected L1 target, PSNR aggregation, FID proxy and risk-control guarantees; made later step-edge experiment self-contained | 6 / pass after namespace fix |
| `dl-04d-blind-real-world-super-resolution` | Blind degradation estimation, domain mismatch, claimed versus measured restoration | Reviewed; no concrete correction identified | 4 / pass |
| `dl-04e-one-step-diffusion-super-resolution` | One-step pipeline cost, restoration versus generation, guide limitations | Reviewed; no concrete correction identified | 1 / pass |
| `dl-05-low-light-enhancement` | Illumination/reflectance assumptions, enhancement evaluation, named guides | Reviewed; no concrete correction identified | 14 / pass |
| `dl-05b-image-fusion` | Task-aware fusion, original results versus synthetic loss demonstration | Reviewed; SeAFusion numerical-table evidence remains blocked, explicitly retained below | 9 / pass |
| `dl-06-object-tracking` | Association, identity metrics and detection/tracking coupling | Reviewed; no concrete correction identified | 9 / pass |
| `dl-06a-pose-keypoints` | Coordinate conventions, matching/visibility and pose metrics | Reviewed; no concrete correction identified | 4 / pass |
| `dl-06b-video-understanding` | Temporal sampling/aggregation, recognition versus localization evaluation | Reviewed; no concrete correction identified | 7 / pass |
| `dl-07-world-models` | Prediction/planning assumptions, learned simulator limitations and source framing | Reviewed; no concrete correction identified | 5 / pass |
| `dl-08-3d-reconstruction` | Camera/appearance assumptions across SfM, NeRF and DUSt3R | Replaced incorrect claim that all methods require known cameras and direction-independent appearance | 11 / pass |
| `dl-08a-depth-optical-flow` | Depth/flow ambiguity, scales, supervision and evaluation conventions | Reviewed; no concrete correction identified | 7 / pass |
| `dl-09-3d-object-detection` | Sparse observations, benchmark difficulty labels and evaluation membership | Corrected Waymo LEVEL_2 assignment and its cumulative evaluation | 10 / pass |
| `dl-10-image-generation` | Generative objectives, likelihood/perception evaluation, conditioning | Reviewed; no concrete correction identified | 2 / pass |
| `dl-10a-video-generation` | Temporal metrics, historical papers versus current product availability | Distinguished Sora's 2024 introduction from release; added verified 2026 discontinuation dates and current selection-guide link | 5 / pass |
| `dl-10b-one-step-generation` | Few/one-step objective distinctions, quality-cost comparison assumptions | Reviewed; no concrete correction identified | 1 / pass |
| `dl-11-robustness` | Threat-model scope, adversarial versus distributional claims and evaluation | Reviewed; no concrete correction identified | 5 / pass |
| `dl-11a-text-in-images` | OCR matching, ignored regions, sequence/image metrics | Original MLT-2019 Eq. 1 confirms the existing ignored-region area denominator; intentionally unchanged | 9 / pass |

## Corrections and verified primary evidence

- **Gradient accumulation:** with global item count `N` and `R` ranks, a sum-reduced summed-loss gradient is divided by `N`; default DDP's rank average is multiplied by `R/N`. Require `N > 0`. Added reference 48 to the opened [PyTorch 2.15.0 DDP source documentation](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/nn/parallel/distributed.py). Custom hooks and uneven-input policies need their own reduction check.
- **Building blocks:** the RegNet multiplier condition now matches the existing paper-guide derivation (`w_m > 1`); a multiplier of 1 makes the logarithmic quantization undefined. Restormer's block/head/channel arrays use inline code rather than numeric citation syntax.
- **Verification curves:** removed an unjustified universal claim about logarithmic ROC axes. The [original DET paper record](https://www.isca-archive.org/eurospeech_1997/martin97b_eurospeech.html) was opened; DET's normal-deviate definition remains unchanged.
- **Detection:** a small area does not bound both side lengths of a thin object. The [official COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py), `evaluateImg(imgId, catId, aRng, maxDet)`, sorts and truncates detections per image/category in the default category-aware mode. Its source is now linked in existing reference 40; a detector's separate total-output cap is distinguished.
- **Segmentation:** the old Dice discussion compared a derivative with respect to probability against BCE's derivative with respect to a logit. The text now compares like variables and explains the sigmoid chain rule versus the full multiclass softmax Jacobian. Independently checked the added binary-logit formula against central finite differences; maximum absolute error was approximately `5.6e-11`.
- **Generative restoration:** separable L1 targets coordinate-wise conditional medians, whereas squared L2 targets the conditional mean. The factor-two expected-MSE identity requires exact conditionally independent posterior sampling under the same model; its roughly 3 dB consequence concerns PSNR formed from those expected MSEs, not automatically mean per-image PSNR. Opened [Blau and Michaeli](https://arxiv.org/abs/1711.06077). FID is a Gaussian feature-distribution proxy, clarified using the [original FID paper](https://arxiv.org/abs/1706.08500). Sample spread is a diagnostic, not a free calibrated error map. Read Section 1.1 and Definition 2.1 of [Angelopoulos et al., ICML 2022](https://proceedings.mlr.press/v162/angelopoulos22a/angelopoulos22a.pdf): the calibrated risk concerns the expected fraction of uncovered pixels on same-distribution fresh images, with confidence over calibration; it is not a guarantee for every pixel/image or arbitrary distribution shift.
- **3D reconstruction:** opened [COLMAP's official project](https://demuc.de/colmap/), [NeRF](https://arxiv.org/abs/2003.08934) and [DUSt3R](https://arxiv.org/abs/2312.14132). SfM estimates poses; original NeRF uses known poses but models view-dependent color; DUSt3R does not require prior camera calibration/poses. The failure section now distinguishes those assumptions.
- **3D detection:** opened the [official Waymo challenge definitions](https://waymo.com/open/challenges/2020/3d-detection/). LEVEL_2 labels include objects with 1–5 points or explicit difficulty flags; LEVEL_2 evaluation includes both difficulty levels. Corrected the contradictory failure-mode sentence; the already-correct main metric explanation remains.
- **Video generation:** opened the [official Sora discontinuation notice](https://help.openai.com/en/articles/20001152-what-to-know-about-the-sora-discontinuation). Web/app ended 26 April 2026; API ended 24 September 2026. Added reference 54 and distinguished historical research from current service selection. Independent Open-Sora projects are not the discontinued OpenAI service.
- **OCR verification without a change:** opened [MLT-2019's original paper](https://arxiv.org/pdf/1907.00945). Equation 1 uses the ignored ground-truth region's area in the overlap denominator; the current explanation already follows that definition.

## Executable-example validation

Executed all **217 Python blocks across 29 English chapters**, in chapter order and a shared namespace within each chapter, using the repository `.venv`. Environment caps were `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, with a noninteractive Matplotlib backend. Existing explicit thread settings could override those environment caps. Each chapter had a 90-second timeout.

The first sweep passed 27 chapters. Two problems were diagnosed and corrected in both translations:

1. Detection's teaching network explicitly requested four CPU threads and exceeded the 90-second chapter cap in the shared environment. With one thread, its standalone training example completed in 25.3 seconds and produced mean IoU 0.882, 99.9% at IoU ≥ 0.5 and 97.3% at IoU ≥ 0.75. These observed values replace the stale displayed run, with an explicit hardware/library variability note. The complete eight-block chapter rerun passed in 19.1 seconds; timing is not a benchmark claim.
2. Generative restoration's DPS example reassigns `A` from a blur function to a matrix. Its later step-edge experiment incorrectly assumed the original function still existed. The later experiment now recreates its small setup explicitly. The complete six-block rerun passed in 3.0 seconds, and the displayed seed/sample statistics match its output.

Final outcome: all 29 chapter sequences pass, with no remaining skipped blocks or timed-out chapters. This smoke check verifies execution and the two changed examples' observed outputs, not every printed historical result or research benchmark. No model checkpoints or full-paper training experiments were reproduced. Additional small algebra checks verified the DDP normalization on unequal item counts and the Dice logit derivative.

Temporary run evidence is in `/tmp/dlcv-smoke/`, `/tmp/dlcv-smoke-rerun/` and `/tmp/dlcv-detection-one-thread.log` for this workspace session; these paths are not permanent repository artifacts.

## Remaining source limitation: SeAFusion

Retried the [official SeAFusion repository](https://github.com/Linfeng-Tang/SeAFusion), the author's publication record, and the repository's [original publisher link](https://www.sciencedirect.com/science/article/pii/S1566253521002542). The publisher full article remains inaccessible (403 in the available research tool). Targeted title/PDF searches did not yield an identity-verified full original paper. Search snippets and later papers' comparison rows are insufficient evidence for original tables. The previously rejected ResearchGate text belongs to a different paper and is not used.

Consequently, **the original SeAFusion numerical comparison and ablation tables are still unverified**. No original mIoU, timing, parameter count or ablation delta is invented. The existing original-method explanation, attributed figure, released-code objective and executed synthetic loss demonstration remain. This is an explicit source-access blocker, not a claim that the missing original-table evidence has been completed.

## Source and bilingual checks

- All 58 MDX files compile with the repository MDX/KaTeX checker.
- All 29 pairs retain matching heading-depth sequences, complete identical fenced code, display-equation counts and figure counts.
- No new headings, formal paper guides, figures, registry entries or curriculum edits were introduced.
- `git diff --check` passes for the owned tutorial files and this worklog.
- No full build, commit, push or deployment was performed by this agent.
