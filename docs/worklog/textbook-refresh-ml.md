# ML textbook refresh — November 2026

Scope: all 15 English/Traditional Chinese ML chapter pairs, including their 21
paper guides. Baseline: `116d724`, branch `feat/textbook-refresh-2026-11`.
Read `AGENTS.md` and the generated ML content map before inspecting chapter
structure, prerequisite links, examples, mathematical qualifications, evaluation
claims and optional paper material. This is a targeted content and executable
example audit, not a fresh reproduction of every cited paper's experiments.

Seven pairs received substantive corrections; eight pairs were reviewed and
left unchanged. No new chapter, heading, paper guide, figure, dependency or code
example was added. All existing heading text/IDs, Python blocks, figure imports
and paper-guide structure were preserved. Shared indexes, curriculum, UI and
figure registries were left to the integrating agent.

## Chapter-by-chapter findings and actions

| Chapter file stem | Reviewed focus | Finding and action |
| --- | --- | --- |
| `ml-01-introduction` | Learning paradigms, full-batch/mini-batch distinction, train/validation/test use, numerical least squares, Domingos guide | **Updated.** Convexity does not imply unique parameters without full column rank. Removed the incorrect claim that `np.polyfit` forms normal equations; linked its actual least-squares/SVD documentation. Restricted the monotonic-loss diagnostic to the deterministic quadratic and explained why stochastic losses fluctuate. |
| `ml-01a-linear-algebra` | Shapes, conditioning, SVD, centered PCA, training-only fit, PCA review guide | **Reviewed unchanged.** The existing SVD/least-squares explanation and training-only PCA warning address the key assumptions; no new section needed. |
| `ml-01b-probability-information` | Density versus probability, conditioning, likelihood/MAP, KL support, entropy units, Shannon guide | **Reviewed unchanged.** Definitions, examples and the guide distinguish continuous/discrete quantities and finite-data estimation. The deliberately overflowing exponential example behaves as documented. |
| `ml-01c-numerical-computation` | Floating-point limits, condition numbers, finite differences, mixed precision and its guide | **Updated.** Replaced the general eigenvalue-ratio assertion with the singular-value definition, qualified actual error amplification, and corrected the claim about detached tensors: finite differences can catch a dependency cut. Added primary NumPy and PyTorch documentation references. |
| `ml-02-neural-networks` | Output/loss pairing, normalization modes, backpropagation, optimization, Adam guide and convergence caveat | **Reviewed unchanged.** Existing code and guide distinguish optimization from generalization and acknowledge the Adam convergence correction. |
| `ml-03-convolutional-networks` | Correlation/convolution conventions, tensor dimensions, residual matching, held-out model selection, ResNet guide | **Reviewed unchanged.** The CPU classifier selects on validation and evaluates the test set after selection; the guide keeps architecture and evaluation protocol separate. |
| `ml-03a-recurrent-recursive-networks` | BPTT, state detachment, padding, teacher forcing, streaming restrictions, original versus modern LSTM | **Reviewed unchanged.** Examples pass sequentially; the LSTM guide explicitly distinguishes the original cell from later forget-gate variants. |
| `ml-04-vision-transformers` | Token shapes, attention masks, positional information, small-data comparison, ViT guide | **Updated.** The suggested switch from `MultiheadAttention` to functional SDPA needs inverted Boolean-mask semantics. Added explicit evaluation dropout handling and the MHA reference. Verified conversion with matched projections. |
| `ml-05-representation-learning` | Positive-pair assumptions, collapse, train-only fitting, held-out transfer, SimCLR guide | **Reviewed unchanged.** The examples separate representation fitting from test evaluation; augmentation and collapse caveats are already substantive. |
| `ml-05a-factor-models-autoencoders` | PCA/FA/NMF assumptions, train/test reconstruction, autoencoder sampling limits, PPCA guide | **Updated.** Removed “PCA wins … as it must” for a held-out comparison. The theorem applies to the fitted training matrix; held-out ranking is empirical. |
| `ml-06-generative-models` | Likelihood units, VAE bound, adversarial assumptions, change of variables, VAE/GAN guides | **Updated.** Distinguished discrete coding bpd from a continuous density; supplied the 8-bit dequantization/rescaling correction and identified the result as a bound, not exact discrete likelihood. |
| `ml-07-diffusion-flow-matching` | Time conventions, parameterization, sampling, MeanFlow/iMF/JiT/pMF distinctions, six paper guides, evaluation | **Updated.** Distinguished an exact density identity from numerical ODE and divergence-estimation error; qualified ODE/SDE marginal equivalence for learned approximate scores. Separated ODE likelihood evaluation from DDPM variational bounds and reminded readers to state dequantization. Recent detailed paper guides remain intact. |
| `ml-08-autoregressive-vision` | Target shifting/masking, token versus pixel likelihood, sampling, VQ-VAE guide | **Reviewed unchanged.** Existing text already warns against comparing lossy token likelihood directly with pixel bpd and distinguishes exact factorization from modeling assumptions. |
| `ml-09-foundation-models` | Contrastive supervision, frozen adaptation, prompt protocol, contamination, CLIP guide | **Reviewed unchanged.** Existing caveats distinguish zero-shot transfer from no supervision and similarities from calibrated confidence. |
| `ml-10-evaluation-deployment` | Split unit, nested selection, uncertainty, calibration, distribution shift, efficiency, temperature-scaling guide | **Updated.** Precision's prevalence dependence leaves recall/FPR unchanged only under fixed class-conditional distributions and threshold. Added that assumption and changed the existing label-shift reference to the verified ICML proceedings. |

No supplementary-note dependency was introduced into the main learning path.
Paper guides remain optional and contain their existing thesis, attributed
figure, method, evidence, design analysis, reading checkpoint and limitations.
Their baseline headings and imported figures were retained.

## Primary sources reopened for these corrections

- NumPy developers, `numpy.polyfit`: https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
- NumPy developers, `numpy.linalg.cond`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html
- Bottou, Curtis and Nocedal, *Optimization Methods for Large-Scale Machine Learning*, revised 2018 manuscript: https://arxiv.org/abs/1606.04838
- PyTorch contributors, `gradcheck`: https://docs.pytorch.org/docs/2.15/generated/torch.autograd.gradcheck.gradcheck.html
- PyTorch contributors, functional SDPA: https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch contributors, `MultiheadAttention`: https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html
- Jolliffe and Cadima, *Principal component analysis: a review and recent developments*, 2016: https://pmc.ncbi.nlm.nih.gov/articles/PMC4792409/ (publisher request returned 403; full published article verified through PMC).
- Theis, van den Oord and Bethge, *A note on the evaluation of generative models*, 2016 revision: https://arxiv.org/abs/1511.01844
- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR 2021 manuscript, especially Appendix D.2: https://arxiv.org/html/2011.13456v2
- Lipton, Wang and Smola, *Detecting and Correcting for Label Shift with Black Box Predictors*, ICML, PMLR 80:3122–3130, 2018: https://proceedings.mlr.press/v80/lipton18a.html

Access date: 2026-11-05. Documentation behavior relevant to the mask and gradient
corrections was also checked against the installed PyTorch 2.7.1+cpu runtime.
Existing reference numbering was preserved; new references were appended only
where needed. No unverified contemporary model claim was added.

## Validation

- `node scripts/check-mdx.mjs src/content/tutorials/en/ml/*.mdx src/content/tutorials/zh-tw/ml/*.mdx`: **30/30 pass**.
- All **102 existing Python blocks** executed in chapter order, sharing a namespace within each chapter; **15/15 chapter processes pass**, no skip or timeout. Ran the project `.venv/bin/python` with `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and a 180-second chapter limit. Slowest chapter: 52.98 seconds.
- EN/zh-tw Python ASTs match for all 102 blocks, so the same executable behavior covers both languages without repeating training.
- All existing heading text, Python blocks and image imports match baseline `116d724`. Bilingual heading counts and external reference URL sequences match across all 15 pairs.
- Independent local assertions confirm: an eigenvalue ratio can miss poor conditioning; an accidental detach can fail gradcheck; training-optimal PCA need not minimize test reconstruction error; uniform normalized density still corresponds to 8 bpd for 8-bit bins; inverted Boolean masks reproduce MHA output through SDPA.
- `git diff --check` for the owned ML files: pass.
- Expected/nonfatal warnings: the probability chapter intentionally overflows a naive exponential; the evaluation chapter's fixed-budget MLP reaches its 300-iteration limit and emits scikit-learn's convergence warning. No assertion or process failed in the project-environment run.
- The first environment probe used the system Python, which lacks PyTorch. It was superseded by the complete project-venv run and is not counted as a chapter failure.

Reproducible audit scripts, extracted chapter programs, complete stdout/stderr,
per-chapter timing and structural inventory are retained outside the repository:
`/workspace/scratch/cvblog-work/textbook-refresh/ml/` (`run_snippets.py`,
`results.json`, `structure.json`, `correction_checks.py`, chapter `.py`/`.log`
files). Integration-wide build, index, paper coverage, links and PDF validation
are handled by the parent agent. No commit or build performed by this worker.
