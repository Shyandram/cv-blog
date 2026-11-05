# Foundations and Pattern Recognition textbook refresh

Baseline: `116d724`, branch `feat/textbook-refresh-2026-11`.
Scope: all 12 English/Traditional Chinese chapter pairs in Foundations (2) and
Pattern Recognition (10). Existing chapter headings, optional-page headings,
figures, curriculum positions, prerequisites and paper IDs are retained.

The review used the generated track maps, chapter outlines, mathematical
assumptions, code/data flow, exercises, optional notes and the existing PR 09
paper guide. This is a chapter-wide teaching/correctness scan with targeted
repairs, not a claim to have reverified every historical bibliography entry or
reproduced every cited paper's experiments. No new paper guide or figure was
needed. Each chapter had a concrete issue; no generic per-page filler was added.

## Chapter-by-chapter findings and actions

| Chapter pair | Finding | Completed action |
| --- | --- | --- |
| Foundations: getting-started | Promised every block was independent, despite sequential arrays/capstone examples; track table and some routes used old chapter numbers; claimed one dependency list covered all examples. | Explain notebook execution order, fresh kernels and setup dependencies using actual chapter examples. Align ML 01–11 / DLCV 01–14 and vision/language/world-model routes with the current curriculum. Distinguish introductory dependencies from specialized extras. |
| Foundations: images-as-arrays | Called all pixel access a view, including scalar integer indexing; did not distinguish advanced-index reads from assignment. | Explain scalar versus remaining-axis views, copies from advanced reads, and writes through direct indexed assignment; add verified NumPy reference. |
| PR 01: Bayes decisions | Ratio inequality omitted cost-sign assumptions; doubling a probability was equated with doubling odds; English ROC description incorrectly varied the iso-cost slope. | State positive-prior and positive cost-difference assumptions, direct-risk fallback and reversed inequality; distinguish prior odds from probability; explain parallel fixed-slope iso-cost lines. |
| PR 02: density estimation | Interpreted the 1-NN error bound as “half the information”; high-dimensional neighborhood failure was presented as universal. | Distinguish classification-error bounds from information quantities; qualify distance concentration by irrelevant/diffuse coordinates and intrinsic structure/sample size. |
| PR 03: linear models | Ordinary distance to a hyperplane could be negative; attributed squared conditioning to inversion rather than forming normal equations; Gaussian squared loss equated exactly to NLL. Reference 3 was uncited. | Correct ordinary/signed distance and closest point; locate conditioning at the Gram matrix; include scale/additive-constant qualification; cite the existing linear-classifier textbook at the geometry claim. |
| PR 04: nonlinear classifiers | “Always standardize” conflicted with the capstone's validated raw-pixel choice; slack equal to one was called correctly classified. | Explain scaling as a validated modeling choice fitted inside training folds, with official SVM documentation; distinguish margin interior, decision boundary and errors at optimum. |
| PR 05: features/dimensionality | PCA reconstruction identity missed the sample-covariance normalization factor; regularized whitening was described as identity covariance. | Define per-sample versus per-coordinate error, include `(N−1)/N`, and add a runnable assertion; give diagonal variance `lambda/(lambda+epsilon)` and zero-eigenvalue handling. |
| PR 06: templates/structured models | CTC helper indexed beyond an empty transcript and did not handle a zero-frame input. | Handle empty target and zero-frame cases, assert empty/all-blank and impossible repeated-label behavior, explain probability-space underflow limits. Existing NCC, DTW, HMM/CRF and CTC structure retained. |
| PR 07: clustering/mixtures | Suggested diagonal/spherical covariance alone cures mixture collapse, and described EM convergence too categorically. | Explain that the collapsing spherical component is already diagonal; require a positive variance floor/constraint or suitable prior, distinguish fewer parameters from bounded likelihood, and qualify convergence. |
| PR 08: approximate inference | Monte Carlo rate omitted finite-variance requirements; self-normalized evidence/support assumptions were incomplete; Jensen equality required full coverage; finite grids were labeled exact. | State integrability/finite variance and dimension-dependent constants/cost; strengthen proposal support for denominator/evidence; qualify Jensen equality and retain the general ELBO bound; label quadrature a refinement-checked approximation. |
| PR 09: ensembles/validation | Existing double-descent paper guide equated parameter count `N>=n` with interpolation feasibility. | Add full-row-rank/column-space condition and distinguish minimum-norm least squares from exact interpolation, with NumPy's primary solver specification. Existing paper figure/IDs and experiment remain. |
| PR 10: recognition capstone | Independent open-set block overwrote `X`, `y`, `test` before the original model's stress test; a post-selection corrected t value was treated as a statistical tie; OOF diagnostics omitted selection reuse. | Put the independent experiment in `open_set_demo()`; preserve the original locked split/model; label the test exploratory and explain that overlap correction does not fix selection bias or prove equivalence; clarify OOF fit versus selection exclusion. |

## Source checks

Opened primary sources on 2026-11-05:

- NumPy, “Indexing on ndarrays”: https://numpy.org/doc/stable/user/basics.indexing.html
  (scalar/basic/advanced indexing and assignment semantics; new Arrays ref. 13).
- scikit-learn, “Support Vector Machines”: https://scikit-learn.org/stable/modules/svm.html
  (scaling recommendations and training/test transform consistency; new PR 04 ref. 27).
- scikit-learn, “GaussianMixture”: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
  (`reg_covar` and covariance configurations; new PR 07 ref. 26).
- NumPy, “numpy.linalg.lstsq”: https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
  (rank-deficient minimum-norm least squares; new PR 09 ref. 27).
- scikit-learn, decomposition and common-pitfalls documentation:
  https://scikit-learn.org/stable/modules/decomposition.html and
  https://scikit-learn.org/stable/common_pitfalls.html
  (cross-checks for PCA/whitening and leakage).
- Cawley and Talbot, “On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation,” JMLR 11(70), 2010:
  https://www.jmlr.org/papers/v11/cawley10a.html
  (existing capstone ref. 7, now cited at selection-reuse qualifications).
- Bishop's official PRML page:
  https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/
  (existing textbook scope/reference cross-check).

Arrays reference 8 was already cited with a section locator (`[8, Sec. 2.5.1]` /
`[8, 第 2.5.1 節]`); it was not an unused reference. No reference renumbering.

## Validation

- All 24 changed MDX files passed `scripts/check-mdx.mjs`.
- Header comparison against baseline and Python-block comparison across all 12
  EN/zh-tw pairs passed: no heading changes and identical executable Python.
- New/changed PCA and CTC snippets executed successfully. CTC's original printed
  results remain unchanged; added boundary assertions pass.
- All Images-as-Arrays blocks executed sequentially.
- Entire nine-block PR 10 workflow executed sequentially. Explicit assertions
  verify `X`, `y`, development/test indices and the original model are unchanged
  after `open_set_demo()`. The final stress results reproduce the published
  clean/noise/translation/contrast values: 0.993, 0.991, 0.969, 0.544, 0.907.
- Additional numerical checks passed for scalar/view assignment, ordinary versus
  signed hyperplane distance, and parameter count without full row rank.
- Full sequential Python smoke: **68 blocks across all 12 chapters passed**, zero failures. Each chapter ran in a fresh namespace, in printed order, with CPU thread limits of one. Total execution time was 62.6 seconds; the slowest block took 11.2 seconds. Results: `all-snippets.json`.
- All 12 bilingual reference-URL sequences match; Traditional Chinese terminology scan found no prohibited Simplified terms. `git diff --check` passes.
- Build, generated index, site links and integration are owned by the parent.

Detailed temporary execution logs and scripts:
`/workspace/scratch/cvblog-work/textbook-foundations-pr/`.
