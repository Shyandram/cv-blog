# Optimization and information-theory refresh — 2026-11-05

Baseline: `116d724`. Scope: all 8 OPT and 8 IT chapter pairs, including their
mathematical assumptions, worked examples, code, modern-view/reference sections,
and bilingual structure. These tracks had no paper-guide sections to extend.
The review preserves existing headings and keeps new detailed examples optional.
It does not claim an independent proof audit of every theorem or fresh retrieval
of every unchanged historical reference.

## Chapter-by-chapter review

| Chapter | Reviewed focus and finding | Action |
| --- | --- | --- |
| OPT 01 `opt-01-modeling-convex-sets` | Feasibility, attainment versus finite infimum, closed/convex projection assumptions, scaled residuals; distinctions already explicit. | Reviewed; unchanged. Both Python blocks pass. |
| OPT 02 `opt-02-convex-functions` | Open-domain Hessian test, strict versus strong convexity, composition/domain rules, conjugate certificates; no missing qualification identified in these checks. | Reviewed; unchanged. Three Python blocks pass. |
| OPT 03 `opt-03-convex-programs` | LP bounds, nonseparable QP clipping counterexample, Schur-complement domain and logarithmic GP transformation; numerical examples agree with stated models. | Reviewed; unchanged. Three Python blocks pass. |
| OPT 04 `opt-04-duality-kkt` | Slater/attainment, complementary slackness, approximate feasible pairs and corrected residual reasoning; false negative-gap example already explained. | Reviewed; unchanged. Three Python blocks pass. |
| OPT 05 `opt-05-fitting-estimation` | Convex likelihood does not guarantee a finite fitted coefficient; the Bernoulli boundary discussion lacked a concrete separated-regression example. | Updated both languages: optional two-observation logistic example, finite-unattained-infimum distinction, explicit regularization and executable root check. Four Python blocks pass. |
| OPT 06 `opt-06-gradient-newton` | Fixed-step assumptions, strong-convexity gap, local Newton conditions and derivative checks; new OPT 05 note links here for the regularized case. | Reviewed; unchanged. Both Python blocks pass. |
| OPT 07 `opt-07-constrained-interior-point` | Reduced Hessian, KKT indefiniteness, central-path gap versus approximate centering, phase-I boundary cases; qualifications already present. | Reviewed; unchanged. Three Python blocks pass. |
| OPT 08 `opt-08-applications-modern-methods` | Proximal domain, ADMM dual state/refactorization, exact and inexact Frank–Wolfe certificates, communication costs; checks support the examples. | Reviewed; unchanged. Both Python blocks pass. |
| IT 01 `it-01-entropy-mutual-information` | Probability support, infinite KL, average conditioning, data processing and empirical entropy bias; masking and parity tests behave as described. | Reviewed; unchanged. Three Python blocks pass. |
| IT 02 `it-02-typicality-entropy-rates` | Typical probability versus cardinality, finite-block tolerance, Markov entropy rate and nonergodic mixture; examples distinguish the assumptions. | Reviewed; unchanged. Three Python blocks pass. |
| IT 03 `it-03-lossless-universal-coding` | Prefix decoding, tree/length side information, arithmetic interval round trip and adaptive model synchronization; no completion gap found in these checks. | Reviewed; unchanged. Three Python blocks pass. |
| IT 04 `it-04-channel-capacity` | Channel orientation, BA upper/lower gap, average-message error, finite repetition code and input constraints; numerical limits already stated. | Reviewed; unchanged. Both Python blocks pass. |
| IT 05 `it-05-continuous-gaussian-channels` | Gaussian entropy helper accepted any matrix with positive determinant, including a negative-definite matrix. | Fixed both languages: square/finite/symmetry checks plus positive-definite Cholesky factorization; reject negative-definite, singular and asymmetric matrices. Explain why adding noise changes the model. Three Python blocks pass. |
| IT 06 `it-06-rate-distortion` | Backward binary/Gaussian constructions, penalty units, update tolerance versus a certificate, quantizer entropy versus actual rate; analytic comparisons hold. | Reviewed; unchanged. Four Python blocks pass. |
| IT 07 `it-07-statistics-decision-growth` | CR regularity is stated, but its support caveat lacked a concrete failure mechanism. | Updated both languages: optional uniform-endpoint example derives an unbiased maximum estimator, exposes nonzero expected score and invalid information-additivity substitution, and checks moments by simulation. Six Python blocks pass. |
| IT 08 `it-08-networks-information-learning` | Separate versus joint observations, multiple-access sum constraints, XOR marginal information, bounded-loss information calculation and parity-decoding accounting. | Reviewed; unchanged. Both Python blocks pass. |

## Verified sources for changes

Opened on 2026-11-05:

- Boyd and Vandenberghe, *Convex Optimization*, 2004: official book page and
  full PDF, particularly likelihood/model existence context.
  https://web.stanford.edu/~boyd/cvxbook/
  https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
- SAS Institute, “Existence of Maximum Likelihood Estimates,” *SAS/STAT 12.1
  User's Guide*: finite-estimate and separation distinctions. The fixed
  documented version is explicit; no claim about latest software behavior.
  https://support.sas.com/documentation/cdl/en/statug/65328/HTML/default/statug_logistic_details10.htm
- Cover and Thomas, *Elements of Information Theory*, second edition, Wiley,
  copyright 2006: official book/edition record.
  https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X
- Tsachy Weissman, Stanford EE276 Lecture 8, 2024: continuous-variable and
  Gaussian entropy setting.
  https://web.stanford.edu/class/ee276/files/previous_notes/lecture_8.pdf
- NumPy, `numpy.linalg.cholesky`: requires positive-definite input and does not
  itself verify symmetry; this is why the example explicitly checks symmetry.
  https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html
- MIT OpenCourseWare, *14.381 Statistical Method in Economics*, Fall 2018,
  Lecture 6, pp. 3–6: score regularity and the parameter-dependent uniform-support
  counterexample. The new note derives its moments rather than copying prose.
  https://ocw.mit.edu/courses/14-381-statistical-method-in-economics-fall-2018/15224973dfc1c2b4287804c8712681f7_MIT14_381F18_lec6.pdf

## Validation

- All 16 bilingual pairs: heading-level structure, Python code blocks and
  external-reference URL sequences match; prerequisite source files exist.
- All 46 original English Python blocks passed before edits. All 48 blocks pass
  after additions, including the changed Gaussian helper and its rejection tests.
  Chinese code is byte-identical, so no separate numerical run is represented.
- All 6 changed MDX files pass syntax/KaTeX/frontmatter checks.
- Existing headings are preserved; two new optional-note headings and their
  bilingual pointers require normal parent-level content-index regeneration and
  built-route verification.
- No shared index, curriculum, figure registry, original paper asset, UI file,
  commit or build was changed/performed by this subtask.

Detailed per-block output and structural results:
`/workspace/scratch/cvblog-work/textbook-refresh-opt-it/snippet-results.json` and
`structure-results.json`. The runner is `run-snippets.py` in the same directory.
