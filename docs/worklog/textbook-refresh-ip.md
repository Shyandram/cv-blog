# Image-processing textbook refresh — 2026-11-05

Baseline: `116d724`, branch `feat/textbook-refresh-2026-11`.
Scope: all 13 English/Traditional Chinese image-processing chapter pairs. Read
`AGENTS.md` and the image-processing content map first. This pass scanned chapter
structure, core explanations, optional-note/paper-guide coverage, numerical
assumptions, and executable examples. It is a targeted textbook correctness pass,
not a new reproduction of every cited experiment or an external recheck of every
unchanged reference.

## Chapter-by-chapter disposition

| Chapter | Checked | Finding and action |
| --- | --- | --- |
| IP 01 — Introduction | Image-array vocabulary; acquisition/processing/analysis roadmap; modality notes; modern imaging context; all 6 Python blocks | Reviewed, unchanged. Array, quantization, CT, photon-energy, coin-counting and storage examples execute. The chapter already explains that imaging physics changes the appropriate processing method. |
| IP 02 — Fundamentals | Sampling, interpolation, coordinate conventions, connectivity, arithmetic, homography and PSNR notes; review reading map; all 13 blocks | Updated both languages. Bilinear interpolation silently accepted negative coordinates through NumPy indexing; added domain checks and an explicit border-policy explanation. Corrected the exercise hint that incorrectly described rigid rotation as stretching distances. The project venv lacks `ProjectiveTransform.from_estimate`; use the documented instance `estimate` interface, check success, and update the quoted numerical output from an actual run. Added API reference 21. |
| IP 03 — Intensity/spatial filtering | Point transforms, histogram methods, convolution/correlation, smoothing/sharpening; guided/bilateral notes and original-figure guides; all 13 blocks | Reviewed, unchanged. The existing detailed guided and bilateral comparisons run, including the edge-gradient and ramp behavior examples. No additional generic method survey was needed. |
| IP 04 — Frequency domain | Sampling/aliasing, DFT properties, phase, zero-phase filtering, padding, spectral-learning reading map; all 10 blocks | Updated both languages. Replaced one-argument arctangent in both complex-number and Fourier-phase formulas with quadrant-aware `atan2`; added a tested quadrant example and the undefined-phase-at-zero qualification. Corrected the claim that every real symmetric transfer function is zero-phase: negative gains add π. Added NumPy reference 21. |
| IP 05 — Restoration/reconstruction | Noise/degradation assumptions, inverse/Wiener/CLS models, CT, non-local-means/TV notes, NLM/BM3D guides; all 11 blocks | Reviewed, unchanged. Noise, filter, NLM and TV examples execute. Existing notes already distinguish regularization/priors from the known measurement model and demonstrate smoothing/edge tradeoffs. |
| IP 06 — Color | Color spaces, encoded versus linear signals, white balance, vector filtering, colorization reading map; all 10 blocks | Updated English only: the paper-guide lead-in incorrectly named histogram rules and edge-aware enhancement from another chapter. Replaced it with color constancy, vector filtering, color mapping and automatic colorization, matching the already-correct Chinese lead-in. No Chinese prose change was necessary. |
| IP 07 — Wavelets/transforms | Bases/frames, transforms, DWT filter banks, thresholding, boundary handling and review guide; all 7 blocks | Updated both languages. Qualified the claim that an orthonormal DWT always returns exactly M coefficients: finite-array extension can add boundary coefficients. Added a verified PyWavelets boundary-mode reference and the need to use matching analysis/synthesis modes. Removed an unsupported universal “fastest strong baseline” claim and qualified shrinkage guarantees. Corrected the unrelated English reading-map introduction to match Chinese. |
| IP 08 — Compression/watermarking | Coding/entropy, JPEG/DWT pipelines, rate versus coefficient count, LSB/DCT watermark examples and survey guide; all 10 blocks | Updated both languages. Derived LSB PSNR from the changed-pixel fraction instead of implying one universal value; 51.14 dB corresponds to half the pixels changing and 48.13 dB to every pixel changing. Clarified that small transform coefficients may contain real detail. The unkeyed LSB toy demonstrates fragility, not authenticated integrity. Corrected the unrelated English survey lead-in to match Chinese. Existing figure's approximately 51 dB caption remains correct for its actual mark. |
| IP 09 — Morphology | Set definitions/reflection, border consistency, reconstruction, skeleton/pruning and grayscale notes; all 11 blocks | Reviewed, unchanged. The explicit duality/border and idempotence examples pass. Optional notes already separate thinning, morphological skeletons and reconstruction. |
| IP 10 — Segmentation | Gradient/threshold definitions, clustering, graph/watershed methods, optional derivations and U-Net guide; all 12 blocks | Reviewed, unchanged. Code executes, including original U-Net size arithmetic and marker-controlled segmentation. Gradient direction already correctly uses `atan2`. |
| IP 11 — Active contours | Snake/GVF and level-set updates, sign convention, numerical notes, exercise hints and Chan–Vese guide; all 6 blocks | Updated both languages. The claimed μ ≤ 1/4 condition ignored the reaction term. Added a conservative adaptive explicit time step, a normalized-map regression example, and a derived sufficient boundedness condition in the existing GVF note. Corrected the exercise hint to use evolution time nΔt rather than treating iterations as time or μ > 0.25 as automatically unstable. |
| IP 12 — Feature extraction | Shape/Fourier descriptors, texture, PCA, Harris/MSER/SIFT, RANSAC note and SIFT guide; all 9 blocks | Updated both languages. First-harmonic normalization can divide by zero or a numerically tiny value; added a guard and explained sampling/traversal assumptions. Tested a constant contour and a reversed complex-circle contour in addition to the existing transformed-horse invariance example. |
| IP 13 — Pattern classification | Prototype/Bayes rules, perceptron/backprop/CNN examples, data splits and review guide; all 6 blocks | Updated both languages. Corrected the main text and optional derivation that called perceptron learning ordinary backpropagation through a step activation. A hard step has zero derivative away from its threshold and no derivative at the threshold; the mistake-driven perceptron rule is distinct. Reverified the existing original papers. |

Result: **8 chapters updated, 5 reviewed without changes; 15 tutorial files
changed**. All original section headings, note headings and paper-guide headings
are preserved. No new paper guide or architecture diagram was added; existing
figure provenance, IDs, curriculum, shared index and UI were left for the parent
integration process.

## Primary-source verification

New documentation references:

- IP 02 reference 21: scikit-image contributors, `ProjectiveTransform.estimate`,
  version 0.25.2 documentation:
  <https://scikit-image.org/docs/0.25.x/api/skimage.transform.html#skimage.transform.ProjectiveTransform.estimate>.
- IP 04 reference 21: NumPy developers, `numpy.angle`; confirms use of `arctan2`
  and zero-magnitude conventions:
  <https://numpy.org/doc/stable/reference/generated/numpy.angle.html>.
- IP 07 reference 21: PyWavelets developers, signal extension modes; confirms
  redundancy and matching reconstruction modes:
  <https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html>.

Reopened sources supporting corrections to existing material:

- Xu and Prince, *Snakes, Shapes, and Gradient Vector Flow*, IEEE TIP 7(3),
  359–369, 1998: author project <https://iacl.ece.jhu.edu/static/gvf/> and
  original PDF <https://iacl.ece.jhu.edu/pubs/p084j.pdf>. The new conservative
  bound is explicitly derived from the displayed discrete update; it is not
  presented as a quoted sharp bound from the paper.
- SciPy `ndimage.laplace` documentation, including replicated (`nearest`)
  boundary semantics:
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.laplace.html>.
- Rumelhart, Hinton and Williams, *Learning representations by back-propagating
  errors*, Nature 323, 1986, DOI `10.1038/323533a0`; author PDF opened at
  <https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf> and metadata verified
  through <https://api.crossref.org/works/10.1038/323533a0>.
- Rosenblatt, *The perceptron: A probabilistic model for information storage and
  organization in the brain*, Psychological Review, 1958, DOI
  `10.1037/h0042519`; publisher access was blocked, so title, author, year and
  venue were verified through <https://api.crossref.org/works/10.1037/h0042519>.

All accessed 5 November 2026. Elementary changes to rotation, LSB distortion,
Fourier normalization and convex update weights are direct mathematical/code
checks, not new empirical paper claims.

## Validation and numerical evidence

- **124 Python blocks across all 13 English chapters executed successfully** in
  chapter order using `.venv/bin/python`, with no final stderr output. Some
  existing snippets reuse imports/variables from earlier blocks; this check does
  not claim that each block is a standalone program. English and Chinese Python
  block contents are identical across all 13 pairs, so identical translated
  snippets were not redundantly executed a second time.
- Rechecked the final Fourier-descriptor edit separately after the complete
  chapter run. Existing transformed-horse difference is
  `1.1102230246251565e-16`; both degenerate normalization cases raise `ValueError`.
- Interpolation tests cover the last pixel, negative coordinates, coordinates
  beyond either boundary, and NaN. Fourier-angle tests cover all four quadrants.
- GVF checks cover constant, normalized checker and random maps with μ values
  0.05, 0.25 and 2.0; outputs remain finite and within the initial component
  bounds. On the 2×2 checker map, the old μ=0.25/unit-step update grows to
  `max |u| = 38351.01184657682` after 30 steps. The new code converges to rows
  `[0.8, 0.8]` and `[-0.8, -0.8]`, matching its documented output.
- A 16-sample `db2` DWT produces 16 coefficients with `periodization` and 18 with
  `symmetric`; both reconstruct using their matching inverse mode.
- Direct LSB calculations give `51.14 dB` for changed fraction 0.5 and
  `48.13 dB` for fraction 1. Recreating the existing figure's mark without
  rewriting the image gives changed fraction `0.4961967468261719` and
  `51.17426448251497 dB`, consistent with its rounded caption.
- Changed MDX files pass syntax/KaTeX/frontmatter validation.
- All 13 bilingual pairs have matching reference counts and exact Python-code
  parity. A comparison against `116d724` confirms no heading changes. No new
  Simplified Chinese/mainland terminology was introduced.
- `git diff --check` passes for the owned tutorial paths.

Scratch execution scripts and logs are under `/tmp/cvblog-ip-audit/` and are not
repository deliverables. Full production build, generated content indexes,
site links and browser validation remain the parent agent's integration checks.
No commit or push was performed by this agent.
