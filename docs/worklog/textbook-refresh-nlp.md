# NLP textbook refresh — 2026-11-05

Baseline: `116d724`; scope: all 12 English/Traditional Chinese NLP chapter pairs.
Changes are confined to NLP tutorial sources and this report. Existing headings,
public routes, paper figures and legacy IDs are preserved. Existing frontmatter
update dates are later than the environment date; they were not moved backward.

## Coverage and findings

All 24 files were scanned for section structure, bilingual Python parity,
reference-URL parity and available image imports. Conceptual review focused on
prediction contracts, probabilistic quantities, numerical examples, masking,
evaluation boundaries and the existing paper-guide explanations. This is not a
claim that every historical benchmark in every cited paper was reproduced or
that every existing external reference was downloaded again.

| Chapter | Review focus and finding | Action |
| --- | --- | --- |
| NLP 01 — Introduction | Task definitions, lossy preprocessing, Unicode/Chinese distinctions, split boundaries and the four paper-reading routes already make their assumptions explicit. | Reviewed; unchanged. Both Python blocks pass. |
| NLP 02 — Linguistic essentials | Morphology/tokenization, ambiguity/scope, annotation agreement, corpus selection and Zipf examples are internally consistent. | Reviewed; unchanged. Seven Python blocks pass. |
| NLP 03 — Neural networks for NLP | Training-only vocabulary/TF–IDF fitting, classification loss, negation controls, CNN padding and uncertainty evaluation have executable examples. | Reviewed; unchanged. Five Python blocks pass. |
| NLP 04 — Word embeddings | The distinction between normalized softmax and negative sampling, collision conventions and the shifted-PMI interpretation is already explicit. One Chinese paper-guide phrase used 採樣. | Changed that phrase to Taiwan usage 取樣; substantive explanation retained. Eight Python blocks pass. |
| NLP 05 — Recurrent networks | Corpus perplexity was defined correctly, but batch aggregation lacked a worked demonstration and the cross-tokenizer comparison advice omitted lossless-preprocessing conditions. | Added a token-weighted aggregation example distinguishing three averages; clarified what changing units cannot repair. Four final Python blocks pass. |
| NLP 06 — Sequence labeling | HMM joint/posterior probabilities, maximum path versus marginal decisions, span metrics, CRF normalization and recurrent padding are explained separately. | Reviewed; unchanged. Four Python blocks pass. |
| NLP 07 — Attention/Transformers | A generic Boolean mask can invert semantics when moved between PyTorch APIs. The old all-masked-row statement generalized manual-softmax behavior too broadly. | Added a tested SDPA/MultiheadAttention convention comparison, explicit dropout handling and precise padding-only-row caveats. Four final Python blocks pass. |
| NLP 08 — Seq2seq | Conditional likelihood, teacher forcing, beam-search limits, translation metrics and dialogue state have distinct acceptance criteria and source boundaries. | Reviewed; unchanged. Eight Python blocks pass. |
| NLP 09 — Language modeling | The top-p equation minimized arbitrary set cardinality, permitting a set that skips higher-probability tokens; the existing code used a sorted prefix. | Made the equation and explanation specify the shortest descending prefix; defined ties and valid threshold range; added a counterexample assertion. Twelve final Python blocks pass. |
| NLP 10 — BERT | Token/segment/position semantics, MLM target selection and probing controls are present. The English WordPiece text pointed to section 9 for Chinese BERT, while that topic is section 11. | Corrected the English cross-reference; the Chinese version already points to 11. Three Python blocks pass. |
| NLP 11 — BERT classification | Retrieval evidence, answerability, conditional error chains, calibration and supervised test boundaries are distinguished. | Reviewed; unchanged. Three Python blocks pass. |
| NLP 12 — Syntax/semantics | The PCFG demonstration set all lexical factors to one, then called its inside total a sentence probability. This leaves parent expansions unnormalized. | Added normalized lexical probabilities and per-parent mass checks, recomputed printed outputs, and distinguished best-tree joint probability, sentence probability and parse posterior. Eight final Python blocks pass. |

Six chapter pairs have edits (NLP 04 and NLP 10 require only one language's small
correction); six pairs are reviewed and unchanged. Four pairs have substantive
mathematical or operational additions/corrections.

## Verified sources for changes

Opened during this refresh:

- Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020:
  [record](https://arxiv.org/abs/1904.09751) and
  [full text, section 3.1](https://arxiv.org/html/1904.09751v2).
  Its implementation explanation selects highest-probability tokens; the course
  now makes that descending-prefix requirement explicit. The OpenReview venue
  URL encountered a browser challenge, so the verified arXiv record is retained.
- PyTorch 2.15 [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)
  and [`MultiheadAttention`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html):
  Boolean-mask semantics and functional dropout behavior. Added as NLP 07
  references 22–23, with matching numbers in both languages.
- NLTK [`PCFG` API](https://www.nltk.org/api/nltk.grammar.PCFG.html):
  probabilities of productions sharing a left-hand side must sum to one.
  Added as NLP 12 reference 40 in both languages.
- Jurafsky and Martin, [August 19, 2026 book release](https://web.stanford.edu/~jurafsky/slp3/)
  and [chapter 19](https://web.stanford.edu/~jurafsky/slp3/19.pdf):
  existing grammar/chart-parsing context and the probability-model distinction.

The perplexity example and probability calculations are original synthetic
calculations from the definitions already given in the course. No external
benchmark scores or model-performance claims were added.

## Validation

- All 24 NLP MDX files pass syntax/KaTeX checks; the final small reference and
  terminology edits were checked again.
- All 66 pre-existing Python blocks execute sequentially within each chapter,
  with `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and a
  180-second per-chapter timeout. No block errors or timeouts occurred.
- The two new standalone examples pass. All blocks in the four substantively
  changed chapters were then rerun in chapter order; final distinct coverage is
  68 Python blocks. No dataset/model downloads were used.
- The corrected PCFG outputs are `best=0.004551, total=0.006827` for recursive
  probability 0.2 and `best=0.001389, total=0.002500` for 0.5. Assertions verify
  each parent's normalization, the inside sum and the best-parse posterior.
- All 12 pairs retain identical Python blocks, heading structure and external
  URL sequences. All 58 imported image files exist. No heading was added or
  renamed; existing paper guides and figure registries are unchanged.
- `git diff --check` passes for this scope. Parent integration owns the content
  index, site build, internal-link audit and rendered-page checks.

Scratch logs, reproducible snippet runners and parity reports:
`/workspace/scratch/cvblog-work/textbook-refresh-nlp/`.
