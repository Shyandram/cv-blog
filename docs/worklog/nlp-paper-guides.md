# NLP paper guides — 2026-11-06 (Asia/Taipei)

## Pull and rescan

`git pull --ff-only` on `local/site-subchapters-tree`, tracking `origin/wip/site-subchapters-tree`, returned **Already up to date**. The starting local HEAD was `66e83cb`, eight commits ahead of that upstream. Remote main contains the earlier PR #4 merge; its merge result adds no content relative to its second parent. No branch merge or deployment was necessary for this task.

The pre-change paper coverage check passed for 155 guide units across 40 chapters: 126 original-source figures, 22 comparative maps, seven teaching diagrams, and zero missing figures. A fresh scan of all twelve NLP chapter pairs found substantial existing main text, examples, exercises and references, but **zero independent paper guides**. English/Traditional Chinese heading-depth order, figure counts, display-math counts and code-block counts matched; no TODO/FIXME markers were found in those chapters. This structural scan does not establish a fresh scientific audit of every existing reference or reproduce the chapter experiments.

## Completed content scope

This first NLP batch adds twelve original-paper guides across NLP 02–12, plus an optional four-stage reading route in NLP 01. It extends the previous paper-guide standard to NLP rather than claiming that every paper cited anywhere in NLP now has an independent guide. Existing main chapters remain the prerequisite material.

Every new guide has six real, foldable subsections: challenge, method, results, ablations/design evidence, conclusion and limitations. Each states the thesis, explains the source figure, defines the objective's symbols, ties reported evidence to its evaluation conditions, and gives a reading checkpoint or next step. The bilingual pages share figures, formulas, citation numbers and section order. Main-text pointers lead to the relevant standalone paper page. No new training snippets or model-reproduction claims were added.

| Chapter | Guide | Verified primary version and figure | Evidence distinction retained |
| --- | --- | --- | --- |
| NLP 02 | BPE, Sennrich/Haddow/Birch | ACL 2016 proceedings; figure 1, PDF p. 4 | Figure 1 and Algorithm 1 use different toy dictionaries; rare-word F1 improves while OOV F1 can fall; joint/separate BPE use different vocabulary budgets |
| NLP 03 | TextCNN, Kim | EMNLP 2014 proceedings; figure 1, p. 2 | Pretraining versus adaptation; multichannel helps SST-2 but hurts TREC in the reported comparison |
| NLP 04 | word2vec, Mikolov/Chen/Corrado/Dean | ICLR 2013 workshop manuscript, arXiv:1301.3781v3; figure 1, p. 5 | Hierarchical softmax in this early paper versus later negative sampling; semantic/syntactic analogy tradeoff; restricted-vocabulary scaling study |
| NLP 04 | GloVe, Pennington/Socher/Manning | EMNLP 2014 proceedings; figure 1, p. 4 | Weighted log-count regression; matched 6B-token comparison versus changed 42B-token corpus; dimension/window studies |
| NLP 05 | Seq2Seq, Sutskever/Vinyals/Le | NeurIPS 2014 author manuscript, arXiv:1409.3215v3; figure 1, p. 2 | Source reversal versus ensembling/search; direct translation versus rescoring an SMT candidate list |
| NLP 06 | BiLSTM-CRF, Huang/Xu/Yu | 2015 manuscript, arXiv:1508.01991v1; figure 7, p. 4 | Pretrained/random embeddings; 88.83 NER F1 versus 90.10 with gazetteer; original engineered features remain part of the setup |
| NLP 07 | Transformer, Vaswani et al. | NeurIPS 2017 proceedings; figure 1, p. 3 | Three attention routes; parallel training versus autoregressive generation; newstest2014 results versus newstest2013 development ablations |
| NLP 08 | Bahdanau attention | ICLR 2015 conference manuscript, arXiv:1409.0473v7; figure 1, p. 3 | Full test versus no-unknown-word subset; training length/duration versus architecture; soft alignment is not causal explanation |
| NLP 09 | InstructGPT, Ouyang et al. | NeurIPS 2022 proceedings; figure 2, p. 4 | SFT, reward model and policy have different roles; common SFT comparison baseline in figure 1; preference, truthfulness and toxicity are different outcomes |
| NLP 10 | BERT, Devlin et al. | NAACL 2019 proceedings; figure 1, p. 3 | Selected-position masking percentages; base development results; bidirectionality and NSP controls within the original recipe |
| NLP 11 | Sentence-BERT, Reimers/Gurevych | EMNLP-IJCNLP 2019 proceedings; adjacent figures 1–2, p. 3 | NLI classification training versus cosine inference; no STS-specific training does not mean no labels; development pooling study versus task evaluation |
| NLP 12 | End-to-end coreference, Lee/He/Lewis/Zettlemoyer | EMNLP 2017 proceedings; figures 1 and 2, p. 3, as separate excerpts | Span construction/pruning versus antecedent links; dummy has several roles; marginalize valid gold antecedents; test scores versus development ablations |

The first-page titles/authors and relevant method, result and design-study sections were inspected in the downloaded primary PDFs. Published proceedings PDFs are used where available; arXiv manuscript revisions are identified above and in the manifest. The InstructGPT review uses the published main paper, not an uninspected appendix. Existing bibliography numbers are preserved; Seq2Seq is appended as NLP 05 reference 25 in both languages.

## Source figures and reading behavior

Thirteen original-source PNG excerpts serve the twelve guides. Sentence-BERT's adjacent original figures share one excerpt; coreference uses two separate crops so intervening body text is excluded. GloVe's weighting-function diagram and BPE's merge-operation figure explain those methods better than an invented network diagram. No teaching redraw was needed.

All crops were visually inspected against the original pages. Source URLs, SHA-256 hashes, versions, PDF pages, figure numbers and crop rectangles are recorded in `docs/paper-figure-sources.json`. Images meet the width and file-size limits and retain original labels. Reproduce them with:

```bash
python scripts/figures/extract_paper_figures.py \
  --cache-dir /tmp/cvblog-paper-cache \
  --only nlp-bpe nlp-textcnn nlp-word2vec nlp-glove nlp-seq2seq \
  nlp-bilstm-crf nlp-transformer nlp-bahdanau nlp-instructgpt \
  nlp-bert nlp-sbert nlp-coref nlp-coref-links
```

The NLP 01 route uses four foldable stages, each with prerequisites, reading links and a question to answer. A four-column draft passed width checks but split words awkwardly on a 375-pixel screen; visual inspection prompted these stacked stages instead. It groups representation, local/sequence decisions, attention/pretraining, and task/structure objectives. Within the attention history, Seq2Seq → Bahdanau → Transformer supplies the motivation even though the curriculum first teaches attention mechanics in NLP 07. Existing guide titles, chapter paths and the legacy subsection-ID baseline are unchanged.

## Validation

The following checks passed on the final source state:

- All 24 changed tutorial MDX files compile, including KaTeX. The four files touched by the image-import capitalization fix were rechecked afterward. English/Traditional Chinese heading-depth order, exact display equations, code blocks and figure variables match; new Chinese prose passes the terminology scan.
- `node scripts/content-index.mjs --check`: current. `node scripts/check-paper-guides.mjs`: 167 guides across 51 chapters, 139 original-source figures, 22 comparative maps, seven teaching diagrams, zero missing figures and matching bilingual sections/citations. All twelve new guides have six subsections.
- All twelve primary bibliography landing links return HTTP 200. All thirteen NLP figure-manifest entries match the downloaded PDF hashes; extracted images are at most 1,591 pixels wide and below 1 MB each.
- `node scripts/tests/paper-heading-ids.test.ts`: all four regression cases pass. The static built-page audit retains all 828 legacy subsection IDs and texts; the baseline file is unchanged.
- Final `npm run build`: 1,172 content pages, 1,171 Pagefind pages. `npx astro check`: zero errors, zero warnings, three pre-existing hints.
- `python scripts/check-site-links.py`: 2,137 HTML pages, 143,553 internal links, zero broken links.

The twelve paper guides pass 48 English/Traditional Chinese desktop/mobile layouts at 1,440/375 pixels: full-size image loading, language/parent links, horizontal bounds, fold-all/open-all, keyboard operation, fragment reveal and no JavaScript errors. Twelve legacy browser forwarding routes also pass. See [the guide browser report](nlp-paper-browser-validation.json).

The revised introductory route passes all 48 paper-link clicks across the same four language/viewport combinations. Each of its four stages folds by keyboard and through the floating console. In a fresh session, all ten NLP 03 code blocks start closed; the console opens and closes them successfully. There is no viewport overflow or JavaScript error. See [the reading-route report](nlp-reading-route-validation.json).

Original crops and rendered pages were visually inspected, including mobile Sentence-BERT, desktop Transformer, and the final English/Traditional Chinese mobile reading routes. The last build contains the stacked, foldable route rather than the rejected narrow-column draft. No original model training or benchmark reproduction is claimed. This task creates a local commit only; it performs no push, merge or deployment.

## Remaining scope

This is a complete first batch of twelve introductory NLP guides, not coverage of every reference in the twelve main chapters. Further paper-specific guides can extend the existing chapter material on tokenization variants, newer recurrent models, retrieval, adaptation and parsing. No new broad claim about all NLP methods being experimentally verified is made.

The previously documented SeAFusion original numerical comparison/ablation-table gap remains outside this NLP batch; see [remaining-supplements-audit.md](remaining-supplements-audit.md). The prior IP audit remains in [paper-review-ip-expansion.md](paper-review-ip-expansion.md). This work does not close that unrelated source limitation.
