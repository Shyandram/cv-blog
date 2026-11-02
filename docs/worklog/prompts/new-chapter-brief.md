# Brief: write a NEW deep-learning tutorial chapter (English version + figure script + figures)

A separate agent translates it to zh-tw afterwards, so write translatable English (no wordplay).

Read FIRST and follow exactly: /home/user/cv-blog/AGENTS.md (all sections; git safety: you do NOT commit, no git
stash/reset/checkout/restore/clean; content rules: every named method/paper/dataset/metric/challenge needs a VERIFIED reference
you actually opened — arXiv abs page, CVF open access, OpenReview, PMLR/NeurIPS proceedings, DOI via
https://api.crossref.org/works/<doi>, or the official project/docs page; prefer the final venue; no invented numbers; original
writing; house style section 3 with its exact section order, including the Supplementary notes paragraph; MDX/KaTeX gotchas
section 5). Imitate the style of /home/user/cv-blog/src/content/tutorials/en/deep-learning/dl-03-object-detection.mdx and
dl-04-image-restoration.mdx (read the relevant one fully; do not duplicate their generic material — link to it, after confirming
each anchor with github-slugger: `import GithubSlugger from '/home/user/cv-blog/node_modules/github-slugger/index.js'`).
Node: `export PATH=$HOME/.nvm/versions/node/v22.23.3/bin:$PATH`. Python: /home/user/cv-blog/.venv/bin/python (numpy, scipy,
scikit-image, scikit-learn, opencv-headless, matplotlib, PyWavelets; torch is NOT installed, so every snippet is NumPy/SciPy/
scikit-image/scikit-learn/OpenCV, CPU, runs in seconds, seeded, no downloads).

## The owner's principles (highest priority)
1. A reader must understand the chapter QUICKLY. Fast understanding first; details are secondary.
2. Every section opens with a plain-language statement (1-3 sentences, an everyday analogy where it helps), then the precise
   version with every symbol defined right after each equation, then a short tested snippet and/or figure where they help.
3. Secondary detail (long derivations, version-by-version minutiae, extra variants, long method catalogues, historical asides)
   goes into the chapter's final `## Supplementary notes` section (the site shows it on a separate page). Where such detail would
   have been, the main text keeps a SIMPLER explanation (2-5 sentences: what it is, why it matters, the one thing to remember)
   plus one italic pointer line: `*Optional: [Heading](#slug) (in the Supplementary notes). One sentence on what it adds.*`
   Nothing in the main text, code, figures or exercises may depend on a note.
4. Main-text English prose about 5,000-7,000 words; notes shorter than the main text.

## Files you create (only these; edit nothing else)
- the chapter MDX under /home/user/cv-blog/src/content/tutorials/en/deep-learning/<slug>.mdx
- /home/user/cv-blog/scripts/figures/<slug with underscores>.py (one script may write several PNGs)
- PNGs only in /home/user/cv-blog/src/assets/deep-learning/<slug>/
Figures: matplotlib Agg, white background, font >= 10, dpi 150, width <= 1600 px, each PNG < 1 MB, English text, data only
from skimage.data, scikit-learn toy data or synthetic data; never reproduce a paper's figure. Use
`<Figure src={img} alt="..." caption="Figure n — ..." />` with imports like the other chapters (ImageCompare for one
before/after pair of same-size images if useful).

## Required sections, in order
What you'll learn (4-6 bullets) · The big picture (ELI5 `<Callout type="tip" title="Explain like I'm five">` + why it matters) ·
topic sections · Modern view (surveys: what each covers, taxonomy, takeaways; 2023-2026 state of the art; open problems; no hype) ·
Key takeaways (5-8) · Exercises (4-6 original; hints in `<details>` with `<summary>` on its own line) · Supplementary notes ·
References (numbered `1. Authors, "Title," Venue, Year. [link](url)`, all cited, cited as [n]).
Never quote a benchmark number you did not read in the paper or official page itself; you may avoid numbers. Mark clearly
which numbers come from your own toy runs.

## Validation
`node scripts/check-mdx.mjs <your mdx>` passes; run the figure script; check images exist and are < 1 MB; extract and run every
code block and confirm the pasted outputs match; verify every internal link target exists (file and heading slug); a small script
checks citations vs entries. Do NOT run `npm run build` (others are editing). Final report: paths, word count (main and notes),
section list, every number in the text with its source, every reference with the URL you opened, internal links, uncertainties.

## UPDATE (owner): two kinds of optional sections
Keep concept details and paper study APART. Concept/sub-field/algorithm details → `## Supplementary notes` / `## 補充資料`
(page …/notes/). A reading guide to ONE paper or survey (problem, idea, what it showed, how to read its tables, limits, why it
matters here) → its own section `## Paper guides` / `## 論文導讀` (page …/papers/), placed after the Supplementary notes and before
References, opening with one sentence: EN "These guides are optional: each walks through one paper the chapter builds on — the
problem it poses, the idea, what it showed and what to check when you read it." / zh-tw「這些導讀是選讀內容：每篇帶你讀一篇本章
依據的論文——它要解決的問題、核心想法、結果，以及閱讀時該注意什麼。」 Pointer lines to a paper guide say "(in the Paper guides)" /
「（見「論文導讀」頁）」. If an existing Supplementary-notes item in your files is really a paper/survey guide (e.g. "U-Net: a
reading guide", "Deep Image Prior", "Guide to the Modern view surveys", "Survey papers"), move it into the Paper guides section
(both languages) and update its pointer line. Remove a section that becomes empty.

## UPDATE (owner): what a Supplementary note is like
A note does NOT need to cover everything like a chapter (no full house structure, no "what you'll learn"). It takes ONE
specific point — a sub-field, an algorithm, a derivation, a design choice — and goes into FINER detail than the main text:
precise definitions with every symbol defined, the steps, edge cases and pitfalls, a worked example or a tested snippet where
it helps. Open with one plain sentence saying what the note is about. Paper guides keep their reading-guide shape.

## UPDATE (owner): notes need no experiment
A Supplementary note does not need an experiment or code; a tested snippet is welcome but optional.
