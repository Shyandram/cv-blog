# Agent and generative-model blog series

Goal: add beginner-friendly bilingual blog articles about agent development,
use, evolution, skills, harness engineering, OpenClaw, tool selection and
workflows; explain model evaluation, representative organizations/contributors,
and image/video model selection. Add blog topic lists and make learning books
closed by default, opening only the reader's chosen book.

Baseline: `c347842`. Original author-written posts are preserved. PR #5 was
already merged externally as `aa890ec` when its current state was checked;
this batch will be reviewed separately. No merge or deployment is performed
for this batch.

## Content

Eight EN/zh-tw pairs, with matching sections and references:

- `agents-from-prompts-to-systems`: evolution and system design.
- `agent-skills-and-harnesses`: reusable procedures, recovery, and evidence.
- `openclaw-practical-agent-operations`: sessions, tools, skills and operation.
- `choosing-agent-tools`: editors, terminals, frameworks and automation.
- `agent-workflows`: task contracts, execution, verification and handoff.
- `models-benchmarks-and-people`: deployment choices and evaluation literacy.
- `choosing-image-generation-models`: image tools, weights and usable assets.
- `choosing-video-generation-models`: motion, control and usable footage.

Three reproducible original teaching diagrams accompany the agent fundamentals
series. The offline Python harness example checks replay, permission denial,
completion evidence, stale evidence, malformed input, budgets and checkpoints.
Its output is reproduced in both language versions. It does not call a model or
claim production-grade transaction guarantees.

OpenClaw documentation is pinned to official source commit
`34bc6434914728a8d13e8e2dcf6dd98bce6d1d88`. Development snapshot versus installed
release is explicit. Automation/heartbeat migration and the distinct skill
refresh mechanisms were checked. No authenticated OpenClaw installation or paid
image/video generation was performed.

Product sources were checked on 2026-11-05. Articles distinguish code licenses,
weight licenses and hosted-service terms. Sora's official sunset is reflected;
Luma uses the current Agents model documentation. Downloadable releases are
representative examples, not claims about the latest release or a benchmark
ranking. Toolformer and Reflexion link to verified final proceedings.

## Interface

The blog groups posts into five native, collapsible topic lists. Series order is
explicit; posts link back to the relevant topic. Existing posts use the default
site category without changing their original text.

Learning index and complete course tree start with all eight books closed.
Nested chapter expansion also starts closed; old expansion preferences no longer
reopen branches. Reading progress remains independent and preserved. An explicit
fragment link opens its requested group. The complete tree retains expand-all
and collapse-all controls. Keyboard focus and mobile touch targets are retained.

## Validation

- All 16 new MDX files pass syntax/KaTeX checks. All eight bilingual pairs have
  matching heading structure, external-reference URLs and code blocks.
- Offline Python harness output matches both articles; its assertions pass.
  Three figures are 1200 × 900 and below 100 KB each, visually inspected.
- Astro check: 0 errors, 0 warnings, 3 pre-existing unused-variable/import hints.
- Content map regeneration and `--check` pass. All 30 existing test files,
  content-statistics checks and playground coverage checks pass.
- Paper coverage remains 188 guides across 66 chapters, with 0 missing figures;
  existing original-source and legacy-ID records are unchanged.
- Final production build: 1,260 pages; Pagefind indexes 1,259. The existing
  Shiki/CSP advisory remains unchanged.
- Final built-site link audit: 2,225 HTML pages, 149,491 internal links,
  0 broken targets or fragments.
- Blog browser validation: 36 EN/zh-tw layouts at 375 and 1440 px, plus native
  topic toggles without JavaScript in both languages. Category order, fragment
  reveal, translation links, images, code defaults and keyboard controls pass.
- Final learning browser validation: 8 layouts across both languages, both
  widths and both interfaces, plus 4 no-JavaScript native book-toggle cases.
  Default folding, keyboard interaction, old expansion-state isolation,
  preserved reading progress, fragment navigation and all-open/all-closed pass.
- No page-level horizontal overflow or JavaScript errors. Final mobile/desktop
  screenshots were inspected; a mobile long-title chevron-wrap issue was fixed
  and the production build and learning suite rerun successfully.

Structured evidence is in `docs/worklog/agent-blog-browser-validation.json`.
Browser screenshot artifacts remain in the scratch directories below.

## Source and artifact notes

Primary references are cited in each article. Source excerpts and scratch
artifacts are in `/workspace/scratch/cvblog-work/agent-blog-2026-11-06/`;
browser evidence is in `/workspace/scratch/cvblog-work/agent-blog-validation/`
and `/workspace/scratch/cvblog-work/learning-disclosure/`.

New figures are labeled original teaching syntheses, not paper reproductions.
Existing paper-guide registries and legacy heading-ID baselines are unchanged.
Downloadable course PDFs are unchanged; the new material is in the online blog.
