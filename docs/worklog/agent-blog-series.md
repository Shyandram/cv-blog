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

## Revision 2026-11-06: more general, more informative (English only)

Owner's request: "make the blog part more general and more informative". The eight
English posts were restructured; the zh-tw twins are untouched and await translation.

Pattern applied to every post: frontmatter `question` (the one question the post
answers) and `checked` (date product facts were re-verified, both new optional
schema fields); a "Key points" list at the top; product-neutral main text
(mechanisms, trade-offs, evaluation, failure modes); a dated
"Current options, checked 6 November 2026" section holding named products,
versions, licenses and releases; a "Further reading" list of primary sources;
renumbered references. Slugs are unchanged, so no redirects were needed.

- Part 1 (agents-from-prompts-to-systems): new sections on the tool-calling round
  trip, the context window as a budget (tested snippet), and an agent failure-mode
  table (incl. indirect prompt injection); research directions moved to the dated
  section. New figure `tool-call-context.png`.
- Part 2 (agent-skills-and-harnesses): progressive-disclosure table from the Agent
  Skills spec, explicit loop exit conditions, idempotency-key design; harness code
  unchanged (output re-run, matches).
- Part 3 (openclaw-practical-agent-operations): retitled "operating an agent
  harness (OpenClaw as a worked example)"; organized by sessions, tools and
  permissions, skills, memory, scheduling, recovery and observability, each with a
  general principle and an "In OpenClaw" paragraph; operating checklist. OpenClaw
  docs re-pinned from `34bc643` to `27f2437` (main on 2026-11-06, package
  2026.11.1 = latest stable tag); differences between the two snapshots in the
  cited pages were diffed and are small (noted in the post).
- choosing-agent-tools: five kinds of tool, a dimension table, trial protocol;
  product tables moved to the dated section with licenses read from each
  repository's license file (OpenHands MIT, Cline Apache-2.0, LangGraph/Agents
  SDK/CrewAI/Browser Use MIT, n8n Sustainable Use License, OpenClaw MIT).
- agent-workflows: pattern-choice questions, stage gates, approval placement,
  budgets, failure-type recovery, a handoff template; checkpoint mechanics now
  link to part 2 instead of repeating them.
- models-benchmarks-and-people: retitled "Choosing models: reading benchmarks,
  licenses and evidence"; licensing layers (figure `licensing-layers.png`),
  anatomy of a benchmark, contamination, Wilson intervals, pass@k vs pass^k
  (figure `pass-at-k.png`, tested snippet), KV-cache memory estimate (tested
  snippet); model/benchmark tables and the people reading map moved to the dated
  section.
- choosing-image-generation-models: operations table, mechanism primer (latent
  diffusion, rectified flow, masked tokens, SDEdit), capability axes, what FID,
  CLIPScore, compositional benchmarks and arenas measure, provenance (C2PA).
- choosing-video-generation-models: inputs table, mechanism primer, capability
  axes, FVD/VBench/VBench-2.0, cost per accepted second (tested snippet).

Stale facts corrected on 2026-11-06: OpenAI image catalog now lists
`gpt-image-2.6-sunburst`/`-flare`; `gpt-image-1` shut down 23 Oct 2026; Google
lists `gemini-4-argon` (stable) and `gemini-nano-banana-2.1`; Veo 3.1 Lite
preview on Google Cloud; Sora/Videos API removal now cited to OpenAI's
deprecations page (help-center page returns 403 to automated checks); the
Midjourney "V8.2 default" claim was removed because the official page could not
be opened (403) for re-verification; xAI Grok Imagine Video 1.5 and Runway's
`wan3` added. Older representative model cards (Qwen3-8B, DeepSeek-R1,
Llama-3.3, Mistral Small 3.1) and the OpenAI "About" page were dropped.

Blog index: "Where to start" box (three entry points), series labels
("Agent fundamentals · Part n of 3"), each post's question instead of its
description when present, and "Sources checked" dates; post pages show the
question and the checked date. English topic descriptions updated; zh-tw topic
descriptions left for the translator.

New figure script: `scripts/figures/blog_general_guides.py` (writes only to
`src/assets/blog/general-guides/`; three PNGs, 1500 px wide, under 200 KB).

Validation (2026-11-06): check-mdx on the 8 files; all 5 Python blocks re-run and
match their printed output; `astro check` 0 errors / 0 warnings (3 existing
hints); production build 1,268 pages, Pagefind 1,267; built-site link audit
2,233 pages, 150,207 internal links, 0 broken; content map regenerated and
`--check` passes; unit tests 336 pass; content-stats, playground and paper-guide
checks unchanged. External links: 205 URLs checked; non-200 responses are
bot-blocking 403s (GitHub web pages, openai.com, arena.ai, Midjourney, Adobe, DOI
resolvers). GitHub targets were confirmed through raw files or a rendering fetch,
DOIs through Crossref. openai.com/index/harness-engineering, the Arena
text-to-image leaderboard, the Midjourney version page and the Adobe Firefly page
could not be opened in this session; their (unchanged, non-numeric) citations rest
on the 2026-11-05 verification.
Playwright at 1440 and 375 px: blog index (collapsed and expanded), part 1,
the models post, part 3 and the zh-tw index; no horizontal overflow, no JS
errors.
