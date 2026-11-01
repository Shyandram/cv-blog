# Brief: build interactive experiments (image labs / SVG experiments) for new chapters

You work in YOUR OWN git worktree (the Agent tool created it for you; `pwd` shows it). First run there:
`ln -s /home/user/cv-blog/node_modules node_modules; ln -s /home/user/cv-blog/.venv .venv` (both are git-ignored), and
`export PATH=$HOME/.nvm/versions/node/v22.23.3/bin:$PATH`. Never touch /home/user/cv-blog itself. Commit your finished work in your
worktree's branch (explicit paths only; message ending with the two lines
`Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` and
`Claude-Session: https://claude.ai/code/session_01LDLcNdAG6dTWpQHbdjJauG`); do NOT push, do NOT open PRs. The coordinator merges your
branch. No git stash/reset/checkout/restore/clean.

Read first, completely: /home/user/cv-blog/AGENTS.md (content rules, language rules, references), /home/user/cv-blog/docs/playground.md
(how experiments are built, the 9 principles, the checks), and one existing lab end to end as a template:
src/lib/playground/labs/segmentation-threshold.ts (image lab) and src/lib/playground/labs/tracking-association.ts (lab with plots and
pointer interaction), plus src/lib/playground/labs/types.ts, registry.ts, loaders.ts, assets.ts, sources.ts, src/lib/playground/
coverage.ts and scripts/check-labs.py. Use `docs/content-map/deep-learning.md` and `npm run rag -- "<topic>"` to find the chapter
sections your lab illustrates.

## What each lab must be (from docs/playground.md, strictly)
- ONE misconception or decision per lab, stated as a question in the title; the default state already shows the idea.
- Real or synthetic data with a known answer (skimage.data images via scripts/figures/playground_images.py-style asset scripts, or
  data generated in TypeScript with a seeded RNG); before/after plus one number next to the picture.
- At most three controls, each with a visible effect; a `predict` question whose answer you VERIFIED by running the code.
- Bilingual (en / zh-tw, Taiwanese terminology, no Simplified characters) title, description, challenge, note, control labels,
  predict text, plot labels; legible at 375 px.
- Pure computation in exported functions with Node unit tests (scripts/tests/lab-<id>.test.ts). If the lab shows a number that the
  chapter text also quotes, it must match the chapter's Python (or say clearly it is the lab's own toy).
- `related` lists the chapter slug(s); `sources` cites any named method with a verified reference (open the page).
- Images, if any: a NEW script scripts/figures/playground_<id>.py writing only src/assets/playground/<id>-*.png (≤ 256 px per side,
  small files), registered in labs/assets.ts with a credit in labs/sources.ts.

## Wiring (do it in your worktree so you can test; the coordinator re-applies it when merging)
- add the lab to labs/registry.ts and labs/loaders.ts; in coverage.ts move the chapter from `readingOnlyChapters` to `chapterPlan`
  (or add the lab to its existing chapterPlan entry; max 2 per chapter);
- add a `check_<id>` function and an EXPECT entry in scripts/check-labs.py asserting the default state shows the idea;
- embed `<Lab id="<id>" />` in BOTH language files of the chapter, after the figure or paragraph that introduces the idea, with
  `import Lab from '../../../../components/Lab.astro';` among the imports (check it is not already imported). Do not change other
  chapter text.

## Validate in your worktree before committing
`node scripts/check-mdx.mjs <changed mdx>`; `node scripts/check-playground.mjs`;
`node --experimental-strip-types --test scripts/tests/*.test.ts`; `npx astro check` (0 errors, 0 warnings); `npm run build`;
`.venv/bin/python scripts/check-labs.py` (real browser: loads, default informs, predict works, no overflow, en and zh-tw);
`node scripts/content-index.mjs` (commit the regenerated docs/content-map and docs/content-index.json too). Look at a screenshot of
each lab at 1440 and 375 px yourself (Playwright, Chromium at glob('/opt/pw-browsers/chromium-*/chrome-linux*/chrome'), --no-sandbox).

## Final report
Your worktree path and branch name, the commit hash(es), per lab: id, chapter, the question, controls, default-state numbers and the
predict answer with how you verified them, files added/changed, test results, screenshots taken, anything uncertain.
