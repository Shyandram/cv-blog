# Playground: how experiments are built

The Playground has two kinds of experiments. Both are optional per chapter (see `chapterPlan` in `src/lib/playground/coverage.ts`).

| Kind | Use it for | Code | Draws with |
| --- | --- | --- | --- |
| **SVG experiment** | Concepts: a curve, a table, a small diagram, a decision rule | `src/lib/playground/*.ts`, `compute(state, lang)` returns SVG, a summary and metrics | SVG string |
| **Image lab** | Anything about images or real data: run a real algorithm on a real (or synthetic) image | `src/lib/playground/labs/<id>.ts` | canvas, plus optional SVG plots |

## Principles (why a demo earns its place)

1. One misconception or decision per demo, stated as a question in the title.
2. Real data for vision: images from `skimage.data` or synthetic images with a known answer, never an 8 × 8 stand-in when the chapter is about images. Tabular data from scikit-learn toy sets.
3. Before and after, plus one number (PSNR, IoU, accuracy, fraction nonzero) next to the picture.
4. The default state already shows the idea. The learner turns it off, not on.
5. Predict, then reveal. Every image lab has a `predict` question whose answer was verified by running the code.
6. At most three controls, each with a visible effect.
7. Embed it in the chapter with `<Lab id="..." />` (it also appears on the Playground page).
8. Plot text is legible at 375 px; zh-tw labels are localized.
9. A demo is optional: if the chapter's own figure carries the idea, the chapter is reading-only.

## Adding an image lab

1. **Images.** Add what you need to `scripts/figures/playground_images.py` (or a new script that writes only `src/assets/playground/`), keep each image at 256 px or less per side, and register it in `src/lib/playground/labs/assets.ts`. Credit it: `labs/sources.ts` has the licence line shown under every lab.
2. **Kit.** Algorithms live in `src/lib/imagekit/` (FFT, DCT, Haar, Gaussian, median, Sobel, Otsu, local threshold, PSNR, IoU, seeded noise). Add a function there if you need one, and a golden test: `scripts/figures/playground_goldens.py` computes the reference with NumPy, SciPy or scikit-image into `scripts/tests/golden/imagekit.json`, and `scripts/tests/imagekit.test.ts` compares. Numbers shown in a lab must match the Python used in the chapter text.
3. **Lab module.** Copy `labs/segmentation-threshold.ts`. It exports `lab: LabDef` with:
   * `id` (reuse the id of the demo it replaces so `#id` links keep working), `track`, bilingual `title`, `description`, `challenge`, `note`, `related` chapter slugs;
   * `controls` (1 to 3 `range` or `select`), `predict`, `assets`, `panes`, `sources`;
   * `create({ lang, images })` returns `{ render(state, canvases) }` which draws to the canvases and returns `{ summary, metrics, plots }`. Keep pure computation in exported functions so Node tests can call them.
4. **Register.** Add the lab to `labs/registry.ts` (build time) and `labs/loaders.ts` (browser, lazy), and list it in `chapterPlan`. At most two experiments per chapter.
5. **Embed.** Put `<Lab id="..." />` in both language files of the chapter, after the figure that introduces the idea, and add `import Lab from '../../../../components/Lab.astro';`.
6. **Test.** Unit tests in `scripts/tests/`, a threshold for the default state in `scripts/check-labs.py` (`EXPECT`), then `npm run build` and `.venv/bin/python scripts/check-labs.py`.

## Checks

```bash
export PATH=$HOME/.nvm/versions/node/v22.22.0/bin:$PATH
node scripts/check-playground.mjs                        # coverage plan, cap, aliases, predict options
node --experimental-strip-types --test scripts/tests/*.test.ts
npx astro check && npm run build
.venv/bin/python scripts/check-labs.py                   # real browser: labs load, defaults inform, predict works, no overflow
```

Content rules from `AGENTS.md` apply: every named method needs a verified reference (open the link; confirm title, authors, year, venue via arXiv or Crossref), no invented numbers (quote only what the code produced), original text, and zh-tw with Taiwanese terminology and the same structure as English.
