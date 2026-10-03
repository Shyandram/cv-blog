# Pixels → Perception

A bilingual (English / 繁體中文) computer vision tutorial site, from basic image processing to modern deep learning.
Built with [Astro 7](https://astro.build) + MDX, KaTeX math, Pagefind search, and deployed to GitHub Pages.

- Survey of website systems: [`docs/01-survey.md`](docs/01-survey.md)
- Site design: [`docs/02-design.md`](docs/02-design.md)
- Visual mockup: [`design/mockup/index.html`](design/mockup/index.html)

## Deploying (GitHub Pages)

1. Create a GitHub repo and push this folder to its `main` branch.
2. The workflow enables GitHub Pages (source: GitHub Actions) on its first run. If that run fails with a
   Pages permission error, open **Settings → Pages**, set **Source** to **GitHub Actions**, and re-run it.

Every push to `main` runs `.github/workflows/deploy.yml`. It builds the site and publishes it to
`https://<user>.github.io/<repo>/`; the base path is detected automatically. Pull requests are built but not deployed.

## Writing a tutorial

Create the same slug under both languages (the English one is required; a missing translation falls back to English with a notice):

```
src/content/tutorials/en/<track>/<slug>.mdx
src/content/tutorials/zh-tw/<track>/<slug>.mdx
```

Frontmatter (validated at build time, see `src/content.config.ts`):

```yaml
title: Convolution & Image Filtering
description: One sentence shown on cards and in search results.
track: image-processing   # foundations | image-processing | classical-cv | ml | deep-learning
order: 3                  # position within the track
level: beginner           # beginner | intermediate | advanced
prereqs: [foundations/images-as-arrays]
tags: [filtering, opencv]
notebook: notebooks/image-processing/convolution.ipynb   # optional, adds "Open in Colab"
published: 2026-10-04
draft: false
```

Math uses `$...$` and `$$...$$`. Components available in MDX (import them at the top of the file):

| Component | Use |
|---|---|
| `<KernelPlayground preset="sobel-x" lang="en" />` | Live editable 3×3 kernel on a test image |
| `<ImageCompare before={a} after={b} beforeLabel afterLabel />` | Before/after slider (no images = built-in blur demo) |
| `<Callout type="note\|tip\|warn\|math" title="…">` | Aside box |
| `<Figure src={img} alt caption />` | Optimized, captioned image |

Blog posts go in `src/content/posts/<lang>/<slug>.md(x)`.

## Local preview (optional)

Only needed if you want to look at changes before pushing. Requires Node 22+.

```bash
npm install
npm run dev        # live-reload at http://localhost:4321/en/
npm run build && npm run preview   # production build, including search
npm run check      # type and frontmatter checks
```

## Layout

```
src/content/      tutorials/ and posts/, per language
src/components/   MDX components and interactive demos
src/pages/[lang]/ home, learn, blog, playground, search, about, rss.xml
src/i18n/ui.ts    UI strings for both languages
src/lib/          site helpers, image-processing helpers (cv.ts), track list
notebooks/        runnable companions for tutorials (Colab)
```
