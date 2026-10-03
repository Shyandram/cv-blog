# Pixels → Perception

A bilingual (English / 繁體中文) computer vision tutorial site, from basic image processing to modern deep learning.
Built with [Astro 7](https://astro.build) + MDX, KaTeX math, Pagefind search, and deployed to Cloudflare Pages.

**Live site: <https://blog.shyandram.dev/>** ([English](https://blog.shyandram.dev/en/) · [繁體中文](https://blog.shyandram.dev/zh-tw/))

- Survey of website systems: [`docs/01-survey.md`](docs/01-survey.md)
- Site design: [`docs/02-design.md`](docs/02-design.md)
- Visual mockup: [`design/mockup/index.html`](design/mockup/index.html)

## Deploying (Cloudflare Pages)

The site is hosted on Cloudflare Pages, which builds it from this GitHub repo on every push.

One-time setup in the Cloudflare dashboard (**Workers & Pages → Create → Pages → Connect to Git**):

| Setting | Value |
|---|---|
| Repository | `Shyandram/cv-blog` (the repo can stay private) |
| Production branch | `main` |
| Framework preset | Astro |
| Build command | `npm run build` |
| Build output directory | `dist` |
| Environment variable | `SITE` = `https://blog.shyandram.dev` |

Node 22 is picked up from `.nvmrc`. Pushes to `main` go to production; other branches and pull requests
get their own preview URL. `public/_headers` sets long-term caching for built assets.

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

Blog posts go in `src/content/posts/<lang>/<slug>.md(x)`. A post that exists in only one language
is still listed on the other language's blog, labelled with its language. Add `original: <url>` to
show an "Originally published on …" link.

### Importing from Medium

```bash
curl -L -o feed.xml https://medium.com/feed/@shyangen104   # the 10 most recent posts, full text
python scripts/import_medium.py feed.xml my-slug=<id at the end of the Medium URL>
```

The script converts the post to Markdown, downloads its images next to it, and fills in the frontmatter.
Review the description and any links to other Medium posts afterwards.

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
