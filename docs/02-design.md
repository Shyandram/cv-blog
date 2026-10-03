# 02 — Site Design (draft)

Working name: **Pixels → Perception** (placeholder, easy to change).
This design is written against Option A (Astro + MDX; built on Astro 7, released after the survey). Section 8 lists
what changes under Quarto.

## 1. Audience and promise
- **Readers:** students and engineers who know some Python and want to understand CV
  from pixels up to modern deep models.
- **Promise:** every tutorial has the intuition, the math, code you can run, and something
  you can play with.

## 2. Information architecture

```
Home
├── Learn (curriculum)                     /learn/
│   ├── 0  Foundations                     /learn/#foundations
│   │      getting started, textbook selection, images as arrays
│   ├── 1  Image Processing                /learn/#image-processing
│   │      histograms, point ops, convolution & filtering, edges, morphology,
│   │      frequency domain (FFT), geometric transforms, denoising
│   ├── 2  Pattern Recognition             /learn/#pattern-recognition
│   │      Bayes decisions, densities, linear/kernel methods, features, context,
│   │      clustering, inference, ensembles and evaluation; Theodoridis/Koutroumbas + Bishop
│   ├── 3  ML/DL                           /learn/#ml
│   │      mathematics, neural networks, merged filtering/CNNs, recurrence, transformers,
│   │      representations, factor models, VAEs/GANs/flows, diffusion, AR, foundation models
│   ├── 4  DL for Vision                   /learn/#deep-learning
│   │      classification, detection, restoration, low light, tracking and world models
│   └── 5  NLP                             /learn/#nlp
│          tokenization, language modeling, transformers and a further reading roadmap
├── Blog (news, paper notes, opinion)      /blog/
├── Playground (all interactive demos)     /playground/
├── Tags                                   /tags/<tag>/
├── About                                  /about/
└── RSS                                    /rss.xml
```

**Two content types:**
- **Tutorial**: part of a track, ordered, with a level, prerequisites, and prev/next navigation.
- **Post**: dated blog entry (paper reviews, "what's new in…"), outside the curriculum.

## 3. Tutorial page template

1. **Header**: track badge, level (Beginner / Intermediate / Advanced), reading time,
   prerequisites as links, and "Open in Colab" / "View notebook" buttons
2. **What you'll learn**: 3–5 bullets
3. **Intuition**: plain language plus one key figure
4. **The math**: KaTeX, with symbols defined right after each equation
5. **Code**: NumPy from scratch first, then the OpenCV/PyTorch one-liner
6. **Try it**: an interactive demo island (optional)
7. **Results and pitfalls**: before/after compare, common bugs
8. **Exercises**: collapsible hints and solutions
9. **References**: papers, docs
10. **Footer**: prev/next in track, related tutorials, edit-on-GitHub link, comments (Giscus)

Desktop layout is three columns: track nav (left), article (center, about 72ch), and an "On this page"
table of contents (right). On mobile, the side columns collapse into drawers.

## 4. Frontmatter schema (Astro content collection, Zod-validated)

```yaml
# src/content/tutorials/en/ml/ml-03-convolutional-networks.mdx
title: "ML 03 · Convolution, Image Filtering and CNNs"
description: "From blur and Sobel kernels to learned filters and a small CNN classifier."
track: ml                        # enum of the 6 tracks
order: 30                         # position in track
level: intermediate               # beginner | intermediate | advanced
prereqs: [foundations/images-as-arrays]
tags: [filtering, opencv, numpy]
notebook: notebooks/image-processing/convolution.ipynb   # optional; renders Colab badge
cover: ./cover.png
published: 2026-10-10
updated: 2026-10-12
draft: false
```

## 5. Component library (MDX)

Built so far: `ImageCompare`, `KernelPlayground`, `Figure`, `Callout`, prerequisites (in the page template). The rest are planned, as are the `/tags/` pages.

| Component | Purpose |
|---|---|
| `<ImageCompare before after />` | Drag slider between input and output |
| `<KernelPlayground />` | Edit a 3×3 kernel and see the filtered image live (Canvas) |
| `<Figure src caption />` | Numbered, captioned, responsive figure |
| `<Callout type="note|tip|warn|math" />` | Asides |
| `<CodeTabs>` | NumPy / OpenCV / PyTorch versions of the same snippet |
| `<Equation label>` | Numbered display math with reference links |
| `<Prereqs />` | Auto-rendered from frontmatter |
| `<OnnxDemo model />` | Run a small model in the browser (`onnxruntime-web`), loaded lazily |
| `<PaperCard arxiv />` | Paper reference card |

## 6. Visual design
- **Mood:** clean technical notebook. Lots of white space, a calm neutral palette, and color
  reserved for code, math, and track identity.
- **Typography:** Bricolage Grotesque (headings), Source Serif 4 (article prose), system UI font (interface), JetBrains Mono (code), KaTeX fonts (math).
- **Track colors** (used for badges and progress bars only):
  Foundations slate · Image Processing teal · Pattern Recognition indigo · ML/DL amber · DL for Vision rose · NLP green.
- **Themes:** light and dark via CSS tokens, following the OS setting with a manual toggle. Images get no
  inversion in dark mode, because pixel values matter in this subject.
- See `design/mockup/` for a clickable prototype of the home page and tutorial page.

## 7. Repo layout and deploy

```
cv-blog/
├── src/
│   ├── content/{tutorials,posts}/     # MDX content
│   ├── components/                    # MDX components + demo islands
│   ├── layouts/  pages/  styles/
├── notebooks/                         # runnable companions (Colab badge)
├── public/models/                     # small .onnx files for browser demos
├── docs/                              # this survey + design
├── public/_headers                    # Cloudflare Pages cache rules
└── astro.config.mjs                   # mdx, remark-math, rehype-katex, expressive-code, sitemap, pagefind
```

- **Hosting:** Cloudflare Pages, connected to the GitHub repo. Pushes to `main` deploy to production;
  pull requests get preview URLs. The build fails on invalid frontmatter.
- **Domain:** `<project>.pages.dev` first; a custom domain can be added in the Cloudflare dashboard.
- **Comments:** Giscus (GitHub Discussions), which needs no backend.
- **Analytics (optional):** a privacy-friendly option such as GoatCounter.

## 8. If Quarto is chosen instead
- Content becomes `.qmd`/`.ipynb` under `learn/<track>/`, the same frontmatter keys map to YAML
  front matter, and listings pages replace Astro collections.
- Interactive demos become Observable JS cells or embedded HTML widgets. Simpler,
  but less freedom.
- Deploy with `quarto publish gh-pages` or the `quarto-dev/quarto-actions` workflow, using
  `freeze: auto` so CI doesn't re-run DL notebooks.

## 9. Decisions
- **Engine:** Astro 7 + MDX (decided 2026-10-04). Astro 7 shipped after the survey and was current at
  scaffold time. It defaults to the Sätteri Markdown processor; the site opts into `unified()` from
  `@astrojs/markdown-remark` so remark-math and rehype-katex work.
- **Languages:** English + Traditional Chinese from day one.
  - Routes: `/en/...` and `/zh-tw/...`; `/` detects the browser language and offers a switcher.
  - Content: `src/content/tutorials/<lang>/<track>/<slug>.mdx`. A translation shares its `slug`
    so the language toggle can link the counterpart, and a missing translation shows a
    "not yet translated" notice that links to the English version.
  - UI strings: `src/i18n/{en,zh-tw}.json`. Pagefind indexes each language separately.
  - Typography: add Noto Sans TC / Noto Serif TC as the CJK fallback in each font stack.
- **Hosting:** GitHub repo → Cloudflare Pages (changed from GitHub Pages on 2026-10-04).

## 10. Open questions
1. Site name (placeholder: Pixels → Perception) and custom domain.
2. Comments: Giscus or none.
