# Pixels → Perception

A bilingual (English / 繁體中文) tutorial site covering image processing, pattern recognition, machine learning, deep learning for vision, NLP, optimization, and information theory.
Built with [Astro 7](https://astro.build) + MDX, KaTeX math, Pagefind search, and deployed to Cloudflare Pages.

**Live site: <https://blog.shyandram.dev/>** ([English](https://blog.shyandram.dev/en/) · [繁體中文](https://blog.shyandram.dev/zh-tw/))

- Survey of website systems: [`docs/01-survey.md`](docs/01-survey.md)
- Site design: [`docs/02-design.md`](docs/02-design.md)
- Visual mockup: [`design/mockup/index.html`](design/mockup/index.html)

## Curriculum

The curriculum has eight tracks in English and Traditional Chinese. Start with
[Foundations: getting started and textbooks](src/content/tutorials/en/foundations/getting-started.mdx).

| Track | Coverage |
| --- | --- |
| Foundations (start) | Entry paths, textbook selection, and images as arrays |
| Image Processing (IP 01–13, entry) | Thirteen tutorials following the chapter order of Gonzalez & Woods, from intensity transforms to pattern classification |
| Natural Language Processing (NLP 01–12, entry) | Twelve tutorials following the course syllabus: introduction, linguistic essentials, neural networks for NLP, lexical semantics and word embeddings, RNNs, sequence labeling (POS tagging, information extraction), attention and Transformers, seq2seq (machine translation, chatbots), language modeling and generation, BERT, classification with BERT (sentiment, NLI), and syntactic and semantic tasks |
| Pattern Recognition (PR 01–10, entry) | A complete course based on Theodoridis/Koutroumbas, *Pattern Recognition* (4th ed.), and Bishop, *Pattern Recognition and Machine Learning* |
| Machine Learning (ML 01–10, core) | ML basics; linear algebra; probability/information; numerical computation; feedforward, convolutional, recurrent and recursive networks; selective state-space models and visual Mamba; transformers; representation learning; factor models and autoencoders; VAEs/GANs/flows; diffusion/flow matching, JiT and representation autoencoders; pretrained image/video generators; autoregression; foundation models; evaluation/deployment |
| Deep Learning (DLCV 01–07, core) | Classification, detection, restoration, low-light enhancement, tracking, and world models |
| Optimization (OPT 01–08, theory) | Eight chapters based on Boyd and Vandenberghe: convex sets/functions, problem formulation, duality, estimation, gradient/Newton methods, constrained solvers and applications |
| Information Theory (IT 01–08, theory) | Eight chapters based on Cover and Thomas: entropy, typicality, source coding, channel capacity, Gaussian channels, rate distortion, statistical decisions and networks |

ML 03 incorporates the former **Convolution & Image Filtering** tutorial, including its
interactive kernel and image comparison demos. The old English and Chinese URLs
redirect to the merged chapter; its existing filtering notebook remains available.

### Books, surveys and current methods

The ML/DL expansion uses Goodfellow/Bengio/Courville's *Deep Learning*, Prince's
*Understanding Deep Learning*, Bishop/Bishop's *Deep Learning: Foundations and
Concepts*, *Mathematics for Machine Learning*, and Murphy's probabilistic ML texts
as complementary readings. Each chapter links its books, surveys and original papers.
The Foundations chapter explains which book to choose for a particular starting point.

The NLP course follows Masato Hagiwara's [*Real-World Natural Language Processing*](https://www.manning.com/books/real-world-natural-language-processing) (2021) and Susan McRoy's [*Principles of Natural Language Processing*](https://wisconsin.pressbooks.pub/naturallanguage/) (2021). Optimization follows Stephen Boyd and Lieven Vandenberghe's [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/) (2004). Information Theory follows Thomas M. Cover and Joy A. Thomas's [*Elements of Information Theory*, second edition](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X) (2006). Chapters identify specific book readings and connect them to verified modern surveys and original papers.

| Reading purpose | Chapter and survey/review |
| --- | --- |
| Convolutional architecture | ML 03 · Alzubaidi et al., CNN/deep-learning review (2021) |
| Attention and visual architecture | ML 04 · Khan et al., *Transformers in Vision: A Survey* (2022); visual Mamba review |
| Self-supervised objectives | ML 05 · Liu et al., *Self-supervised Learning: Generative or Contrastive* |
| Explicit density modeling | ML 06 · Papamakarios et al., *Normalizing Flows for Probabilistic Modeling and Inference* (2021) |
| Noise/velocity-based generation | ML 07 · Yang et al., diffusion survey (2023); Lipman et al., flow matching guide (2024) |
| Sequential visual generation | ML 08 · Xiong et al., autoregressive vision survey (2024, revised 2025) |
| Multimodal adaptation | ML 09 · Caffagni et al., multimodal LLM survey (2024) |
| Efficient deployment | ML 10 · Gou et al., distillation survey (2021); Cheng et al., compression survey (2017) |

This is a curated reading map, not an exhaustive review of every publication.
Research coverage is dated **2026-10-04** and includes, where relevant, DINOv3,
SAM 3, Qwen3-VL, V-JEPA 2, VGGT, VAR, MAR, MeanFlow, JiT and representation autoencoders. Text separates a paper's
reported capabilities from general guarantees. Small CPU experiments and synthetic
figures teach mechanisms; they do not claim to reproduce those large-model benchmarks.

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
title: ML 03 · Convolution, Image Filtering and CNNs
description: One sentence shown on cards and in search results.
track: ml                 # foundations | image-processing | nlp | pattern-recognition | ml | deep-learning | optimization | information-theory
order: 30                 # position within the track
level: intermediate       # beginner | intermediate | advanced
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
