# 01 — Website System Survey

**Goal:** pick a platform for a computer-vision tutorial blog that covers
basic image processing → classical CV → machine learning → deep learning.
**Hosting:** a GitHub repo, deployed with GitHub Pages + GitHub Actions.
**Surveyed:** October 2026. (Update: Astro 7 was already current when the site was scaffolded; the Astro row applies to it too.)

## 1. What a CV tutorial site needs

General blog features (posts, tags, RSS) are the easy part. These are the
requirements that actually separate the candidates:

| # | Requirement | Why it matters for CV tutorials |
|---|---|---|
| R1 | **Authoring model**: Markdown/MDX vs. Jupyter notebooks | Most CV/ML code is written and checked in notebooks. If outputs (plots, filtered images) can't come from the notebook, every figure has to be exported by hand. |
| R2 | **Math** (KaTeX/MathJax) | Convolution, Fourier transforms, homography, backprop, and loss functions all need display math. |
| R3 | **Code blocks** with highlighting, copy button, line highlights, and file tabs | Readers copy the code to run it. |
| R4 | **Figures**: before/after image compare, captions, responsive images | "Input vs. output" is the core visual of image processing. |
| R5 | **In-browser interactive demos** (Canvas/WebGL, `onnxruntime-web`, TF.js) | A live kernel or threshold slider teaches much more than a static image. |
| R6 | **Learning structure**: tracks, levels, prerequisites, series navigation | The site is a curriculum, not a feed of unrelated posts. |
| R7 | **Search** (client-side, works on static hosting) | Readers look up specific topics like "Canny" or "IoU". |
| R8 | **GitHub Pages deploy** through Actions | Hosting decision is already made. |
| R9 | **Toolchain cost and maturity** | Long-lived content needs a stable engine. |

## 2. Candidates

| System | Language | Authoring | Math | Interactive demos (R5) | Notebook outputs (R1) | Blog + curriculum (R6) | Search | Maturity / risk |
|---|---|---|---|---|---|---|---|---|
| **Astro 6 + MDX** (custom theme) | Node | Markdown + MDX components | remark-math + rehype-katex | ★★★ Islands: any JS/React/Svelte component inside a post, shipped only where used | ★ Link to Colab/GitHub, or convert with `nbconvert` | ★★★ Content collections with typed frontmatter make tracks and prereqs straightforward | Pagefind | Very active; v6 released March 2026 |
| **Astro Starlight** | Node | MD/MDX | Same plugins | ★★★ Same islands | ★ | ★★ Docs-shaped sidebar; a blog needs a community plugin | Pagefind built in | Mature docs theme |
| **Quarto** | Single binary (+ Python/Jupyter) | `.qmd` / `.ipynb` | Built-in | ★★ Observable JS cells; arbitrary JS is possible but clunky | ★★★ Executes notebooks and embeds outputs; `freeze` caches results | ★★ Built-in blog listing; tracks via listings and sidebars | Built in | Mature (Posit) |
| **Jupyter Book 2 / MyST-MD** | Python wrapper + Node engine | MyST MD / `.ipynb` | Built-in | ★★ "Anywidgets" and Thebe/JupyterLite live kernels | ★★★ | ★★ Book/TOC-shaped, weak blog features | Built in | Rewritten on MyST; needs Node anyway |
| **Docusaurus 3** | Node (React) | MD/MDX | remark-math | ★★★ React components | ★ | ★★ Docs + blog plugins included | Algolia (external) or local plugin | Mature (Meta) |
| **Hugo** | Single Go binary | Markdown + shortcodes | KaTeX via partials | ★ Raw JS through shortcodes | ★ | ★★ Taxonomies and sections | Pagefind / Fuse | Very mature and fast; Go templates are awkward |
| **MkDocs Material → Zensical** | Python | Markdown | Built-in | ★ | ★★ via `mkdocs-jupyter` | ★★ | Built in | **Risky right now:** Material is in maintenance mode (fixes until Nov 2026); Zensical is v0.0.x and its blog plugin isn't officially available yet |
| **Jekyll / al-folio** | Ruby | Markdown | MathJax | ★ | ★ (Jupyter include in al-folio) | ★ Academic-portfolio shape | Basic | Mature but old; the only system GitHub Pages builds without Actions |
| **Next.js / Nextra** | Node | MDX | KaTeX | ★★★ | ★ | ★★ | Built in (Nextra) | An app framework, so more upkeep than a content site needs |
| **Hosted CMS** (Ghost, Medium, Substack) | — | WYSIWYG | Poor or none | ✗ | ✗ | ✗ | Hosted | Doesn't fit a GitHub repo, and you don't own the content pipeline |

★★★ strong · ★★ workable · ★ possible with effort · ✗ not practical

## 3. Shortlist

Only two candidates fit the requirements well. The deciding factor is
**how the content gets written**, not how popular the framework is.

### Option A — Astro 6 + MDX, custom theme (recommended)
- **Strengths:** full design control; interactive CV demos become ordinary components
  (`<KernelPlayground/>`, `<ImageCompare/>`, `<OnnxDemo model="yolo.onnx"/>`) that
  ship JS only on pages that use them; typed frontmatter keeps the curriculum consistent;
  fast static output; Pagefind search; official GitHub Pages Action.
- **Weaknesses:** notebooks aren't first-class. The workflow is: write in a notebook,
  then write the prose in MDX and link a Colab/GitHub notebook. Figures can be
  generated by a small `scripts/` step if needed. Requires Node 22+ (`nvm` install on WSL).

### Option B — Quarto
- **Strengths:** notebook-native. Write `.ipynb`/`.qmd`, and Quarto runs the code and embeds
  outputs, so the figures never drift out of sync with the code. No Node needed. Built-in blog, math,
  search, and cross-references.
- **Weaknesses:** a less distinctive look, and custom interactive demos are harder to build. Running
  DL notebooks in CI is slow or impossible without a GPU, so you'd rely on `freeze: auto`
  and commit the rendered outputs.

### Hybrid (possible later)
Keep `notebooks/` in the repo as the runnable companion to every tutorial
(with an "Open in Colab" badge), whichever engine renders the site. This gives
most of Quarto's reproducibility while keeping Astro's interactivity.

## 4. Environment notes
- This machine has **no Node.js**. Option A needs `nvm install 22`. Option B needs the Quarto binary.
- Local Python is **3.9.7 (end-of-life)**. Create a fresh env for the tutorial code
  (e.g. `conda create -n cvblog python=3.12 opencv pytorch …`) whichever site engine you choose.

## Sources
- [Astro 6 release (Netlify write-up, Mar 2026)](https://blogs.apievangelist.com/blogs/netlify-2026-03-10-astro-6-just-works-on-netlify/)
- [Astro 2025 year in review](https://astro.build/blog/year-in-review-2025)
- [Material for MkDocs maintenance mode](https://pydevtools.com/handbook/reference/mkdocs-material/) · [discussion](https://discuss.privacyguides.net/t/mkdocs-material-is-now-in-maintenance-mode/32736) · [GitHub issue #8523](https://github.com/squidfunk/mkdocs-material/issues/8523)
- [Zensical 0.0.31 package](https://simple-repository.app.cern.ch/project/zensical) · [Trying out Zensical (Jan 2026)](https://blog.lhns.de/blog/2026-01-10-trying-out-zensical/) · [Zensical vs Docusaurus](https://docsio.co/blog/zensical-vs-docusaurus)
- [Jupyter Book 2 alpha announcement](https://jupyterbook.org/blog/posts/2024-11-15-jupyter-book-2-alpha) · [Jupyter Book toolchain](https://next.jupyterbook.org/about/toolchain) · [2i2c post](https://2i2c.org/blog/jupyter-book-2/)
- [Publishing Jupyter notebooks with Quarto](https://aicouncil.com/talks/publishing-jupyter-notebooks-with-quarto) · [Notebook tools 2026 comparison](https://www.youngju.dev/blog/culture/2026-05-16-notebooks-2026-jupyter-marimo-quarto-observable-deepnote-hex-positron-deep-dive)
