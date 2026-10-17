# LaTeX book export

This exporter reads the published English and Traditional Chinese tutorials and blog posts directly from the content source. It does not execute MDX. The website's text is preserved; interactive widgets become links, image comparisons become static image pairs, and exercise hints are expanded.

The books use an 8 × 10 inch page, a designed title page, contents, eight curriculum parts, a final blog part, running headers, native mathematical typesetting, embedded fonts, and hyperlinks. Reference numbering remains local to each tutorial. Generated files are written to the ignored `book-output/` directory and are not deployed with the website.

## Requirements

- Node.js 22 and this repository's installed dependencies.
- [Pandoc](https://pandoc.org/installing.html), tested with 3.12.
- [Tectonic](https://tectonic-typesetting.github.io/book/latest/installation/), tested with 0.17.0. Its first build downloads LaTeX packages and the TeX Gyre fonts.
- Noto Serif CJK TC and Noto Sans CJK TC, each in Regular and Bold static OTF files, from the [official Noto CJK repository](https://github.com/notofonts/noto-cjk). Include their licenses in the same font directory.
- DejaVu Sans Mono (`DejaVuSansMono.ttf`) and its license in that font directory.

The required font filenames are `NotoSerifCJKtc-Regular.otf`, `NotoSerifCJKtc-Bold.otf`, `NotoSansCJKtc-Regular.otf`, `NotoSansCJKtc-Bold.otf`, and `DejaVuSansMono.ttf`.

## Generate and compile

```bash
export PANDOC=/path/to/pandoc
export BOOK_FONTS=/path/to/book-fonts
node scripts/book/export.mjs
cd book-output
/path/to/tectonic --untrusted --keep-logs --keep-intermediates pixels-perception-en.tex
/path/to/tectonic --untrusted --keep-logs --keep-intermediates pixels-perception-zh-tw.tex
```

To export just one language, pass `en` or `zh-tw` to the Node script. `BOOK_OUTPUT` overrides the output directory; `BOOK_DATE` overrides the snapshot date. Generated TeX files use paths relative to their output directory, so compile there. Tectonic runs the passes needed to resolve contents and hyperlinks.

`manifest.json` records the source commit, date, chapter list, source files, figure counts, code blocks, and display equations for each edition. Unknown MDX constructs or unresolved images fail the export instead of being silently omitted. Local links to included chapters stay inside the PDF; links to specific web sections and interactive features remain online.

The source archive should include both `.tex` files, `manifest.json`, and the complete `assets/` directory. Keep font licenses with the fonts. Build logs and intermediate TeX files do not need to be distributed.

## Validate the PDFs

Install PyMuPDF and Pillow in a Python environment, then run:

```bash
python3 scripts/book/check_pdf.py
```

This checks chapter coverage, code/equation/figure counts against the manifest, embedded fonts, internal link destinations, unresolved references, missing glyphs, and text outside page boundaries. It writes `validation.json` and preview contact sheets under `book-output/previews/`. Review those previews along with the LaTeX logs before distributing an updated edition.

## Publish the PDFs on the site

The site has a download page at `/en/download/` and `/zh-tw/download/` (also linked from the footer and the Learn page). It lists the PDFs in `public/books/` using the sizes and counts in `src/data/books.json`. Cloudflare builds the site from `main` and cannot compile LaTeX, so the PDFs are built here and committed.

```bash
export BOOK_FONTS=/path/to/book-fonts TECTONIC=/path/to/tectonic     # see Requirements above
npm run book                                                          # both languages, or: npm run book -- en
```

`npm run book` runs the export, shrinks the figures in the output folder (`optimize_figures.py`, needs Pillow: plots stay palette PNGs, photographs become JPEG, width capped at 800 px; the originals in `src/assets` are untouched), rewrites the LaTeX sources for the renamed files, compiles with Tectonic, and copies `pixels-perception-<lang>.pdf` to `public/books/`. It refuses to publish a PDF over 25 MiB, the Cloudflare Pages limit for a single file. Commit `public/books/*.pdf` and `src/data/books.json` together, after reviewing the PDFs with `check_pdf.py`. `--skip-build` publishes PDFs that are already compiled in `BOOK_OUTPUT`.

Pandoc 3.8 renamed `--no-highlight` to `--syntax-highlighting=none`; the exporter detects the installed version and uses the right flag.
