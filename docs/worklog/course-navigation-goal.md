# Course navigation goal

Status: completed and validated for all 97 currently available tutorials and all eight downloadable PDF volumes. The remaining source-verification item is tracked separately in [the supplements audit](remaining-supplements-audit.md).

重新編排完整課程與導覽，讓新手知道從哪裡開始、需要哪些先備知識，以及接下來可以學什麼。

Completion criteria: one course tree controls ordering and numbering; main chapters expand into sub-chapters; prerequisites and previous/next navigation remain visible; bilingual mobile and desktop navigation works; legacy URLs and fragments resolve; search, retrieval and the content map use canonical URLs; website build and internal-link checks pass.

The accepted tree includes the completed non-blind, blind/real-world and one-step super-resolution chapters, the one-step generation chapter, standalone image fusion, and merged low-light/lightweight enhancement. Source filenames remain stable identifiers while public URLs omit numbers. Major headings on the main learning path and individual supplementary-note/paper pages can fold; keyboard navigation, hash reveal, printing and reading without JavaScript are preserved.

Validated curriculum and supplement checkpoint: `fde8eaa`. Astro check reports zero errors and warnings; all 28 Node test files pass; playground coverage includes 97 tutorials, 73 experiments and 26 image labs. Desktop/mobile browser checks cover 307 old bookmarks, 56 updated chapter/optional-page layouts and language switches, eight folding layouts, search/overview and all image labs. The built site contains 1,076 content pages; Pagefind indexes 1,075. The cycle-aware link check passes 2,041 HTML pages and 130,517 internal links with no broken targets or fragments. Local screenshots and machine-readable browser results are under `artifacts/`.

All eight PDFs were refreshed and validated from checkpoint `fde8eaa`: 97 tutorials and two original posts per language, 4,392 pages, 12,145 links, embedded fonts and valid page destinations; largest file 14.40 MiB. The final artifact hashes and edition metadata are recorded in `pdf-validation-2026-11-02.json` and `src/data/books.json`. Optional notes and paper guides remain online and are linked from the PDFs. The supplements audit records the subsequent 12 focused desktop/mobile checks, bilingual inline-lab regression check and public notebook-link verification.

No remote push, merge or deployment was performed. `main` is unchanged.
