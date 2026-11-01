# Course navigation goal

重新編排完整課程與導覽，讓新手知道從哪裡開始、需要哪些先備知識，以及接下來可以學什麼。

Completion criteria: one course tree controls ordering and numbering; main chapters expand into sub-chapters; prerequisites and previous/next navigation remain visible; bilingual mobile and desktop navigation works; legacy URLs and fragments resolve; search, retrieval and the content map use canonical URLs; website build and internal-link checks pass.

Implemented for all 95 currently available tutorials. Other paused content-writing branches remain separate; this change reorganizes the published curriculum without inventing missing chapters. Source filenames remain stable identifiers while public URLs omit numbers. The low-light parent and lightweight-enhancement child remain separate until their content merge; new SR chapters can be inserted into the existing restoration branch when written.

Validation: Astro build succeeds; Astro check reports 0 errors and 0 warnings; all tutorial MDX validates; 28 Node test files pass; playground coverage passes; 28 bilingual layouts at 1440/375 px pass with no horizontal overflow; old main-page, notes-index and numbered-URL bookmark fragments forward correctly in Chromium; 1,846 HTML pages and 103,487 internal links pass with zero broken targets/fragments. Search excludes redirects and indexes 921 content pages. Screenshots and machine-readable results are under artifacts/ (local, not committed).

PDF export has produced all eight LaTeX volumes using curriculum hierarchy and online-only optional-item links. PDF compilation/publication is being completed separately after the website source checkpoint.

No remote push, merge or deployment is authorized by this work.
