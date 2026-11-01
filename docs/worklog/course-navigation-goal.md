# Course navigation goal

Status: completed and validated for the 95 currently available tutorials, including all eight downloadable PDF volumes.

重新編排完整課程與導覽，讓新手知道從哪裡開始、需要哪些先備知識，以及接下來可以學什麼。

Completion criteria: one course tree controls ordering and numbering; main chapters expand into sub-chapters; prerequisites and previous/next navigation remain visible; bilingual mobile and desktop navigation works; legacy URLs and fragments resolve; search, retrieval and the content map use canonical URLs; website build and internal-link checks pass.

Implemented for all 95 currently available tutorials. Other paused content-writing branches remain separate; this change reorganizes the published curriculum without inventing missing chapters. Source filenames remain stable identifiers while public URLs omit numbers. The low-light parent and lightweight-enhancement child remain separate until their content merge; new SR chapters can be inserted into the existing restoration branch when written.

Validation: Astro build succeeds; Astro check reports 0 errors and 0 warnings; all tutorial MDX validates; 28 Node test files pass; playground coverage passes; 28 bilingual layouts at 1440/375 px pass with no horizontal overflow; old main-page, notes-index and numbered-URL bookmark fragments forward correctly in Chromium; 1,846 HTML pages and 103,487 internal links pass with zero broken targets/fragments. Search excludes redirects and indexes 921 content pages. Screenshots and machine-readable results are under artifacts/ (local, not committed).

All eight PDF volumes have been compiled with XeLaTeX (three passes per volume), using curriculum hierarchy and online-only optional-item links. No missing glyphs or undefined references remain; all 1,219 internal PDF destinations resolve to valid pages. Files range from 1.7 to 14.4 MiB, below the 25 MiB limit, and local download metadata points to source checkpoint b7e4395. The source export used Pandoc 3.1.11.1 and xeCJK 3.9.1; Tectonic’s remote bundle returned HTTP 403 in this environment, so the installed XeLaTeX engine was used instead.

No remote push, merge or deployment is authorized by this work.
