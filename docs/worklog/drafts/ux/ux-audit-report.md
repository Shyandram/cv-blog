# UX / learning-design audit — implementation plan (saved by the coordinator from the auditor's report)

Screenshots: shots/ (names `<page>-<view>-<d|m>-<light|dark>.png`), measurements: metrics.json, scripts: shoot.py,
metrics.py, contrast.py, printpdf.py (re-run them against a new dist/ to verify).

NOTE (new since the audit): Supplementary notes now live on a separate page per tutorial (`…/<slug>/notes/`), built by
src/lib/rehypeSupplementZones.mjs + src/components/zones/* + the `mode` prop in src/pages/[lang]/learn/[...slug].astro.
Keep that working (main page shows the NotesTeaser card; notes page shows only notes + references).

## Findings (evidence in shots/ and metrics.json)
F1 High: <kernel-playground class="demo"> and <image-compare class="demo"> are inline custom elements without display:block,
   so the .demo frame renders as empty slivers (shots/dip03-kernel-d-light.png; ImageLab.astro:148 does set it).
F2 High: track-coloured small text fails contrast in light mode: image-processing #0f8b8d 3.87 on --bg / 3.66 on --code-bg;
   ml #c27c0e 3.19/3.02 (also warn callout title); deep-learning #d1366b 4.43/4.20; foundations #64748b 4.47/4.23. Dark passes.
   Tokens global.css:19-25; used in global.css:66,69, Callout.astro strong, .rail h4, learn index .num.
F3 High: --line form-control borders 1.29:1 on white (WCAG 1.4.11): playground search/select, kernel cells, Pagefind inputs.
F4 High: at 375 px the sticky header wraps to 3 rows (138 px of 812) and hides focused elements (WCAG 2.4.11); no
   scroll-padding-top. Header.astro:58,72; global.css:95.
F5 High: no skip link; 8 header + 13 rail tab stops before the article; <main> has no id (Base.astro:60-65).
F6 High: figures (1400-1600 px PNGs) shown at 343 px on phones → labels ~4-5 px; not links, cannot be enlarged (Figure.astro:8).
F7 Medium: HeroSobel requestAnimationFrame loop never stops (WCAG 2.2.2), runs off-screen (HeroSobel.astro:28).
F8 Medium: targets < 24 px: footer links 25×16, breadcrumbs 16 px tall, playground summaries 23 px, permalink 21×23,
   Predict radios 13×13, ImageCompare range 16 px, copy button 48×22; header controls 34-36 px.
F9 Medium: theme toggle has no aria-pressed and static icon; "Copied" not announced; aria-label="Main" (Header.astro:23) and
   "Pagination" ([...slug].astro) are English on zh-tw; citations announced as bare numbers.
F10 Low: outline starts with <h4> (rail title, "On this page"); TOC is an unnamed <aside>, not <nav>.
L1 Medium: below 1200 px orientation fades in 35-42 minute reads (DIP-03 37,000 px tall at 1440, 50,000 at 375; 15 h2, 23 h3);
   TOC is one <details> at the top; progress bar aria-hidden.
L2 Medium: time and resume are page-level only (ReaderTools remember()).
L3 Medium: at 375 px DIP-03 has 15/33 display equations and 14/15 code blocks overflowing with no cue; DLCV-03 8/17 and 10/12.
L4 Low: .katex-display spans the 916 px column while prose is 720 px, so equations sit ~100 px right of their paragraph (global.css:90).
L5 Medium: citations jump ~30,000 px with no preview and no way back (rehypeCitations.mjs).
L6 Medium: exercise hints contain full answers; no "try first" prompt; "✓ Read" at ≥95 % scroll rewards scrolling to references;
   the Predict component is used in no tutorial.
L7 Medium: the chapter experiment is offered once, above "What you'll learn", linking to /playground/#id; never again later.
L8 Medium: switching EN↔中文 drops you at the top (Header.astro:32-34,52-54); sections mirror by rule so map by heading index.
L9 Medium: search excerpts show raw TeX and tripled glyphs ("R_{\text{normal}}Rlow˜I=…", "TTT"); no filters.
L10 Medium: learn index: duplicate numbering ("01" + "IP 01 ·"); descriptions 102-128 chars/line; no minutes/prereqs;
   progress is an 11 px "42%" that squeezes text on phones; "Continue reading" only on /learn/.
L11 Medium: zh-tw DIP-03 downloads 38 font files, 2,854 KB; every page ships 313 KB render-blocking CSS (106 KB gz), mostly
   213 Noto Sans TC + Noto Serif TC @font-face rules; --f-ui includes Noto Sans TC so English pages fetch a CJK subset.
L12 Low: white figures glare in dark mode. L13 Low: print unstyled. L14 Low: footer floats on short pages; "先備知識:" uses ASCII
   colon; back-to-top covers captions on phones; one 1556 px figure file to phones.
KEEP: Source Serif 4 at 18/1.7 with 720 px measure; 2 px focus ring; no horizontal page scroll; CLS ≤ 0.02; lazy WebP figures;
   house structure; <details> hints, prereqs line, scroll-spy, Ctrl-K, reduced motion respected. No restyle (skill → Swiss/minimal,
   matches current design).

## P0 (must fix)
P0-1 global.css: `kernel-playground, image-compare, predict-reveal { display:block; }` (check the actual custom element names).
P0-2 light tokens: --t-foundations #5f6e84, --t-image-processing #0d797b, --t-ml #97610b, --t-deep-learning #c72e62 (≥4.6 on
   --code-bg and --bg); add --line-strong #7d8a9b light / #6b7a90 dark for `input:not([type=range],[type=radio],[type=checkbox]),
   select, textarea` borders and the Pagefind border variables. Verify with contrast.py, also check nlp/pr/opt/it tokens.
P0-3 compact mobile header at ≤620 px: not sticky (position: relative), smaller logo, tools 44×44, nav one scrollable row
   (nowrap, overflow-x:auto, padding 11px 10px); short language label on phones ("EN"/"中文") with aria-label; global.css
   `html { scroll-padding-top: var(--anchor-offset) }` and --anchor-offset 16px at ≤620 px (tutorial pages raise it if a sticky
   section bar is added). Verify header ≤ ~100 px and no obscured focus at 375.
P0-4 skip link in Base.astro (`<a class="skip-link" href="#main">` + `<main id="main" tabindex="-1">`, hidden until focus);
   i18n labels: nav.main Main/主選單, tut.pager "Previous and next tutorials"/上一篇與下一篇, skip "Skip to content"/跳到主要內容,
   copy "Copy code"/複製程式碼, citation "Reference {n}"/參考文獻 {n}; theme toggle aria-pressed set on load/click; copy result in a
   role="status" sr-only element; rehypeCitations adds aria-label (language from file.path); rail/TOC titles become <p>, TOC a <nav aria-label>.
P0-5 Figure.astro: wrap the image in `<a class="fig-zoom" href={full-size src}>` with aria-label "Open full-size figure: …"/
   「開啟原尺寸圖片：…」 (lang from Astro.url.pathname), cursor zoom-in; optional "⤢ tap to enlarge"/「⤢ 點擊放大」 badge ≤620 px.
P0-6 HeroSobel: one sweep (~4 s) then hold the final frame; "Replay the Sobel sweep"/「重播 Sobel 掃描」 button; pause off-screen
   with IntersectionObserver; respect reduced motion.
P0-7 target sizes: footer + breadcrumb links inline-block padding-block 5px; playground summaries padding-block 10px; permalink
   ≥24×24; Predict labels min-height 28px, radios 18px; .rt-copy and range inputs ≥28 px.

## P1 (high learning value)
P1-1 orientation below 1200 px: make .rt-mnav sticky (top 0, body max-height 70dvh overflow auto); summary shows the current h2
   and "4/15 · 18 min left"/「第 4/15 節・剩 18 分鐘」 updated by spy(); clicking a link closes it; per-section minutes (split
   entry.body on /^## /m, readingMinutes per chunk, zip with TOC; "6 min"/「6 分鐘」); visited sections ✓ (pp:s:<path>, sr-only
   "(read)"/「（已讀）」); pp:last stores the active heading so "Continue reading" links to path#id (handle /notes/ pages).
P1-2 overflow: `.prose > .katex-display { max-width: var(--measure) }`; scroll shadows on .katex-display and pre (background
   gradients with attachment local/scroll; --code-bg for code); display math 0.92em at ≤620 px; overflowing equations get
   tabindex=0, role=group, aria-label "Equation (scrolls sideways)"/「方程式（可左右捲動）」 (client-side, only when overflowing).
P1-3 citation previews: id="cite-n" on first use, ↩ back-link in each reference item ("Back to the text"/「回到正文」); one
   role="tooltip" element filled with a clone of #ref-n on hover/focus (Esc closes, hoverable, WCAG 1.4.13); on touch keep the jump.
P1-4 retrieval practice: under the Exercises/練習 heading a prompt (EN "Try each exercise before opening its hint: hints contain
   worked answers. Tick an exercise when you can solve it without looking." / zh「每題先自己做，再打開提示：提示內含解答。不看提示也能
   解出來時，就勾選完成。」) and a "Solved"/「已解出」 checkbox per exercise (pp:x:<path>), inserted by script/component, not by
   editing prose; a "Check your understanding"/「檢查理解」 panel after the prose: exercises solved "2 of 6"/「已解出 2／6 題」, the
   chapter experiments "Try it now"/「動手試試」, next tutorial; learn index shows "2/6 exercises" and the ✓ comes from exercises.
P1-5 keep place across languages: on language-link click store {to, headingIndex, count, ratio} in sessionStorage; on load scroll
   to that heading if counts match, else the ratio.
P1-6 search: Pagefind index excludes '.katex-html', '.katex-mathml annotation', '.rt-copy' (or mark them data-pagefind-ignore /
   index only the TeX-free text); data-pagefind-filter for track, level, type (Tutorial/教學, Blog/部落格); empty state with track
   chips and sample queries "Try"/「試試看」. Verify "histogram equalization" excerpts show no TeX.
P1-7 learn index + home: drop duplicate counter, .desc max-width 72ch; per-item minutes, level, "Needs IP 02"; 3 px progress bar +
   sr-only "42% read"/「已讀 42%」; track header "13 tutorials · ≈ 8 h"/「13 篇・約 8 小時」; shared progress script; "Continue
   reading" also in the home hero.
P1-8 lighter CJK fonts: remove Noto Sans TC from --f-ui; system CJK sans for zh headings ('PingFang TC','Noto Sans CJK TC',
   'Microsoft JhengHei'); drop the noto-sans-tc import; keep Noto Serif TC for zh body unless it dominates; measure first.
   Target zh DIP-03 fonts well under 1.5 MB, English pages load no TC files.

## P2
dark-mode figures brightness(.9) + 1 px frame (no invert); print stylesheet (hide chrome, white bg, open details on beforeprint,
break-inside avoid); responsive figure widths [480, 960, full] with sizes; spaced review on learn index ("Revisit 2 exercises from
IP 03"/「回頭再做 IP 03 的 2 題練習」 after 3/10/30 days); TOC shows h3s of the active section only; sticky footer on short pages;
zh full-width colon 「先備知識：」; back-to-top clear of captions; defer Pagefind component CSS.

## Order
1) P0-1, P0-2, P0-4, P0-5, P0-6, P0-7. 2) P0-3 with P1-1. 3) P1-2, P1-6. 4) P1-3, P1-5, P1-4, P1-7, P1-8. Then P2 if time.
After each batch: npm run build, npx astro check (0/0), re-run metrics.py, shoot.py, contrast.py, compare: no horizontal scroll,
only inline links < 24 px, no obscured focus at 375, contrast ≥4.5 text / ≥3 control borders both themes, zh font bytes,
screenshots 1440 & 375 light/dark EN & zh-tw.
