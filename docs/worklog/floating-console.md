# Compact floating reading console — 2026-11-04

The user clarified that the reading tools should be a persistent floating console, with a few buttons covering most operations. The console now stays visible as a 64px-high bar with four primary actions: Contents, Sections, Code and Top. Contents opens an attached tray with reading progress, a native section picker, Previous/Next and Reset position. The tray starts closed on every page. Clicking elsewhere keeps the console available; Escape inside it closes the tray and returns focus to Contents.

Section/code actions collapse all when any are open, and expand all when none are open. Accessible labels include the intended action and open/total counts; pressed state also reflects partial selections. Code expansion reveals parent chapters. Individual native disclosures, initially folded code and copy-while-closed remain intact. Pages without code keep that action disabled.

Drag the grip with a pointer or touch; use arrow keys for 16px movement, Shift for 48px and Home to reset. Normalized position is saved locally, with storage failures tolerated. Position is clamped to the visual viewport and safe areas after resizing. The console is hidden for printing, and existing article print-state restoration remains intact. Without JavaScript, static navigation and native code disclosures continue working.

The previous popover and its positioning dependency have been removed. The existing back-to-top button yields to the console's Top action. Both languages and themes retain the site's shared typography, track colors and focus indicators.

Validated against the production build:

- Build: 1,114 pages; Pagefind: 1,113 indexed pages. Astro check: 0 errors, 0 warnings, 3 existing hints.
- Console: 20 layouts across both languages/themes at 320, 375, 768, 1440 and landscape 844px. Verified four primary buttons, default closed tray, fold states, section selection, persistent outside-click behavior, pointer/keyboard movement, reload position, reset, viewport clamping, Top and print visibility. Additional checks cover code-free paper pages in both languages, blocked storage and real mobile touch events. See [floating-console-validation.json](floating-console-validation.json).
- Reading: 72 layouts; code: 12 layouts plus 1,730 initially folded blocks across 348 pages, native no-JS disclosure/printing, source-preserving copy, pending/success/failure feedback and search indexing; chapter folding: 8 layouts; course navigation: 28 layouts. All passed.
- Internal links: 2,079 HTML pages, 139,181 links, 0 broken. Content index regenerated without changes; paper-guide coverage remains 149 guides across 40 chapters with no missing figures.
- Reviewed phone/desktop and light/dark screenshots. Screenshots are local artifacts under `artifacts/console-*-compact.png` and `artifacts/console-*-contents.png`.

Regression scripts now use the public console actions through a shared helper. Broader paper/supplement/chapter scripts had their selectors migrated and were syntax-checked; their full matrices were not rerun for this interface-only change. Browser validation used Chromium and does not imply a complete cross-browser accessibility audit. No article text or PDF layout changed; this work was committed locally without pushing or deploying.
