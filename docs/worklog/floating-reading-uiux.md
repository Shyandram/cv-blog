# Floating reading tools and visual polish — 2026-11-04

The course's separate chapter/code actions now share one reading toolbar. Each group shows its open/total count, short action labels retain contextual accessible names, and code instructions appear on demand. Rounded surfaces, restrained track-color highlights, grouped chapter tabs, a clear current-course marker and next-page cards create a consistent reading interface.

After the inline toolbar scrolls above the viewport, a floating entry opens the same controls in a native nonmodal popover. `@floating-ui/dom` is loaded on demand and uses offset, flip, shift and available-height sizing. A placeholder preserves document space while the original toolbar moves into the popover. It is restored on close and before printing, preserving fold states and listeners.

Escape and the close button return focus appropriately; outside clicks keep the outside target's focus. The trigger handles direct jumps as well as normal scrolling, and the floating controls leave room for the existing back-to-top button and safe-area inset. Native popover support is feature-detected; the inline toolbar remains the fallback. CSS provides bounded placement if the optional positioning chunk fails to load.

Copy controls display pending, success and failure states. Pending writes cannot overlap. Native disclosures, code initially folded, full-code printing and the content itself retain their existing behavior. Motion is limited to short color/border transitions and respects reduced-motion settings.

Validation completed against the production build:

- Build: 1,114 pages, 1,113 indexed for search. Astro check: 0 errors, 0 warnings and 3 existing unused-symbol hints.
- Floating tools: 20 Chromium layouts across both languages/themes at 320, 375, 768, 1440 and landscape 844px; plus two section-only paper pages. Verified lazy loading, count synchronization, placement, resize, space preservation, Escape/close focus, outside-click focus, print restoration and clipboard pending/success behavior. See [floating-tools-validation.json](floating-tools-validation.json).
- Additional manual browser checks passed with the positioning chunk blocked and with native popover support removed; inline controls and CSS fallback placement remained usable.
- Reading: 72 layouts passed. Code: 1,730 blocks on 348 pages remain initially folded; 12 operation layouts, no-JS disclosure/print and code search passed. Chapter folding: 8 layouts passed. Course navigation: 28 layouts passed.
- Internal links: 2,079 HTML pages, 139,181 links, 0 broken. Content index regenerated without changes; all 149 paper guides retain their figure coverage.
- Mobile/desktop and light/dark screenshots were visually inspected. The screenshot review also reduced the spacing between the tools and the opening section.

The tests caught a direct-scroll jump that did not cross an IntersectionObserver threshold. The floating trigger now also checks scroll and resize events. Outside-click focus and reopening after a resize have dedicated handling. Validation here is Chromium-based and does not claim a complete cross-browser accessibility audit.
