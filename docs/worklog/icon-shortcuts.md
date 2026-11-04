# Icon shortcuts — 2026-11-04

The floating reading console now uses four consistent outline icons for Contents, expand/collapse sections, Code and Top. The section icon changes with the intended action. Localized tooltips and accessible names explain each control; visually hidden text remains available to assistive technology. The icons are static inline SVGs with no external asset or icon-package dependency.

The closed console is 238px wide and 54px high, with four 44px action targets and a separate 44px drag grip. Opening Contents retains the wider navigation tray. Mouse, touch and keyboard behavior, position persistence and default folded code are preserved.

Validation: production build and Pagefind completed; Astro check reports 0 errors, 0 warnings and 3 existing hints. The 20-layout console regression passed, including touch dragging, storage fallback, positioning, folding and print. An additional eight icon layouts across both languages, both themes and 375/1440px widths verified rendered SVGs, localized names/tooltips, minimum target sizes, changing fold icons, Enter/Space activation and focus restoration. Content index regeneration and coverage of all 149 paper guides passed without content changes.

Reviewed the rendered icon strip. Local previews: `artifacts/icon-bar-*.png` and `artifacts/icon-console-*.png`. Browser validation used Chromium. Committed locally without pushing or deploying.
