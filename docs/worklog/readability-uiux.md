# Reading UI/UX — 2026-11-04

Applied the user-requested [UI/UX Pro Max skill](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill/blob/main/.claude/skills/ui-ux-pro-max/SKILL.md) to the existing bilingual course interface. The design decisions are recorded in [readability.md](../design/readability.md).

## Changes

- Larger prose, bounded line length, more generous paragraph/list spacing and distinct paper-guide heading levels.
- Full chapter names in the course tree, separate 44px disclosure targets, larger chapter tabs and mobile navigation links.
- A visible prerequisite card and vertically stacked previous/next cards on phones.
- An always-visible code-copy toolbar above the code, with clipboard contents checked in Chromium.
- Clearer table headers, larger figure captions and aligned desktop contents controls.
- Wrapping for unbroken reference/API names: the Goyal guide's `torch.optim.lr_scheduler.OneCycleLR` reference previously widened a 375px viewport to 394px after the typography change. The revised prose rule restores the viewport width without clipping the reference.

The changes affect shared web presentation. Original posts' text, paper claims, source figures, course URLs and standalone book layouts were not edited.

## Validation

- Production build: 1,114 pages; Pagefind indexed 1,113 pages.
- Astro check: 0 errors, 0 warnings, 3 existing unused-symbol hints.
- Content index regenerated with no diff; paper coverage: 149 guides in 40 chapters, all with figures.
- Internal links: 2,079 HTML pages, 139,181 links, 0 broken.
- Reading regression: 72 layouts across both languages, both themes, three representative pages and six viewports (320, 375, 768, 1024, 1440 and landscape 844px). Checks cover prose size, semantic text contrast, horizontal bounds, touch targets, code toolbar placement, desktop contents alignment, folding, mobile navigation, reduced motion and print line height. See [readability-validation.json](readability-validation.json).
- Course navigation: 28 layouts, tree controls and mobile menus passed.
- Section folding: 8 layouts, keyboard, deep links, print restoration and no-JS reading passed.
- Full paper-guide browser regression: all 149 guides in both languages at 1440px and 375px (596 layouts) passed. Checks include full-size images, language/parent links, horizontal bounds, folding, keyboard access, fragment reveal and JavaScript errors; 828 subsection bookmarks retained. See [paper-guide-browser-validation.json](paper-guide-browser-validation.json).

Manual screenshots cover English and Traditional Chinese, light and dark themes, mobile entry, paper reading, desktop chapter navigation and the code toolbar. The screenshot review caught a contents disclosure wrapping below its heading; the control now has its own grid column. These checks target the changed interface and do not constitute a full accessibility certification.
