# Reading interface — 2026-11-04

Goal: make the bilingual course and its paper guides comfortable for beginners to read, locate and navigate on phones and desktops.

Applied skill: [UI/UX Pro Max](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill/blob/main/.claude/skills/ui-ux-pro-max/SKILL.md). The initial broad design-system search returned a marketing/testimonial pattern unsuitable for this course. A narrower product search, `documentation knowledge base`, matched Knowledge Base/Documentation with Minimalism & Swiss Style + Accessible & Ethical. Typography searches matched bounded line length, adequate line height and balanced headings. Astro guidance confirmed preserving static/no-JS navigation. No marketing pattern or new font dependency was adopted.

Decisions:
- Keep semantic light/dark colors and track identities; test primary and secondary text against both page and surface backgrounds at 4.5:1.
- Bound prose to 40rem. Use 19px desktop / 18px phone text, 1.8 line height, and 1.85 for Traditional Chinese. The CJK system font leads Chinese paragraphs for a consistent reading texture; Latin technical names retain available system fallbacks.
- Give paper-guide h4 headings their own 20px tier and clear section spacing. Maintain independently wide figures, equations, tables and code.
- Show complete chapter titles with wrapping. Use distinct 44px disclosure targets without hit-area overlap, 44px chapter tabs, and larger mobile navigation rows.
- Align desktop contents disclosures beside their headings in a two-column grid, keeping nested lists across both columns.
- Make prerequisite information easy to find and stack previous/next cards on small screens.
- Code starts folded using native HTML disclosures. Show the language and line count before opening, with a separate 44px copy button usable while closed. Page-level code controls and a short usage hint support readers who want to try the examples or search all code. Keep no-JS disclosure and full-code printing; restore the reader's open/closed choices after printing. These controls apply to course, optional-reading and blog prose without changing source text.
- Retain keyboard controls, fragment reveal, reduced-motion support and existing course URLs.
- Allow unusually long unbroken reference/API names to wrap within prose; keep code and wide equations in their own horizontal scrollers.
- Group chapter and code controls in one reading toolbar, with live open/total counts and an optional usage disclosure. Short visual button labels retain full, contextual accessible names.
- Once the inline toolbar is above the viewport, show a floating reading-tools button. Reuse the original controls in a native popover; load `@floating-ui/dom` on demand for offset, flip, shift and available-height sizing. Preserve document space, keyboard focus, outside-click behavior and printing. Unsupported popover browsers retain the inline controls.
- Use consistent rounded surfaces, restrained track-color highlights and clear current-page states for tabs, chapter disclosures, the course tree and next-page cards. Copy feedback distinguishes pending, success and failure and prevents overlapping writes. Motion stays optional under reduced-motion preferences.

Verification uses `scripts/check-readability.py` at 320, 375, 768, 1024, 1440 and landscape 844px in both languages and themes, plus existing course/folding/paper-guide regressions. The standalone PDF layout is unchanged by web CSS.
