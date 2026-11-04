import './legacy-fragments';
// Enhance the page's major sections while preserving content, ids and source heading levels.
const prose = document.querySelector<HTMLElement>('.prose[data-fold-sections]');
if (prose) {
  const zh = document.documentElement.lang.startsWith('zh');
  // Extracted note/paper items keep their original h4/h5 headings. Their page title is
  // rendered separately, so use the shallowest remaining body level on those pages.
  const bodyHeadings = [...prose.querySelectorAll<HTMLElement>(':scope > h2, :scope > h3, :scope > h4, :scope > h5, :scope > h6')];
  const level = prose.dataset.foldLevel === 'item' && bodyHeadings.length
    ? Math.min(...bodyHeadings.map((heading) => Number(heading.tagName.slice(1))))
    : 2;
  const headings = [...prose.querySelectorAll<HTMLElement>(`:scope > h${level}, :scope > .zone-wrap > h2`)];
  const sections: HTMLDetailsElement[] = [];
  for (const heading of headings) {
    const section = document.createElement('details');
    section.className = 'chapter-section';
    section.open = true;
    const summary = document.createElement('summary');
    const body = document.createElement('div');
    body.className = 'section-body';
    heading.before(section);
    summary.append(heading);
    const action = document.createElement('span');
    action.className = 'section-action';
    action.setAttribute('aria-hidden', 'true');
    action.dataset.open = zh ? '摺疊' : 'Collapse';
    action.dataset.closed = zh ? '展開' : 'Expand';
    summary.append(action);
    // Optional-reading cards and reference wrappers remain independent of the preceding section.
    let next = section.nextSibling;
    while (next && !(next instanceof HTMLElement && (
      (/^H[2-6]$/.test(next.tagName) && Number(next.tagName.slice(1)) <= level)
      || next.matches('.notes-teaser, .zone-wrap')
    ))) {
      const following = next.nextSibling;
      body.append(next);
      next = following;
    }
    section.append(summary, body);
    sections.push(section);
  }
  if (sections.length) {
    const controls = document.createElement('div');
    controls.className = 'section-controls';
    controls.dataset.pagefindIgnore = '';
    for (const [label, open] of [[zh ? '展開本頁章節' : 'Expand all sections', true], [zh ? '收合本頁章節' : 'Collapse all sections', false]] as const) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = label;
      button.addEventListener('click', () => sections.forEach((section) => { section.open = open; }));
      controls.append(button);
    }
    prose.before(controls);
  }
  const reveal = (hash: string, scroll = false) => {
    let id: string;
    try { id = decodeURIComponent(hash.slice(1)); } catch { return; }
    const target = document.getElementById(id);
    if (!target || !prose.contains(target)) return;
    for (let parent = target.parentElement; parent; parent = parent.parentElement) {
      if (parent instanceof HTMLDetailsElement) parent.open = true;
    }
    if (scroll) requestAnimationFrame(() => target.scrollIntoView());
  };
  document.addEventListener('click', (event) => {
    const link = (event.target as Element | null)?.closest<HTMLAnchorElement>('a[href]');
    if (!link || event.defaultPrevented) return;
    const target = new URL(link.href, location.href);
    if (target.origin === location.origin && target.pathname === location.pathname && target.hash) reveal(target.hash);
  }, true);
  window.addEventListener('hashchange', () => reveal(location.hash, true));
  if (location.hash) reveal(location.hash, true);
}
