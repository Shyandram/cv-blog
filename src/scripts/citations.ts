// Citation previews and back-links.
// - Every citation link (a.cite → #ref-n, added at build time by rehypeCitations) gets an id; each reference entry gets a
//   "↩" link back to the citation the reader came from (the first one on the page until a citation is used).
// - With a mouse or the keyboard, a citation shows the reference in one shared tooltip (role="tooltip"): it stays while the
//   pointer is on the citation or the tooltip, and Esc closes it (WCAG 1.4.13). Touch keeps the plain jump to the list.
const zh = document.documentElement.lang.startsWith('zh');
const backLabel = zh ? '回到正文' : 'Back to the text';
const cites = [...document.querySelectorAll<HTMLAnchorElement>('a.cite[href^="#ref-"]')];
const refOf = (a: HTMLAnchorElement) => document.getElementById(decodeURIComponent(a.hash.slice(1)));

if (cites.length > 0) {
  // Ids: the first citation of reference n is cite-n, later ones cite-n-2, cite-n-3, …
  const seen = new Map<string, number>();
  for (const a of cites) {
    const n = a.hash.slice('#ref-'.length);
    const k = (seen.get(n) ?? 0) + 1;
    seen.set(n, k);
    if (!a.id) a.id = k === 1 ? `cite-${n}` : `cite-${n}-${k}`;
  }
  const backOf = (li: HTMLElement) => li.querySelector<HTMLAnchorElement>(':scope > a.cite-back');
  for (const a of cites) {
    const li = refOf(a);
    if (!li || backOf(li)) continue;
    const back = document.createElement('a');
    back.className = 'cite-back';
    back.href = `#${a.id}`;
    back.title = backLabel;
    back.setAttribute('aria-label', backLabel);
    back.dataset.pagefindIgnore = '';
    back.textContent = '↩';
    li.append(' ', back);
  }

  // The tooltip.
  const tip = document.createElement('div');
  tip.id = 'cite-tip';
  tip.className = 'cite-tip';
  tip.setAttribute('role', 'tooltip');
  tip.hidden = true;
  document.body.append(tip);
  let owner: HTMLAnchorElement | null = null;
  let hideTimer: number | undefined;
  let showTimer: number | undefined;

  const place = (a: HTMLAnchorElement) => {
    const r = a.getBoundingClientRect();
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    const left = Math.min(Math.max(8, r.left + r.width / 2 - 40), document.documentElement.clientWidth - w - 8);
    const below = r.bottom + 8 + h <= window.innerHeight - 8 || r.top - 8 - h < 8;
    tip.style.left = `${Math.max(8, left) + window.scrollX}px`;
    tip.style.top = `${(below ? r.bottom + 8 : r.top - 8 - h) + window.scrollY}px`;
  };
  const hide = () => {
    clearTimeout(showTimer);
    clearTimeout(hideTimer);
    tip.hidden = true;
    owner?.removeAttribute('aria-describedby');
    owner = null;
  };
  const show = (a: HTMLAnchorElement) => {
    clearTimeout(hideTimer);
    const li = refOf(a);
    if (!li) return;
    if (owner !== a) {
      const num = document.createElement('span');
      num.className = 'cite-tip-n';
      num.textContent = `[${a.hash.slice('#ref-'.length)}] `;
      tip.replaceChildren(num, ...[...li.childNodes].map((c) => c.cloneNode(true)));
      tip.querySelectorAll('.cite-back').forEach((x) => x.remove());
      tip.querySelectorAll('[id]').forEach((x) => x.removeAttribute('id'));
      // Links in the preview work with the mouse; keyboard users reach them in the list itself.
      tip.querySelectorAll('a').forEach((x) => { x.tabIndex = -1; });
      owner?.removeAttribute('aria-describedby');
      owner = a;
      a.setAttribute('aria-describedby', tip.id);
    }
    tip.hidden = false;
    place(a);
  };
  const hideSoon = () => { clearTimeout(showTimer); clearTimeout(hideTimer); hideTimer = window.setTimeout(hide, 250); };

  for (const a of cites) {
    a.addEventListener('pointerenter', (e) => {
      if (e.pointerType !== 'mouse') return;
      clearTimeout(hideTimer);
      clearTimeout(showTimer);
      showTimer = window.setTimeout(() => show(a), owner ? 0 : 120);
    });
    a.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') hideSoon(); });
    a.addEventListener('focus', () => { if (a.matches(':focus-visible')) show(a); });
    a.addEventListener('blur', () => { if (owner === a && !tip.matches(':hover')) hide(); });
    // Following a citation: the reference's back-link returns to this citation.
    a.addEventListener('click', () => {
      hide();
      const li = refOf(a);
      const back = li && backOf(li);
      if (back) back.href = `#${a.id}`;
    });
  }
  tip.addEventListener('pointerenter', () => clearTimeout(hideTimer));
  tip.addEventListener('pointerleave', hideSoon);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !tip.hidden) hide(); });
  window.addEventListener('resize', () => { if (!tip.hidden) hide(); });
}
