// Blog index: the topic chips filter the list. Without JavaScript every topic stays visible and the chips are plain
// links to each topic's section (`/blog/#agents` also keeps working for the breadcrumb on post pages).
const chips = Array.from(document.querySelectorAll<HTMLAnchorElement>('[data-blog-chips] a'));
const sections = Array.from(document.querySelectorAll<HTMLElement>('[data-blog-list] > [data-topic]'));
const start = document.querySelector<HTMLElement>('[data-blog-start]');

function apply(topic: string) {
  const known = sections.some((s) => s.dataset.topic === topic);
  const active = known ? topic : 'all';
  for (const s of sections) s.hidden = active !== 'all' && s.dataset.topic !== active;
  if (start) start.hidden = active !== 'all';
  for (const c of chips) {
    if (c.dataset.topic === active) c.setAttribute('aria-current', 'true');
    else c.removeAttribute('aria-current');
  }
}

function fromHash() {
  let id = '';
  try { id = decodeURIComponent(location.hash.slice(1)); } catch { /* malformed hash */ }
  apply(id || 'all');
}

for (const c of chips) {
  c.addEventListener('click', (e) => {
    e.preventDefault();
    const topic = c.dataset.topic ?? 'all';
    apply(topic);
    try { history.replaceState(null, '', topic === 'all' ? location.pathname + location.search : `#${topic}`); } catch { /* ignore */ }
  });
}
window.addEventListener('hashchange', fromHash);
fromHash();
