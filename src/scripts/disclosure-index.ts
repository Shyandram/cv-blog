// Explicit fragment navigation opens only the requested group. Ordinary visits
// always start closed, without restoring a prior browsing session's choices.
function revealIndexTarget() {
  let id: string;
  try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
  if (!id) return;
  const target = document.getElementById(id);
  if (!target?.closest('[data-disclosure-index]')) return;
  let parent: HTMLElement | null = target;
  while (parent) {
    if (parent instanceof HTMLDetailsElement) parent.open = true;
    parent = parent.parentElement;
  }
  requestAnimationFrame(() => target.scrollIntoView({ block: 'start' }));
}
revealIndexTarget();
window.addEventListener('hashchange', revealIndexTarget);
