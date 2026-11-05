import aliases from '../content/legacy-chapter-fragments.json';

// Hashes are not sent to the server. Keep bookmarks when a topic moves between chapters.
const mappings = aliases as Record<string, Record<string, string>>;
function followMovedTopic() {
  let id: string;
  try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
  const target = mappings[location.pathname]?.[id];
  if (!target) return;
  const destination = new URL(target, location.origin);
  if (destination.href === location.href) return;
  // Existing optional-page forwarding already handles unchanged heading ids. Let it
  // navigate once instead of starting a second, competing location.replace().
  for (const card of document.querySelectorAll<HTMLElement>('[data-zone-base][data-zone-ids]')) {
    const ids = JSON.parse(card.dataset.zoneIds || '{}') as Record<string, string>;
    if (!ids[id]) continue;
    const automatic = new URL(`${card.dataset.zoneBase}${ids[id]}/${location.hash}`, location.href);
    if (automatic.pathname === destination.pathname && (!destination.hash || automatic.hash === destination.hash)) return;
  }
  location.replace(destination.href);
}
followMovedTopic();
window.addEventListener('hashchange', followMovedTopic);
