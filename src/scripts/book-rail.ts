// The chapter page's left contents sidebar can be hidden to widen the reading column (>= 900 px; below that it is the
// "Contents" sheet of the section bar). The choice is a per-viewer convenience kept in localStorage (pp:rail); without
// JavaScript or storage the sidebar is simply shown and the buttons stay hidden.
import { load, save } from './progress';

const page = document.querySelector<HTMLElement>('.tut');
const rail = document.getElementById('book-rail');
const hide = document.querySelector<HTMLButtonElement>('.rail-hide');
const show = document.querySelector<HTMLButtonElement>('.rail-show');
if (page && rail && hide && show) {
  const set = (open: boolean, focus = false) => {
    page.dataset.rail = open ? 'open' : 'closed';
    hide.hidden = false;
    hide.setAttribute('aria-expanded', String(open));
    show.hidden = open;
    show.setAttribute('aria-expanded', String(open));
    if (focus) (open ? hide : show).focus({ preventScroll: true });
  };
  set(load('pp:rail') !== 'closed');
  hide.addEventListener('click', () => { set(false, true); save('pp:rail', 'closed'); });
  show.addEventListener('click', () => { set(true, true); save('pp:rail', 'open'); });
  // Keep the current chapter in view inside the sidebar's own scroll area (the page itself does not move).
  const cur = rail.querySelector<HTMLElement>('.ct-node.cur > .ct-row');
  if (cur && rail.scrollHeight > rail.clientHeight) {
    const top = cur.getBoundingClientRect().top - rail.getBoundingClientRect().top;
    if (top > rail.clientHeight * 0.6) rail.scrollTop = top - rail.clientHeight / 3;
  }
}
