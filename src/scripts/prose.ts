// Display equations that are wider than the column scroll sideways. Only those become a focusable, labelled group, so
// keyboard users can scroll them and screen-reader users hear why there is a scrollable region. Re-checked on resize.
const label = document.documentElement.lang.startsWith('zh') ? '方程式（可左右捲動）' : 'Equation (scrolls sideways)';
const equations = [...document.querySelectorAll<HTMLElement>('.prose .katex-display')];
const overflows = (el: HTMLElement) => el.scrollWidth > el.clientWidth + 1;
const mark = (el: HTMLElement) => {
  // An equation wider than the text measure may use the whole article column before it has to scroll.
  if (overflows(el) && el.parentElement?.matches('.prose, .section-body, .deeper-body')) el.classList.add('eq-wide');
  const wide = overflows(el);
  if (wide && !el.hasAttribute('tabindex')) {
    el.tabIndex = 0;
    el.setAttribute('role', 'group');
    el.setAttribute('aria-label', label);
  } else if (!wide && el.getAttribute('role') === 'group') {
    el.removeAttribute('tabindex');
    el.removeAttribute('role');
    el.removeAttribute('aria-label');
  }
};
if (equations.length > 0) {
  const ro = new ResizeObserver((entries) => entries.forEach((e) => mark(e.target as HTMLElement)));
  equations.forEach((el) => ro.observe(el));
}

// Printing: open every hint and collapsed block so the printout has the answers, then restore what the reader had open.
let closedForPrint: HTMLDetailsElement[] = [];
window.addEventListener('beforeprint', () => {
  closedForPrint = [...document.querySelectorAll<HTMLDetailsElement>('main details:not([open])')].filter((d) => !d.classList.contains('rt-mnav'));
  closedForPrint.forEach((d) => { d.open = true; });
});
window.addEventListener('afterprint', () => { closedForPrint.forEach((d) => { d.open = false; }); closedForPrint = []; });
