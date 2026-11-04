// Native popover supplies Escape/light dismissal; Floating UI supplies collision-aware placement.
export function setupFloatingReadingTools(toolbar: HTMLElement) {
  if (!('showPopover' in HTMLElement.prototype)) return;
  const zh = document.documentElement.lang.startsWith('zh');
  const host = document.createElement('div');
  host.className = 'reader-toolbar-host';
  toolbar.before(host);
  host.append(toolbar);
  const trigger = document.createElement('button');
  trigger.type = 'button';
  trigger.className = 'reader-float-trigger';
  trigger.hidden = true;
  trigger.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M4 7h7m4 0h5M4 17h3m4 0h9"/><circle cx="13" cy="7" r="2"/><circle cx="9" cy="17" r="2"/></svg>';
  trigger.append(document.createTextNode(zh ? '閱讀工具' : 'Reading tools'));
  trigger.setAttribute('aria-expanded', 'false');
  trigger.setAttribute('aria-haspopup', 'dialog');
  const popover = document.createElement('div');
  popover.id = 'reader-floating-tools';
  popover.className = 'reader-popover';
  popover.popover = 'auto';
  popover.setAttribute('role', 'dialog');
  popover.setAttribute('aria-labelledby', 'reader-popover-title');
  trigger.setAttribute('popovertarget', popover.id);
  const header = document.createElement('div');
  header.className = 'reader-popover-header';
  const title = document.createElement('strong');
  title.id = 'reader-popover-title';
  title.textContent = zh ? '閱讀工具' : 'Reading tools';
  const close = document.createElement('button');
  let returnFocus = false;
  close.type = 'button';
  close.textContent = '×';
  close.setAttribute('aria-label', zh ? '關閉閱讀工具' : 'Close reading tools');
  close.addEventListener('click', () => { returnFocus = true; popover.hidePopover(); });
  header.append(title, close);
  popover.append(header);
  trigger.dataset.pagefindIgnore = '';
  popover.dataset.pagefindIgnore = '';
  document.body.append(trigger, popover);
  let stopPositioning: (() => void) | undefined;
  const isOpen = () => popover.matches(':popover-open');
  const visibility = () => {
    const hidden = !isOpen() && host.getBoundingClientRect().bottom > 0;
    if (trigger.hidden !== hidden) trigger.hidden = hidden;
  };
  new IntersectionObserver(visibility).observe(host);
  // A bookmark can jump from below the viewport to above it without crossing an
  // IntersectionObserver threshold. Track those jumps as well as ordinary scrolling.
  let scheduled = false;
  window.addEventListener('scroll', () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => { scheduled = false; visibility(); });
  }, { passive: true });
  window.addEventListener('resize', visibility);
  popover.addEventListener('beforetoggle', (event) => {
    const opening = (event as ToggleEvent).newState === 'open';
    trigger.setAttribute('aria-expanded', String(opening));
    if (opening) {
      Object.assign(popover.style, { left: '', top: '', right: '', bottom: '', maxHeight: '' });
      const trackColor = getComputedStyle(toolbar).getPropertyValue('--tc');
      if (trackColor) popover.style.setProperty('--tc', trackColor);
      const navigation = document.querySelector<HTMLDetailsElement>('.rt-mnav');
      if (navigation) navigation.open = false;
      host.style.height = `${host.getBoundingClientRect().height}px`;
      popover.append(toolbar);
      trigger.hidden = false;
    } else {
      stopPositioning?.();
      stopPositioning = undefined;
      const restoreFocus = returnFocus;
      returnFocus = false;
      host.append(toolbar);
      host.style.height = '';
      requestAnimationFrame(() => {
        visibility();
        if (restoreFocus) (trigger.hidden ? toolbar.querySelector('button') : trigger)?.focus({ preventScroll: true });
      });
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && isOpen()) returnFocus = true;
  });
  popover.addEventListener('toggle', async () => {
    if (!isOpen()) return;
    close.focus({ preventScroll: true });
    try {
      const { computePosition, autoUpdate, offset, flip, shift, size } = await import('@floating-ui/dom');
      if (!isOpen()) return;
      stopPositioning?.();
      stopPositioning = autoUpdate(trigger, popover, async () => {
        const { x, y } = await computePosition(trigger, popover, {
          strategy: 'fixed', placement: 'top-end',
          middleware: [offset(12), flip({ padding: 12 }), shift({ padding: 12 }), size({ padding: 12,
            apply({ availableHeight, elements }) { elements.floating.style.maxHeight = `${Math.max(0, availableHeight)}px`; },
          })],
        });
        if (isOpen()) Object.assign(popover.style, { left: `${x}px`, top: `${y}px`, right: 'auto', bottom: 'auto' });
      });
    } catch {
      // The CSS viewport-bounded placement remains usable if the optional chunk cannot load.
    }
  });
  window.addEventListener('beforeprint', () => { if (isOpen()) popover.hidePopover(); });
}
