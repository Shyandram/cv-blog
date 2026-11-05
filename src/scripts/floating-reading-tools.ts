import { load, save } from './progress';

// A persistent, modeless tool window. It owns the original controls and never
// dismisses when the reader interacts with the article.
export function setupFloatingReadingTools(prose: HTMLElement) {
  const zh = document.documentElement.lang.startsWith('zh');
  const key = 'pp:reading-console';
  const clamp = (n: number, min = 0, max = 1) => Math.min(max, Math.max(min, n));
  let stored: { x?: number; y?: number } = {};
  try { stored = JSON.parse(load(key) ?? '{}') ?? {}; } catch { /* defaults below */ }
  let x = typeof stored.x === 'number' && Number.isFinite(stored.x) ? clamp(stored.x) : 1;
  let y = typeof stored.y === 'number' && Number.isFinite(stored.y) ? clamp(stored.y) : 1;
  let collapsed = true;
  const remember = () => save(key, JSON.stringify({ x, y }));
  const panel = document.createElement('aside');
  panel.className = 'reader-console';
  panel.setAttribute('aria-label', zh ? '閱讀控制台' : 'Reading console');
  panel.dataset.pagefindIgnore = '';
  const track = prose.closest<HTMLElement>('[style*="--tc"]')?.style.getPropertyValue('--tc');
  if (track) panel.style.setProperty('--tc', track);
  const header = document.createElement('div');
  header.className = 'reader-console-header';
  const handle = document.createElement('button');
  handle.type = 'button';
  handle.className = 'reader-console-handle';
  handle.innerHTML = '<svg viewBox="0 0 16 24" width="16" height="24" fill="currentColor" aria-hidden="true"><circle cx="5" cy="6" r="1.4"/><circle cx="11" cy="6" r="1.4"/><circle cx="5" cy="12" r="1.4"/><circle cx="11" cy="12" r="1.4"/><circle cx="5" cy="18" r="1.4"/><circle cx="11" cy="18" r="1.4"/></svg>';
  handle.setAttribute('aria-label', zh ? '移動控制台：方向鍵移動，Home 歸位' : 'Move console: arrow keys to move, Home to reset');
  handle.title = zh ? '拖曳移動；方向鍵微調，Home 歸位' : 'Drag to move; arrow keys to adjust, Home to reset';
  // Static outline icons share a viewBox and stroke; names remain available to assistive technology.
  const icons = {
    contents: '<path d="M9 6h11M9 12h11M9 18h11"/><circle cx="4" cy="6" r="1"/><circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/>',
    collapse: '<path d="m8 3 4 4 4-4M8 21l4-4 4 4M4 12h16"/>',
    expand: '<path d="m8 7 4-4 4 4m-8 10 4 4 4-4M4 12h16"/>',
    code: '<path d="m8 7-5 5 5 5m8-10 5 5-5 5M14 4l-4 16"/>',
    top: '<path d="M5 4h14M12 20V8m-6 6 6-6 6 6"/>',
  };
  const iconMarkup = (kind: keyof typeof icons) => `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${icons[kind]}</svg>`;
  const action = (name: string, symbol: keyof typeof icons, className: string) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = className;
    const icon = document.createElement('span');
    icon.className = 'reader-console-icon';
    icon.innerHTML = iconMarkup(symbol);
    icon.setAttribute('aria-hidden', 'true');
    const label = document.createElement('span');
    label.className = 'reader-console-label sr-only';
    label.textContent = name;
    button.append(icon, label);
    button.title = name;
    return button;
  };
  const toggle = action(zh ? '目錄' : 'Contents', 'contents', 'reader-console-toggle');
  toggle.setAttribute('aria-controls', 'reader-console-body');
  const sectionsButton = action(zh ? '章節' : 'Sections', 'collapse', 'reader-console-fold-sections');
  const codeButton = action(zh ? '程式碼' : 'Code', 'code', 'reader-console-fold-code');
  sectionsButton.disabled = true;
  codeButton.disabled = true;
  const top = action(zh ? '頁首' : 'Top', 'top', 'reader-console-top');
  top.setAttribute('aria-label', zh ? '回到頁首' : 'Back to top');
  top.title = top.getAttribute('aria-label')!;
  header.append(handle, toggle, sectionsButton, codeButton, top);
  const body = document.createElement('div');
  body.id = 'reader-console-body';
  body.className = 'reader-console-body';
  const progressRow = document.createElement('div');
  progressRow.className = 'reader-console-progress';
  const progressLabel = document.createElement('span');
  progressLabel.textContent = zh ? '閱讀進度' : 'Reading progress';
  const percent = document.createElement('span');
  const progress = document.createElement('progress');
  progress.max = 100;
  progress.value = 0;
  progress.setAttribute('aria-label', progressLabel.textContent);
  progressRow.append(progressLabel, percent, progress);
  const select = document.createElement('select');
  select.className = 'reader-console-sections';
  select.setAttribute('aria-label', zh ? '跳至章節' : 'Jump to section');
  const nav = document.createElement('div');
  nav.className = 'reader-console-nav';
  const previous = document.createElement('button');
  previous.type = 'button';
  previous.textContent = zh ? '← 上一節' : '← Previous';
  const dock = document.createElement('button');
  dock.type = 'button';
  dock.className = 'reader-console-dock';
  dock.textContent = zh ? '控制台歸位' : 'Reset position';
  const next = document.createElement('button');
  next.type = 'button';
  next.textContent = zh ? '下一節 →' : 'Next →';
  nav.append(previous, dock, next);
  const title = document.createElement('strong');
  title.className = 'reader-console-title';
  title.textContent = zh ? '閱讀控制台' : 'Reading console';
  body.append(title, progressRow, select, nav);
  panel.append(body, header);
  document.body.append(panel);

  const bounds = () => {
    const viewport = window.visualViewport;
    const style = getComputedStyle(panel);
    const edge = (side: string) => Math.max(12, parseFloat(style.getPropertyValue(`--console-safe-${side}`)) || 0);
    const left = (viewport?.offsetLeft ?? 0) + edge('left');
    const upper = (viewport?.offsetTop ?? 0) + edge('top');
    const width = Math.max(0, (viewport?.width ?? innerWidth) - edge('left') - edge('right'));
    const height = Math.max(0, (viewport?.height ?? innerHeight) - edge('top') - edge('bottom'));
    panel.style.maxWidth = `${width}px`;
    panel.style.maxHeight = `${height}px`;
    return { left, top: upper, dx: Math.max(0, width - panel.offsetWidth), dy: Math.max(0, height - panel.offsetHeight) };
  };
  const place = () => {
    if (matchMedia('print').matches) return;
    panel.style.right = 'auto';
    panel.style.bottom = 'auto';
    const b = bounds();
    panel.style.left = `${b.left + x * b.dx}px`;
    panel.style.top = `${b.top + y * b.dy}px`;
  };
  const setCollapsed = (value: boolean, persist = true) => {
    collapsed = value;
    body.hidden = value;
    panel.classList.toggle('is-collapsed', value);
    toggle.setAttribute('aria-label', value ? (zh ? '展開章節目錄' : 'Open section contents') : (zh ? '收合章節目錄' : 'Close section contents'));
    toggle.title = toggle.getAttribute('aria-label')!;
    toggle.setAttribute('aria-expanded', String(!value));
    place();
    if (persist) remember();
  };
  toggle.addEventListener('click', () => setCollapsed(!collapsed));
  const reset = () => { x = 1; y = 1; place(); remember(); };
  dock.addEventListener('click', reset);
  panel.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !collapsed) {
      event.stopPropagation();
      setCollapsed(true);
      toggle.focus({ preventScroll: true });
    }
  });
  let drag: { id: number; left: number; top: number; pointerX: number; pointerY: number } | undefined;
  const moveTo = (left: number, upper: number) => {
    const b = bounds();
    x = b.dx ? clamp((left - b.left) / b.dx) : 1;
    y = b.dy ? clamp((upper - b.top) / b.dy) : 1;
    place();
  };
  handle.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 || !event.isPrimary) return;
    event.preventDefault();
    handle.focus({ preventScroll: true });
    const rect = panel.getBoundingClientRect();
    drag = { id: event.pointerId, left: rect.left, top: rect.top, pointerX: event.clientX, pointerY: event.clientY };
    handle.setPointerCapture(event.pointerId);
    panel.classList.add('is-dragging');
  });
  handle.addEventListener('pointermove', (event) => {
    if (!drag || event.pointerId !== drag.id) return;
    moveTo(drag.left + event.clientX - drag.pointerX, drag.top + event.clientY - drag.pointerY);
  });
  const endDrag = () => { if (drag) remember(); drag = undefined; panel.classList.remove('is-dragging'); };
  handle.addEventListener('pointerup', endDrag);
  handle.addEventListener('pointercancel', endDrag);
  handle.addEventListener('lostpointercapture', endDrag);
  handle.addEventListener('keydown', (event) => {
    if (event.key === 'Home') { event.preventDefault(); reset(); return; }
    const delta: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    const direction = delta[event.key];
    if (!direction) return;
    event.preventDefault();
    const rect = panel.getBoundingClientRect();
    const step = event.shiftKey ? 48 : 16;
    moveTo(rect.left + direction[0] * step, rect.top + direction[1] * step);
    remember();
  });

  let headings: HTMLElement[] = [];
  let active = 0;
  const update = () => {
    const found = [...document.querySelectorAll<HTMLElement>('.prose .chapter-section > summary > :is(h2,h3,h4,h5,h6)')];
    const items = found.length ? found : [...document.querySelectorAll<HTMLElement>('.prose h2[id]')];
    if (items.length !== headings.length || items.some((h, i) => h !== headings[i])) {
      headings = items;
      select.replaceChildren(...headings.map((heading, i) => new Option(`${i + 1}. ${heading.textContent?.trim() ?? ''}`, String(i))));
    }
    select.disabled = !headings.length;
    const headerHeight = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--header-h')) || 0;
    const mobileBar = document.querySelector<HTMLElement>('.rt-mnav > summary');
    const line = Math.max(headerHeight, mobileBar?.offsetParent ? mobileBar.getBoundingClientRect().bottom : 0) + 24;
    active = 0;
    headings.forEach((heading, i) => { if (heading.getClientRects().length && heading.getBoundingClientRect().top <= line) active = i; });
    if (headings.length) select.value = String(active);
    previous.disabled = !headings.length || active === 0;
    next.disabled = !headings.length || active === headings.length - 1;
    const max = document.documentElement.scrollHeight - innerHeight;
    const ratio = Math.round(max > 0 ? clamp(scrollY / max) * 100 : 0);
    percent.textContent = `${ratio}%`;
    progress.value = ratio;
  };
  const jump = (index: number) => {
    const target = headings[index];
    if (!target) return;
    for (let parent = target.parentElement; parent; parent = parent.parentElement) {
      if (parent instanceof HTMLDetailsElement) parent.open = true;
    }
    target.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    if (target.id) history.replaceState(null, '', `#${encodeURIComponent(target.id)}`);
    update();
  };
  select.addEventListener('change', () => jump(Number(select.value)));
  previous.addEventListener('click', () => jump(active - 1));
  next.addEventListener('click', () => jump(active + 1));
  top.addEventListener('click', () => window.scrollTo({ top: 0, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' }));
  let scheduled = false;
  const schedule = () => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => { scheduled = false; update(); });
  };
  window.addEventListener('scroll', schedule, { passive: true });
  const resize = () => { place(); schedule(); };
  window.addEventListener('resize', resize);
  window.visualViewport?.addEventListener('resize', resize);
  window.visualViewport?.addEventListener('scroll', place);
  document.addEventListener('toggle', schedule, true);
  document.addEventListener('DOMContentLoaded', schedule);
  document.fonts.ready.then(schedule);
  new ResizeObserver(place).observe(panel);
  setCollapsed(collapsed, false);
  schedule();
  return {
    register(kind: 'sections' | 'code', folds: HTMLDetailsElement[], setOpen: (open: boolean) => void) {
      const button = kind === 'sections' ? sectionsButton : codeButton;
      const sync = () => {
        const count = folds.filter((fold) => fold.open).length;
        const expanded = count > 0;
        const subject = kind === 'sections' ? (zh ? '章節' : 'sections') : (zh ? '程式碼' : 'code');
        const label = zh ? `${expanded ? '收合' : '展開'}全部${subject}` : `${expanded ? 'Collapse' : 'Expand'} all ${subject}`;
        button.disabled = !folds.length;
        button.dataset.expanded = String(expanded);
        button.setAttribute('aria-label', `${label} (${count}/${folds.length})`);
        button.setAttribute('aria-pressed', count === 0 ? 'false' : count === folds.length ? 'true' : 'mixed');
        button.title = label;
        button.querySelector('.reader-console-label')!.textContent = zh ? (kind === 'sections' ? '章節' : '程式碼') : (kind === 'sections' ? 'Sections' : 'Code');
        if (kind === 'sections') button.querySelector('.reader-console-icon')!.innerHTML = iconMarkup(expanded ? 'collapse' : 'expand');
      };
      button.addEventListener('click', () => { setOpen(!folds.some((fold) => fold.open)); sync(); });
      folds.forEach((fold) => fold.addEventListener('toggle', sync));
      sync();
      schedule();
    },
  };
}
