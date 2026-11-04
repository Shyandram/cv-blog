import { useT } from '../i18n/ui';

const t = useT(document.documentElement.lang.startsWith('zh') ? 'zh-tw' : 'en');
const prose = document.querySelector<HTMLElement>('.prose');
const folds = [...document.querySelectorAll<HTMLDetailsElement>('.prose .code-fold')];
if (prose && folds.length) {
  const status = document.createElement('p');
  status.className = 'sr-only';
  status.setAttribute('role', 'status');
  status.dataset.pagefindIgnore = '';
  prose.after(status);

  const openParents = (target: Element) => {
    for (let parent = target.parentElement; parent; parent = parent.parentElement) {
      if (parent instanceof HTMLDetailsElement) parent.open = true;
    }
  };
  const controls = document.createElement('div');
  controls.className = 'code-controls';
  controls.dataset.pagefindIgnore = '';
  const help = document.createElement('p');
  help.textContent = t('tut.codeHelp');
  controls.append(help);
  for (const [label, open] of [[t('tut.codeExpand'), true], [t('tut.codeCollapse'), false]] as const) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.addEventListener('click', () => folds.forEach((fold) => {
      if (open) openParents(fold);
      fold.open = open;
    }));
    controls.append(button);
  }
  prose.before(controls);

  folds.forEach((fold) => {
    const pre = fold.querySelector('pre');
    const wrap = fold.parentElement;
    if (!pre || !wrap) return;
    // textContent preserves indentation and line breaks even while the code is hidden.
    const source = (pre.querySelector('code') ?? pre).textContent?.replace(/\n$/, '') ?? '';
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'rt-copy';
    button.textContent = t('tut.copy');
    button.setAttribute('aria-label', t('tut.copyCode'));
    button.dataset.pagefindIgnore = '';
    let timer: number | undefined;
    button.addEventListener('click', async () => {
      let message = t('tut.copied');
      try { await navigator.clipboard.writeText(source); }
      catch { message = t('tut.copyFailed'); fold.open = true; openParents(fold); pre.focus(); }
      button.textContent = message;
      status.textContent = '';
      window.setTimeout(() => { status.textContent = message; }, 50);
      clearTimeout(timer);
      timer = window.setTimeout(() => { button.textContent = t('tut.copy'); }, 1600);
    });
    wrap.classList.add('has-copy');
    wrap.append(button);
  });

  // Shared by chapters, optional-reading pages and posts, including direct bookmarks.
  const reveal = () => {
    let id: string;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    const target = document.getElementById(id);
    if (target?.closest('.code-fold')) {
      openParents(target);
      requestAnimationFrame(() => target.scrollIntoView());
    }
  };
  window.addEventListener('hashchange', reveal);
  reveal();
}
