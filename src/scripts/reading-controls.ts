// Both enhancers can mount first; keep one predictable toolbar beside the article.
import { setupFloatingReadingTools } from './floating-reading-tools';
const toolbars = new WeakMap<HTMLElement, HTMLElement>();
export function mountReadingControls(prose: HTMLElement, controls: HTMLElement, folds: HTMLDetailsElement[], kind: 'sections' | 'code') {
  const zh = document.documentElement.lang.startsWith('zh');
  let toolbar = toolbars.get(prose);
  if (!toolbar) {
    toolbar = document.createElement('div');
    toolbar.className = 'reader-toolbar';
    toolbar.dataset.pagefindIgnore = '';
    toolbar.setAttribute('role', 'group');
    toolbar.setAttribute('aria-label', zh ? '閱讀工具' : 'Reading tools');
    prose.before(toolbar);
    toolbars.set(prose, toolbar);
    setupFloatingReadingTools(toolbar);
  }
  const title = kind === 'sections' ? (zh ? '章節' : 'Sections') : (zh ? '程式碼' : 'Code');
  controls.setAttribute('role', 'group');
  const heading = document.createElement('div');
  heading.className = 'reader-control-heading';
  const name = document.createElement('strong');
  name.textContent = title;
  const count = document.createElement('span');
  count.className = 'reader-count';
  heading.append(name, count);
  controls.prepend(heading);
  controls.querySelectorAll('button').forEach((button, i) => {
    button.setAttribute('aria-label', button.textContent ?? '');
    button.textContent = i === 0 ? (zh ? '展開' : 'Expand') : (zh ? '收合' : 'Collapse');
  });
  const update = () => {
    const open = folds.filter((fold) => fold.open).length;
    const state = zh ? `${open} / ${folds.length} 已展開` : `${open} / ${folds.length} open`;
    count.textContent = state;
    controls.setAttribute('aria-label', `${title} · ${state}`);
  };
  folds.forEach((fold) => fold.addEventListener('toggle', update));
  update();
  // Sections first, code second, regardless of script execution order.
  if (kind === 'sections') toolbar.prepend(controls);
  else toolbar.append(controls);
}
