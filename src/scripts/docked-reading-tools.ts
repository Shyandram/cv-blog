// Chapter pages: the "collapse all sections" and "expand all code" actions live as two quiet buttons in the chapter
// header (.reader-dock) instead of a floating console. The buttons are in the HTML but hidden until a group registers,
// so without JavaScript nothing inert is shown. The label always names what a click will do.
export function setupDockedReadingTools(dock: HTMLElement) {
  const zh = document.documentElement.lang.startsWith('zh');
  return {
    register(kind: 'sections' | 'code', folds: HTMLDetailsElement[], setOpen: (open: boolean) => void) {
      const button = dock.querySelector<HTMLButtonElement>(`.reader-console-fold-${kind}`);
      if (!button || !folds.length) return;
      const label = button.querySelector<HTMLElement>('.rd-label') ?? button;
      const sync = () => {
        const count = folds.filter((fold) => fold.open).length;
        const expanded = count > 0;
        const subject = kind === 'sections' ? (zh ? '章節' : 'sections') : (zh ? '程式碼' : 'code');
        label.textContent = zh ? `${expanded ? '收合' : '展開'}全部${subject}` : `${expanded ? 'Collapse' : 'Expand'} all ${subject}`;
        button.dataset.expanded = String(expanded);
        button.title = zh ? `已展開 ${count}／${folds.length}` : `${count} of ${folds.length} open`;
      };
      button.addEventListener('click', () => { setOpen(!folds.some((fold) => fold.open)); sync(); });
      // One delegated listener per group keeps the label truthful when a single fold is toggled.
      let queued = false;
      document.addEventListener('toggle', (event) => {
        if (queued || !(event.target instanceof HTMLDetailsElement) || !folds.includes(event.target)) return;
        queued = true;
        requestAnimationFrame(() => { queued = false; sync(); });
      }, true);
      sync();
      button.hidden = false;
      dock.hidden = false;
    },
  };
}
