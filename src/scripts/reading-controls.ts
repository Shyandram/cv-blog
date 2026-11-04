import { setupFloatingReadingTools } from './floating-reading-tools';
const consoles = new WeakMap<HTMLElement, ReturnType<typeof setupFloatingReadingTools>>();

// Register existing chapter/code behavior with one persistent console.
export function mountReadingControls(prose: HTMLElement, folds: HTMLDetailsElement[], kind: 'sections' | 'code', setOpen: (open: boolean) => void) {
  let console = consoles.get(prose);
  if (!console) {
    console = setupFloatingReadingTools(prose);
    consoles.set(prose, console);
  }
  console.register(kind, folds, setOpen);
}
