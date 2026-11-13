import { setupFloatingReadingTools } from './floating-reading-tools';
import { setupDockedReadingTools } from './docked-reading-tools';
type Tools = { register(kind: 'sections' | 'code', folds: HTMLDetailsElement[], setOpen: (open: boolean) => void): void };
const consoles = new WeakMap<HTMLElement, Tools>();

// Register existing chapter/code behavior with one set of controls: the buttons docked in a chapter page's header
// (.reader-dock, see src/pages/[lang]/learn/[...slug].astro), or else one persistent floating console (other prose pages).
export function mountReadingControls(prose: HTMLElement, folds: HTMLDetailsElement[], kind: 'sections' | 'code', setOpen: (open: boolean) => void) {
  let console = consoles.get(prose);
  if (!console) {
    const dock = document.querySelector<HTMLElement>('.reader-dock');
    console = dock ? setupDockedReadingTools(dock) : setupFloatingReadingTools(prose);
    consoles.set(prose, console);
  }
  console.register(kind, folds, setOpen);
}
