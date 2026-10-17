import type { Lang } from '../../../i18n/ui';
import type { Track } from '../../taxonomy';
import type { Gray, Rgb } from '../../imagekit/types.ts';
import type { Control, Localized, Predict, State } from '../types.ts';

/**
 * An image lab runs a real algorithm on a real (or synthetic) image in the browser and draws the result to canvases.
 * Metadata (this interface minus `create`) is used at build time to render the card; `create` only runs in the browser.
 * A lab has at most three controls and always a predict step: see AGENTS.md and docs/playground.md.
 */
export interface LabPane {
  key: string;
  kind: 'canvas' | 'svg';
  label: Localized;
  /** Intrinsic canvas size (used for the aspect ratio before the lab has drawn); for svg panes the viewBox height at width 320. */
  width: number;
  height: number;
  /** Span the full row (used for plots). */
  wide?: boolean;
}
export interface LabOutput {
  summary: string;
  metrics: { label: string; value: string }[];
  /** SVG markup for each `svg` pane, keyed by pane key. */
  plots?: Record<string, string>;
}
export interface LabInstance {
  render(state: State, canvases: Record<string, HTMLCanvasElement>): LabOutput;
  /**
   * Optional direct manipulation (click or drag on a canvas). Called once, before the first render. Use `setControl` to
   * write the result into a control (for example the x and y sliders of a probe), so the keyboard-accessible sliders stay
   * the single source of truth and a pointer is only a shortcut.
   */
  attach?(canvases: Record<string, HTMLCanvasElement>, setControl: (key: string, value: string | number) => void): void;
}
export interface LabEnv { lang: Lang; images: Record<string, Gray | Rgb> }
export interface LabDef {
  kind: 'image';
  id: string;
  track: Track;
  title: Localized;
  description: Localized;
  challenge: Localized;
  note: Localized;
  related: string[];
  controls: Control[];
  predict: Predict;
  /** Asset key (see assets.ts) to decoding mode. */
  assets: Record<string, 'gray' | 'rgb'>;
  panes: LabPane[];
  /** Minimum width in px of one canvas pane; larger values give fewer, bigger canvases per row (default 220). */
  paneMin?: number;
  /** Extra method sources shown under the card, in addition to the test-image credit. */
  sources: { title: string; url: string }[];
  create(env: LabEnv): LabInstance;
}
