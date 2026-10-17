// Small canvas plotting helpers for labs whose plots must stay legible at phone width (a canvas pane scales with its column,
// while an svg pane keeps a 460 px minimum width). Colors follow the page theme through the --plot-* custom properties.
export interface PlotColors { ink: string; muted: string; grid: string; bg: string; a: string; b: string }

export function plotColors(canvas: HTMLCanvasElement): PlotColors {
  const cs = getComputedStyle(canvas), get = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback;
  return {
    ink: get('--plot-ink', '#1b1b1b'), muted: get('--plot-muted', '#6b6b6b'), grid: get('--plot-grid', '#c9c9c9'),
    bg: get('--plot-bg', '#ffffff'), a: get('--plot-a', '#2466c2'), b: get('--plot-b', '#c35b25'),
  };
}
/** Resize (only when needed), clear to the theme background and return the context. */
export function startPlot(canvas: HTMLCanvasElement, w: number, h: number, colors: PlotColors): CanvasRenderingContext2D {
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  canvas.style.imageRendering = 'auto';
  const ctx = canvas.getContext('2d')!;
  ctx.setLineDash([]); ctx.globalAlpha = 1; ctx.fillStyle = colors.bg; ctx.fillRect(0, 0, w, h);
  return ctx;
}
export interface TextOpts { size?: number; bold?: boolean; align?: CanvasTextAlign; color: string; halo?: string }
export function label(ctx: CanvasRenderingContext2D, s: string, x: number, y: number, o: TextOpts) {
  ctx.font = `${o.bold ? '600 ' : ''}${o.size ?? 15}px system-ui, -apple-system, 'Segoe UI', 'Noto Sans TC', sans-serif`;
  ctx.textAlign = o.align ?? 'left'; ctx.textBaseline = 'alphabetic';
  if (o.halo) { ctx.lineWidth = 4; ctx.strokeStyle = o.halo; ctx.lineJoin = 'round'; ctx.strokeText(s, x, y); }
  ctx.fillStyle = o.color; ctx.fillText(s, x, y);
}
export function line(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, color: string, width = 1, dash: number[] = []) {
  ctx.setLineDash(dash); ctx.lineWidth = width; ctx.strokeStyle = color; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); ctx.setLineDash([]);
}
