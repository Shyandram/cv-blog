import type { Gray, Rgb } from './types.ts';

function paint(canvas: HTMLCanvasElement, w: number, h: number, fill: (px: Uint8ClampedArray) => void) {
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  const im = ctx.createImageData(w, h);
  fill(im.data);
  ctx.putImageData(im, 0, 0);
}
const byte = (v: number) => Math.max(0, Math.min(255, Math.round(v * 255)));

/** Draw a gray image. `signed` maps [-1, 1] to [0, 1] so that zero is mid-gray. */
export function drawGray(canvas: HTMLCanvasElement, g: Gray | { w: number; h: number; data: ArrayLike<number> }, signed = false) {
  paint(canvas, g.w, g.h, px => {
    for (let i = 0; i < g.w * g.h; i++) {
      const v = byte(signed ? 0.5 + 0.5 * g.data[i] : g.data[i]);
      px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = v; px[4 * i + 3] = 255;
    }
  });
}
export function drawRgb(canvas: HTMLCanvasElement, c: Rgb) {
  paint(canvas, c.w, c.h, px => {
    for (let i = 0; i < c.w * c.h; i++) {
      px[4 * i] = byte(c.data[3 * i]); px[4 * i + 1] = byte(c.data[3 * i + 1]); px[4 * i + 2] = byte(c.data[3 * i + 2]); px[4 * i + 3] = 255;
    }
  });
}
/** Draw a boolean mask as black (foreground) on white, or as a colored overlay of `base` when given. */
export function drawMask(canvas: HTMLCanvasElement, w: number, h: number, mask: ArrayLike<boolean | number>, base?: ArrayLike<number>, rgb: [number, number, number] = [214, 39, 40]) {
  paint(canvas, w, h, px => {
    for (let i = 0; i < w * h; i++) {
      const on = !!mask[i];
      if (base) {
        const g = byte(base[i]);
        px[4 * i] = on ? Math.round(0.45 * g + 0.55 * rgb[0]) : g; px[4 * i + 1] = on ? Math.round(0.45 * g + 0.55 * rgb[1]) : g; px[4 * i + 2] = on ? Math.round(0.45 * g + 0.55 * rgb[2]) : g;
      } else { const v = on ? 20 : 245; px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = v; }
      px[4 * i + 3] = 255;
    }
  });
}
/** Highlight pixels whose value lies outside [lo, hi] in red or blue over a dimmed copy of the image. */
export function drawOutOfRange(canvas: HTMLCanvasElement, g: Gray, lo: number, hi: number, tol = 0.02) {
  paint(canvas, g.w, g.h, px => {
    for (let i = 0; i < g.w * g.h; i++) {
      const v = g.data[i], base = byte(0.35 + 0.5 * Math.max(0, Math.min(1, v)));
      const over = v > hi + tol, under = v < lo - tol;
      px[4 * i] = over ? 230 : under ? 40 : base; px[4 * i + 1] = over ? 60 : under ? 90 : base; px[4 * i + 2] = over ? 40 : under ? 230 : base; px[4 * i + 3] = 255;
    }
  });
}
