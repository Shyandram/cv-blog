/** Single-channel image, values nominally in [0, 1], row-major. */
export interface Gray { w: number; h: number; data: Float32Array }
/** Three-channel image, interleaved R, G, B, values nominally in [0, 1]. */
export interface Rgb { w: number; h: number; data: Float32Array }

export const gray = (w: number, h: number, data = new Float32Array(w * h)): Gray => ({ w, h, data });
export const clip01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

export function cloneGray(g: Gray): Gray { return { w: g.w, h: g.h, data: Float32Array.from(g.data) }; }
export function mapGray(g: Gray, f: (v: number, i: number) => number): Gray {
  const out = new Float32Array(g.data.length);
  for (let i = 0; i < out.length; i++) out[i] = f(g.data[i], i);
  return { w: g.w, h: g.h, data: out };
}
/** Luma with the ITU-R BT.601 weights, as used by OpenCV's gray conversion. */
export function rgbToGray(c: Rgb): Gray {
  const out = new Float32Array(c.w * c.h);
  for (let i = 0; i < out.length; i++) out[i] = 0.299 * c.data[3 * i] + 0.587 * c.data[3 * i + 1] + 0.114 * c.data[3 * i + 2];
  return { w: c.w, h: c.h, data: out };
}
/** Crop a rectangle (no bounds padding: callers pass valid ranges). */
export function crop(g: Gray, x0: number, y0: number, w: number, h: number): Gray {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = g.data[(y0 + y) * g.w + x0 + x];
  return { w, h, data: out };
}
