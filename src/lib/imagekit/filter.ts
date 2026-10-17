import type { Gray } from './types.ts';

/** Index into [0, n) with scipy's 'reflect' boundary (d c b a | a b c d | d c b a). */
export function reflectIndex(i: number, n: number): number {
  const period = 2 * n;
  let k = ((i % period) + period) % period;
  return k < n ? k : period - 1 - k;
}

export function gaussianKernel1d(sigma: number, truncate = 4): Float64Array {
  const radius = Math.max(1, Math.floor(truncate * sigma + 0.5));
  const k = new Float64Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) { k[i + radius] = Math.exp(-(i * i) / (2 * sigma * sigma)); sum += k[i + radius]; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return k;
}

/** Separable Gaussian blur with reflected borders, matching scipy.ndimage.gaussian_filter (truncate 4). */
export function gaussianBlur(src: Float32Array, w: number, h: number, sigma: number): Float32Array {
  if (sigma <= 0) return Float32Array.from(src);
  const k = gaussianKernel1d(sigma), r = (k.length - 1) >> 1;
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let i = -r; i <= r; i++) s += k[i + r] * src[y * w + reflectIndex(x + i, w)];
    tmp[y * w + x] = s;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = -r; j <= r; j++) s += k[j + r] * tmp[reflectIndex(y + j, h) * w + x];
    out[y * w + x] = s;
  }
  return out;
}

/** Median of the 3×3 neighbourhood with edge replication. */
export function median3(src: Float32Array, w: number, h: number): Float32Array {
  const out = new Float32Array(w * h), win = new Float32Array(9);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let n = 0;
    for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++)
      win[n++] = src[Math.min(h - 1, Math.max(0, y + j)) * w + Math.min(w - 1, Math.max(0, x + i))];
    win.sort();
    out[y * w + x] = win[4];
  }
  return out;
}

/** Sobel gradient with edge replication; `ix` and `iy` are derivatives along x (columns) and y (rows), unscaled. */
export function sobel(src: Float32Array, w: number, h: number): { ix: Float32Array; iy: Float32Array } {
  const ix = new Float32Array(w * h), iy = new Float32Array(w * h);
  const at = (x: number, y: number) => src[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    ix[y * w + x] = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
    iy[y * w + x] = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
  }
  return { ix, iy };
}

/** Box-average downsample by an integer factor (used to make small demo images from larger ones). */
export function downsample(g: Gray, factor: number): Gray {
  const w = Math.floor(g.w / factor), h = Math.floor(g.h / factor), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = 0; j < factor; j++) for (let i = 0; i < factor; i++) s += g.data[(y * factor + j) * g.w + x * factor + i];
    out[y * w + x] = s / (factor * factor);
  }
  return { w, h, data: out };
}
